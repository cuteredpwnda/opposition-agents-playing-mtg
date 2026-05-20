"""
High-level game runner — sets up and runs a complete MTG game.

Coordinates agent initialization, game state setup, deck loading,
and the main game loop via LangGraph.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import MTGAgent
from src.engine.card_database import CardDatabase
from src.engine.game_state import CardInstance, GameState, Phase, PlayerState, Zone
from src.engine.rules_engine import RulesEngine
from src.orchestrator.priority_loop import (
    advance_priority,
    get_priority_order,
    priority_action_result,
    run_priority_loop,
)

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for a game."""

    format: str = "commander"  # "commander" | "standard"
    starting_life: int = 40
    max_turns: int = 100
    mulligan_enabled: bool = True
    max_mulligans: int = 3
    # Optional WotC Commander Bracket (1=Exhibition .. 5=cEDH); informational
    # only at runtime — used by deck loaders / matchmaking, not the rules.
    bracket: int | None = None
    # Optional callback invoked at the start of each CLEANUP phase, before
    # watcher state is reset.  Receives the live GameState so callers can
    # snapshot per-turn data (e.g. GoldfishRunner collects board stats here).
    on_turn_end: Callable[["GameState"], None] | None = None


@dataclass
class GameResult:
    """Result of a completed game."""

    winner: str | None
    turns: int
    game_over: bool = False
    log: list[str] = field(default_factory=list)


class GameRunner:
    """Runs a complete MTG game between agents."""

    def __init__(
        self,
        config: GameConfig | None = None,
        card_db: CardDatabase | None = None,
        self_play_collector: object | None = None,
    ):
        self.config = config or GameConfig()
        self.card_db = card_db or CardDatabase()
        self.engine = RulesEngine()
        self.self_play_collector = self_play_collector
        self._tools_kg = None  # cached MTGKnowledgeGraph for tools.set_kg

    def _wire_external_tools(self, agents: dict[str, "MTGAgent"]) -> None:
        """Best-effort: hand the live KG / judge to ``src.agents.tools``
        so LangChain-style ``query_knowledge_graph`` / ``judge_question``
        tools resolve to real data while the game runs.

        Silent on failure — agents that need the KG already fall back
        to non-KG behaviour, and we don't want a game to crash because
        Neo4j is offline.
        """
        # Reuse a KG handle from any agent that already has one.
        kg_handle = None
        for agent in agents.values():
            cand = getattr(agent, "_kg", None) or getattr(agent, "kg", None)
            if cand is not None:
                kg_handle = cand
                break

        if kg_handle is None:
            try:
                from src.knowledge.knowledge_graph import MTGKnowledgeGraph
                kg_handle = MTGKnowledgeGraph()
                self._tools_kg = kg_handle  # so we can close it later if needed
            except Exception as e:
                logger.debug("KG unreachable; tools.set_kg skipped: %s", e)
                return

        try:
            from src.agents import tools as agent_tools
            agent_tools.set_kg(kg_handle)
        except Exception as e:
            logger.debug("set_kg failed: %s", e)

        # Optional: wire judge if available.
        try:
            from src.agents import tools as agent_tools
            if hasattr(agent_tools, "set_judge"):
                from src.judge.judge import Judge  # type: ignore
                agent_tools.set_judge(Judge())
        except Exception:
            pass

    def _resolve_timeout_winner(self, game_state: GameState) -> PlayerState | None:
        """Pick a winner on max-turn timeout; return None if fully tied."""
        if len(game_state.players) < 2:
            return None

        def score(player: PlayerState) -> tuple[int, int, int, int]:
            pid = player.player_id
            battlefield = sum(
                1 for c in game_state.cards
                if c.zone == Zone.BATTLEFIELD and c.controller_id == pid
            )
            hand = sum(
                1 for c in game_state.cards
                if c.zone == Zone.HAND and c.controller_id == pid
            )
            library = sum(
                1 for c in game_state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == pid
            )
            return (player.life_total, battlefield, hand, library)

        p1, p2 = game_state.players[0], game_state.players[1]
        s1, s2 = score(p1), score(p2)
        if s1 > s2:
            return p1
        if s2 > s1:
            return p2
        return None

    def _opening_hand_is_keepable(self, hand_cards: list[CardInstance]) -> bool:
        lands = sum(1 for c in hand_cards if c.is_land())
        non_lands = len(hand_cards) - lands
        return 2 <= lands <= 5 and non_lands >= 1

    def _apply_london_mulligan(
        self,
        cards: list[CardInstance],
        shuffle: bool,
        max_mulligans: int,
        keep_fn=None,
        bottom_fn=None,
    ) -> int:
        import random

        if keep_fn is None:
            def keep_fn(hand, mulligans_taken, max_mulligans):
                return self._opening_hand_is_keepable(hand)

        mulligans_taken = 0
        while True:
            for c in cards:
                if c.zone != Zone.COMMAND_ZONE:
                    c.zone = Zone.LIBRARY

            if shuffle:
                random.shuffle(cards)

            drawables = [c for c in cards if c.zone == Zone.LIBRARY]
            hand = drawables[:7]
            for c in hand:
                c.zone = Zone.HAND

            if mulligans_taken >= max_mulligans or keep_fn(
                    hand, mulligans_taken, max_mulligans):
                break
            mulligans_taken += 1

        if mulligans_taken > 0:
            hand = [c for c in cards if c.zone == Zone.HAND]
            if bottom_fn is not None:
                try:
                    to_bottom = list(bottom_fn(hand, mulligans_taken))[:mulligans_taken]
                except Exception:
                    to_bottom = []
            else:
                to_bottom = []

            if not to_bottom:
                def bottom_priority(card: CardInstance) -> tuple[int, float]:
                    return (0 if card.is_land() else 1, float(card.cmc or 0.0))

                to_bottom = sorted(hand, key=bottom_priority, reverse=True)[:mulligans_taken]

            for card in to_bottom:
                card.zone = Zone.LIBRARY
                if card in cards:
                    cards.remove(card)
                cards.append(card)

        return mulligans_taken

    @staticmethod
    def _agent_mulligan_callbacks(agent):
        """Build (keep_fn, bottom_fn) for an ``MTGAgent`` with a safe fallback."""

        def keep_fn(hand, mulligans_taken: int, max_mulligans: int) -> bool:
            if agent is not None and hasattr(agent, "decide_mulligan"):
                try:
                    return bool(agent.decide_mulligan(
                        hand, mulligans_taken, max_mulligans))
                except Exception:
                    pass
            from src.agents.mulligan import should_keep
            return should_keep(hand,
                               strategy=getattr(agent, "strategy", None),
                               mulligans_taken=mulligans_taken,
                               max_mulligans=max_mulligans)

        def bottom_fn(hand, n):
            if agent is not None and hasattr(agent, "select_bottom_cards"):
                try:
                    return list(agent.select_bottom_cards(hand, n))
                except Exception:
                    pass
            from src.agents.mulligan import select_bottom_cards
            return select_bottom_cards(hand, n,
                                       strategy=getattr(agent, "strategy", None))

        return keep_fn, bottom_fn

    async def run_game(
        self,
        agents: dict[str, MTGAgent],
        decks: dict[str, list[dict[str, Any]]],
    ) -> GameResult:
        """Run a full game to completion with zero external API calls.

        All card data is pre-loaded from decks into the local CardDatabase during setup.
        The game executes deterministically with no external API dependencies after initialization.

        Args:
            agents: mapping of player_id → MTGAgent instance
            decks: mapping of player_id → list of card dicts (from Scryfall)
                   All cards in both decks must be pre-fetched before calling this method.

        Returns:
            GameResult with winner, final state, and turn log

        Note:
            After _setup_game completes, all game logic uses only locally cached card data.
            This ensures reproducible games and prevents API rate limiting during long play sessions.
        """
        game_state = self._setup_game(agents, decks)
        logger.info(f"Game started: {list(agents.keys())}")

        # Make the KG / judge handles available to LangChain-style tools
        # (``src.agents.tools``) for the duration of this game.  Best-effort:
        # if the KG isn't reachable we just skip — agents that need it will
        # fall back to non-KG behaviour.
        self._wire_external_tools(agents)

        while not game_state.game_over and game_state.turn_number <= self.config.max_turns:
            game_state = await self._play_turn(game_state, agents)

        # Max-turn guard for non-terminal games
        if not game_state.game_over:
            game_state.game_over = True
            timeout_winner = self._resolve_timeout_winner(game_state)
            if timeout_winner is not None:
                game_state.winner = timeout_winner
                game_state.log(
                    f"Game ended: max turn limit reached, winner by tie-break = {timeout_winner.player_id}"
                )
            else:
                game_state.log("Game drawn: max turn limit reached (fully tied)")

        winner_name = game_state.winner.player_id if game_state.winner else None

        if self.self_play_collector is not None:
            # Finalize self-play trajectory
            winner_idx = None
            if winner_name is not None:
                player_ids = [p.player_id for p in game_state.players]
                winner_idx = player_ids.index(winner_name) if winner_name in player_ids else None
            self.self_play_collector.finish_game(winner=winner_idx, num_turns=game_state.turn_number)

        result = GameResult(
            winner=winner_name,
            turns=game_state.turn_number,
            game_over=game_state.game_over,
            log=game_state.game_log,
        )
        logger.info(f"Game ended: winner={winner_name}, turns={result.turns}")
        return result

    def _setup_game(
        self,
        agents: dict[str, MTGAgent],
        decks: dict[str, list[dict[str, Any]]],
    ) -> GameState:
        """Initialize game state, shuffle libraries, draw opening hands."""
        import random
        from src.engine.game_state import Zone

        # PHASE 1: Populate card database (zero API calls after this point)
        # Pre-load all cards from both decks into local cache
        for pid, deck_data in decks.items():
            for card_dict in deck_data:
                card_name = card_dict.get("name", "Unknown")
                # Store card data by name for quick lookup
                # (Multiple instances of same card name reference same data)
                if not self.card_db.card_exists(card_name):
                    self.card_db.add_card(card_name, card_dict)

        # Create players list. Use the agent's display name (if any) as the
        # PlayerState name so the human-readable game log shows deck/seat
        # labels like "Krenko · player1 casts Goblin Lackey" rather than
        # the raw player_id everywhere.
        player_ids = list(agents.keys())
        players: list[PlayerState] = []
        for pid in player_ids:
            agent_name = getattr(agents.get(pid), "name", "") or ""
            display = agent_name if agent_name and agent_name != pid else pid
            players.append(
                PlayerState(
                    player_id=pid,
                    name=display,
                    life_total=self.config.starting_life,
                )
            )

        # Load cards for each player (all data from pre-loaded database)
        all_cards: list[CardInstance] = []
        for pid, deck_data in decks.items():
            player_cards: list[CardInstance] = []
            for i, card_dict in enumerate(deck_data):
                card = CardInstance(
                    instance_id=f"{pid}_{i}",
                    card_data=card_dict,  # Use pre-loaded card data
                    zone=Zone.LIBRARY,
                    owner_id=pid,
                    controller_id=pid,
                )
                player_cards.append(card)

            # Commanders: lift every card flagged as a commander (or, by
            # legacy convention, just the deck-index-0 card) into the
            # command zone BEFORE shuffling — otherwise random.shuffle
            # would let pop(0) crown a random card as "the commander",
            # breaking colour-identity checks for the rest of the game.
            #
            # Decklist loaders may flag a card as a commander by setting
            # ``card_dict["is_commander"] = True`` before passing the deck
            # in.  This supports partner / partner-with / friends-forever
            # / background pairs (1 or 2 commanders per player).
            commanders_for_player: list[CardInstance] = []
            if self.config.format == "commander" and player_cards:
                tagged = [c for c in player_cards if c.card_data.get("is_commander")]
                if not tagged:
                    # Legacy fallback: deck index 0 is the (sole) commander.
                    tagged = [player_cards[0]]
                for c in tagged:
                    player_cards.remove(c)
                    c.zone = Zone.COMMAND_ZONE
                    c.card_data["is_commander"] = True
                    commanders_for_player.append(c)

            # Shuffle library (commanders are held aside, not in player_cards yet)
            random.shuffle(player_cards)

            # Re-insert commanders so all_cards/state.cards still contains them.
            for c in commanders_for_player:
                player_cards.insert(0, c)

            player_state = next((p for p in players if p.player_id == pid), None)
            if self.config.mulligan_enabled:
                keep_fn, bottom_fn = self._agent_mulligan_callbacks(agents.get(pid))
                mulligans_taken = self._apply_london_mulligan(
                    player_cards,
                    shuffle=True,
                    max_mulligans=self.config.max_mulligans,
                    keep_fn=keep_fn,
                    bottom_fn=bottom_fn,
                )
                if player_state is not None:
                    player_state.mulligans_taken = mulligans_taken
            else:
                drawables = [c for c in player_cards if c.zone == Zone.LIBRARY]
                for card in drawables[:7]:
                    card.zone = Zone.HAND

            all_cards.extend(player_cards)

        # Create game state
        game_state = GameState(
            format=self.config.format,
            turn_number=1,
            active_player_index=0,
            priority_player_index=0,
            phase=Phase.UNTAP,
            players=players,
            cards=all_cards,
        )

        # Commander post-setup: register every commander instance for each player.
        if self.config.format == "commander":
            for p in players:
                for c in all_cards:
                    if c.owner_id == p.player_id and c.zone == Zone.COMMAND_ZONE:
                        game_state.add_commander(p.player_id, c.instance_id)

        return game_state

    def _tick_sagas(self, game_state: GameState) -> None:
        """At the start of the active player's pre-combat main phase, add a
        lore counter to each saga they control and fire the matching chapter
        ability (CR 714.2/.3). Sacrifice the saga when its lore counter
        exceeds its final chapter (CR 714.4)."""
        import re as _re
        from src.engine.game_state import StackItem, Trigger, TriggerType, Zone
        from src.engine.zones import move_card

        active_id = game_state.active_player.player_id
        roman_to_int = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}

        for card in list(game_state.cards):
            if card.zone != Zone.BATTLEFIELD or card.controller_id != active_id:
                continue
            type_line = (card.type_line or "").lower()
            if "saga" not in type_line:
                continue
            oracle = card.oracle_text or ""
            chapters: dict[int, str] = {}
            for m in _re.finditer(
                r"^\s*([IVX, ]+?)\s*[\u2014\-]\s*(.+?)$",
                oracle,
                _re.MULTILINE,
            ):
                roman_group = m.group(1).strip().lower()
                effect = m.group(2).strip()
                for tok in [t.strip() for t in roman_group.split(",")]:
                    n = roman_to_int.get(tok)
                    if n:
                        chapters[n] = effect
            if not chapters:
                continue
            card.counters["lore"] = card.counters.get("lore", 0) + 1
            current = card.counters["lore"]
            game_state.log(f"  \u271a {card.name}: lore counter added (chapter {current})")
            if current in chapters:
                effect = chapters[current]
                trig = Trigger(
                    source_card_id=card.instance_id,
                    controller_id=card.controller_id,
                    trigger_type=TriggerType.UPKEEP,
                    description=effect,
                )
                game_state.triggered_abilities.append(trig)
                item = StackItem(
                    source_card_id=card.instance_id,
                    controller_id=card.controller_id,
                    is_spell=False,
                    card_data={
                        "name": f"[Saga] {card.name} chapter {current}: {effect}",
                        "type_line": "Ability",
                    },
                )
                game_state.stack.append(item)
                game_state.log(f"[TRIGGER (SAGA)] {card.name} chapter {current}: {effect}")
            final_chapter = max(chapters.keys())
            if current > final_chapter:
                move_card(game_state, card.instance_id, Zone.BATTLEFIELD,
                          Zone.GRAVEYARD, card.owner_id)
                game_state.log(f"  \u232b {card.name} is sacrificed (final chapter)")

        while game_state.stack:
            game_state = self.engine.resolve_stack_item(game_state)
            self.engine.check_state_based_actions(game_state)
            if game_state.game_over:
                return

    async def _play_turn(
        self, game_state: GameState, agents: dict[str, MTGAgent]
    ) -> GameState:
        """Play a single turn (all phases)."""
        from src.engine.phases import PHASE_ORDER, advance_phase, is_main_phase
        from src.engine.game_state import ActionType

        # Actually play through each phase of the turn
        # Banner at the top of every turn so the log reads like a real game.
        active = game_state.active_player
        active_label = active.name or active.player_id
        bar = "═" * 70
        game_state.log(f"\n╔{bar}╗")
        game_state.log(
            f"║  Turn {game_state.turn_number:>3}  —  {active_label}  "
            f"(life {active.life_total})"
        )
        game_state.log(f"╚{bar}╝")
        # Board snapshot for every player, so it's clear what's in play.
        for p in game_state.players:
            hand_n = sum(1 for c in game_state.cards
                         if c.zone == Zone.HAND and c.controller_id == p.player_id)
            lib_n = sum(1 for c in game_state.cards
                        if c.zone == Zone.LIBRARY and c.controller_id == p.player_id)
            gy_n = sum(1 for c in game_state.cards
                       if c.zone == Zone.GRAVEYARD and c.controller_id == p.player_id)
            bf = [c for c in game_state.cards
                  if c.zone == Zone.BATTLEFIELD and c.controller_id == p.player_id]
            lands = [c for c in bf if "Land" in (c.card_data.get("type_line") or "")]
            others = [c for c in bf if c not in lands]

            def _fmt_perm(c) -> str:
                # Face-down permanents hide their identity (CR 707)
                if getattr(c, "face_down", False) or c.card_data.get("_face_down"):
                    mode = getattr(c, "_face_down_mode", "face-down")
                    tag = "T" if c.tapped else "U"
                    return f"[{mode}][{tag}]"
                tag = "T" if c.tapped else "U"
                # Summoning sickness indicator
                sick = ""
                if c.is_creature() and getattr(c, "summoning_sick", False):
                    sick = "~"
                pt = ""
                if c.is_creature():
                    try:
                        from src.engine.keywords import effective_power as _ep, effective_toughness as _et
                        eff_p = _ep(c, game_state)
                        eff_t = _et(c, game_state)
                        base_p = c.card_data.get("power", "?")
                        base_t = c.card_data.get("toughness", "?")
                        dmg = getattr(c, "damage_marked", 0)
                        dmg_str = f" ⚔{dmg}" if dmg > 0 else ""
                        try:
                            base_p_int = int(base_p)
                            base_t_int = int(base_t)
                            if (eff_p, eff_t) != (base_p_int, base_t_int):
                                pt = f" {eff_p}/{eff_t}(base {base_p}/{base_t}){dmg_str}"
                            else:
                                pt = f" {eff_p}/{eff_t}{dmg_str}"
                        except (TypeError, ValueError):
                            pt = f" {base_p}/{base_t}{dmg_str}"
                    except Exception:
                        pwr = c.card_data.get("power", "?")
                        tgh = c.card_data.get("toughness", "?")
                        pt = f" {pwr}/{tgh}"
                # Counter summary (only non-empty buckets)
                counters = getattr(c, "counters", {})
                ctr_str = ""
                if counters:
                    parts = []
                    for k, v in counters.items():
                        if v:
                            parts.append(f"{v}{k}")
                    if parts:
                        ctr_str = " {" + ",".join(parts) + "}"
                return f"{sick}{c.name}{pt}{ctr_str}[{tag}]"

            land_part = (f"{len(lands)} lands "
                         f"({sum(1 for l in lands if not l.tapped)} untapped)")
            you_label = p.name or p.player_id
            # Poison counters + commander damage annotations
            extras: list[str] = []
            if getattr(p, "poison_counters", 0):
                extras.append(f"☠ {p.poison_counters} poison")
            cmd_dmg = getattr(p, "commander_damage_received", {})
            if cmd_dmg:
                for src_id, amt in cmd_dmg.items():
                    if amt >= 5:  # only log once it's a real threat
                        # Try to get commander name
                        cmd_card = next(
                            (c for c in game_state.cards if c.instance_id == src_id),
                            None
                        )
                        cmd_name = cmd_card.name if cmd_card else src_id
                        extras.append(f"{amt} cmd dmg from {cmd_name}")
            extras_str = ("  [" + ", ".join(extras) + "]") if extras else ""
            game_state.log(
                f"  • {you_label}: life {p.life_total}, "
                f"hand {hand_n}, lib {lib_n}, gy {gy_n} | {land_part}{extras_str}"
            )
            if others:
                # Group identical permanents for compactness.
                from collections import Counter as _Counter
                grouped = _Counter(_fmt_perm(c) for c in others)
                line = ", ".join(
                    (n if k == 1 else f"{k}× {n}") for n, k in grouped.items()
                )
                game_state.log(f"      board: {line}")
            # Hand contents — useful for debugging logs by hand. Sorted for
            # determinism; truncated to 12 entries to keep the snapshot tidy.
            hand_cards = [c for c in game_state.cards
                          if c.zone == Zone.HAND and c.controller_id == p.player_id]
            if hand_cards:
                from collections import Counter as _Counter
                names = _Counter(c.name for c in hand_cards)
                items = sorted(names.items())
                hand_line = ", ".join(
                    (n if k == 1 else f"{k}× {n}") for n, k in items[:12]
                )
                if len(items) > 12:
                    hand_line += f", … (+{len(items) - 12} more)"
                game_state.log(f"      hand: {hand_line}")

        phase_idx = 0
        while phase_idx < len(PHASE_ORDER):
            if game_state.game_over:
                return game_state
            
            phase = PHASE_ORDER[phase_idx]
            game_state.phase = phase
            # Skip the visual noise of empty steps; only log phases that
            # players regularly interact with.
            VERBOSE_PHASES = {
                Phase.MAIN_1, Phase.MAIN_2, Phase.COMBAT_BEGIN,
                Phase.COMBAT_ATTACKERS, Phase.COMBAT_BLOCKERS,
                Phase.COMBAT_DAMAGE, Phase.DRAW, Phase.END_STEP,
            }
            if phase in VERBOSE_PHASES:
                game_state.log(f"  ┌─ {phase.value.upper()} ─")

            # In untap step, creatures remove summoning sickness (gained pre-eot)
            if phase == Phase.UNTAP:
                from src.engine.counters import apply_stun_on_untap
                for card in game_state.cards:
                    if card.zone != Zone.BATTLEFIELD:
                        continue
                    if card.is_creature():
                        # Creatures that entered in a previous turn lose summoning sickness
                        if card.summoning_sick and card.turn_entered < game_state.turn_number:
                            card.summoning_sick = False
                    # Only untap active player's permanents (CR 502.1).
                    if card.controller_id != game_state.active_player.player_id:
                        continue
                    # Stun counter: remove one instead of untapping (CR 701.49).
                    if apply_stun_on_untap(card):
                        game_state.log(f"{card.name}: stun counter removed (skips untap)")
                        continue
                    card.tapped = False
                
                # Reset land play allocation for all players
                for player in game_state.players:
                    player.land_plays_remaining = 1

            # Upkeep + end step: fire phase-based triggers.
            if phase == Phase.UPKEEP or phase == Phase.END_STEP:
                # Suspend: tick time counters on suspended cards at upkeep
                if phase == Phase.UPKEEP:
                    from src.engine.alternate_costs import tick_suspend_counters, cast_suspended_card
                    freed = tick_suspend_counters(game_state, game_state.active_player.player_id)
                    for sus_card in freed:
                        cast_suspended_card(game_state, sus_card)
                    # Fire NEXT_UPKEEP delayed triggers (H5)
                    try:
                        from src.engine.delayed_triggers import (
                            get_delayed_registry, TriggerPoint
                        )
                        get_delayed_registry(game_state).fire(
                            game_state, TriggerPoint.NEXT_UPKEEP
                        )
                    except Exception:
                        pass
                if phase == Phase.END_STEP:
                    # Fire NEXT_END_STEP delayed triggers (H5)
                    try:
                        from src.engine.delayed_triggers import (
                            get_delayed_registry, TriggerPoint
                        )
                        get_delayed_registry(game_state).fire(
                            game_state, TriggerPoint.NEXT_END_STEP
                        )
                    except Exception:
                        pass
                from src.engine.triggers import check_phase_triggers
                from src.engine.game_state import TriggerType, StackItem
                ttype = TriggerType.UPKEEP if phase == Phase.UPKEEP else TriggerType.END_STEP
                phase_triggers = check_phase_triggers(
                    game_state, ttype, game_state.active_player.player_id
                )
                for trig in phase_triggers:
                    src_card = next(
                        (c for c in game_state.cards if c.instance_id == trig.source_card_id),
                        None,
                    )
                    src_name = src_card.name if src_card else "Ability"
                    item = StackItem(
                        source_card_id=trig.source_card_id,
                        controller_id=trig.controller_id,
                        is_spell=False,
                        card_data={
                            "name": f"[Trigger] {src_name}: {trig.description}",
                            "type_line": "Ability",
                        },
                    )
                    game_state.stack.append(item)
                    game_state.triggered_abilities.append(trig)
                    game_state.log(
                        f"[TRIGGER ({ttype.value.upper()})] {src_name}: {trig.description} added to stack"
                    )
                # Resolve any phase-based triggers immediately (LIFO).
                while game_state.stack:
                    game_state = self.engine.resolve_stack_item(game_state)
                    self.engine.check_state_based_actions(game_state)
                    if game_state.game_over:
                        return game_state

            # In draw step, active player draws
            if phase == Phase.DRAW:
                player = game_state.active_player
                library = [c for c in game_state.cards if c.zone == Zone.LIBRARY and c.owner_id == player.player_id]
                if library:
                    card_to_draw = library[0]
                    from src.engine.zones import move_card
                    game_state = move_card(game_state, card_to_draw.instance_id, Zone.LIBRARY, Zone.HAND, player.player_id)
                    player.has_drawn_for_turn = True
                else:
                    # CR 104.3c / 704.5b: drawing from empty library loses.
                    loser_idx = game_state.active_player_index
                    winner_idx = (loser_idx + 1) % len(game_state.players)
                    game_state.winner = game_state.players[winner_idx]
                    game_state.game_over = True
                    game_state.log(f"{player.player_id} attempted to draw from empty library and lost")
                    return game_state

            # Combat phases
            if phase == Phase.COMBAT_BEGIN:
                # Initialize combat state at the start of combat
                from src.engine.game_state import CombatState
                game_state.combat = CombatState()
            
            if phase == Phase.COMBAT_ATTACKERS:
                # Active player declares attackers with stack/priority support
                game_state.priority_player_index = game_state.active_player_index
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
                if game_state.game_over:
                    return game_state
            
            if phase == Phase.COMBAT_BLOCKERS:
                # Defending player declares blockers with stack/priority support
                defending_player_idx = (game_state.active_player_index + 1) % len(game_state.players)
                game_state.priority_player_index = defending_player_idx
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
                if game_state.game_over:
                    return game_state
            
            if phase == Phase.COMBAT_DAMAGE:
                from src.engine.combat import resolve_combat_damage
                # Resolve combat damage
                resolve_combat_damage(game_state)
                # Fire END_OF_COMBAT delayed triggers (myriad token exile etc.)
                try:
                    from src.engine.delayed_triggers import (
                        get_delayed_registry, TriggerPoint
                    )
                    get_delayed_registry(game_state).fire(
                        game_state, TriggerPoint.END_OF_COMBAT
                    )
                except Exception:
                    pass
            
            if phase == Phase.CLEANUP:
                # Watcher-based turn summary — fires before reset so data is still live.
                try:
                    from src.engine.watchers import (
                        get_watcher_registry,
                        SpellsCastThisTurnWatcher,
                        LifeLostThisTurnWatcher,
                        LifeGainedThisTurnWatcher,
                        PlayerAttackedThisTurnWatcher,
                        LandPlayedThisTurnWatcher,
                    )
                    reg = get_watcher_registry(game_state)
                    spells_w = reg.get(SpellsCastThisTurnWatcher)
                    lost_w = reg.get(LifeLostThisTurnWatcher)
                    gained_w = reg.get(LifeGainedThisTurnWatcher)
                    atk_w = reg.get(PlayerAttackedThisTurnWatcher)
                    land_w = reg.get(LandPlayedThisTurnWatcher)
                    summary_lines: list[str] = []
                    for p in game_state.players:
                        pid = p.player_id
                        parts: list[str] = []
                        sc = spells_w.count_for(pid) if spells_w else 0
                        lc = land_w.count_for(pid) if land_w else 0
                        gained = gained_w.gained_by(pid) if gained_w else 0
                        lost = lost_w.lost_by(pid) if lost_w else 0
                        attacked = atk_w.did_attack(pid) if atk_w else False
                        if sc:
                            parts.append(f"{sc} spell{'s' if sc != 1 else ''}")
                        if lc:
                            parts.append(f"{lc} land{'s' if lc != 1 else ''}")
                        if attacked:
                            n_atk = len([
                                a for a in (atk_w.attacker_ids if atk_w else set())
                                if any(
                                    c.instance_id == a and c.controller_id == pid
                                    for c in game_state.cards
                                )
                            ])
                            parts.append(f"attacked ({n_atk or '?'} creature{'s' if n_atk != 1 else ''})")
                        life_parts: list[str] = []
                        if gained:
                            life_parts.append(f"+{gained}")
                        if lost:
                            life_parts.append(f"-{lost}")
                        if life_parts:
                            net = gained - lost
                            sign = "+" if net >= 0 else ""
                            parts.append(f"life {'/'.join(life_parts)} (net {sign}{net})")
                        if parts:
                            label = p.name or pid
                            summary_lines.append(f"  │ {label}: {', '.join(parts)}")
                    if summary_lines:
                        game_state.log("  ┌─── TURN SUMMARY ───")
                        for sl in summary_lines:
                            game_state.log(sl)
                        game_state.log("  └────────────────────")
                except Exception:
                    pass
                # Notify the optional per-turn hook BEFORE watcher reset so
                # it can still read turn statistics (e.g. GoldfishRunner).
                if self.config.on_turn_end is not None:
                    try:
                        self.config.on_turn_end(game_state)
                    except Exception:
                        pass
                # Empty all players' mana pools and discard down to max hand size
                from src.engine.mana import empty_mana_pool
                from src.engine.zones import move_card
                # End-of-turn cleanup: damage wears off, "until end of turn"
                # P/T bonuses expire (CR 514.2). Vehicles revert via
                # equip.clear_crew_eot.
                for c in game_state.cards:
                    if c.zone != Zone.BATTLEFIELD:
                        continue
                    c.damage_marked = 0
                    if hasattr(c, "eot_power_bonus"):
                        c.eot_power_bonus = 0
                    if hasattr(c, "eot_toughness_bonus"):
                        c.eot_toughness_bonus = 0
                try:
                    from src.engine.equip import clear_crew_eot
                    clear_crew_eot(game_state)
                except Exception:
                    pass
                # Reset per-turn counters used by storm/spectacle/surge/raid/etc.
                game_state.spells_cast_this_turn = 0
                for player in game_state.players:
                    if hasattr(player, "life_lost_this_turn"):
                        player.life_lost_this_turn = 0
                    if hasattr(player, "attacked_this_turn"):
                        player.attacked_this_turn = False
                # Expire end-of-turn continuous effects (layer system)
                try:
                    from src.engine.continuous_effects import expire_end_of_turn as _cfeot
                    _cfeot(game_state)
                except Exception:
                    pass
                # Expire end-of-turn watcher reset (H3)
                try:
                    from src.engine.watchers import get_watcher_registry
                    get_watcher_registry(game_state).reset_turn()
                except Exception:
                    pass
                # Fire end-of-turn delayed triggers (H5)
                try:
                    from src.engine.delayed_triggers import get_delayed_registry, TriggerPoint
                    reg = get_delayed_registry(game_state)
                    reg.fire(game_state, TriggerPoint.END_OF_TURN)
                    reg.expire_end_of_turn()
                except Exception:
                    pass
                # Clear summoning-sick flag on permanents that have been
                # under their controller's control since their last upkeep
                # (handled at untap, but normalize attacked-this-turn here).
                for c in game_state.cards:
                    if hasattr(c, "attacked_this_turn"):
                        c.attacked_this_turn = False
                for player in game_state.players:
                    empty_mana_pool(player)
                    hand_cards = [
                        c for c in game_state.cards
                        if c.zone == Zone.HAND and c.controller_id == player.player_id
                    ]
                    excess = max(0, len(hand_cards) - player.max_hand_size)
                    if excess > 0:
                        hand_cards.sort(key=lambda c: c.card_data.get("cmc", 0), reverse=True)
                        for card in hand_cards[:excess]:
                            game_state = move_card(
                                game_state,
                                card.instance_id,
                                Zone.HAND,
                                Zone.GRAVEYARD,
                                player.player_id,
                            )
                            game_state.log(f"{player.name} discards {card.name} (cleanup)")

            # In main phases, use full priority loop (enables stack + instant-speed)
            if is_main_phase(phase):
                # Saga: at the beginning of active player's pre-combat main,
                # add a lore counter and fire the matching chapter ability
                # (CR 714.2). We trigger this on MAIN_1 only.
                if phase == Phase.MAIN_1:
                    self._tick_sagas(game_state)
                    if game_state.game_over:
                        return game_state

                # Initialize priority to active player
                game_state.priority_player_index = game_state.active_player_index
                
                # Run full priority loop until stack empties and all pass
                game_state = await run_priority_loop(
                    game_state,
                    agents,
                    self.engine,
                    collector=self.self_play_collector,
                )
                
                if game_state.game_over:
                    return game_state
            
            # Check SBAs after each phase
            sba_events = self.engine.check_state_based_actions(game_state)
            if sba_events or game_state.game_over:
                logger.info(f"SBA after {phase.value}: {sba_events}, game_over={game_state.game_over}")
            if game_state.game_over:
                logger.info(f"Game over triggered after phase {phase.value}")
                return game_state

            # CR 106.4: at the end of every step and phase, all mana in
            # players' mana pools empties. We already drain at cleanup; this
            # makes the rule visible everywhere else too. We log the loss so
            # players can see when they let mana float and lose it.
            from src.engine.mana import empty_mana_pool
            from src.orchestrator.priority_loop import _format_mana
            for pl in game_state.players:
                if any(v > 0 for v in pl.mana_pool.values()):
                    lost = _format_mana(pl.mana_pool)
                    pname = pl.name or pl.player_id
                    game_state.log(
                        f"      ✗ {pname} loses {lost} from mana pool "
                        f"(end of {phase.value})"
                    )
                    empty_mana_pool(pl)

            # Move to next phase
            phase_idx += 1
        
        # After all phases complete, advance the active player. The turn
        # number only increases when we wrap back to the first seat — i.e.
        # one full trip around the table is one "turn" in multiplayer
        # parlance. (In a 2-player game this still increments every player's
        # play, matching standard usage.)
        next_idx = (game_state.active_player_index + 1) % len(game_state.players)
        game_state.active_player_index = next_idx
        if next_idx == 0:
            game_state.turn_number += 1

        return game_state

"""Deterministic heuristic baseline agent.

Strategically simple but deterministic given a fixed seed: it always picks
the highest-priority legal action class, breaking ties with a stable RNG.
Used as a stronger floor than `RandomAgent` for benchmark comparisons.
"""
from __future__ import annotations

import random
from typing import Iterable

from src.agents.base_agent import MTGAgent, AgentStrategy
from src.engine_legacy.game_state import Action, ActionType, GameState

# Action-type priority. Higher = picked first.
_PRIORITY: dict[ActionType, int] = {
    ActionType.PLAY_LAND: 100,
    ActionType.CAST_SPELL: 90,
    ActionType.DECLARE_ATTACKERS: 80,
    ActionType.DECLARE_BLOCKERS: 70,
    ActionType.ACTIVATE_ABILITY: 60,
    ActionType.PASS_PRIORITY: 1,
    ActionType.CONCEDE: 0,
}


class HeuristicAgent(MTGAgent):
    """Picks actions by a fixed priority over `ActionType`.

    Deterministic given the random seed used to construct the agent; useful
    as a benchmark baseline that is *both* stronger than uniform-random and
    fully reproducible across runs.
    """

    def __init__(
        self,
        player_id: str,
        name: str = "",
        seed: int | None = None,
        prefer_aggressive: bool = True,
    ):
        super().__init__(player_id=player_id, name=name or f"Heuristic({player_id})")
        self._rng = random.Random(seed)
        self._prefer_aggressive = prefer_aggressive

    @property
    def strategy(self) -> AgentStrategy | None:
        """Return the mulligan strategy based on play style."""
        return AgentStrategy.AGGRESSIVE if self._prefer_aggressive else AgentStrategy.CONTROL

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        if not legal_actions:
            return Action(
                action_type=ActionType.PASS_PRIORITY, player_id=self.player_id
            )

        # Counterspell discipline: don't waste a counter on our own spells
        # or fire it when nothing's on the stack. Filter out CAST_SPELL
        # actions for cards whose oracle text contains "counter target"
        # unless the topmost stack item is a spell controlled by an
        # opponent.
        original_count = len(legal_actions)
        legal_actions = self._filter_counter_actions(game_state, legal_actions)
        filtered_counters = original_count - len(legal_actions)

        ranked = sorted(
            legal_actions,
            key=lambda a: (
                -_PRIORITY.get(a.action_type, 5),
                # tie-break by deterministic hash of action repr
                self._tiebreak(a),
            ),
        )

        # When aggressive, prefer the action that targets opponent or
        # taps for mana to enable more plays.
        top_priority = _PRIORITY.get(ranked[0].action_type, 5)
        candidates: Iterable[Action] = (
            a for a in ranked if _PRIORITY.get(a.action_type, 5) == top_priority
        )
        candidates = list(candidates)

        chosen: Action
        chose_via = "top_priority"
        if self._prefer_aggressive:
            for a in candidates:
                if a.action_type == ActionType.DECLARE_ATTACKERS and a.targets:
                    self._record_reasoning(
                        legal_actions, a,
                        priority=top_priority,
                        candidate_count=len(candidates),
                        filtered_counters=filtered_counters,
                        rationale=f"aggressive override → DECLARE_ATTACKERS with {len(a.targets)} target(s)",
                        chose_via="aggressive_override",
                    )
                    return a

        # When blocking is on the table, block the biggest unblocked attacker
        # with the smallest creature first (chump-block heuristic).
        if candidates and candidates[0].action_type == ActionType.DECLARE_BLOCKERS:
            blocker = self._pick_block(game_state, candidates)
            if blocker is not None:
                self._record_reasoning(
                    legal_actions, blocker,
                    priority=top_priority,
                    candidate_count=len(candidates),
                    filtered_counters=filtered_counters,
                    rationale="chump-block: smallest blocker → largest attacker",
                    chose_via="chump_block",
                )
                return blocker

        chosen = candidates[0]
        self._record_reasoning(
            legal_actions, chosen,
            priority=top_priority,
            candidate_count=len(candidates),
            filtered_counters=filtered_counters,
            rationale=f"highest-priority action class ({chosen.action_type.value}, prio={top_priority})",
            chose_via=chose_via,
        )
        return chosen

    def _record_reasoning(
        self,
        legal_actions: list[Action],
        chosen: Action,
        *,
        priority: int,
        candidate_count: int,
        filtered_counters: int,
        rationale: str,
        chose_via: str,
    ) -> None:
        from src.agents.reasoning import ReasoningTrace
        try:
            idx = legal_actions.index(chosen)
        except ValueError:
            idx = -1
        self.set_reasoning(ReasoningTrace(
            agent_kind="heuristic",
            rationale=rationale,
            legal_action_count=len(legal_actions),
            chosen_index=idx,
            beliefs={
                "top_priority": priority,
                "tied_candidate_count": candidate_count,
                "filtered_counterspells": filtered_counters,
                "chose_via": chose_via,
                "prefer_aggressive": self._prefer_aggressive,
            },
        ))

    def _pick_block(
        self, game_state: GameState, candidates: list[Action]
    ) -> Action | None:
        """Pick a single blocker assignment for the most threatening attacker."""
        from src.engine_legacy.game_state import Zone

        def _power(card_id: str) -> int:
            card = next((c for c in game_state.cards if c.instance_id == card_id), None)
            try:
                return int(card.power) if card and card.power else 0
            except ValueError:
                return 0

        # Group candidates by attacker target.
        by_attacker: dict[str, list[Action]] = {}
        for a in candidates:
            if a.action_type != ActionType.DECLARE_BLOCKERS or not a.targets:
                continue
            by_attacker.setdefault(a.targets[0], []).append(a)
        if not by_attacker:
            return None

        # Choose attacker with greatest power.
        worst_attacker = max(by_attacker, key=_power)
        # Choose smallest blocker for it.
        return min(by_attacker[worst_attacker], key=lambda a: _power(a.card_instance_id or ""))

    def _tiebreak(self, action: Action) -> int:
        # Stable deterministic tiebreak: hash action serialisation with RNG.
        return self._rng.getrandbits(32) ^ hash(
            (action.action_type, action.player_id, tuple(action.targets or ()))
        )

    def _filter_counter_actions(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> list[Action]:
        """Remove ``Cast(counterspell)`` from the menu unless an opponent
        spell is on the stack.

        Without this, the heuristic happily casts Counterspell on its own
        spells (or whenever it has the mana) since `CAST_SPELL` > `PASS`.
        """
        # Find the controller of the topmost spell on the stack, if any.
        opp_spell_on_top = False
        if game_state.stack:
            top = game_state.stack[-1]
            if (
                getattr(top, "is_spell", False)
                and getattr(top, "controller_id", None) != self.player_id
            ):
                opp_spell_on_top = True

        if opp_spell_on_top:
            return legal_actions

        # No opponent spell to counter — drop Cast(counterspell) actions.
        kept: list[Action] = []
        for a in legal_actions:
            if a.action_type == ActionType.CAST_SPELL and a.card_instance_id:
                card = next(
                    (c for c in game_state.cards if c.instance_id == a.card_instance_id),
                    None,
                )
                if card and self._is_counterspell(card.oracle_text or ""):
                    continue
            kept.append(a)
        # Never strip the entire menu down to nothing.
        return kept or legal_actions

    @staticmethod
    def _is_counterspell(oracle: str) -> bool:
        text = oracle.lower()
        # CR 701.5: "counter target spell" or "counter target X spell".
        return (
            "counter target spell" in text
            or "counter target activated" in text
            or "counter target triggered" in text
        )

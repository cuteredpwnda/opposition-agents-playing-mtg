"""Goldfish simulator — runs a deck solo against a do-nothing opponent.

"Goldfishing" is a classic MTG deck-testing technique: play your deck
without any opponent interaction to measure its raw consistency and speed.
Inspired by the cEDH Rhystic Goldfish Simulator (cedhdata.com/simulator).

Usage (async)::

    from src.agents.goldfish_runner import GoldfishRunner

    runner = GoldfishRunner(runs=100, max_turns=8, seed=42, target_card="Rhystic Study")
    stats = await runner.run(decklist)
    print(stats.summary())

Usage (sync)::

    stats = GoldfishRunner(runs=100).run_sync(decklist)

Usage (generate training trajectories)::

    runs = await GoldfishRunner(runs=200).run_raw(decklist)
    trajectories = GoldfishRunner.to_trajectories(runs)

The null opponent never casts spells, attacks, or blocks.  The active
player's deck is run under a ``HeuristicAgent`` by default so it makes
reasonable plays.

Key metrics returned in :class:`GoldfishStats`:

* ``win_rate`` — fraction of runs where the deck killed the opponent inside
  ``max_turns``.
* ``avg_kill_turn`` — average turn of kill across winning runs.
* ``kill_turn_distribution`` — CDF: ``{turn: fraction_of_all_runs_killed_by_that_turn}``.
* ``win_rate_convergence`` — running win rate after each successive game (convergence plot).
* ``target_card_by_turn`` — if ``target_card`` is set, fraction of runs where
  that card was cast by turn N.
* ``top_winning_lines`` — most common ordered sequences of spell names in winning runs.
* ``avg_spells_per_turn``, ``avg_power_per_turn`` — per-turn consistency.
* ``curve_hit_rate`` — fraction of turns (active player only) with ≥1 spell.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.engine_legacy.card_database import CardDatabase
from src.orchestrator_legacy.game_runner import GameConfig, GameRunner

logger = logging.getLogger(__name__)

# ─── Data containers ─────────────────────────────────────────────────────────


@dataclass
class TurnSnapshot:
    """Per-turn state captured during a goldfish run (active player's turns only)."""

    turn: int
    spells_cast: int
    land_played: int
    creatures: int
    total_power: int
    opponent_life: int
    spell_names: list[str] = field(default_factory=list)
    """Names of spells cast this turn (ordered); empty if none."""


@dataclass
class GoldfishRun:
    """Result of a single goldfish game."""

    kill_turn: int | None
    """Turn the active player won; ``None`` if max_turns reached."""
    turns_played: int
    snapshots: list[TurnSnapshot] = field(default_factory=list)
    winning_line: list[str] = field(default_factory=list)
    """Ordered list of all spell names cast across the whole game (winning runs only)."""
    target_card_turn: int | None = None
    """Turn the target card was first cast (if ``GoldfishRunner.target_card`` is set)."""

    @property
    def won(self) -> bool:
        return self.kill_turn is not None


@dataclass
class GoldfishStats:
    """Aggregated statistics over many goldfish runs."""

    runs: int
    wins: int
    kill_turns: list[int]
    """Raw kill-turn values from winning runs."""

    # Per-turn averages — index 0 = turn 1.
    avg_spells_per_turn: list[float]
    avg_power_per_turn: list[float]
    avg_creatures_per_turn: list[float]

    curve_hit_rate: float
    """Fraction of (active player) turns where at least one spell was cast."""

    win_rate_convergence: list[float] = field(default_factory=list)
    """Running win rate after game i (length == runs).  Useful for convergence plots."""

    target_card_by_turn: dict[int, float] = field(default_factory=dict)
    """Fraction of runs where the target card was cast by (or on) turn N."""

    top_winning_lines: list[list[str]] = field(default_factory=list)
    """Up to 10 most common winning spell sequences (Winning Lines Console)."""

    @property
    def win_rate(self) -> float:
        return self.wins / self.runs if self.runs else 0.0

    @property
    def avg_kill_turn(self) -> float | None:
        return statistics.mean(self.kill_turns) if self.kill_turns else None

    @property
    def kill_turn_distribution(self) -> dict[int, float]:
        """CDF: probability of having killed by (or on) turn N."""
        if not self.kill_turns:
            return {}
        max_turn = max(self.kill_turns)
        cdf: dict[int, float] = {}
        for t in range(1, max_turn + 1):
            cdf[t] = sum(1 for k in self.kill_turns if k <= t) / self.runs
        return cdf

    def summary(self, deck_name: str = "deck") -> str:
        lines: list[str] = [
            f"Goldfish analysis: {deck_name}  ({self.runs} runs)",
            f"  Win rate : {self.win_rate:.1%}  ({self.wins}/{self.runs})",
        ]
        if self.avg_kill_turn is not None:
            kt_parts = "  ".join(
                f"T{t}: {p:.1%}"
                for t, p in sorted(self.kill_turn_distribution.items())
                if t <= 8
            )
            lines.append(f"  Kill turn: avg {self.avg_kill_turn:.2f}  |  {kt_parts}")
        else:
            lines.append("  Kill turn: — (no wins)")
        lines.append(f"  Curve efficiency: {self.curve_hit_rate:.1%} of turns had ≥1 spell")

        if self.avg_spells_per_turn:
            sp_parts = "  ".join(
                f"T{i + 1}: {v:.1f}" for i, v in enumerate(self.avg_spells_per_turn[:8])
            )
            lines.append(f"  Avg spells/turn: {sp_parts}")
        if self.avg_power_per_turn:
            pw_parts = "  ".join(
                f"T{i + 1}: {v:.1f}" for i, v in enumerate(self.avg_power_per_turn[:8])
            )
            lines.append(f"  Avg power/board: {pw_parts}")

        if self.target_card_by_turn:
            tc_parts = "  ".join(
                f"T{t}: {p:.1%}" for t, p in sorted(self.target_card_by_turn.items())
            )
            lines.append(f"  Target card by turn: {tc_parts}")

        if self.top_winning_lines:
            lines.append("  Winning lines (top 3):")
            for wl in self.top_winning_lines[:3]:
                lines.append("    " + " → ".join(wl[:8]) + (" …" if len(wl) > 8 else ""))

        return "\n".join(lines)


# ─── Helper — build null deck ─────────────────────────────────────────────────

_PLAINS_DATA: dict[str, Any] = {
    "name": "Plains",
    "type_line": "Basic Land — Plains",
    "mana_cost": "",
    "colors": [],
    "oracle_text": "{T}: Add {W}.",
    "power": None,
    "toughness": None,
}


def _make_null_deck(n: int = 40) -> list[dict[str, Any]]:
    """Return N copies of basic Plains — a deck that never does anything useful."""
    return [dict(_PLAINS_DATA) for _ in range(n)]


# ─── GoldfishRunner ──────────────────────────────────────────────────────────


class GoldfishRunner:
    """Run a deck solo against a :class:`~src.agents.null_agent.NullAgent`.

    Parameters
    ----------
    agent_type:
        Short name of the agent driving the active deck.  Defaults to
        ``"heuristic"`` which gives reasonable plays; use ``"random"`` for
        a cheap lower-bound.
    runs:
        Number of independent games to simulate.
    max_turns:
        Hard cap per game (turns for the *active player*; actual game turns
        may be up to 2×).
    seed:
        Optional RNG seed for reproducibility.  When set, each game uses a
        derived seed ``seed + game_index`` so games differ but the suite is
        deterministic.
    starting_life:
        Life totals for both players (default 20 for standard goldfish).
    """

    ACTIVE_PID = "goldfish_player"
    NULL_PID = "null_opponent"

    def __init__(
        self,
        agent_type: str = "heuristic",
        runs: int = 100,
        max_turns: int = 10,
        seed: int | None = None,
        starting_life: int = 20,
        target_card: str | None = None,
    ) -> None:
        self.agent_type = agent_type
        self.runs = runs
        self.max_turns = max_turns
        self.seed = seed
        self.starting_life = starting_life
        self.target_card = target_card  # optional card to track (cEDH-style)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        decklist: list[dict[str, Any]],
        card_db: object | None = None,
    ) -> GoldfishStats:
        """Simulate ``self.runs`` goldfish games and return aggregated stats."""
        all_runs = await self.run_raw(decklist, card_db=card_db)
        return self._aggregate(all_runs, target_card=self.target_card)

    async def run_raw(
        self,
        decklist: list[dict[str, Any]],
        card_db: object | None = None,
    ) -> list[GoldfishRun]:
        """Return raw per-game results without aggregation.

        Useful for feeding into :meth:`to_trajectories` or custom analysis.
        """
        all_runs: list[GoldfishRun] = []
        for i in range(self.runs):
            run_result = await self._run_one(decklist, game_index=i, card_db=card_db)
            all_runs.append(run_result)
        return all_runs

    def run_sync(
        self,
        decklist: list[dict[str, Any]],
        card_db: object | None = None,
    ) -> GoldfishStats:
        """Synchronous wrapper around :meth:`run` for non-async callers."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Inside an existing event loop (e.g. Jupyter) — use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, self.run(decklist, card_db))
                    return future.result()
            return loop.run_until_complete(self.run(decklist, card_db))
        except RuntimeError:
            return asyncio.run(self.run(decklist, card_db))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_one(
        self,
        decklist: list[dict[str, Any]],
        game_index: int,
        card_db: object | None,
    ) -> GoldfishRun:
        from src.agents.heuristic_agent import HeuristicAgent
        from src.agents.null_agent import NullAgent
        from src.agents.random_agent import RandomAgent

        # Per-run snapshot accumulator (closed over by the hook)
        snapshots: list[TurnSnapshot] = []
        target_card_lower = self.target_card.lower().strip() if self.target_card else None

        def _on_turn_end(state: "GameState") -> None:  # type: ignore[name-defined]  # noqa: F821
            """Snapshot watcher data for the active player's turn."""
            try:
                from src.engine_legacy.game_state import Zone
                from src.engine_legacy.keywords import effective_power
                from src.engine_legacy.watchers import (
                    LandPlayedThisTurnWatcher,
                    SpellsCastThisTurnWatcher,
                    get_watcher_registry,
                )

                # Only record snapshots for the deck-under-test.
                active_pid = (
                    state.players[state.active_player_index].player_id
                    if state.players
                    else None
                )
                if active_pid != self.ACTIVE_PID:
                    return

                reg = get_watcher_registry(state)
                spells_w = reg.get(SpellsCastThisTurnWatcher)
                land_w = reg.get(LandPlayedThisTurnWatcher)

                spells = spells_w.count_for(active_pid) if spells_w else 0
                lands = land_w.count_for(active_pid) if land_w else 0

                # Collect spell names from this turn's SPELL_CAST events
                spell_names: list[str] = []
                if spells_w:
                    # Build a lookup from instance_id → card name
                    id_to_name: dict[str, str] = {
                        c.instance_id: c.card_data.get("name", c.instance_id)
                        for c in state.cards
                    }
                    for ev in spells_w.spells:
                        if ev.player_id == active_pid:
                            name = id_to_name.get(ev.source_id, ev.source_id)
                            spell_names.append(name)

                my_creatures = [
                    c
                    for c in state.cards
                    if c.zone == Zone.BATTLEFIELD
                    and c.controller_id == active_pid
                    and "creature" in c.card_data.get("type_line", "").lower()
                ]
                total_power = sum(effective_power(c, state) for c in my_creatures)

                opponent = next(
                    (p for p in state.players if p.player_id != active_pid), None
                )
                opp_life = opponent.life_total if opponent else 0

                snapshots.append(
                    TurnSnapshot(
                        turn=state.turn_number,
                        spells_cast=spells,
                        land_played=lands,
                        creatures=len(my_creatures),
                        total_power=total_power,
                        opponent_life=opp_life,
                        spell_names=spell_names,
                    )
                )
            except Exception:
                pass
            except Exception:
                pass

        derived_seed: int | None = (
            None if self.seed is None else self.seed + game_index
        )

        # Active agent
        if self.agent_type == "heuristic":
            active_agent = HeuristicAgent(
                player_id=self.ACTIVE_PID,
                name="Goldfish",
                seed=derived_seed,
            )
        else:
            active_agent = RandomAgent(
                player_id=self.ACTIVE_PID,
                name="Goldfish",
            )

        null_agent = NullAgent(player_id=self.NULL_PID, name="Null")

        db = card_db or CardDatabase()
        config = GameConfig(
            format="standard",
            starting_life=self.starting_life,
            max_turns=self.max_turns * 2,  # game turns are shared between both players
            mulligan_enabled=True,
            max_mulligans=2,
            on_turn_end=_on_turn_end,
        )
        runner = GameRunner(config=config, card_db=db)

        # Tag the active player's deck; give null opponent 40 basic lands
        active_deck = list(decklist)
        null_deck = _make_null_deck(40)

        result = await runner.run_game(
            agents={self.ACTIVE_PID: active_agent, self.NULL_PID: null_agent},
            decks={self.ACTIVE_PID: active_deck, self.NULL_PID: null_deck},
        )

        kill_turn: int | None = None
        if result.winner == self.ACTIVE_PID:
            # The kill happened on the active player's turn that ended the game.
            # Best proxy: the last snapshot's turn number.
            if snapshots:
                kill_turn = snapshots[-1].turn
            else:
                kill_turn = result.turns

        # Build winning line — all spells cast across all turns (if won)
        winning_line: list[str] = []
        if kill_turn is not None:
            for snap in snapshots:
                winning_line.extend(snap.spell_names)

        # Detect which turn the target card was first cast (if configured)
        target_card_turn: int | None = None
        if target_card_lower:
            for snap in snapshots:
                if any(name.lower() == target_card_lower for name in snap.spell_names):
                    target_card_turn = snap.turn
                    break

        return GoldfishRun(
            kill_turn=kill_turn,
            turns_played=result.turns,
            snapshots=snapshots,
            winning_line=winning_line,
            target_card_turn=target_card_turn,
        )

    @staticmethod
    def _aggregate(
        runs: list[GoldfishRun],
        target_card: str | None = None,
    ) -> GoldfishStats:
        wins = sum(1 for r in runs if r.won)
        kill_turns = [r.kill_turn for r in runs if r.kill_turn is not None]

        # Determine the widest number of turns covered
        max_snap_turn = max(
            (snap.turn for r in runs for snap in r.snapshots), default=0
        )

        # Per-turn aggregation
        def _avg_by_turn(attr: str) -> list[float]:
            result: list[float] = []
            for t in range(1, max_snap_turn + 1):
                vals = [
                    getattr(snap, attr)
                    for r in runs
                    for snap in r.snapshots
                    if snap.turn == t
                ]
                result.append(statistics.mean(vals) if vals else 0.0)
            return result

        avg_spells = _avg_by_turn("spells_cast")
        avg_power = _avg_by_turn("total_power")
        avg_creatures = _avg_by_turn("creatures")

        # Curve hit rate: fraction of all active-player turns with ≥1 spell
        all_snaps = [snap for r in runs for snap in r.snapshots]
        if all_snaps:
            curve_hit_rate = sum(1 for s in all_snaps if s.spells_cast >= 1) / len(all_snaps)
        else:
            curve_hit_rate = 0.0

        # Win rate convergence: running win rate after each game
        win_rate_convergence: list[float] = []
        running_wins = 0
        for i, r in enumerate(runs, 1):
            if r.won:
                running_wins += 1
            win_rate_convergence.append(running_wins / i)

        # Target card by-turn CDF
        target_card_by_turn: dict[int, float] = {}
        if target_card and any(r.target_card_turn is not None for r in runs):
            max_tc_turn = max(
                (r.target_card_turn for r in runs if r.target_card_turn is not None),
                default=0,
            )
            n = len(runs)
            for t in range(1, max_tc_turn + 1):
                target_card_by_turn[t] = sum(
                    1 for r in runs if r.target_card_turn is not None and r.target_card_turn <= t
                ) / n

        # Top winning lines — count unique spell sequences in winning runs
        from collections import Counter
        wl_counter: Counter[tuple[str, ...]] = Counter(
            tuple(r.winning_line) for r in runs if r.won and r.winning_line
        )
        top_winning_lines = [list(seq) for seq, _ in wl_counter.most_common(10)]

        return GoldfishStats(
            runs=len(runs),
            wins=wins,
            kill_turns=kill_turns,
            avg_spells_per_turn=avg_spells,
            avg_power_per_turn=avg_power,
            avg_creatures_per_turn=avg_creatures,
            curve_hit_rate=curve_hit_rate,
            win_rate_convergence=win_rate_convergence,
            target_card_by_turn=target_card_by_turn,
            top_winning_lines=top_winning_lines,
        )

    # ------------------------------------------------------------------
    # Trajectory export — feed goldfish traces into WorldModel training
    # ------------------------------------------------------------------

    @staticmethod
    def to_trajectories(
        runs: list[GoldfishRun],
        tokenizer: object | None = None,
    ) -> list["Trajectory"]:  # type: ignore[name-defined]  # noqa: F821
        """Convert goldfish runs into :class:`~src.world_model.trajectory.Trajectory` objects.

        Each ``TurnSnapshot`` becomes a ``Transition`` with lightweight
        state features (no neural encoding required).  The trajectories are
        tagged ``source="goldfish"`` so training pipelines can distinguish
        them from self-play data.

        Parameters
        ----------
        runs:
            Raw run results from :meth:`run_raw`.
        tokenizer:
            Optional ``GameTokenizer`` for richer state encoding.  When
            ``None`` the transitions carry a minimal hand-crafted feature dict
            instead, which is still useful for imitation-learning on action
            sequences.
        """
        import numpy as np
        from src.world_model.trajectory import Trajectory, Transition

        trajectories: list[Trajectory] = []
        for game_idx, run in enumerate(runs):
            traj = Trajectory(
                game_id=f"goldfish_{game_idx:06d}",
                winner=0 if run.won else 1,   # 0 = active player won
                num_turns=run.turns_played,
                source="goldfish",
                metadata={
                    "kill_turn": run.kill_turn,
                    "winning_line": run.winning_line,
                    "target_card_turn": run.target_card_turn,
                },
            )
            for snap in run.snapshots:
                # Lightweight state encoding from snapshot data
                state_features: dict[str, np.ndarray] = {
                    "turn": np.array([snap.turn], dtype=np.float32),
                    "spells_cast": np.array([snap.spells_cast], dtype=np.float32),
                    "land_played": np.array([snap.land_played], dtype=np.float32),
                    "creatures": np.array([snap.creatures], dtype=np.float32),
                    "total_power": np.array([snap.total_power], dtype=np.float32),
                    "opponent_life": np.array([snap.opponent_life], dtype=np.float32),
                }
                # Represent the whole-turn action as the count of spells played
                action_enc = np.array([float(snap.spells_cast)], dtype=np.float32)
                # Reward: proportional life pressure applied this turn
                reward = max(0.0, (20.0 - snap.opponent_life) / 20.0)
                transition = Transition(
                    state_features=state_features,
                    action_encoding=action_enc,
                    reward=reward,
                    done=(snap == run.snapshots[-1] and run.won),
                    action_type="GOLDFISH_TURN",
                    card_name=run.winning_line[0] if run.winning_line else None,
                    metadata={
                        "turn": snap.turn,
                        "spell_names": snap.spell_names,
                    },
                )
                traj.add(transition)
            trajectories.append(traj)
        return trajectories

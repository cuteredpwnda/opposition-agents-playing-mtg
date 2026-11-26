"""Extended BenchmarkSuite — Phase B harness.

Builds on top of `src.training.benchmark.BenchmarkSuite` to provide:

- **Standard archetype matchups** drawn from `src.training.archetype_decks`.
- **Deterministic replay** via per-game seed derivation from a master seed.
- **Parallel execution** via `asyncio.gather` with a configurable concurrency.
- **Per-game CSV metrics** (winner, turns, decision-time, archetype pair).
- **Aggregate JSON summary** with per-agent ELO and per-archetype winrates.
- **Ablation toggles**: a configurable set of feature flags can be passed to
  agent factories so a single run can compare e.g. ``kg_on/kg_off`` agents.

The class is intentionally agent-agnostic: callers pass a mapping of
``name -> factory(player_id, seed, ablation)`` and the suite handles the rest.
"""
from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from src.agents.base_agent import MTGAgent
from src.orchestrator_legacy.game_runner import GameConfig, GameResult, GameRunner
from src.training.archetype_decks import ARCHETYPES, list_archetypes

logger = logging.getLogger(__name__)


AgentFactory = Callable[..., MTGAgent]


# ---------------------------------------------------------------------------
# Config / records
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    games_per_match: int = 10
    max_turns: int = 50
    archetypes: list[str] = field(default_factory=list_archetypes)
    seed: int = 0
    parallel: int = 4
    output_dir: Path = field(default_factory=lambda: Path("logs/benchmark"))
    elo_k: float = 32.0
    elo_init: float = 1000.0
    ablation: dict[str, Any] | None = None

    def derived_game_seed(self, idx: int) -> int:
        return (self.seed * 1_000_003) ^ (idx * 2654435761)


@dataclass
class GameRecord:
    match_id: str
    game_idx: int
    agent_a: str
    agent_b: str
    archetype_a: str
    archetype_b: str
    winner: str | None
    turns: int
    decision_time_a_s: float
    decision_time_b_s: float
    seed: int

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------
class StandardBenchmarkSuite:
    """Multi-agent, multi-archetype benchmark with deterministic replay."""

    def __init__(
        self,
        agents: Mapping[str, AgentFactory],
        config: BenchmarkConfig | None = None,
    ):
        if not agents:
            raise ValueError("StandardBenchmarkSuite needs at least one agent factory")
        self.agents: dict[str, AgentFactory] = dict(agents)
        self.config = config or BenchmarkConfig()
        self.records: list[GameRecord] = []
        self.elo: dict[str, float] = {name: self.config.elo_init for name in self.agents}

    # ------------------------------------------------------------------ run
    async def run(self) -> dict[str, Any]:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        names = list(self.agents)
        archs = cfg.archetypes
        if not archs:
            raise ValueError("BenchmarkConfig.archetypes must not be empty")

        match_specs: list[tuple[str, str, str, str]] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for arch_a in archs:
                    for arch_b in archs:
                        match_specs.append((a, b, arch_a, arch_b))

        sem = asyncio.Semaphore(max(1, cfg.parallel))
        tasks: list[Awaitable[list[GameRecord]]] = []
        for spec in match_specs:
            tasks.append(self._run_matchup(*spec, sem=sem))

        all_records: list[list[GameRecord]] = await asyncio.gather(*tasks)
        for batch in all_records:
            self.records.extend(batch)
            for rec in batch:
                self._update_elo(rec)

        summary = self._build_summary()
        self._write_outputs(summary)
        return summary

    # ------------------------------------------------------------ matchup
    async def _run_matchup(
        self,
        a: str,
        b: str,
        arch_a: str,
        arch_b: str,
        sem: asyncio.Semaphore,
    ) -> list[GameRecord]:
        cfg = self.config
        match_id = f"{a}_{arch_a}_vs_{b}_{arch_b}"
        out: list[GameRecord] = []
        for game_idx in range(cfg.games_per_match):
            async with sem:
                seed = cfg.derived_game_seed(
                    hash(match_id) ^ (game_idx + 1)
                ) & 0x7FFFFFFF
                rec = await self._run_single(
                    match_id, game_idx, a, b, arch_a, arch_b, seed
                )
                if rec is not None:
                    out.append(rec)
        return out

    async def _run_single(
        self,
        match_id: str,
        idx: int,
        name_a: str,
        name_b: str,
        arch_a: str,
        arch_b: str,
        seed: int,
    ) -> GameRecord | None:
        cfg = self.config
        random.seed(seed)
        try:
            agent_a = self._make_agent(name_a, "player_1", seed)
            agent_b = self._make_agent(name_b, "player_2", seed + 1)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Factory failure for %s/%s: %s", name_a, name_b, exc)
            return None

        runner = GameRunner(GameConfig(max_turns=cfg.max_turns))
        decks = {
            "player_1": ARCHETYPES[arch_a](),
            "player_2": ARCHETYPES[arch_b](),
        }
        timed = _DecisionTimer(agent_a, agent_b)
        timed.attach()
        t0 = time.perf_counter()
        try:
            result: GameResult = await runner.run_game(
                {"player_1": agent_a, "player_2": agent_b}, decks
            )
        except Exception as exc:
            logger.warning(
                "Game crashed (%s vs %s): %s", name_a, name_b, exc, exc_info=True
            )
            return None
        finally:
            timed.detach()
        elapsed = time.perf_counter() - t0
        winner_name: str | None
        if result.winner == "player_1":
            winner_name = name_a
        elif result.winner == "player_2":
            winner_name = name_b
        else:
            winner_name = None

        return GameRecord(
            match_id=match_id,
            game_idx=idx,
            agent_a=name_a,
            agent_b=name_b,
            archetype_a=arch_a,
            archetype_b=arch_b,
            winner=winner_name,
            turns=result.turns,
            decision_time_a_s=timed.total_a,
            decision_time_b_s=timed.total_b,
            seed=seed,
        )

    # ---------------------------------------------------------------- ELO
    def _update_elo(self, rec: GameRecord) -> None:
        cfg = self.config
        ra, rb = self.elo[rec.agent_a], self.elo[rec.agent_b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        if rec.winner == rec.agent_a:
            sa = 1.0
        elif rec.winner == rec.agent_b:
            sa = 0.0
        else:
            sa = 0.5
        delta = cfg.elo_k * (sa - ea)
        self.elo[rec.agent_a] = ra + delta
        self.elo[rec.agent_b] = rb - delta

    # ------------------------------------------------------------- agents
    def _make_agent(self, name: str, player_id: str, seed: int) -> MTGAgent:
        factory = self.agents[name]
        try:
            return factory(player_id=player_id, seed=seed, ablation=self.config.ablation)
        except TypeError:
            try:
                return factory(player_id=player_id, seed=seed)
            except TypeError:
                return factory(player_id)

    # --------------------------------------------------------------- summary
    def _build_summary(self) -> dict[str, Any]:
        win_counts: dict[str, dict[str, int]] = {n: {} for n in self.agents}
        for rec in self.records:
            for a, opp in ((rec.agent_a, rec.agent_b), (rec.agent_b, rec.agent_a)):
                win_counts[a].setdefault(opp, 0)
        per_agent: dict[str, dict[str, Any]] = {
            n: {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            for n in self.agents
        }
        per_archetype: dict[tuple[str, str], dict[str, int]] = {}

        for rec in self.records:
            per_agent[rec.agent_a]["games"] += 1
            per_agent[rec.agent_b]["games"] += 1
            arch_key = tuple(sorted((rec.archetype_a, rec.archetype_b)))
            per_archetype.setdefault(arch_key, {"games": 0, "decisive": 0})
            per_archetype[arch_key]["games"] += 1
            if rec.winner == rec.agent_a:
                per_agent[rec.agent_a]["wins"] += 1
                per_agent[rec.agent_b]["losses"] += 1
                per_archetype[arch_key]["decisive"] += 1
            elif rec.winner == rec.agent_b:
                per_agent[rec.agent_b]["wins"] += 1
                per_agent[rec.agent_a]["losses"] += 1
                per_archetype[arch_key]["decisive"] += 1
            else:
                per_agent[rec.agent_a]["draws"] += 1
                per_agent[rec.agent_b]["draws"] += 1

        for stats in per_agent.values():
            stats["winrate"] = (
                stats["wins"] / stats["games"] if stats["games"] else 0.0
            )

        return {
            "config": {
                "seed": self.config.seed,
                "games_per_match": self.config.games_per_match,
                "max_turns": self.config.max_turns,
                "archetypes": self.config.archetypes,
                "parallel": self.config.parallel,
                "ablation": self.config.ablation,
            },
            "per_agent": per_agent,
            "elo": dict(sorted(self.elo.items(), key=lambda kv: -kv[1])),
            "per_archetype_pair": {
                f"{k[0]}__vs__{k[1]}": v for k, v in per_archetype.items()
            },
            "total_games": len(self.records),
        }

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # CSV (one row per game)
        csv_path = out / "games.csv"
        if self.records:
            fieldnames = list(self.records[0].as_row().keys())
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                for rec in self.records:
                    w.writerow(rec.as_row())

        # JSON summary
        with (out / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Benchmark wrote %d games to %s", len(self.records), out)


# ---------------------------------------------------------------------------
# Decision-time instrumentation helper
# ---------------------------------------------------------------------------
class _DecisionTimer:
    """Wraps two agents' `decide_action` to accumulate wall-clock time."""

    def __init__(self, agent_a: MTGAgent, agent_b: MTGAgent):
        self.a = agent_a
        self.b = agent_b
        self._orig_a = None
        self._orig_b = None
        self.total_a = 0.0
        self.total_b = 0.0

    def attach(self) -> None:
        self._orig_a = self.a.decide_action
        self._orig_b = self.b.decide_action

        async def wrap_a(state, legal):
            t = time.perf_counter()
            res = await self._orig_a(state, legal)
            self.total_a += time.perf_counter() - t
            return res

        async def wrap_b(state, legal):
            t = time.perf_counter()
            res = await self._orig_b(state, legal)
            self.total_b += time.perf_counter() - t
            return res

        self.a.decide_action = wrap_a  # type: ignore[assignment]
        self.b.decide_action = wrap_b  # type: ignore[assignment]

    def detach(self) -> None:
        if self._orig_a is not None:
            self.a.decide_action = self._orig_a  # type: ignore[assignment]
        if self._orig_b is not None:
            self.b.decide_action = self._orig_b  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default factory registry
# ---------------------------------------------------------------------------
def default_baseline_factories(
    include_llm: bool = False,
) -> dict[str, AgentFactory]:
    """Return the standard baseline agent set used by the paper's tables."""
    from src.agents.heuristic_agent import HeuristicAgent
    from src.agents.random_agent import RandomAgent

    factories: dict[str, AgentFactory] = {
        "Random": lambda player_id, seed=0, ablation=None: RandomAgent(
            player_id=player_id
        ),
        "Heuristic": lambda player_id, seed=0, ablation=None: HeuristicAgent(
            player_id=player_id, seed=seed
        ),
    }
    if include_llm:  # optional, requires an Ollama server
        try:
            from src.agents.llm_agent import OllamaAgent

            factories["LLM"] = lambda player_id, seed=0, ablation=None: OllamaAgent(
                player_id=player_id
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM baseline unavailable: %s", exc)
    return factories


async def run_default_benchmark(
    games_per_match: int = 5,
    max_turns: int = 30,
    parallel: int = 2,
    seed: int = 0,
    archetypes: Iterable[str] | None = None,
    output_dir: str | Path = "logs/benchmark",
    include_llm: bool = False,
) -> dict[str, Any]:
    """One-call entry point used by the pipeline and the tests."""
    cfg = BenchmarkConfig(
        games_per_match=games_per_match,
        max_turns=max_turns,
        archetypes=list(archetypes) if archetypes else list_archetypes()[:3],
        seed=seed,
        parallel=parallel,
        output_dir=Path(output_dir),
    )
    suite = StandardBenchmarkSuite(default_baseline_factories(include_llm), cfg)
    return await suite.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description="Run the standard MTG agent benchmark suite",
    )
    p.add_argument("--games-per-match", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--archetypes",
        nargs="*",
        default=None,
        help="Subset of archetype names; defaults to the first three.",
    )
    p.add_argument("--output-dir", default="logs/benchmark")
    p.add_argument(
        "--include-llm",
        action="store_true",
        help="Include the Ollama LLM baseline (requires a running Ollama server).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_argparser().parse_args(argv)
    summary = asyncio.run(
        run_default_benchmark(
            games_per_match=args.games_per_match,
            max_turns=args.max_turns,
            parallel=args.parallel,
            seed=args.seed,
            archetypes=args.archetypes,
            output_dir=args.output_dir,
            include_llm=args.include_llm,
        )
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

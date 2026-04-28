#!/usr/bin/env python
"""First ablation: Two small LLM agents playing against each other (no training).

Pure rule-based games where two LLMs reason about each decision via Ollama.
By default, alternates which model goes first across games to remove the
going-first bias (MTG strongly favors the player who goes first).

Usage:
    python examples/ablation_llm_only.py
    python examples/ablation_llm_only.py --model1 gemma --model2 llama --games 4
    python examples/ablation_llm_only.py --games 6 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

# Force UTF-8 stdout/stderr on Windows (avoid cp1252 UnicodeEncodeError on
# punctuation like em-dash inside game logs / agent outputs).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_agent import OllamaAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.utils.seeding import set_global_seed


MODEL_ALIASES = {
    "gemma": "gemma4:e2b",
    "gemma4": "gemma4:e2b",
    "gemma4-e2b": "gemma4:e2b",
    "llama": "llama3.2:1b",
    "qwen": "qwen2.5-coder:1.5b",
    "phi": "phi:latest",
    "stable-code": "stable-code:3b-code-q4_0",
    "nemotron": "nemotron-3-nano:4b",
    "1b": "llama3.2:1b",
    "1.5b": "qwen2.5-coder:1.5b",
    "2b": "gemma4:e2b",
    "3b": "stable-code:3b-code-q4_0",
    "4b": "nemotron-3-nano:4b",
}


@dataclass
class AgentStats:
    model: str
    seat: str  # "first" or "second"
    llm_calls_total: int = 0
    llm_calls_success: int = 0
    llm_calls_failed: int = 0
    fallback_invocations: int = 0
    actions_chosen: dict = field(default_factory=dict)
    mulligan_decisions: list = field(default_factory=list)


@dataclass
class GameRecord:
    game_num: int
    model_first: str
    model_second: str
    winner_seat: Optional[str]
    winner_model: Optional[str]
    turns: int
    elapsed_sec: float
    stats_first: Optional[AgentStats] = None
    stats_second: Optional[AgentStats] = None
    final_log_lines: int = 0
    log_path: Optional[str] = None

    def summary(self) -> str:
        if self.winner_model:
            return (f"Game {self.game_num}: {self.winner_model} won as "
                    f"{self.winner_seat} in {self.turns} turns "
                    f"({self.elapsed_sec:.1f}s)")
        return (f"Game {self.game_num}: {self.winner_seat} in "
                f"{self.turns} turns ({self.elapsed_sec:.1f}s)")


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def _build_simple_deck() -> list[dict]:
    return (
        [{"name": "Mountain", "type_line": "Basic Land - Mountain", "cmc": 0}
         for _ in range(10)]
        + [{"name": "Goblin Guide", "type_line": "Creature - Goblin Scout", "cmc": 1}
           for _ in range(6)]
        + [{"name": "Lightning Bolt", "type_line": "Instant", "cmc": 1}
           for _ in range(6)]
        + [{"name": "Goblin King", "type_line": "Creature - Goblin", "cmc": 2}
           for _ in range(5)]
        + [{"name": "Chandra", "type_line": "Planeswalker", "cmc": 3}
           for _ in range(3)]
    )


def _agent_stats_from(agent: OllamaAgent, model_label: str, seat: str) -> AgentStats:
    s = agent.stats
    return AgentStats(
        model=model_label,
        seat=seat,
        llm_calls_total=s["llm_calls_total"],
        llm_calls_success=s["llm_calls_success"],
        llm_calls_failed=s["llm_calls_failed"],
        fallback_invocations=s["fallback_invocations"],
        actions_chosen=dict(s["actions_chosen"]),
        mulligan_decisions=list(s["mulligan_decisions"]),
    )


async def play_one_game(model_first: str, model_second: str, game_num: int,
                        log_dir: Optional[Path], verbose: bool = False) -> GameRecord:
    """Play one game where model_first is seated as player1 (goes first)."""
    start = time.time()
    m1_tag = resolve_model_name(model_first)
    m2_tag = resolve_model_name(model_second)

    agent_first = OllamaAgent("player1", name=f"LLM[{model_first}]", model=m1_tag)
    agent_second = OllamaAgent("player2", name=f"LLM[{model_second}]", model=m2_tag)

    if not agent_first._ollama_available or not agent_second._ollama_available:
        print(f"[WARN] Game {game_num}: One or both models unavailable, skipping")
        return GameRecord(game_num=game_num, model_first=model_first,
                          model_second=model_second, winner_seat="error",
                          winner_model=None, turns=0, elapsed_sec=0.0)

    deck = _build_simple_deck() * 2

    config = GameConfig(max_turns=30, mulligan_enabled=True, max_mulligans=3)
    runner = GameRunner(config)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Game {game_num}: {model_first} (FIRST) vs {model_second} (SECOND)")
        print(f"{'='*70}")

    try:
        result = await runner.run_game(
            agents={"player1": agent_first, "player2": agent_second},
            decks={"player1": deck, "player2": deck},
        )
        elapsed = time.time() - start

        if result.winner == "player1":
            winner_seat, winner_model = "first", model_first
        elif result.winner == "player2":
            winner_seat, winner_model = "second", model_second
        else:
            winner_seat, winner_model = "draw", None

        log_path: Optional[str] = None
        log_lines = result.log if isinstance(result.log, list) else []
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_first = model_first.replace(':', '-').replace('/', '-')
            safe_second = model_second.replace(':', '-').replace('/', '-')
            log_file = log_dir / f"game_{game_num:03d}_{safe_first}_vs_{safe_second}.log"
            log_file.write_text("\n".join(str(line) for line in log_lines), encoding="utf-8")
            log_path = str(log_file)

        record = GameRecord(
            game_num=game_num,
            model_first=model_first,
            model_second=model_second,
            winner_seat=winner_seat,
            winner_model=winner_model,
            turns=result.turns,
            elapsed_sec=elapsed,
            stats_first=_agent_stats_from(agent_first, model_first, "first"),
            stats_second=_agent_stats_from(agent_second, model_second, "second"),
            final_log_lines=len(log_lines),
            log_path=log_path,
        )

        if verbose:
            _print_game_diagnostics(record)
        return record

    except Exception as e:
        elapsed = time.time() - start
        print(f"[ERROR] Game {game_num} error: {e}")
        import traceback
        traceback.print_exc()
        return GameRecord(game_num=game_num, model_first=model_first,
                          model_second=model_second, winner_seat="error",
                          winner_model=None, turns=0, elapsed_sec=elapsed)


def _print_game_diagnostics(rec: GameRecord) -> None:
    print(f"\n--- Diagnostics for game {rec.game_num} ---")
    for stats in (rec.stats_first, rec.stats_second):
        if stats is None:
            continue
        total = stats.llm_calls_total + stats.fallback_invocations
        actions = ", ".join(f"{k}:{v}" for k, v in stats.actions_chosen.items() if v)
        success_rate = (stats.llm_calls_success / max(stats.llm_calls_total, 1)) * 100
        print(f"  [{stats.seat:6s}] {stats.model:20s} "
              f"LLM={stats.llm_calls_success}/{stats.llm_calls_total} "
              f"({success_rate:.0f}% ok), fallback={stats.fallback_invocations}, "
              f"total={total}")
        if actions:
            print(f"           actions: {actions}")
        if stats.mulligan_decisions:
            mull = ", ".join(f"#{i}:{('keep' if k else 'mull')}"
                             for i, k in stats.mulligan_decisions)
            print(f"           mulligan: {mull}")
    if rec.log_path:
        print(f"  Game log: {rec.log_path}")
    print()


async def run_ablation(model1: str, model2: str, num_games: int,
                       seed: Optional[int], log_dir: Optional[Path],
                       no_alternate: bool = False, verbose: bool = False) -> None:
    if seed is not None:
        set_global_seed(seed)

    print(f"\n{'='*70}")
    print(f"LLM-ONLY ABLATION: {model1} vs {model2}")
    print(f"{'='*70}")
    print(f"Games: {num_games}")
    print(f"Alternate first player: {not no_alternate}")
    print(f"Seed: {seed}")
    print(f"Log dir: {log_dir or '(not saved)'}")
    print(f"{'='*70}\n")

    records: list[GameRecord] = []
    for game_num in range(1, num_games + 1):
        if no_alternate or game_num % 2 == 1:
            first, second = model1, model2
        else:
            first, second = model2, model1
        rec = await play_one_game(first, second, game_num,
                                  log_dir=log_dir, verbose=verbose)
        records.append(rec)
        print(rec.summary())

    _print_summary(model1, model2, records, log_dir=log_dir)


def _print_summary(model1: str, model2: str, records: list[GameRecord],
                   log_dir: Optional[Path]) -> None:
    valid = [r for r in records if r.winner_seat not in (None, "error")]
    if not valid:
        print("\nNo valid games to summarize.")
        return

    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    model_wins = Counter(r.winner_model for r in valid if r.winner_model)
    seat_wins = Counter(r.winner_seat for r in valid
                        if r.winner_seat in ("first", "second"))
    draws = sum(1 for r in valid if r.winner_seat == "draw")

    print(f"Total games: {len(valid)}")
    print()
    print("Wins by MODEL (averaged across seats):")
    print(f"  {model1:25s} {model_wins.get(model1, 0)}")
    print(f"  {model2:25s} {model_wins.get(model2, 0)}")
    print(f"  Draws                     {draws}")
    print()
    print("Wins by SEAT (going-first bias check):")
    print(f"  First player wins:  {seat_wins.get('first', 0)}")
    print(f"  Second player wins: {seat_wins.get('second', 0)}")

    avg_turns = sum(r.turns for r in valid) / len(valid)
    avg_time = sum(r.elapsed_sec for r in valid) / len(valid)
    print()
    print(f"Avg turns: {avg_turns:.1f}")
    print(f"Avg time/game: {avg_time:.1f}s")

    llm_total = llm_success = fallback = 0
    for rec in valid:
        for stats in (rec.stats_first, rec.stats_second):
            if stats is None:
                continue
            llm_total += stats.llm_calls_total
            llm_success += stats.llm_calls_success
            fallback += stats.fallback_invocations
    print()
    print("LLM diagnostics (across all games):")
    if llm_total > 0:
        print(f"  LLM calls succeeded: {llm_success}/{llm_total} "
              f"({100*llm_success/llm_total:.1f}%)")
    print(f"  Fallback invocations: {fallback}")
    if fallback > llm_success * 0.1 and llm_success > 0:
        print(f"  [WARN] High fallback rate -- LLM may be timing out")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "model1": model1, "model2": model2,
            "model_wins": dict(model_wins), "seat_wins": dict(seat_wins),
            "draws": draws, "avg_turns": avg_turns, "avg_time": avg_time,
            "games": [
                {**asdict(r),
                 "stats_first": asdict(r.stats_first) if r.stats_first else None,
                 "stats_second": asdict(r.stats_second) if r.stats_second else None}
                for r in records
            ],
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nSummary JSON: {summary_path}")
    print(f"{'='*70}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model1", default="gemma4:e2b",
                   help="First model (default: gemma4:e2b)")
    p.add_argument("--model2", default="gemma4:e2b",
                   help="Second model (default: gemma4:e2b — mirror match)")
    p.add_argument("--games", type=int, default=2,
                   help="Number of games (default: 2 — one per side if alternating)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility")
    p.add_argument("--no-alternate", action="store_true",
                   help="Don't alternate first player (model1 always first)")
    p.add_argument("--log-dir", type=Path,
                   default=Path("runs/ablation/llm_only"),
                   help="Directory to write per-game logs and summary.json")
    p.add_argument("--no-log", action="store_true",
                   help="Don't write any game logs to disk")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-game diagnostics")
    return p


async def main(args) -> int:
    log_dir = None if args.no_log else args.log_dir
    await run_ablation(args.model1, args.model2, args.games, args.seed,
                       log_dir=log_dir, no_alternate=args.no_alternate,
                       verbose=args.verbose)
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main(args)))

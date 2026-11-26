#!/usr/bin/env python
"""Benchmark trained world-model checkpoints against fixed baselines.

This script closes the gap between "we have checkpoints" and
"we have a benchmark that compares trained agents against heuristic,
random, and pure-LLM baselines".

It also supports selecting the best checkpoint with an
active-inference-inspired score from ``src.training.world_model_selection``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_matchups import play_one_game
from src.training.world_model_selection import (
    ActiveInferenceSelectionConfig,
    BenchmarkCell,
    WorldModelCandidate,
    score_candidate,
    select_best_candidate,
)
from src.utils.seeding import derive_seed, set_global_seed


async def _run_checkpoint_vs_baseline(
    checkpoint: str,
    baseline: str,
    deck_paths: list[Path],
    games: int,
    max_turns: int,
    seed: int,
) -> BenchmarkCell:
    wins = 0
    turns: list[float] = []
    elapsed: list[float] = []

    for game_idx in range(games):
        pair = ["world_model", baseline] if game_idx % 2 == 0 else [baseline, "world_model"]
        rec = await play_one_game(
            agent_names=pair,
            deck_paths=deck_paths,
            format="standard",
            max_turns=max_turns,
            seed=derive_seed(seed, checkpoint, baseline, game_idx),
            log_dir=None,
            game_idx=game_idx + 1,
            wm_checkpoint=checkpoint,
        )
        if rec.get("winner_agent") == "world_model":
            wins += 1
        turns.append(float(rec.get("turns", 0)))
        elapsed.append(float(rec.get("elapsed_sec", 0.0)))

    return BenchmarkCell(
        baseline=baseline,
        win_rate=wins / games if games else 0.0,
        avg_turns=sum(turns) / len(turns) if turns else 0.0,
        avg_elapsed_sec=sum(elapsed) / len(elapsed) if elapsed else 0.0,
        games=games,
    )


async def run(args: argparse.Namespace) -> int:
    set_global_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    deck_paths = [Path(p) for p in args.decks]

    candidates: list[WorldModelCandidate] = []
    rows: list[dict[str, object]] = []

    for checkpoint in args.checkpoints:
        candidate = WorldModelCandidate(checkpoint=checkpoint)
        for baseline in args.baselines:
            cell = await _run_checkpoint_vs_baseline(
                checkpoint=checkpoint,
                baseline=baseline,
                deck_paths=deck_paths,
                games=args.games,
                max_turns=args.max_turns,
                seed=args.seed,
            )
            candidate.cells.append(cell)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "baseline": baseline,
                    "win_rate": round(cell.win_rate, 4),
                    "avg_turns": round(cell.avg_turns, 3),
                    "avg_elapsed_sec": round(cell.avg_elapsed_sec, 3),
                    "games": cell.games,
                }
            )
        candidates.append(candidate)

    selector_cfg = ActiveInferenceSelectionConfig(
        pragmatic_weight=args.pragmatic_weight,
        epistemic_weight=args.epistemic_weight,
        latency_weight=args.latency_weight,
        horizon_weight=args.horizon_weight,
        loss_weight=args.loss_weight,
    )
    best = select_best_candidate(candidates, config=selector_cfg)
    selection_rows = []
    for candidate in candidates:
        scored = score_candidate(candidate, config=selector_cfg)
        selection_rows.append(
            {
                "checkpoint": scored.checkpoint,
                "expected_free_energy": round(scored.expected_free_energy, 6),
                "pragmatic_value": round(scored.pragmatic_value, 4),
                "epistemic_value": round(scored.epistemic_value, 4),
                "latency_cost": round(scored.latency_cost, 4),
                "horizon_cost": round(scored.horizon_cost, 4),
                "loss_cost": round(scored.loss_cost, 4),
            }
        )

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with (out_dir / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)

    summary = {
        "checkpoints": args.checkpoints,
        "baselines": args.baselines,
        "games_per_baseline": args.games,
        "best_checkpoint": best.checkpoint,
        "best_expected_free_energy": best.expected_free_energy,
        "selection_config": vars(args) | {
            "decks": args.decks,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["heuristic", "random", "llm", "active_inference"],
    )
    parser.add_argument(
        "--decks",
        nargs="+",
        default=[
            "data/decks/modern/modern_mono_red_burn.txt",
            "data/decks/modern/modern_azorius_control.txt",
        ],
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="runs/trained_benchmark")
    parser.add_argument("--pragmatic-weight", type=float, default=1.0)
    parser.add_argument("--epistemic-weight", type=float, default=0.6)
    parser.add_argument("--latency-weight", type=float, default=0.15)
    parser.add_argument("--horizon-weight", type=float, default=0.05)
    parser.add_argument("--loss-weight", type=float, default=0.25)
    return parser


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

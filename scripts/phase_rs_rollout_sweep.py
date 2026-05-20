#!/usr/bin/env python
"""Run phase-rs rollout sweeps for policy benchmarking.

This is an *episode-level rollout* harness: each rollout is a full game on
phase-rs (Rust engine), driven by one of our Python pickers against a built-in
phase-ai difficulty.

It does not yet do in-turn branch dreaming from arbitrary intermediate states.
That requires a full Action/GameState translator and server-side clone/fork
support.

Examples:

  python scripts/phase_rs_rollout_sweep.py --games-per-cell 3 --autostart
  python scripts/phase_rs_rollout_sweep.py --pickers ollama --difficulties Hard VeryHard --autostart
  python scripts/phase_rs_rollout_sweep.py --our-deck-file data/decks/modern/modern_mono_red_burn.txt --autostart
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.integrations.phase_rs import (
    AgentActionPicker,
    HeuristicActionPicker,
    OllamaActionPicker,
    PhaseServerConfig,
    PreferNonPassPicker,
    RandomActionPicker,
    STARTER_DECK_NAMES,
    load_deck_data,
    run_game_sync,
)
from src.agents import make_agent


@dataclass
class RolloutRow:
    picker: str
    ai_difficulty: str
    game_index: int
    seed: int
    winner_seat: int | None
    our_seat: int
    result: str
    reason: str
    turns: int
    actions: int
    elapsed_sec: float


def _build_picker(name: str, seed: int, args: argparse.Namespace):
    if name == "random":
        return RandomActionPicker(seed=seed)
    if name == "prefer-nonpass":
        return PreferNonPassPicker(seed=seed)
    if name == "heuristic":
        return HeuristicActionPicker(seed=seed)
    if name == "ollama":
        return OllamaActionPicker(
            seed=seed,
            model=args.ollama_model,
            base_url=args.ollama_url,
        )
    if name.startswith("agent:"):
        agent_name = name.split(":", 1)[1].strip()
        if not agent_name:
            raise ValueError("agent picker requires a name, e.g. agent:heuristic")
        agent = make_agent(agent_name, player_id="seat0", seed=seed)
        return AgentActionPicker(agent=agent, name=f"phase_rs_agent:{agent_name}")
    raise ValueError(f"unsupported picker: {name}")


def _load_our_deck(args: argparse.Namespace) -> dict:
    if args.our_deck_file:
        return load_deck_data(args.our_deck_file)
    return {"main_deck": [], "sideboard": [], "commander": []}


def run(args: argparse.Namespace) -> int:
    for deck_name in args.ai_decks:
        if deck_name not in STARTER_DECK_NAMES:
            raise SystemExit(
                f"--ai-decks entries must be in {STARTER_DECK_NAMES}; got {deck_name!r}"
            )

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[RolloutRow] = []
    cell_summary: list[dict[str, object]] = []
    max_retries = args.max_retries if hasattr(args, "max_retries") else 2

    for picker_name in args.pickers:
        for ai_difficulty in args.difficulties:
            for ai_deck in args.ai_decks:
                wins = 0
                losses = 0
                draws = 0
                timeouts = 0
                elapsed_total = 0.0
                turns_total = 0
                actions_total = 0

                for g in range(args.games_per_cell):
                    seed = args.seed + g
                    result = None
                    for attempt in range(1, max_retries + 1):
                        picker = _build_picker(picker_name, seed, args)
                        cfg = PhaseServerConfig(uri=args.uri, stream_timeout_s=args.stream_timeout)

                        started = time.perf_counter()
                        result = run_game_sync(
                            deck=_load_our_deck(args),
                            picker=picker,
                            config=cfg,
                            ai_difficulty=ai_difficulty,
                            ai_deck_name=ai_deck,
                            max_actions=args.max_actions,
                            autostart_server=args.autostart,
                            format_name=args.format,
                        )
                        elapsed = time.perf_counter() - started
                        # If succeeded or not a transient failure, stop retrying.
                        if result.reason != "stream_timeout" or attempt == max_retries:
                            break
                        print(f"[retry {attempt}/{max_retries}] cell {picker_name}/{ai_difficulty}/{ai_deck} game {g+1}: {result.reason}")
                    if result is None:
                        continue  # Skip this game

                    if result.winner_seat is None:
                        outcome = "draw"
                        draws += 1
                    elif result.winner_seat == result.our_seat:
                        outcome = "win"
                        wins += 1
                    else:
                        outcome = "loss"
                        losses += 1

                    if result.reason == "stream_timeout":
                        timeouts += 1

                    row = RolloutRow(
                        picker=picker_name,
                        ai_difficulty=ai_difficulty,
                        game_index=g + 1,
                        seed=seed,
                        winner_seat=result.winner_seat,
                        our_seat=result.our_seat,
                        result=outcome,
                        reason=result.reason,
                        turns=result.turns_observed,
                        actions=result.actions_sent,
                        elapsed_sec=elapsed,
                    )
                    all_rows.append(row)

                    if result.trace:
                        trace_dir = out_dir / "traces"
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        safe_picker = picker_name.replace(":", "-").replace("/", "-").replace(" ", "_")
                        safe_deck = ai_deck.replace(":", "-").replace("/", "-").replace(" ", "_")
                        (trace_dir / f"{safe_picker}_{ai_difficulty}_{safe_deck}_{g+1:04d}.jsonl").write_text(
                            "\n".join(json.dumps(evt) for evt in result.trace) + "\n",
                            encoding="utf-8",
                        )

                    elapsed_total += elapsed
                    turns_total += result.turns_observed
                    actions_total += result.actions_sent

                    print(
                        f"[{picker_name}/{ai_difficulty}/{ai_deck}] "
                        f"game {g+1}/{args.games_per_cell}: {outcome} "
                        f"(reason={result.reason}, turns={result.turns_observed}, {elapsed:.2f}s)"
                    )

                n = args.games_per_cell
                cell_summary.append(
                    {
                        "picker": picker_name,
                        "ai_difficulty": ai_difficulty,
                        "ai_deck": ai_deck,
                        "games": n,
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "timeouts": timeouts,
                        "win_rate": (wins / n) if n else 0.0,
                        "avg_elapsed_sec": (elapsed_total / n) if n else 0.0,
                        "avg_turns": (turns_total / n) if n else 0.0,
                        "avg_actions": (actions_total / n) if n else 0.0,
                    }
                )

    with (out_dir / "rollouts.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(asdict(row)) + "\n")

    csv_path = out_dir / "summary.csv"
    if cell_summary:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cell_summary[0].keys()))
            writer.writeheader()
            writer.writerows(cell_summary)

    summary = {
        "pickers": args.pickers,
        "difficulties": args.difficulties,
        "ai_decks": args.ai_decks,
        "format": args.format or "Standard",
        "games_per_cell": args.games_per_cell,
        "autostart": args.autostart,
        "seed": args.seed,
        "uri": args.uri,
        "stream_timeout": args.stream_timeout,
        "cells": cell_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== phase-rs rollout sweep complete ===")
    print(json.dumps(summary, indent=2))
    print(f"output: {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pickers",
        nargs="+",
        default=["random", "prefer-nonpass", "heuristic"],
        help=(
            "One or more: random, prefer-nonpass, heuristic, ollama, "
            "agent:<name> (e.g. agent:heuristic, agent:active_inference)"
        ),
    )
    p.add_argument(
        "--difficulties",
        nargs="+",
        default=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
    )
    p.add_argument("--ai-decks", nargs="+", default=["Red Deck Wins"])
    p.add_argument("--games-per-cell", type=int, default=3)
    p.add_argument("--our-deck-file", default=None)
    p.add_argument("--format", default=None, help="Game format (Standard, Pioneer, Modern, Commander, etc.). Defaults to Standard.")
    p.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    p.add_argument("--autostart", action="store_true")
    p.add_argument(
        "--stream-timeout",
        type=float,
        default=180.0,
        help="Per-message timeout in seconds while game is streaming (default 180s). "
        "Increase if phase-ai decisions chain longer than expected.",
    )
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-retries", type=int, default=2, help="Retry transient failures per game cell.")
    p.add_argument("--output-dir", default="runs/phase_rs_rollout_sweep")
    p.add_argument("--ollama-model", default="gemma4:e2b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

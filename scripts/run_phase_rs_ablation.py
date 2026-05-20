#!/usr/bin/env python
"""Ablation runner: our phase-rs picker policies vs phase-rs built-in AI.

This keeps the rules backend and opponent policy inside phase-rs, while our
contribution is the decision layer on the Python side (LLM picker, future
KG-aware pickers, etc.).

Examples:

  python scripts/run_phase_rs_ablation.py --games 4 --picker random --autostart
  python scripts/run_phase_rs_ablation.py --games 6 --picker ollama --ollama-model gemma4:e2b --autostart
  python scripts/run_phase_rs_ablation.py --games 8 --picker ollama --ai-difficulty Hard --our-deck-file data/decks/modern_burn.txt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
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
class GameRow:
    game_index: int
    winner_seat: int | None
    our_seat: int
    reason: str
    turns: int
    actions: int


def _make_picker(args: argparse.Namespace, seed: int):
    if args.picker == "random":
        return RandomActionPicker(seed=seed)
    if args.picker == "prefer-nonpass":
        return PreferNonPassPicker(seed=seed)
    if args.picker == "heuristic":
        return HeuristicActionPicker(seed=seed)
    if args.picker == "ollama":
        return OllamaActionPicker(
            seed=seed,
            model=args.ollama_model,
            base_url=args.ollama_url,
        )
    if args.picker.startswith("agent:"):
        agent_name = args.picker.split(":", 1)[1].strip()
        if not agent_name:
            raise ValueError("agent picker requires a name, e.g. agent:heuristic")
        # seat id is corrected by AgentActionPicker at runtime.
        agent = make_agent(agent_name, player_id="seat0", seed=seed)
        return AgentActionPicker(agent=agent, name=f"phase_rs_agent:{agent_name}")
    raise ValueError(f"unsupported picker: {args.picker}")


def _load_our_deck(args: argparse.Namespace) -> dict:
    if args.our_deck_file:
        return load_deck_data(args.our_deck_file)
    # Starter deck data lives server-side; for the host seat we keep the
    # existing adapter behavior and send an empty DeckData placeholder.
    return {"main_deck": [], "sideboard": [], "commander": []}


def run(args: argparse.Namespace) -> int:
    if args.ai_deck not in STARTER_DECK_NAMES:
        raise SystemExit(
            f"--ai-deck must be one of {STARTER_DECK_NAMES}, got {args.ai_deck!r}"
        )

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rows: list[GameRow] = []
    max_retries = args.max_retries if hasattr(args, "max_retries") else 2

    for i in range(args.games):
        game_seed = args.seed + i
        result = None
        for attempt in range(1, max_retries + 1):
            picker = _make_picker(args, seed=game_seed)
            cfg = PhaseServerConfig(uri=args.uri, stream_timeout_s=args.stream_timeout)
            result = run_game_sync(
                deck=_load_our_deck(args),
                picker=picker,
                config=cfg,
                ai_difficulty=args.ai_difficulty,
                ai_deck_name=args.ai_deck,
                max_actions=args.max_actions,
                autostart_server=args.autostart,
            )
            # If succeeded or not a transient failure, stop retrying.
            if result.reason != "stream_timeout" or attempt == max_retries:
                break
            print(f"[retry {attempt}/{max_retries}] game {i+1}: {result.reason}")
        if result is None:
            continue  # Skip this game
        row = GameRow(
            game_index=i + 1,
            winner_seat=result.winner_seat,
            our_seat=result.our_seat,
            reason=result.reason,
            turns=result.turns_observed,
            actions=result.actions_sent,
        )
        rows.append(row)

        if result.trace:
            trace_dir = out_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"game_{i+1:04d}.jsonl").write_text(
                "\n".join(json.dumps(evt) for evt in result.trace) + "\n",
                encoding="utf-8",
            )

        outcome = "win" if result.winner_seat == result.our_seat else "loss"
        if result.winner_seat is None:
            outcome = "draw/unknown"
        print(
            f"game {i+1}/{args.games}: {outcome} "
            f"(our_seat={result.our_seat}, winner={result.winner_seat}, turns={result.turns_observed})"
        )

    total = len(rows)
    wins = sum(1 for r in rows if r.winner_seat == r.our_seat)
    losses = sum(1 for r in rows if r.winner_seat is not None and r.winner_seat != r.our_seat)
    draws = total - wins - losses
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row.reason] = reason_counts.get(row.reason, 0) + 1
    summary = {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / total) if total else 0.0,
        "reason_counts": reason_counts,
        "picker": args.picker,
        "ai_difficulty": args.ai_difficulty,
        "ai_deck": args.ai_deck,
        "autostart": args.autostart,
        "seed": args.seed,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "games.jsonl").write_text(
        "\n".join(json.dumps(asdict(r)) for r in rows) + "\n",
        encoding="utf-8",
    )

    print("\n=== phase-rs ablation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"output: {out_dir}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument(
        "--picker",
        help=(
            "Policy for our seat: random | prefer-nonpass | heuristic | ollama "
            "| agent:<name> (e.g. agent:heuristic, agent:world_model, "
            "agent:active_inference, agent:fusion)"
        ),
        default="random",
    )
    parser.add_argument("--ollama-model", default="gemma4:e2b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--ai-difficulty",
        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
        default="Medium",
    )
    parser.add_argument("--ai-deck", default="Red Deck Wins")
    parser.add_argument("--our-deck-file", default=None)
    parser.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=45.0,
        help="Max seconds to wait for the next server message before ending a game with reason=stream_timeout",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--max-retries", type=int, default=2, help="Retry transient failures per game.")
    parser.add_argument("--output-dir", default="runs/phase_rs_ablation")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

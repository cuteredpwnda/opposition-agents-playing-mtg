#!/usr/bin/env python
"""Collect phase-rs gameplay traces for training datasets.

Each run plays N games on phase-rs and writes:
- `games.jsonl`: one row per game outcome/metadata
- `trace_events.jsonl`: flattened per-event stream with game ids

This is the bridge dataset for training world-model/policy components on
phase-rs transitions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.agents import make_agent
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


@dataclass
class GameRow:
    game_id: str
    game_index: int
    picker: str
    ai_difficulty: str
    ai_deck: str
    seed: int
    winner_seat: int | None
    our_seat: int
    reason: str
    turns: int
    actions: int


def _build_picker(name: str, seed: int, args: argparse.Namespace):
    if name == "random":
        return RandomActionPicker(seed=seed)
    if name == "prefer-nonpass":
        return PreferNonPassPicker(seed=seed)
    if name == "heuristic":
        return HeuristicActionPicker(seed=seed)
    if name == "ollama":
        return OllamaActionPicker(seed=seed, model=args.ollama_model, base_url=args.ollama_url)
    if name.startswith("agent:"):
        agent_name = name.split(":", 1)[1].strip()
        agent = make_agent(agent_name, player_id="seat0", seed=seed)
        return AgentActionPicker(agent=agent, name=f"phase_rs_agent:{agent_name}")
    raise ValueError(f"unsupported picker: {name}")


def _load_our_deck(args: argparse.Namespace) -> dict:
    if args.our_deck_file:
        return load_deck_data(args.our_deck_file)
    return {"main_deck": [], "sideboard": [], "commander": []}


def run(args: argparse.Namespace) -> int:
    if args.ai_deck not in STARTER_DECK_NAMES:
        raise SystemExit(
            f"--ai-deck must be one of {STARTER_DECK_NAMES}, got {args.ai_deck!r}"
        )

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[GameRow] = []
    events_path = out_dir / "trace_events.jsonl"

    with events_path.open("w", encoding="utf-8") as events_file:
        for i in range(args.games):
            seed = args.seed + i
            picker = _build_picker(args.picker, seed, args)
            cfg = PhaseServerConfig(uri=args.uri, stream_timeout_s=args.stream_timeout)
            result = run_game_sync(
                deck=_load_our_deck(args),
                picker=picker,
                config=cfg,
                ai_difficulty=args.ai_difficulty,
                ai_deck_name=args.ai_deck,
                max_actions=args.max_actions,
                autostart_server=args.autostart,
                format_name=args.format,
            )

            game_id = f"g{i+1:05d}"
            rows.append(
                GameRow(
                    game_id=game_id,
                    game_index=i + 1,
                    picker=args.picker,
                    ai_difficulty=args.ai_difficulty,
                    ai_deck=args.ai_deck,
                    seed=seed,
                    winner_seat=result.winner_seat,
                    our_seat=result.our_seat,
                    reason=result.reason,
                    turns=result.turns_observed,
                    actions=result.actions_sent,
                )
            )

            for event in result.trace:
                events_file.write(json.dumps({"game_id": game_id, **event}) + "\n")

            print(
                f"game {i+1}/{args.games}: reason={result.reason}, "
                f"winner={result.winner_seat}, turns={result.turns_observed}"
            )

    (out_dir / "games.jsonl").write_text(
        "\n".join(json.dumps(asdict(r)) for r in rows) + "\n",
        encoding="utf-8",
    )

    summary = {
        "games": args.games,
        "picker": args.picker,
        "ai_difficulty": args.ai_difficulty,
        "ai_deck": args.ai_deck,
        "format": args.format or "Standard",
        "seed": args.seed,
        "uri": args.uri,
        "stream_timeout": args.stream_timeout,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--picker", default="agent:heuristic")
    p.add_argument(
        "--ai-difficulty",
        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
        default="Medium",
    )
    p.add_argument("--ai-deck", default="Red Deck Wins")
    p.add_argument("--our-deck-file", default=None)
    p.add_argument("--format", default=None, help="Game format (Standard, Pioneer, Modern, Commander, etc.). Defaults to Standard.")
    p.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    p.add_argument("--autostart", action="store_true")
    p.add_argument("--stream-timeout", type=float, default=60.0)
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="runs/phase_rs_training_traces")
    p.add_argument("--ollama-model", default="gemma4:e2b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

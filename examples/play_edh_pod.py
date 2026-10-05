"""Run one Python-controlled Commander seat against three native phase-rs AI seats."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from examples.play_phase_rs import _build_picker
from src.integrations.phase_rs import PhaseServerConfig, load_deck_data, run_game_sync

DECKS = (
    "krenko-mob-boss_core.txt", "atraxa-praetors-voice_core.txt",
    "meren-of-clan-nel-toth_core.txt", "urza-lord-high-artificer_core.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="none", help="none, ollama:<tag>, or tev1:<size>")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--max-game-seconds", type=float, default=600)
    parser.add_argument("--ai-difficulty", default="VeryEasy")
    parser.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "edh_pod")
    args = parser.parse_args()
    if args.model == "none":
        picker_name, model = "heuristic", ""
    elif args.model.startswith("ollama:"):
        picker_name, model = "ollama", args.model.removeprefix("ollama:")
    elif args.model.startswith("tev1:"):
        picker_name, model = "tev1", args.model
    else:
        parser.error("--model must be none, ollama:<tag>, or tev1:<size>")
    picker = _build_picker(
        picker_name, args.seed, ollama_model=model, ollama_url="http://localhost:11434",
        tev1_model=model or "tev1:0.8b",
    )
    root = Path(__file__).resolve().parents[1] / "data" / "decks" / "edh"
    decks = [load_deck_data(root / name) for name in DECKS]
    result = run_game_sync(
        deck=decks[0], ai_decks=decks[1:], picker=picker,
        config=PhaseServerConfig(
            uri=args.uri, stream_timeout_s=60, max_message_bytes=16 * 1024 * 1024,
        ),
        format_name="Commander", ai_difficulty=args.ai_difficulty,
        max_actions=args.max_actions, max_turns=args.max_turns,
        max_game_seconds=args.max_game_seconds, autostart_server=args.autostart,
        reconnect_attempts=0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pod_{datetime.now():%Y%m%d_%H%M%S_%f}"
    (args.output_dir / f"{stem}.log").write_text("\n".join(result.log) + "\n", encoding="utf-8")
    (args.output_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in result.trace), encoding="utf-8",
    )
    print(json.dumps({
        "player_count": 4, "reason": result.reason, "winner_seat": result.winner_seat,
        "actions": result.actions_sent, "turns": result.turns_observed,
        "output": str(args.output_dir / stem),
    }, indent=2))
    completed = any(event.get("event") == "game_over" for event in result.trace)
    # An explicitly requested turn-capped smoke is successful execution, not a draw.
    return 0 if completed or (args.max_turns is not None and result.reason == "turn_cap") else 1


if __name__ == "__main__":
    raise SystemExit(main())

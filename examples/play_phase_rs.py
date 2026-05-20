"""Smoke-test the phase-rs adapter end-to-end.

Requires a running ``phase-server`` on ``ws://127.0.0.1:9374/ws``. In the
phase-rs checkout::

    cargo serve

Then from this repo (after ``pip install 'opposition-agents-mtg[phase_rs]'``
or ``pip install websockets>=12``)::

    python examples/play_phase_rs.py --deck "Red Deck Wins" --picker prefer-nonpass

Our seat uses whichever deck was passed in; the AI opponent uses the named
starter deck (resolved server-side from phase-rs's bundled card data).
"""

from __future__ import annotations

import argparse
import logging
import sys
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


def _build_picker(
    name: str,
    seed: int | None,
    *,
    ollama_model: str,
    ollama_url: str,
):
    if name == "random":
        return RandomActionPicker(seed=seed)
    if name in {"prefer-nonpass", "prefer_nonpass"}:
        return PreferNonPassPicker(seed=seed)
    if name == "heuristic":
        return HeuristicActionPicker(seed=seed)
    if name == "ollama":
        return OllamaActionPicker(seed=seed, model=ollama_model, base_url=ollama_url)
    if name.startswith("agent:"):
        agent_name = name.split(":", 1)[1].strip()
        if not agent_name:
            raise SystemExit("agent picker requires a name, e.g. agent:heuristic")
        agent = make_agent(agent_name, player_id="seat0", seed=seed)
        return AgentActionPicker(agent=agent, name=f"phase_rs_agent:{agent_name}")
    raise SystemExit(
        "unknown picker "
        f"{name!r}; try: random | prefer-nonpass | heuristic | ollama"
        f" | agent:<name>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck",
        default="Red Deck Wins",
        help=(
            "Either the name of a phase-rs starter deck (one of: "
            f"{', '.join(STARTER_DECK_NAMES)}) for the AI seat, OR a path to "
            "a decklist file under data/decks/ for our seat. If a name is "
            "given, both seats use that deck."
        ),
    )
    parser.add_argument("--our-deck-file", default=None, help="Path to our deck (.txt)")
    parser.add_argument("--picker", default="prefer-nonpass")
    parser.add_argument("--ollama-model", default="gemma4:e2b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ai-difficulty", default="Medium")
    parser.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    parser.add_argument(
        "--autostart",
        action="store_true",
        help=(
            "Spawn a local phase-server subprocess for the duration of this "
            "run (uses external/phase-rs/target/release/phase-server when "
            "available; falls back to `cargo run`). Adopts an already-"
            "running instance on the same port without killing it."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.our_deck_file:
        deck_path = Path(args.our_deck_file)
        if not deck_path.exists():
            print(f"deck file not found: {deck_path}", file=sys.stderr)
            return 2
        our_deck = load_deck_data(deck_path)
        ai_deck_name = args.deck if args.deck in STARTER_DECK_NAMES else "Red Deck Wins"
    else:
        if args.deck not in STARTER_DECK_NAMES:
            print(
                f"--deck must be one of {STARTER_DECK_NAMES} when --our-deck-file is omitted; "
                f"got {args.deck!r}",
                file=sys.stderr,
            )
            return 2
        # Both seats use the same starter — phase-server resolves it server-side
        # via the AI seat's deckName; for our seat we send an empty DeckData
        # and rely on the matching starter being picked up. If your server
        # rejects this, pass --our-deck-file pointing at a real list.
        our_deck = {"main_deck": [], "sideboard": [], "commander": []}
        ai_deck_name = args.deck

    picker = _build_picker(
        args.picker,
        args.seed,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )
    cfg = PhaseServerConfig(uri=args.uri)

    if args.autostart:
        print("autostarting phase-server (or adopting existing instance) ...")
    print(f"connecting to {cfg.uri} ...")
    result = run_game_sync(
        deck=our_deck,
        picker=picker,
        config=cfg,
        ai_difficulty=args.ai_difficulty,
        ai_deck_name=ai_deck_name,
        max_actions=args.max_actions,
        autostart_server=args.autostart,
    )

    print("\n--- game log ---")
    for line in result.log:
        print(line)
    print("--- result ---")
    print(f"winner_seat: {result.winner_seat}")
    print(f"our_seat:    {result.our_seat}")
    print(f"reason:      {result.reason}")
    print(f"turns:       {result.turns_observed}")
    print(f"actions:     {result.actions_sent}")
    if result.winner_seat is None:
        return 1
    return 0 if result.winner_seat == result.our_seat else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI script — run goldfish simulations for a deck.

Examples::

    # 200 games, heuristic agent, mono-red burn
    python scripts/goldfish.py --deck data/decks/modern/modern_mono_red_burn.txt --runs 200 --max-turns 8 --seed 42

    # Quick sanity check with random agent
    python scripts/goldfish.py --deck data/decks/modern/modern_mono_red_burn.txt --runs 20 --agent random

    # Save results to JSON
    python scripts/goldfish.py --deck data/decks/modern/modern_mono_red_burn.txt --runs 100 --out runs/goldfish
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.goldfish_runner import GoldfishRunner  # noqa: E402


def _load_deck(deck_path: str) -> list[dict]:
    """Load a deck from a plain-text decklist file."""
    try:
        from src.engine.card_database import CardDatabase
        from src.orchestrator.deck_loader import DecklistLoader

        db = CardDatabase()
        loader = DecklistLoader(db)
        deck = loader.load_deck(deck_path)
        if not deck:
            raise ValueError(f"Deck loaded empty from {deck_path!r}")
        return deck
    except Exception as exc:
        # Fall back: try to import via load_real_decklists helper if available
        print(f"[warn] DecklistLoader failed ({exc}); trying fallback loader …", flush=True)
        return _load_deck_fallback(deck_path)


def _load_deck_fallback(deck_path: str) -> list[dict]:
    """Minimal fallback: read a plain-text file and build stub card dicts."""
    cards: list[dict] = []
    with open(deck_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Skip section headers
            if line.lower() in ("mainboard", "sideboard", "commander"):
                continue
            # Parse "N Card Name" or "Card Name"
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                count, name = int(parts[0]), parts[1]
            else:
                count, name = 1, line
            for _ in range(count):
                cards.append({"name": name, "type_line": "Unknown", "mana_cost": ""})
    return cards


def _save_results(stats: "GoldfishStats", out_dir: str, deck_name: str) -> None:  # type: ignore[name-defined]  # noqa: F821
    from src.agents.goldfish_runner import GoldfishStats  # re-import for type safety

    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in deck_name)
    out_path = Path(out_dir) / f"goldfish_{safe_name}.json"
    data = {
        "deck": deck_name,
        "runs": stats.runs,
        "wins": stats.wins,
        "win_rate": stats.win_rate,
        "avg_kill_turn": stats.avg_kill_turn,
        "kill_turn_distribution": {str(k): v for k, v in stats.kill_turn_distribution.items()},
        "curve_hit_rate": stats.curve_hit_rate,
        "avg_spells_per_turn": stats.avg_spells_per_turn,
        "avg_power_per_turn": stats.avg_power_per_turn,
        "avg_creatures_per_turn": stats.avg_creatures_per_turn,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Results saved → {out_path}", flush=True)


async def _main(args: argparse.Namespace) -> None:
    deck_name = Path(args.deck).stem if args.deck else "unknown"
    print(f"Loading deck: {deck_name} …", flush=True)
    decklist = _load_deck(args.deck)
    print(f"  {len(decklist)} cards loaded", flush=True)

    runner = GoldfishRunner(
        agent_type=args.agent,
        runs=args.runs,
        max_turns=args.max_turns,
        seed=args.seed,
        starting_life=args.life,
    )

    print(
        f"Simulating {args.runs} goldfish games  "
        f"(agent={args.agent}, max_turns={args.max_turns}, life={args.life}) …",
        flush=True,
    )

    stats = await runner.run(decklist)
    print()
    print(stats.summary(deck_name=deck_name))

    if args.out:
        _save_results(stats, args.out, deck_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Goldfish deck simulation — play solo against a do-nothing opponent."
    )
    parser.add_argument(
        "--deck", required=True, metavar="PATH",
        help="Path to a plain-text decklist file."
    )
    parser.add_argument(
        "--runs", type=int, default=100, metavar="N",
        help="Number of games to simulate (default 100)."
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, metavar="N",
        help="Maximum turns for the active player before declaring a non-win (default 10)."
    )
    parser.add_argument(
        "--agent", default="heuristic", choices=["heuristic", "random"],
        help="Agent controlling the deck under test (default heuristic)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="RNG seed for deterministic runs."
    )
    parser.add_argument(
        "--life", type=int, default=20, metavar="N",
        help="Starting life total for both players (default 20)."
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Directory to write JSON results file (optional)."
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()

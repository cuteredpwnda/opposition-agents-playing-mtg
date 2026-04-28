#!/usr/bin/env python
"""LLM-vs-LLM game on real Modern decks loaded from local Scryfall cache.

Combines all the working pieces:

- :mod:`src.integrations.offline_card_db` — zero-network Scryfall lookup.
- :mod:`src.integrations.decklist_loader` — parse plaintext decklists.
- :mod:`src.agents.llm_agent.OllamaAgent` — small LLM via Ollama.
- :mod:`src.orchestrator.game_runner.GameRunner` — full priority loop +
  phase progression + counter SBAs etc. (everything that lands in
  ``examples/ablation_llm_only.py`` works here too).

Decks live under ``data/decks/*.txt`` and can be swapped via flags.

Usage::

    # default: burn vs azorius control, 1 game, gemma4:e2b
    python examples/play_real_decks.py

    # different model and more games
    python examples/play_real_decks.py --model gemma --games 3 --seed 7

    # specify decks
    python examples/play_real_decks.py \\
        --deck1 data/decks/modern_mono_red_burn.txt \\
        --deck2 data/decks/modern_azorius_control.txt

Requires the local Scryfall cache (run ``python scripts/fetch_card_data.py``
once) and a local Ollama server at ``http://localhost:11434``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_agent import OllamaAgent
from src.integrations.decklist_loader import DecklistLoader
from src.integrations.offline_card_db import get_default_db
from src.orchestrator.game_runner import GameConfig, GameRunner
from src.utils.seeding import set_global_seed


MODEL_ALIASES = {
    "gemma": "gemma4:e2b",
    "llama": "llama3.2:1b",
    "qwen": "qwen2.5-coder:1.5b",
    "phi": "phi:latest",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def _fallback_card(name: str) -> dict:
    """Minimal stub when a name isn't in the local cache."""
    print(f"  [warn] '{name}' not in local cache, using stub")
    lower = name.lower()
    is_basic = lower in {"mountain", "island", "plains", "swamp", "forest"}
    color = {"mountain": "R", "island": "U", "plains": "W",
             "swamp": "B", "forest": "G"}.get(lower, "C")
    return {
        "name": name,
        "type_line": f"Basic Land — {name.title()}" if is_basic else "Creature",
        "oracle_text": f"({{T}}: Add {{{color}}})" if is_basic else "",
        "mana_cost": "" if is_basic else "{2}",
        "cmc": 0 if is_basic else 2,
        "power": None if is_basic else "2",
        "toughness": None if is_basic else "2",
        "set": "STUB",
    }


def load_deck(path: Path) -> tuple[str, list[dict]]:
    """Parse a decklist file and expand to a flat list of card-data dicts."""
    text = path.read_text(encoding="utf-8")
    deck = DecklistLoader().from_text(text)
    db = get_default_db()
    if len(db) == 0:
        print("[error] Local Scryfall cache is empty. "
              "Run: python scripts/fetch_card_data.py")
        sys.exit(2)

    cards: list[dict] = []
    missing: list[str] = []
    for name, count in deck.mainboard.items():
        data = db.get(name)
        if data is None:
            missing.append(name)
            data = _fallback_card(name)
        # GameRunner expects a list of dicts; one entry per copy.
        for _ in range(count):
            cards.append(dict(data))  # shallow copy so card_data isn't shared

    print(f"  Loaded {sum(deck.mainboard.values())} cards "
          f"({len(deck.mainboard)} unique) from {path.name}")
    if missing:
        print(f"  Missing from cache ({len(missing)}): "
              f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
    return path.stem, cards


async def play_one_game(
    deck1_path: Path, deck2_path: Path, model: str,
    game_num: int, max_turns: int, log_dir: Path | None,
) -> dict:
    deck1_name, deck1 = load_deck(deck1_path)
    deck2_name, deck2 = load_deck(deck2_path)

    model_tag = resolve_model(model)
    agent1 = OllamaAgent("player1", name=f"{deck1_name}", model=model_tag)
    agent2 = OllamaAgent("player2", name=f"{deck2_name}", model=model_tag)
    if not (agent1._ollama_available and agent2._ollama_available):
        print(f"[warn] Ollama not available — agents will fall back to "
              f"random play. Start with: ollama serve")

    config = GameConfig(max_turns=max_turns, mulligan_enabled=True, max_mulligans=3)
    runner = GameRunner(config)

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: {deck1_name}  vs  {deck2_name}  (model={model_tag})")
    print(f"{'='*70}")

    t0 = time.time()
    result = await runner.run_game(
        agents={"player1": agent1, "player2": agent2},
        decks={"player1": deck1, "player2": deck2},
    )
    elapsed = time.time() - t0

    winner_label = {
        "player1": deck1_name, "player2": deck2_name,
    }.get(result.winner, str(result.winner))

    log_path = None
    if log_dir is not None and isinstance(result.log, list):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"game_{game_num:03d}_{deck1_name}_vs_{deck2_name}.log"
        log_path.write_text(
            "\n".join(str(line) for line in result.log), encoding="utf-8",
        )

    summary = {
        "game": game_num,
        "deck1": deck1_name, "deck2": deck2_name,
        "model": model_tag,
        "winner": winner_label,
        "winner_seat": result.winner,
        "turns": result.turns,
        "elapsed_sec": round(elapsed, 2),
        "agent1_actions": dict(agent1.stats["actions_chosen"]),
        "agent2_actions": dict(agent2.stats["actions_chosen"]),
        "agent1_llm_calls": agent1.stats["llm_calls_total"],
        "agent2_llm_calls": agent2.stats["llm_calls_total"],
        "log_path": str(log_path) if log_path else None,
    }
    print(f"  -> {winner_label} wins in {result.turns} turns ({elapsed:.1f}s)")
    return summary


async def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        set_global_seed(args.seed)

    deck1 = Path(args.deck1)
    deck2 = Path(args.deck2)
    if not deck1.exists():
        print(f"[error] deck1 not found: {deck1}"); sys.exit(2)
    if not deck2.exists():
        print(f"[error] deck2 not found: {deck2}"); sys.exit(2)

    log_dir = Path(args.log_dir) if args.log_dir else None
    summaries: list[dict] = []
    for i in range(1, args.games + 1):
        # Alternate seating across games to wash out going-first bias.
        if i % 2 == 1:
            d1, d2 = deck1, deck2
        else:
            d1, d2 = deck2, deck1
        s = await play_one_game(d1, d2, args.model, i, args.max_turns, log_dir)
        summaries.append(s)

    # Aggregate
    print(f"\n{'='*70}")
    print("MATCH SUMMARY")
    print(f"{'='*70}")
    wins = Counter(s["winner"] for s in summaries)
    for name, n in wins.most_common():
        print(f"  {name}: {n} / {args.games}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps({"games": summaries, "wins": dict(wins)}, indent=2),
            encoding="utf-8",
        )
        print(f"  -> wrote {args.summary_json}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deck1", default="data/decks/modern_mono_red_burn.txt")
    p.add_argument("--deck2", default="data/decks/modern_azorius_control.txt")
    p.add_argument("--model", default="gemma", help="Ollama model name or alias")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="runs/real_decks")
    p.add_argument("--summary-json", default="runs/real_decks/summary.json")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))

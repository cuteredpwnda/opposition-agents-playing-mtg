#!/usr/bin/env python
"""4-player Commander pod with LLM agents on real EDHREC decklists.

Runs a free-for-all Commander game with N (default 4) players, each
piloting a different commander deck loaded from plaintext files written
by ``scripts/build_deck_corpus.py`` (or hand-crafted under
``data/decks/edh_pod/``).

Usage::

    # default: krenko / atraxa / urza / meren, gemma4:e2b
    python examples/play_edh_pod.py

    # custom decks + model
    python examples/play_edh_pod.py \\
        --decks data/decks/edh_pod/krenko-mob-boss_core.txt \\
                data/decks/edh_pod/atraxa-praetors-voice_core.txt \\
                data/decks/edh_pod/urza-lord-high-artificer_core.txt \\
                data/decks/edh_pod/meren-of-clan-nel-toth_core.txt \\
        --model gemma --max-turns 25

The runner uses :class:`GameConfig(format="commander", starting_life=40)`,
which already supports N players, command zone, commander tax, and
21-damage tracking. Commander cards are placed first in each library so
``GameRunner._setup_game`` lifts them into the command zone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

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

DEFAULT_DECKS = [
    "data/decks/edh_pod/krenko-mob-boss_core.txt",
    "data/decks/edh_pod/atraxa-praetors-voice_core.txt",
    "data/decks/edh_pod/urza-lord-high-artificer_core.txt",
    "data/decks/edh_pod/meren-of-clan-nel-toth_core.txt",
]


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def _fallback_card(name: str) -> dict:
    print(f"  [warn] '{name}' not in local cache, using stub")
    lower = name.lower()
    is_basic = lower in {"mountain", "island", "plains", "swamp",
                          "forest", "wastes"}
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


def load_edh_deck(path: Path) -> tuple[str, str, list[dict]]:
    """Parse an EDH decklist.

    Returns ``(deck_label, commander_name, cards)`` where ``cards`` has the
    commander as element 0 so :meth:`GameRunner._setup_game` lifts it
    into the command zone.
    """
    text = path.read_text(encoding="utf-8")
    deck = DecklistLoader().from_text(text)
    db = get_default_db()
    if len(db) == 0:
        print("[error] Local Scryfall cache is empty. "
              "Run: python scripts/fetch_card_data.py")
        sys.exit(2)

    if not deck.commander:
        print(f"[error] {path.name} has no Commander section")
        sys.exit(2)
    cmdr_name = deck.commander[0]

    cards: list[dict] = []
    cmdr_data = db.get(cmdr_name) or _fallback_card(cmdr_name)
    cards.append(dict(cmdr_data))  # index 0 → command zone

    missing: list[str] = []
    total = 0
    for name, count in deck.mainboard.items():
        data = db.get(name)
        if data is None:
            missing.append(name)
            data = _fallback_card(name)
        for _ in range(count):
            cards.append(dict(data))
            total += 1

    label = path.stem.replace("-", "_").replace(" ", "_")
    print(f"  Loaded {total + 1} cards (commander='{cmdr_name}', "
          f"{len(deck.mainboard)} unique non-commander) from {path.name}")
    if missing:
        print(f"  Missing from cache ({len(missing)}): "
              f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
    return label, cmdr_name, cards


async def play_pod(deck_paths: list[Path], model: str,
                   game_num: int, max_turns: int,
                   log_dir: Path | None) -> dict:
    decks: dict[str, list[dict]] = {}
    agents: dict[str, OllamaAgent] = {}
    labels: dict[str, str] = {}
    commanders: dict[str, str] = {}

    model_tag = resolve_model(model)
    for i, path in enumerate(deck_paths, start=1):
        pid = f"player{i}"
        label, cmdr, cards = load_edh_deck(path)
        decks[pid] = cards
        labels[pid] = label
        commanders[pid] = cmdr
        agents[pid] = OllamaAgent(pid, name=label, model=model_tag)

    if not all(a._ollama_available for a in agents.values()):
        print("[warn] Ollama not available — agents will fall back to "
              "random play. Start with: ollama serve")

    config = GameConfig(
        format="commander",
        starting_life=40,
        max_turns=max_turns,
        mulligan_enabled=True,
        max_mulligans=3,
    )
    runner = GameRunner(config)

    print(f"\n{'='*72}")
    print(f"POD GAME {game_num}  (model={model_tag}, max_turns={max_turns})")
    for pid, cmdr in commanders.items():
        print(f"  {pid}: {labels[pid]:<32}  commander={cmdr}")
    print(f"{'='*72}")

    t0 = time.time()
    result = await runner.run_game(agents=agents, decks=decks)
    elapsed = time.time() - t0

    winner_label = labels.get(result.winner, str(result.winner))
    log_path = None
    if log_dir is not None and isinstance(result.log, list):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"pod_game_{game_num:03d}.log"
        log_path.write_text(
            "\n".join(str(line) for line in result.log), encoding="utf-8",
        )

    summary = {
        "game": game_num,
        "model": model_tag,
        "players": labels,
        "commanders": commanders,
        "winner": winner_label,
        "winner_seat": result.winner,
        "turns": result.turns,
        "elapsed_sec": round(elapsed, 2),
        "agent_actions": {
            pid: dict(a.stats["actions_chosen"]) for pid, a in agents.items()
        },
        "agent_llm_calls": {
            pid: a.stats["llm_calls_total"] for pid, a in agents.items()
        },
        "log_path": str(log_path) if log_path else None,
    }
    print(f"  -> {winner_label} wins in {result.turns} turns ({elapsed:.1f}s)")
    return summary


async def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        set_global_seed(args.seed)

    deck_paths = [Path(d) for d in args.decks]
    for p in deck_paths:
        if not p.exists():
            print(f"[error] deck not found: {p}")
            sys.exit(2)
    if len(deck_paths) < 2:
        print("[error] need at least 2 decks for a pod"); sys.exit(2)

    log_dir = Path(args.log_dir) if args.log_dir else None
    summaries: list[dict] = []
    for i in range(1, args.games + 1):
        s = await play_pod(deck_paths, args.model, i, args.max_turns, log_dir)
        summaries.append(s)

    print(f"\n{'='*72}")
    print("POD MATCH SUMMARY")
    print(f"{'='*72}")
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
    p.add_argument("--decks", nargs="+", default=DEFAULT_DECKS,
                   help="2..N EDH decklist paths (one commander each)")
    p.add_argument("--model", default="gemma", help="Ollama model name or alias")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="runs/edh_pod")
    p.add_argument("--summary-json", default="runs/edh_pod/summary.json")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))

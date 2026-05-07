"""CLI for the deck-builder agent (G9).

Usage examples::

    # Brew a Krenko deck (no KG, offline card DB must exist)
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Krenko, Mob Boss" --seed 42

    # Brew with KG synergy scoring (requires Neo4j)
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Atraxa, Praetors' Voice" \
        --commander "Krenko, Mob Boss" \
        --kg --seed 1

    # Control pool sample size and output path
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Meren of Clan Nel Toth" \
        --pool-sample 800 --seed 7 \
        --out runs/brews/meren_gen1.txt

Output is written to ``runs/brews/<commander>_<seed>.txt`` by default
(or the path supplied via ``--out``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.deck_builder.agent import DeckBuilderAgent
from src.agents.deck_builder.scorer import CardScorer
from src.integrations.offline_card_db import OfflineCardDB
from src.integrations.decklist_loader import Decklist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _deck_to_text(deck: Decklist) -> str:
    lines: list[str] = []
    if deck.commander:
        lines.append("Commander")
        for name in deck.commander:
            lines.append(f"1 {name}")
        lines.append("")
    if deck.mainboard:
        lines.append("Mainboard")
        for name, count in sorted(deck.mainboard.items()):
            lines.append(f"{count} {name}")
    return "\n".join(lines)


def _default_out(commanders: list[str], seed: int | None) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", commanders[0].lower()).strip("_")
    suffix = f"_seed{seed}" if seed is not None else ""
    path = Path("runs/brews") / f"{slug}{suffix}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _run(args: argparse.Namespace) -> None:
    logger.info("Loading offline card database…")
    card_db = OfflineCardDB.load_default()
    if len(card_db) == 0:
        logger.error(
            "Offline card DB is empty.  Run:  python scripts/fetch_card_data.py"
        )
        sys.exit(1)
    logger.info("Card DB loaded: %d cards", len(card_db))

    kg = None
    if args.kg:
        try:
            from src.knowledge.knowledge_graph import MTGKnowledgeGraph
            kg = MTGKnowledgeGraph()
            logger.info("KG scoring enabled.")
        except Exception as exc:
            logger.warning("KG unavailable (%s); synergy/combo scoring disabled.", exc)

    scorer = CardScorer(
        kg=kg,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    agent = DeckBuilderAgent(
        commander_names=args.commander,
        card_db=card_db,
        scorer=scorer,
        format=args.format,
        seed=args.seed,
        pool_sample=args.pool_sample,
    )

    deck = await agent.brew()

    out_path = Path(args.out) if args.out else _default_out(args.commander, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_deck_to_text(deck), encoding="utf-8")
    logger.info("Deck written to %s", out_path)

    # Quick summary
    total = sum(deck.mainboard.values()) + len(deck.commander)
    print(f"\n--- {', '.join(args.commander)} ---")
    print(f"Total cards : {total}")
    print(f"Commander(s): {', '.join(deck.commander)}")
    print(f"Top 10 non-land mainboard cards:")
    non_land = [
        (n, c) for n, c in deck.mainboard.items()
        if "land" not in n.lower()
    ]
    for name, cnt in sorted(non_land, key=lambda x: x[0])[:10]:
        print(f"  {cnt}  {name}")
    print(f"\nOutput: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy Commander deck builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--commander", "-c",
        action="append",
        required=True,
        metavar="NAME",
        help="Commander name; repeat for partner pairs.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for pool shuffle (omit for non-deterministic).",
    )
    parser.add_argument(
        "--pool-sample", type=int, default=500,
        help="Maximum candidate pool size.",
    )
    parser.add_argument(
        "--format", default="commander",
        help="Legality format key (commander / vintage / legacy …).",
    )
    parser.add_argument(
        "--kg", action="store_true",
        help="Enable KG synergy + combo-completion scoring (requires Neo4j).",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Synergy weight.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Archetype / type-diversity weight.",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0,
        help="Combo-completion weight.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path for the deck list text file.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

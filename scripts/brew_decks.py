"""CLI for the deck-builder agent (G1–G5 integrated).

Usage examples::

    # Brew a Krenko deck (no evaluation/mutation)
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Krenko, Mob Boss" --seed 42

    # Brew + evaluate vs a reference deck
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Krenko, Mob Boss" \
        --eval --eval-reference "data/decks/edh/krenko-mob-boss_core.txt" \
        --eval-games 4 --seed 42

    # Brew + evaluate + run 2 mutation generations (G5 optimization loop)
    .\.venv\Scripts\python.exe scripts/brew_decks.py \
        --commander "Atraxa, Praetors' Voice" \
        --eval --eval-reference "data/decks/edh/atraxa-praetors-voice_core.txt" \
        --eval-games 4 \
        --mutate --mutate-generations 2 --mutate-swaps-per-iter 3 \
        --kg --seed 1

Output is written to ``runs/brews/<commander>_<seed>.txt`` by default.
Mutation results save per-generation: ``runs/brews/<commander>_gen<N>_<seed>.txt``
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

from src.agents.deck_builder import (
    DeckBuilderAgent,
    CardScorer,
    DeckEvaluator,
    DeckMutator,
)
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


def _print_deck_summary(deck: Decklist, commanders: list[str], title: str) -> None:
    """Print a human-readable summary of a deck."""
    total = sum(deck.mainboard.values()) + len(deck.commander)
    print(f"\n--- {title}: {', '.join(commanders)} ---")
    print(f"Total cards : {total}")
    print(f"Commander(s): {', '.join(deck.commander)}")
    print(f"Top 10 non-land mainboard cards:")
    non_land = [
        (n, c) for n, c in deck.mainboard.items()
        if "land" not in n.lower()
    ]
    for name, cnt in sorted(non_land, key=lambda x: x[0])[:10]:
        print(f"  {cnt}  {name}")


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

    logger.info("Building initial deck…")
    deck = await agent.brew()
    _print_deck_summary(deck, args.commander, "Initial deck")

    # Optionally evaluate and mutate
    current_deck = deck
    if args.eval:
        logger.info("Setting up deck evaluator…")
        evaluator = DeckEvaluator(
            reference_decks=[(args.eval_reference, "heuristic")],
            games_per_pair=args.eval_games,
            format=args.format,
            seed=args.seed,
        )
        result = await evaluator.evaluate(current_deck, agent.commander_cards)
        logger.info(
            "Evaluation: %d wins / %d games (%.1f%%), elapsed %.1fs",
            result.wins,
            result.total,
            result.win_rate * 100,
            result.elapsed,
        )
        logger.info("Top 5 contributors:")
        for card, contrib in result.top_contributors(5):
            logger.info("  %s: %+.3f", card, contrib)

        if args.mutate:
            logger.info(
                "Running mutation loop (%d generations, %d swaps/iter)…",
                args.mutate_generations,
                args.mutate_swaps_per_iter,
            )
            mutator = DeckMutator(
                evaluator=evaluator,
                scorer=scorer,
                card_db=card_db,
                max_iterations=args.mutate_generations,
                swaps_per_iter=args.mutate_swaps_per_iter,
                pool_sample=args.pool_sample,
                seed=args.seed,
            )
            best_deck, log = await mutator.mutate(
                current_deck, agent.commander_cards
            )
            logger.info(log.summary())
            current_deck = best_deck
            _print_deck_summary(
                current_deck,
                args.commander,
                f"Best deck after {args.mutate_generations} mutations",
            )

    # Save final deck
    out_path = Path(args.out) if args.out else _default_out(args.commander, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_deck_to_text(current_deck), encoding="utf-8")
    logger.info("Final deck written to %s", out_path)
    print(f"\nOutput: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy Commander deck builder with optional evaluation and mutation",
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

    # Evaluation (G4) options
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate the built deck against reference decks.",
    )
    parser.add_argument(
        "--eval-reference", default="data/decks/edh/krenko-mob-boss_core.txt",
        metavar="PATH",
        help="Reference deck to evaluate against.",
    )
    parser.add_argument(
        "--eval-games", type=int, default=4, metavar="N",
        help="Games to play per reference deck.",
    )

    # Mutation (G5) options
    parser.add_argument(
        "--mutate", action="store_true",
        help="Run the mutation optimization loop (requires --eval).",
    )
    parser.add_argument(
        "--mutate-generations", type=int, default=3, metavar="N",
        help="Number of mutation iterations.",
    )
    parser.add_argument(
        "--mutate-swaps-per-iter", type=int, default=2, metavar="N",
        help="Number of card swaps per iteration.",
    )

    args = parser.parse_args()

    if args.mutate and not args.eval:
        parser.error("--mutate requires --eval")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

"""
Standalone KG enrichment runner.

Loads all self-play trajectories from data/trajectories/ and runs KGEnrichment
to write LearnedSynergyEvidence and LearnedCardOutcome nodes into Neo4j.

Usage:
    .\.venv\Scripts\python.exe scripts\run_kg_enrichment.py [--dry-run] [--min-co N] [--min-lift F]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_kg_enrichment")


async def main(args: argparse.Namespace) -> None:
    from src.knowledge.knowledge_graph import MTGKnowledgeGraph
    from src.knowledge.kg_enrichment import KGEnrichment, EnrichmentConfig
    from src.world_model.trajectory import TrajectoryStore

    # --- Load trajectories --------------------------------------------------
    store = TrajectoryStore(storage_dir="data/trajectories")
    store.load()
    logger.info("Loaded %d trajectories from data/trajectories/", len(store))

    if len(store) == 0:
        logger.error("No trajectories found. Run stage-4 self-play first.")
        return

    # --- Build config -------------------------------------------------------
    config = EnrichmentConfig(
        min_co_occurrence=args.min_co,
        min_synergy_lift=args.min_lift,
        run_id="manual_enrichment_run",
        source="self_play_trajectories",
    )

    if args.dry_run:
        logger.info("DRY RUN — no writes to Neo4j")

    # --- Connect to KG and run enrichment -----------------------------------
    kg = MTGKnowledgeGraph()
    enrichment = KGEnrichment(kg, config)

    if args.dry_run:
        # Just run analysis, skip writes
        trajectories = store.trajectories
        game_cards, card_wins, card_games = enrichment._extract_card_statistics(trajectories)
        synergies = enrichment._discover_synergies(game_cards, card_wins, card_games)
        logger.info("Would write %d synergy edges", len(synergies))
        if synergies:
            logger.info("Top 10 synergies:")
            for s in synergies[:10]:
                logger.info(
                    "  %s + %s  lift=%.2f  wr=%.2f  co-occ=%d",
                    s["card_a"], s["card_b"], s["lift"], s["pair_win_rate"], s["co_occurrences"]
                )
    else:
        report = await enrichment.enrich_from_trajectories(store)
        logger.info("Enrichment complete:")
        logger.info("  Games analyzed:     %d", report.games_analyzed)
        logger.info("  Synergies proposed: %d", report.synergies_proposed)
        logger.info("  Synergies written:  %d", report.synergies_written)
        logger.info("  Combos proposed:    %d", report.combos_proposed)
        logger.info("  Card stats updated: %d", report.card_stats_updated)
        if report.errors:
            logger.warning("  Errors: %d", len(report.errors))
            for e in report.errors[:5]:
                logger.warning("    %s", e)

    await kg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KG enrichment from self-play trajectories")
    parser.add_argument("--dry-run", action="store_true", help="Analyse only, no Neo4j writes")
    parser.add_argument("--min-co", type=int, default=3, help="Min co-occurrence count (default 3)")
    parser.add_argument("--min-lift", type=float, default=1.2, help="Min win-rate lift (default 1.2)")
    args = parser.parse_args()
    asyncio.run(main(args))

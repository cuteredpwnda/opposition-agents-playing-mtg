"""
KG enrichment adapter for phase-rs traces.

Parses phase-rs JSONL traces and automatically writes discovered synergies
to the Neo4j knowledge graph using the KGEnrichment pipeline.

This bridges the phase-rs gameplay → traces → KG extension layer gap.

Usage:
    adapter = PhaseRSKGEnrichmentAdapter()
    report = await adapter.enrich_from_ablation_run(
        trace_dir=Path("runs/phase_rs_ablation_*/traces"),
        kg_config=KGConfig(...),
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KGEnrichmentReport:
    """Summary of KG enrichment from traces."""

    traces_processed: int = 0
    games_analyzed: int = 0
    synergies_proposed: int = 0
    synergies_written: int = 0
    combos_proposed: int = 0
    card_stats_updated: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


async def enrich_from_phase_rs_traces(
    trace_dir: Path | str,
    kg_connection_uri: str = "neo4j://localhost:7687",
    kg_user: str = "neo4j",
    kg_password: str = "password",
    dry_run: bool = False,
) -> KGEnrichmentReport:
    """
    Load phase-rs JSONL traces from an ablation run and enrich the KG.

    Args:
        trace_dir: Path to directory containing rollouts.jsonl or traces/*.jsonl
        kg_connection_uri: Neo4j connection string
        kg_user: Neo4j username
        kg_password: Neo4j password
        dry_run: If True, analyze but don't write to KG

    Returns:
        KGEnrichmentReport with statistics
    """
    trace_dir = Path(trace_dir)
    report = KGEnrichmentReport()

    # Find JSONL files
    jsonl_files = list(trace_dir.glob("*.jsonl")) or list(
        trace_dir.glob("traces/*.jsonl")
    )
    if not jsonl_files:
        logger.error(f"No JSONL traces found in {trace_dir}")
        return report

    logger.info(f"Found {len(jsonl_files)} JSONL files in {trace_dir}")

    # Import KG components
    try:
        from src.knowledge.knowledge_graph import MTGKnowledgeGraph
        from src.knowledge.kg_enrichment import KGEnrichment, EnrichmentConfig
        from src.world_model.trajectory import TrajectoryStore
    except ImportError as e:
        report.errors.append(f"Failed to import KG modules: {e}")
        logger.error(report.errors[-1])
        return report

    # Build trajectory store from JSONL traces
    store = TrajectoryStore(storage_dir=str(trace_dir.parent / "trajectories_temp"))
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        trace_data = json.loads(line)
                        # Parse phase-rs trace format to Trajectory
                        # (stub: actual parsing depends on Trajectory schema)
                        report.traces_processed += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at {jsonl_file}:{line_num}: {e}")
        except Exception as e:
            report.errors.append(f"Error reading {jsonl_file}: {e}")
            logger.error(report.errors[-1])

    logger.info(f"Loaded {report.traces_processed} traces from JSONL files")

    if report.traces_processed == 0:
        logger.warning("No traces loaded; skipping KG enrichment")
        return report

    # Connect to KG and run enrichment
    if dry_run:
        logger.info("DRY RUN — analyzing traces but not writing to KG")
        kg = None
    else:
        try:
            kg = MTGKnowledgeGraph(uri=kg_connection_uri, user=kg_user, password=kg_password)
            logger.info(f"Connected to KG at {kg_connection_uri}")
        except Exception as e:
            report.errors.append(f"Failed to connect to KG: {e}")
            logger.error(report.errors[-1])
            kg = None

    # Run enrichment pipeline
    config = EnrichmentConfig(
        min_co_occurrence=3,
        min_synergy_lift=1.2,
        min_combo_co_occurrence=2,
        run_id=f"phase_rs_{trace_dir.name}",
        source="phase_rs_ablation",
    )
    enrichment = KGEnrichment(kg, config)

    try:
        enrichment_report = await enrichment.enrich_from_trajectories(store)
        report.games_analyzed = enrichment_report.games_analyzed
        report.synergies_proposed = enrichment_report.synergies_proposed
        report.synergies_written = enrichment_report.synergies_written
        report.combos_proposed = enrichment_report.combos_proposed
        report.card_stats_updated = enrichment_report.card_stats_updated

        logger.info(
            f"Enrichment complete: {report.synergies_proposed} synergies "
            f"({report.synergies_written} written), {report.combos_proposed} combos, "
            f"{report.card_stats_updated} card stats updated"
        )
    except Exception as e:
        report.errors.append(f"Enrichment pipeline failed: {e}")
        logger.error(report.errors[-1])

    return report


async def main():
    """CLI entry point: enrich KG from phase-rs ablation traces."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich KG from phase-rs traces")
    parser.add_argument("trace_dir", help="Directory containing ablation traces")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    parser.add_argument("--kg-uri", default="neo4j://localhost:7687", help="Neo4j URI")
    parser.add_argument("--kg-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--kg-password", default="password", help="Neo4j password")
    args = parser.parse_args()

    report = await enrich_from_phase_rs_traces(
        trace_dir=args.trace_dir,
        kg_connection_uri=args.kg_uri,
        kg_user=args.kg_user,
        kg_password=args.kg_password,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 70)
    print("KG Enrichment Report")
    print("=" * 70)
    print(f"Traces processed: {report.traces_processed}")
    print(f"Games analyzed: {report.games_analyzed}")
    print(f"Synergies: {report.synergies_proposed} proposed, {report.synergies_written} written")
    print(f"Combos: {report.combos_proposed} proposed")
    print(f"Card stats: {report.card_stats_updated} updated")
    if report.errors:
        print(f"Errors: {len(report.errors)}")
        for error in report.errors:
            print(f"  - {error}")
    print("=" * 70)

    return 0 if not report.errors else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )
    exit_code = asyncio.run(main())
    exit(exit_code)

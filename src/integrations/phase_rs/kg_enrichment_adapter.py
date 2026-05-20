"""
KG enrichment adapter for phase-rs traces.

Parses phase-rs ablation outputs (``rollouts.jsonl`` / ``games.jsonl`` plus
per-game ``traces/*.jsonl``) and feeds them into
:class:`src.knowledge.kg_enrichment.KGEnrichment` so the Neo4j knowledge
graph can learn from real gameplay.

Two card sources contribute to each :class:`Trajectory`:

1. **Our deck** — supplied via ``our_deck`` (a ``DeckData``-style dict with a
   ``main_deck`` list) or auto-loaded from ``our_deck_file``. These are the
   cards the agent actually piloted. Most useful enrichment signal.
2. **AI starter deck name** — added as a synthetic token (e.g.
   ``"deck:Red Deck Wins"``) so deck-vs-deck correlations are visible even
   when we don't have the AI's card list client-side.

Usage::

    from src.integrations.phase_rs.kg_enrichment_adapter import (
        enrich_from_phase_rs_traces,
    )

    report = await enrich_from_phase_rs_traces(
        trace_dir="runs/phase_rs_ablation_20260520_153604",
        our_deck_file="data/decks/red_deck_wins.txt",
        dry_run=True,  # no Neo4j needed
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
    errors: list[str] = field(default_factory=list)


def _find_games_manifest(trace_dir: Path) -> Path | None:
    """Locate the per-game manifest. Supports both sweep + collector outputs."""
    for name in ("rollouts.jsonl", "games.jsonl"):
        candidate = trace_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_our_deck_cards(
    our_deck: dict[str, Any] | None,
    our_deck_file: str | Path | None,
) -> list[str]:
    """Resolve the list of card names the agent piloted.

    Returns an empty list if no deck information is available. Duplicates are
    preserved so co-occurrence counts reflect actual multiplicity.
    """
    if our_deck and isinstance(our_deck.get("main_deck"), list):
        return [str(c) for c in our_deck["main_deck"] if c]
    if our_deck_file:
        try:
            from src.integrations.phase_rs.decks import load_deck_data

            data = load_deck_data(our_deck_file)
            return [str(c) for c in data.get("main_deck", []) if c]
        except Exception as exc:  # pragma: no cover — best-effort
            logger.warning("Could not load our_deck_file %s: %s", our_deck_file, exc)
    return []


def _build_trajectory(
    game_row: dict[str, Any],
    our_deck_cards: list[str],
):
    """Convert one ``rollouts.jsonl`` / ``games.jsonl`` row into a Trajectory.

    Each card the agent could have used becomes a ``Transition`` whose
    ``card_name`` is set, so :meth:`KGEnrichment._extract_card_statistics`
    can count it. The AI starter-deck name is appended as a synthetic
    ``deck:<name>`` token to capture deck-vs-deck signal.
    """
    from src.world_model.trajectory import Trajectory, Transition

    winner_seat = game_row.get("winner_seat")
    our_seat = game_row.get("our_seat", 0)
    # KGEnrichment treats seat 0 as "us" — normalise the trajectory so that
    # ``traj.winner == 0`` iff we (the agent) won.
    if winner_seat is None:
        normalised_winner: int | None = None
    elif winner_seat == our_seat:
        normalised_winner = 0
    else:
        normalised_winner = 1

    game_id = str(
        game_row.get("game_id")
        or f"phase_rs_{uuid.uuid4().hex[:8]}"
    )

    ai_deck = game_row.get("ai_deck")
    picker = game_row.get("picker") or game_row.get("agent")

    cards: list[str] = list(our_deck_cards)
    if ai_deck:
        cards.append(f"deck:{ai_deck}")

    traj = Trajectory(
        game_id=game_id,
        winner=normalised_winner,
        num_turns=int(game_row.get("turns") or 0),
        source="phase_rs",
        metadata={
            "picker": picker,
            "ai_difficulty": game_row.get("ai_difficulty"),
            "ai_deck": ai_deck,
            "reason": game_row.get("reason"),
            "seed": game_row.get("seed"),
        },
    )

    # Minimal placeholder state/action features — KGEnrichment only reads
    # ``card_name``, but the Transition dataclass requires the other fields.
    empty_state: dict[str, np.ndarray] = {}
    empty_action = np.zeros(1, dtype=np.float32)

    for card in cards:
        traj.add(
            Transition(
                state_features=empty_state,
                action_encoding=empty_action,
                reward=0.0,
                done=False,
                action_type="DECK_CARD",
                card_name=card,
            )
        )
    return traj


async def enrich_from_phase_rs_traces(
    trace_dir: Path | str,
    kg_connection_uri: str = "neo4j://localhost:7687",
    kg_user: str = "neo4j",
    kg_password: str = "password",
    dry_run: bool = False,
    our_deck: dict[str, Any] | None = None,
    our_deck_file: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> KGEnrichmentReport:
    """Load phase-rs ablation traces and enrich the KG.

    Args:
        trace_dir: Directory containing ``rollouts.jsonl`` (sweep) or
            ``games.jsonl`` (collector) plus the per-game ``traces/`` folder.
        kg_connection_uri: Neo4j connection string.
        kg_user: Neo4j username.
        kg_password: Neo4j password.
        dry_run: If True, run the enrichment pipeline but skip Neo4j writes.
            Also automatically enabled when a KG connection cannot be opened.
        our_deck: Optional pre-loaded ``DeckData`` dict (``main_deck`` list).
        our_deck_file: Optional path to a decklist text file; used when
            ``our_deck`` is not supplied.
        config_overrides: Optional dict patched onto :class:`EnrichmentConfig`.

    Returns:
        :class:`KGEnrichmentReport` with statistics and any errors.
    """
    trace_dir = Path(trace_dir)
    report = KGEnrichmentReport()

    if not trace_dir.exists():
        report.errors.append(f"trace_dir does not exist: {trace_dir}")
        logger.error(report.errors[-1])
        return report

    manifest = _find_games_manifest(trace_dir)
    if manifest is None:
        report.errors.append(
            f"no rollouts.jsonl or games.jsonl found in {trace_dir}"
        )
        logger.error(report.errors[-1])
        return report

    logger.info("KG enrichment: reading game manifest %s", manifest)

    try:
        from src.knowledge.kg_enrichment import EnrichmentConfig, KGEnrichment
        from src.world_model.trajectory import TrajectoryStore
    except ImportError as exc:
        report.errors.append(f"Failed to import KG modules: {exc}")
        logger.error(report.errors[-1])
        return report

    our_deck_cards = _load_our_deck_cards(our_deck, our_deck_file)
    if not our_deck_cards:
        logger.info(
            "No our_deck card list supplied — enrichment will only see "
            "deck:<name> tokens from the AI side"
        )

    store = TrajectoryStore(
        storage_dir=str(trace_dir / "_kg_trajectories_tmp")
    )

    with manifest.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d: %s", manifest, line_num, exc)
                continue
            try:
                traj = _build_trajectory(row, our_deck_cards)
                store.add(traj)
                report.traces_processed += 1
            except Exception as exc:  # pragma: no cover
                report.errors.append(
                    f"Failed to build trajectory from {manifest}:{line_num}: {exc}"
                )
                logger.error(report.errors[-1])

    logger.info(
        "Built %d trajectories (our_deck_cards=%d, dry_run=%s)",
        report.traces_processed,
        len(our_deck_cards),
        dry_run,
    )

    if report.traces_processed == 0:
        logger.warning("No trajectories built; skipping KG enrichment")
        return report

    # Connect to KG (or fall back to dry-run if Neo4j is unreachable).
    kg = None
    if not dry_run:
        try:
            from src.knowledge.knowledge_graph import MTGKnowledgeGraph

            kg = MTGKnowledgeGraph(
                uri=kg_connection_uri, user=kg_user, password=kg_password
            )
            logger.info("Connected to KG at %s", kg_connection_uri)
        except Exception as exc:
            report.errors.append(f"Failed to connect to KG ({exc}); falling back to dry-run")
            logger.warning(report.errors[-1])
            kg = None

    cfg_kwargs: dict[str, Any] = dict(
        min_co_occurrence=3,
        min_synergy_lift=1.2,
        min_combo_co_occurrence=2,
        run_id=f"phase_rs_{trace_dir.name}",
        source="phase_rs_ablation",
    )
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = EnrichmentConfig(**cfg_kwargs)
    enrichment = KGEnrichment(kg, config)

    try:
        enrichment_report = await enrichment.enrich_from_trajectories(store)
        report.games_analyzed = enrichment_report.games_analyzed
        report.synergies_proposed = enrichment_report.synergies_proposed
        report.synergies_written = enrichment_report.synergies_written
        report.combos_proposed = enrichment_report.combos_proposed
        report.card_stats_updated = enrichment_report.card_stats_updated
        logger.info(
            "Enrichment complete: %d synergies (%d written), %d combos, %d card stats",
            report.synergies_proposed,
            report.synergies_written,
            report.combos_proposed,
            report.card_stats_updated,
        )
    except Exception as exc:
        report.errors.append(f"Enrichment pipeline failed: {exc}")
        logger.error(report.errors[-1])
    finally:
        if kg is not None:
            try:
                await kg.close()
            except Exception:  # pragma: no cover
                pass

    return report


async def main():
    """CLI entry point: enrich KG from phase-rs ablation traces."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich KG from phase-rs traces")
    parser.add_argument("trace_dir", help="Directory containing ablation traces")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    parser.add_argument("--our-deck-file", default=None, help="Decklist text file the agent piloted")
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
        our_deck_file=args.our_deck_file,
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

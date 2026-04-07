"""
Knowledge Graph auto-enrichment from self-play trajectories.

Analyzes game trajectories to discover:
- Card co-occurrences that correlate with wins → SYNERGIZES_WITH edges
- Repeated multi-card combos in winning games → Combo node proposals
- Per-card win-rate statistics → metagameImportance updates
- Emerging archetypes via decklist clustering

This closes the self-play → KG feedback loop (Phase A.3 of IMPLEMENTATION_PLAN).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.knowledge.knowledge_graph import MTGKnowledgeGraph
    from src.world_model.trajectory import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """Thresholds for KG enrichment proposals."""

    min_co_occurrence: int = 5          # Min games a card pair must co-occur in
    min_synergy_lift: float = 1.3       # Win-rate lift ratio to propose synergy
    min_combo_co_occurrence: int = 3    # Min wins a multi-card set must appear in
    max_combo_size: int = 3             # Max cards in a discovered combo
    min_games_for_stats: int = 10       # Min games before updating card stats
    synergy_edge_weight: float = 0.1    # Increment per co-occurrence


@dataclass
class EnrichmentReport:
    """Summary of a KG enrichment run."""

    games_analyzed: int = 0
    synergies_proposed: int = 0
    synergies_written: int = 0
    combos_proposed: int = 0
    card_stats_updated: int = 0
    errors: list[str] = field(default_factory=list)


class KGEnrichment:
    """Analyzes self-play trajectories and enriches the knowledge graph.

    Usage:
        enrichment = KGEnrichment(kg, config)
        report = await enrichment.enrich_from_trajectories(trajectory_store)
    """

    def __init__(
        self,
        kg: MTGKnowledgeGraph | None = None,
        config: EnrichmentConfig | None = None,
    ):
        self.kg = kg
        self.config = config or EnrichmentConfig()

    async def enrich_from_trajectories(
        self, store: TrajectoryStore
    ) -> EnrichmentReport:
        """Run the full enrichment pipeline on collected trajectories."""
        report = EnrichmentReport(games_analyzed=len(store))
        logger.info(
            "KG enrichment: analyzing %d trajectories", len(store)
        )

        # Step 1: Extract per-game card usage and outcomes
        game_cards, card_wins, card_games = self._extract_card_statistics(
            store.trajectories
        )

        # Step 2: Discover synergies from co-occurrence + win correlation
        synergies = self._discover_synergies(game_cards, card_wins, card_games)
        report.synergies_proposed = len(synergies)

        # Step 3: Discover combos from multi-card co-occurrence in wins
        combos = self._discover_combos(game_cards)
        report.combos_proposed = len(combos)

        # Step 4: Write to KG if available
        if self.kg is not None:
            report.synergies_written = await self._write_synergies(synergies)
            report.card_stats_updated = await self._write_card_stats(
                card_wins, card_games
            )
        else:
            logger.info(
                "No KG connection — %d synergies and %d combos discovered but not written",
                len(synergies),
                len(combos),
            )

        logger.info(
            "Enrichment complete: %d synergies proposed (%d written), "
            "%d combos proposed, %d card stats updated",
            report.synergies_proposed,
            report.synergies_written,
            report.combos_proposed,
            report.card_stats_updated,
        )

        return report

    # ------------------------------------------------------------------
    # Step 1: Extract card statistics
    # ------------------------------------------------------------------

    def _extract_card_statistics(
        self, trajectories: list[Trajectory]
    ) -> tuple[
        list[tuple[set[str], bool]],   # (cards_in_game, won)
        Counter,                        # card → win count
        Counter,                        # card → game count
    ]:
        """Extract cards played per game and win/loss outcomes.

        Returns:
            game_cards: list of (set of card names played, won?)
            card_wins: Counter of card_name → num wins
            card_games: Counter of card_name → num games appeared
        """
        game_cards: list[tuple[set[str], bool]] = []
        card_wins: Counter = Counter()
        card_games: Counter = Counter()

        for traj in trajectories:
            cards_in_game: set[str] = set()
            for transition in traj.transitions:
                if transition.card_name:
                    cards_in_game.add(transition.card_name)

            won = traj.winner is not None and traj.winner == 0
            game_cards.append((cards_in_game, won))

            for card in cards_in_game:
                card_games[card] += 1
                if won:
                    card_wins[card] += 1

        return game_cards, card_wins, card_games

    # ------------------------------------------------------------------
    # Step 2: Synergy discovery
    # ------------------------------------------------------------------

    def _discover_synergies(
        self,
        game_cards: list[tuple[set[str], bool]],
        card_wins: Counter,
        card_games: Counter,
    ) -> list[dict[str, Any]]:
        """Find card pairs whose co-occurrence correlates with winning.

        A synergy is proposed when:
        - The pair co-occurs in >= min_co_occurrence games
        - The win rate when both appear is >= min_synergy_lift × baseline win rate
        """
        pair_wins: Counter = Counter()
        pair_games: Counter = Counter()

        for cards, won in game_cards:
            card_list = sorted(cards)
            for a, b in combinations(card_list, 2):
                pair = (a, b)
                pair_games[pair] += 1
                if won:
                    pair_wins[pair] += 1

        total_games = max(len(game_cards), 1)
        total_wins = sum(1 for _, won in game_cards if won)
        baseline_wr = total_wins / total_games if total_games > 0 else 0.5

        synergies = []
        for pair, count in pair_games.items():
            if count < self.config.min_co_occurrence:
                continue

            pair_wr = pair_wins.get(pair, 0) / count
            lift = pair_wr / baseline_wr if baseline_wr > 0 else 1.0

            if lift >= self.config.min_synergy_lift:
                synergies.append({
                    "card_a": pair[0],
                    "card_b": pair[1],
                    "co_occurrences": count,
                    "pair_win_rate": pair_wr,
                    "lift": lift,
                    "weight": min(lift * self.config.synergy_edge_weight, 1.0),
                })

        synergies.sort(key=lambda s: s["lift"], reverse=True)
        logger.info(
            "Discovered %d synergies (from %d pairs, baseline WR=%.2f)",
            len(synergies),
            len(pair_games),
            baseline_wr,
        )
        return synergies

    # ------------------------------------------------------------------
    # Step 3: Combo discovery
    # ------------------------------------------------------------------

    def _discover_combos(
        self, game_cards: list[tuple[set[str], bool]]
    ) -> list[dict[str, Any]]:
        """Find multi-card sets that consistently appear in winning games.

        Looks for 2-3 card sets that appear together in wins above threshold.
        """
        winning_games = [(cards, won) for cards, won in game_cards if won]
        if not winning_games:
            return []

        combos = []
        for size in range(2, self.config.max_combo_size + 1):
            combo_counts: Counter = Counter()
            for cards, _ in winning_games:
                card_list = sorted(cards)
                for group in combinations(card_list, size):
                    combo_counts[group] += 1

            for group, count in combo_counts.most_common(20):
                if count >= self.config.min_combo_co_occurrence:
                    combos.append({
                        "cards": list(group),
                        "size": size,
                        "win_appearances": count,
                        "description": f"Self-play discovered: {' + '.join(group)}",
                    })

        logger.info("Discovered %d potential combos from winning games", len(combos))
        return combos

    # ------------------------------------------------------------------
    # Step 4: Write to KG
    # ------------------------------------------------------------------

    async def _write_synergies(
        self, synergies: list[dict[str, Any]]
    ) -> int:
        """Write discovered synergies to the knowledge graph."""
        written = 0
        for syn in synergies:
            try:
                await self.kg.add_synergy(
                    syn["card_a"], syn["card_b"], weight=syn["weight"]
                )
                written += 1
            except Exception as e:
                logger.debug("Failed to write synergy %s↔%s: %s",
                             syn["card_a"], syn["card_b"], e)
        return written

    async def _write_card_stats(
        self, card_wins: Counter, card_games: Counter
    ) -> int:
        """Update per-card win rate statistics in the KG."""
        updated = 0
        for card_name, games in card_games.items():
            if games < self.config.min_games_for_stats:
                continue
            wins = card_wins.get(card_name, 0)
            try:
                await self.kg.update_card_stats(card_name, won=(wins > games // 2))
                updated += 1
            except Exception as e:
                logger.debug("Failed to update stats for %s: %s", card_name, e)
        return updated

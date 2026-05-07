"""Greedy Commander deck builder (G3).

Pipeline:

1. Resolve commander(s) from the card database.
2. Build a candidate pool (entire card DB or a supplied name list),
   shuffle it for diversity, then cap to ``pool_sample`` cards.
3. Greedily pick the highest-scoring legal card from the pool until
   the 99-card mainboard is full.
4. Pad remaining slots with basic lands matching the commander's colour
   identity.

The resulting :class:`~src.integrations.decklist_loader.Decklist` is
ready to be written to a plain-text file or passed directly to a
``GameRunner``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from src.agents.deck_builder.constraints import (
    ConstraintSet,
    commander_color_identity,
)
from src.agents.deck_builder.scorer import CardScorer
from src.integrations.decklist_loader import Decklist
from src.integrations.offline_card_db import OfflineCardDB

logger = logging.getLogger(__name__)

# Colour → basic land name map.
_BASIC_FOR_COLOR: dict[str, str] = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}
_COLORLESS_BASIC = "Wastes"


@dataclass
class DeckBuilderAgent:
    """Greedy Commander deck constructor.

    Args:
        commander_names: One or two commander names (partners, background, …).
        card_db:         :class:`OfflineCardDB` supplying Scryfall data.
        pool:            Optional explicit list of card names to draw from.
                         Defaults to the entire *card_db*.
        scorer:          :class:`CardScorer`; constructed with default weights
                         (no KG, no world model) if not supplied.
        format:          Legality format key for
                         ``card["legalities"][format]``.  Default
                         ``"commander"``.
        seed:            RNG seed for pool shuffle (enables determinism).
        pool_sample:     Maximum pool size after shuffling.  Keeping this
                         low (≤ 500) makes scoring tractable.
    """

    commander_names: list[str]
    card_db: OfflineCardDB
    pool: list[str] | None = None
    scorer: CardScorer | None = None
    format: str = "commander"
    seed: int | None = None
    pool_sample: int = 500

    def __post_init__(self) -> None:
        if self.scorer is None:
            self.scorer = CardScorer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def brew(self) -> Decklist:
        """Build and return a 100-card Commander deck asynchronously."""
        commander_cards = self._resolve_commanders()
        if not commander_cards:
            raise ValueError(
                f"No commanders found in card_db: {self.commander_names}"
            )

        cmd_ci = commander_color_identity(commander_cards)
        constraints = ConstraintSet(commander_ci=cmd_ci, format=self.format)

        pool = self._build_pool(
            {c.get("name", "") for c in commander_cards}
        )
        deck_names: list[str] = []

        logger.info(
            "Brewing deck for [%s]  colour=%s  pool=%d candidates",
            ", ".join(c.get("name", "?") for c in commander_cards),
            "".join(sorted(cmd_ci)) or "C",
            len(pool),
        )

        self.scorer.reset_caches()

        # ------------------------------------------------------------------
        # Greedy fill
        # ------------------------------------------------------------------
        while constraints.needs_more() and pool:
            scored = await self.scorer.score_batch(pool, deck_names)
            # Sort descending by score; stable sort preserves shuffle order
            # for ties (= diversity).
            scored.sort(key=lambda x: x[1], reverse=True)

            added = False
            for card_dict, score in scored:
                result = constraints.accept(card_dict)
                if result.ok:
                    constraints.add(card_dict)
                    deck_names.append(card_dict["name"])
                    pool.remove(card_dict)
                    logger.debug(
                        "  +  %-40s  score=%.3f  (%d/99)",
                        card_dict["name"],
                        score,
                        constraints.current_count(),
                    )
                    added = True
                    break

            if not added:
                logger.warning(
                    "No legal card found in remaining pool (%d); stopping.",
                    len(pool),
                )
                break

        # ------------------------------------------------------------------
        # Pad with basics
        # ------------------------------------------------------------------
        remaining = constraints.target_size - constraints.current_count()
        if remaining > 0:
            basics = self._basic_lands(cmd_ci, remaining)
            deck_names.extend(basics)
            logger.info("Padded %d basic land(s).", remaining)

        # ------------------------------------------------------------------
        # Build Decklist
        # ------------------------------------------------------------------
        deck = Decklist()
        for name in self.commander_names:
            if self.card_db.get(name) is not None:
                deck.commander.append(name)
        for name in deck_names:
            deck.mainboard[name] = deck.mainboard.get(name, 0) + 1

        logger.info(
            "Done: %d mainboard cards + %d commander(s).  Total=%d",
            sum(deck.mainboard.values()),
            len(deck.commander),
            sum(deck.mainboard.values()) + len(deck.commander),
        )
        return deck

    def brew_sync(self) -> Decklist:
        """Synchronous wrapper; calls :func:`asyncio.run` on :meth:`brew`."""
        return asyncio.run(self.brew())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_commanders(self) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for name in self.commander_names:
            data = self.card_db.get(name)
            if data is not None:
                cards.append(data)
            else:
                logger.warning("Commander '%s' not found in card_db.", name)
        return cards

    def _build_pool(
        self, exclude_names: set[str]
    ) -> list[dict[str, Any]]:
        """Shuffle + cap the candidate pool, excluding commanders."""
        if self.pool is not None:
            names = [n for n in self.pool if n not in exclude_names]
        else:
            # card_db.by_name keys are lowercase; values carry proper "name".
            names = [
                k for k in self.card_db.by_name
                if self.card_db.by_name[k].get("name", k) not in exclude_names
            ]

        rng = random.Random(self.seed)
        rng.shuffle(names)
        names = names[: self.pool_sample]

        pool: list[dict[str, Any]] = []
        for name in names:
            data = self.card_db.get(name)
            if data is not None:
                pool.append(data)
        return pool

    def _basic_lands(self, cmd_ci: set[str], count: int) -> list[str]:
        colors = [c for c in "WUBRG" if c in cmd_ci]
        if not colors:
            return [_COLORLESS_BASIC] * count
        lands = [_BASIC_FOR_COLOR[c] for c in colors]
        return [lands[i % len(lands)] for i in range(count)]

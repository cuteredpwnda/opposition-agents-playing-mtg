"""
Combo detector — uses KG to detect available and near-miss combos.

Reference: Section 8.4 of PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@dataclass
class ComboResult:
    """A detected combo or near-combo."""

    description: str
    pieces: list[str]
    effects: list[str]
    missing_piece: str | None = None
    draw_probability: float | None = None


class ComboDetector:
    """Uses the knowledge graph to detect combos from available cards."""

    def __init__(self, kg: MTGKnowledgeGraph):
        self.kg = kg

    async def detect_available_combos(
        self, hand: list[str], battlefield: list[str]
    ) -> list[ComboResult]:
        """Find combos where ALL pieces are in hand + battlefield."""
        available = list(set(hand + battlefield))
        results = await self.kg.detect_available_combos(available)
        return [
            ComboResult(
                description=r["description"],
                pieces=r["pieces"],
                effects=r.get("effects", []),
            )
            for r in results
        ]

    async def detect_near_combos(
        self,
        hand: list[str],
        battlefield: list[str],
        library_beliefs: dict[str, float] | None = None,
    ) -> list[ComboResult]:
        """Find combos missing exactly one piece."""
        available = list(set(hand + battlefield))
        results = await self.kg.detect_near_combos(available)
        return [
            ComboResult(
                description=r["description"],
                pieces=r["pieces"],
                effects=[],
                missing_piece=r.get("missingPiece"),
                draw_probability=(
                    library_beliefs.get(r.get("missingPiece", ""), 0.0)
                    if library_beliefs
                    else None
                ),
            )
            for r in results
        ]

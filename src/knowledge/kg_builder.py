"""
KG builder — populate Neo4j from Scryfall, Commander Spellbook, EDHREC.

Uses batch MERGE operations via the neo4j Python driver.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class KGBuilder:
    """Populates the Neo4j knowledge graph from external data sources."""

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self.driver.close()

    # -----------------------------------------------------------------------
    # Scryfall bulk import
    # -----------------------------------------------------------------------

    async def import_scryfall_cards(self, bulk_json_path: str) -> int:
        """Import cards from a Scryfall oracle-cards bulk JSON file.

        Each card becomes a (:Card) node with appropriate sub-labels
        (e.g., :Creature, :Instant) matching the OWL class hierarchy.
        """
        path = Path(bulk_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Scryfall bulk data not found: {path}")

        cards = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(cards)} cards from {path.name}")

        batch_size = 500
        imported = 0

        for i in range(0, len(cards), batch_size):
            batch = cards[i : i + batch_size]
            params = [self._card_to_props(c) for c in batch if c.get("name")]
            await self._merge_card_batch(params)
            imported += len(params)
            if imported % 5000 == 0:
                logger.info(f"  imported {imported}/{len(cards)} cards")

        logger.info(f"Scryfall import complete: {imported} cards")
        return imported

    async def _merge_card_batch(self, cards: list[dict[str, Any]]) -> None:
        """Batch MERGE cards into Neo4j."""
        query = """
        UNWIND $cards AS card
        MERGE (c:Card {cardName: card.cardName})
        SET c += card.props
        WITH c, card
        // Set sub-labels from type_line (matching OWL class hierarchy)
        FOREACH (_ IN CASE WHEN card.isCreature THEN [1] ELSE [] END |
            SET c:Creature)
        FOREACH (_ IN CASE WHEN card.isInstant THEN [1] ELSE [] END |
            SET c:Instant)
        FOREACH (_ IN CASE WHEN card.isSorcery THEN [1] ELSE [] END |
            SET c:Sorcery)
        FOREACH (_ IN CASE WHEN card.isEnchantment THEN [1] ELSE [] END |
            SET c:Enchantment)
        FOREACH (_ IN CASE WHEN card.isArtifact THEN [1] ELSE [] END |
            SET c:Artifact)
        FOREACH (_ IN CASE WHEN card.isPlaneswalker THEN [1] ELSE [] END |
            SET c:Planeswalker)
        FOREACH (_ IN CASE WHEN card.isLand THEN [1] ELSE [] END |
            SET c:Land)
        FOREACH (_ IN CASE WHEN card.isBattle THEN [1] ELSE [] END |
            SET c:Battle)
        // Create keyword relationships
        WITH c, card
        UNWIND card.keywords AS keyword
        MERGE (k:Keyword {name: keyword})
        MERGE (c)-[:HAS_KEYWORD]->(k)
        """
        async with self.driver.session() as session:
            await session.run(query, cards=cards)

    @staticmethod
    def _card_to_props(card: dict[str, Any]) -> dict[str, Any]:
        """Convert a Scryfall card JSON to Neo4j node properties."""
        type_line = card.get("type_line", "")
        colors = card.get("colors", [])
        color_identity = card.get("color_identity", [])
        keywords = card.get("keywords", [])

        return {
            "cardName": card["name"],
            "props": {
                "scryfallId": card.get("id", ""),
                "oracleText": card.get("oracle_text", ""),
                "manaCostText": card.get("mana_cost", ""),
                "manaValue": int(card.get("cmc", 0)),
                "typeLine": type_line,
                "power": card.get("power"),
                "toughness": card.get("toughness"),
                "loyalty": card.get("loyalty"),
                "defense": card.get("defense"),
                "rarity": card.get("rarity", ""),
                "setCode": card.get("set", ""),
                "colors": colors,
                "colorIdentity": color_identity,
                "edhrecRank": card.get("edhrec_rank"),
            },
            "isCreature": "Creature" in type_line,
            "isInstant": "Instant" in type_line,
            "isSorcery": "Sorcery" in type_line,
            "isEnchantment": "Enchantment" in type_line,
            "isArtifact": "Artifact" in type_line,
            "isPlaneswalker": "Planeswalker" in type_line,
            "isLand": "Land" in type_line,
            "isBattle": "Battle" in type_line,
            "keywords": keywords,
        }

    # -----------------------------------------------------------------------
    # Commander Spellbook combo import
    # -----------------------------------------------------------------------

    async def import_combos(self, combos_json_path: str) -> int:
        """Import combos from a Commander Spellbook JSON dump."""
        path = Path(combos_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Combos file not found: {path}")

        combos = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(combos)} combos")

        for combo in combos:
            await self._merge_combo(combo)

        logger.info(f"Combo import complete: {len(combos)} combos")
        return len(combos)

    async def _merge_combo(self, combo: dict[str, Any]) -> None:
        """Merge a single combo and its component card relationships."""
        query = """
        MERGE (combo:Combo {comboId: $combo_id})
        SET combo.comboDescription = $description,
            combo.result = $result
        WITH combo
        UNWIND $card_names AS cardName
        MERGE (c:Card {cardName: cardName})
        MERGE (c)-[:PART_OF_COMBO]->(combo)
        """
        card_names = combo.get("cards", combo.get("uses", []))
        if isinstance(card_names, list) and card_names:
            # Handle both string lists and object lists
            names = [
                c["card"]["name"] if isinstance(c, dict) else c
                for c in card_names
            ]
        else:
            return

        # Stable id: use upstream id if present, otherwise hash of card set + source.
        combo_id = str(combo.get("id") or "")
        if not combo_id:
            import hashlib
            key = "|".join(sorted(n.lower() for n in names if n))
            src = combo.get("source", "merged")
            combo_id = f"{src}:{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"

        async with self.driver.session() as session:
            await session.run(
                query,
                combo_id=combo_id,
                description=combo.get("description", ""),
                result=", ".join(combo.get("produces", combo.get("results", []) or [])),
                card_names=names,
            )

    # -----------------------------------------------------------------------
    # Format legality edges
    # -----------------------------------------------------------------------

    async def import_legalities(self, bulk_json_path: str) -> None:
        """Create LEGAL_IN / BANNED_IN edges from Scryfall legality data."""
        path = Path(bulk_json_path)
        cards = json.loads(path.read_text(encoding="utf-8"))

        query = """
        UNWIND $batch AS item
        MATCH (c:Card {cardName: item.name})
        WITH c, item
        UNWIND item.legal_formats AS fmt
        MERGE (f:Format {name: fmt})
        MERGE (c)-[:LEGAL_IN]->(f)
        """
        batch_size = 500
        for i in range(0, len(cards), batch_size):
            batch = []
            for card in cards[i : i + batch_size]:
                legalities = card.get("legalities", {})
                legal_formats = [
                    fmt for fmt, status in legalities.items() if status == "legal"
                ]
                if legal_formats:
                    batch.append(
                        {"name": card["name"], "legal_formats": legal_formats}
                    )
            if batch:
                async with self.driver.session() as session:
                    await session.run(query, batch=batch)

        logger.info("Legality edges imported")

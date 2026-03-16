"""
Async Scryfall API client with rate limiting.

For most use cases, prefer the Scrython library (pip install scrython).
This async client is for integration with the async game loop.

Reference: Section 11.1 of PLAN.md, Scrython (MIT) — https://github.com/NandaScott/Scrython
"""

from __future__ import annotations

import asyncio
import time

import httpx


class ScryfallClient:
    """Async client for the Scryfall REST API with rate limiting."""

    BASE_URL = "https://api.scryfall.com"
    RATE_LIMIT = 0.1  # 100ms between requests (Scryfall asks for 50-100ms)

    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": "OppositionAgentsMTG/1.0",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._last_request: float = 0

    async def close(self) -> None:
        await self.client.aclose()

    async def get_card_by_name(self, name: str) -> dict:
        """GET /cards/named?exact={name}"""
        await self._rate_limit()
        resp = await self.client.get(
            f"{self.BASE_URL}/cards/named", params={"exact": name}
        )
        resp.raise_for_status()
        return resp.json()

    async def get_rulings(self, card_name: str) -> list[dict]:
        """Fetch rulings for a card by resolving its Scryfall ID first."""
        card = await self.get_card_by_name(card_name)
        rulings_uri = card.get("rulings_uri", "")
        if not rulings_uri:
            return []
        await self._rate_limit()
        resp = await self.client.get(rulings_uri)
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def search_cards(self, query: str) -> list[dict]:
        """GET /cards/search?q={query} using Scryfall search syntax."""
        await self._rate_limit()
        resp = await self.client.get(
            f"{self.BASE_URL}/cards/search", params={"q": query}
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def get_bulk_data_url(
        self, bulk_type: str = "oracle_cards"
    ) -> str:
        """Get download URI for a Scryfall bulk data type."""
        await self._rate_limit()
        resp = await self.client.get(f"{self.BASE_URL}/bulk-data")
        resp.raise_for_status()
        for item in resp.json()["data"]:
            if item["type"] == bulk_type:
                return item["download_uri"]
        raise ValueError(f"Bulk data type '{bulk_type}' not found")

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        wait = self.RATE_LIMIT - (now - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()

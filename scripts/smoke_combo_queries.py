"""Smoke check the new combo-name + outcome KG queries."""
from __future__ import annotations

import asyncio

from src.knowledge.knowledge_graph import MTGKnowledgeGraph


async def main() -> None:
    kg = MTGKnowledgeGraph()
    try:
        print("--- outcome categories (top 10 by combo count) ---")
        for r in (await kg.list_outcome_categories())[:10]:
            print(f"  {r['outcomeId']:>40s}  {r['comboCount']:>5d}  {r['displayName']}")

        print("\n--- combos by outcome (mana, infinite) top 5 ---")
        for r in (await kg.get_combos_by_outcome("mana", "infinite"))[:5]:
            print(f"  [{r['cardCount']}p] {r['comboName']}")

        print("\n--- combos containing Heliod, Sun-Crowned (first 3) ---")
        helps = await kg.get_combos_containing("Heliod, Sun-Crowned")
        for r in helps[:3]:
            outs = [o["displayName"] for o in r["outcomes"] if o.get("displayName")]
            print(f"  {r['comboName']!r}  outcomes={outs}")

        if helps:
            first_id = helps[0]["comboId"]
            print(f"\n--- related to {first_id} by shared outcome (top 3) ---")
            for r in (await kg.get_related_combos_by_outcome(first_id))[:3]:
                print(f"  shared={r['sharedOutcomes']}  {r['comboName']}")
    finally:
        await kg.close()


if __name__ == "__main__":
    asyncio.run(main())

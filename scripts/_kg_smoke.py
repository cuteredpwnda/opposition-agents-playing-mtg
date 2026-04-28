"""Quick KG smoke check."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from src.config import settings


async def main():
    d = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    async with d.session() as s:
        r = await s.run("MATCH (n) RETURN count(n) AS n")
        rec = await r.single()
        print(f"Total nodes: {rec['n']}")

        r = await s.run(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS c "
            "ORDER BY c DESC LIMIT 10"
        )
        async for row in r:
            print(f"  {row['labels']}: {row['c']}")
    await d.close()


if __name__ == "__main__":
    asyncio.run(main())

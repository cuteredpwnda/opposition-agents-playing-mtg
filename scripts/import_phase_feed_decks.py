"""Import MTGGoldfish metagame decks shipped in the phase-rs submodule feeds.

phase-rs vendors scraped MTGGoldfish metagame snapshots under
``external/phase-rs/client/public/feeds/mtggoldfish-<format>.json``. This
script converts them into plain-text decklists under
``data/decks/meta/<format>/`` and writes a provenance manifest that records:

* the feed file SHA-256, its ``updated`` stamp and source URL,
* the phase-rs submodule commit the feed was read from,
* per-deck format legality according to the engine card database, and
* per-deck engine coverage: the share of mainboard copies whose parsed card
  data contains an ``Unimplemented`` effect node.

Decks are never silently "fixed": an illegal or partially implemented deck
is still written, but flagged so benchmark scripts can exclude it.

Usage::

    python scripts/import_phase_feed_decks.py --formats standard modern pioneer \
        --card-data <phase-v374>/data/card-data.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = REPO_ROOT / "external" / "phase-rs"
FEED_DIR = SUBMODULE / "client" / "public" / "feeds"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _contains_unimplemented(node: object) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "Unimplemented":
            return True
        return any(_contains_unimplemented(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_unimplemented(value) for value in node)
    return False


def _fold(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()


class CardResolver:
    """Map feed card names onto engine card-database names.

    Only two lossless rewrites are applied: accent folding and resolving a
    ``Front // Back`` name to its front face. Anything else is left missing.
    """

    def __init__(self, raw_cards: dict[str, dict]) -> None:
        self.cards: dict[str, dict] = {}
        for key, card in raw_cards.items():
            name = str(card.get("name", key))
            self.cards.setdefault(name.lower(), card)
            self.cards.setdefault(_fold(name), card)

    def resolve(self, name: str) -> tuple[str, dict] | None:
        for candidate in (name, name.split(" // ")[0]):
            card = self.cards.get(candidate.lower()) or self.cards.get(_fold(candidate))
            if card is not None:
                return str(card["name"]), card
        return None


def _entries(rows: list[dict]) -> list[tuple[int, str]]:
    return [(int(row["count"]), str(row["name"])) for row in rows if int(row.get("count", 0)) > 0]


def _normalise(
    rows: list[tuple[int, str]], resolver: CardResolver, renames: dict[str, str],
) -> list[tuple[int, str]]:
    out = []
    for count, name in rows:
        hit = resolver.resolve(name)
        if hit is not None and hit[0] != name:
            renames[name] = hit[0]
        out.append((count, hit[0] if hit else name))
    return out


def screen_deck(
    main: list[tuple[int, str]], fmt: str, resolver: CardResolver
) -> dict[str, object]:
    """Return legality and engine-coverage facts for one mainboard."""
    missing: list[str] = []
    illegal: list[str] = []
    unimplemented: list[str] = []
    copies = sum(count for count, _ in main)
    unimplemented_copies = 0
    for count, name in main:
        hit = resolver.resolve(name)
        if hit is None:
            missing.append(name)
            continue
        card = hit[1]
        legality = (card.get("legalities") or {}).get(fmt)
        if legality not in {"legal", "restricted"}:
            illegal.append(name)
        if _contains_unimplemented(card):
            unimplemented.append(name)
            unimplemented_copies += count
    return {
        "mainboard_count": copies,
        "missing_cards": missing,
        "illegal_cards": illegal,
        "unimplemented_cards": unimplemented,
        "implemented_copy_share": round(1 - unimplemented_copies / copies, 4) if copies else 0.0,
        "eligible": not missing and not illegal and copies >= 60,
    }


def _render(
    deck: dict, fmt: str, feed: dict,
    main: list[tuple[int, str]], side: list[tuple[int, str]],
) -> str:
    lines = [
        f"# {deck['name']} ({fmt.title()}) - MTGGoldfish metagame snapshot",
        f"# Source: {feed.get('source')} (feed updated {feed.get('updated')})",
        f"# Author: {deck.get('author') or 'unknown'}; vendored by phase-rs feed {feed.get('id')}",
        "",
    ]
    lines += [f"{count} {name}" for count, name in main]
    if side:
        lines += ["", "Sideboard"] + [f"{count} {name}" for count, name in side]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--formats", nargs="+", default=["standard", "modern", "pioneer"])
    parser.add_argument(
        "--card-data", default=os.environ.get("PHASE_CARD_DATA"),
        help="phase-rs card-data.json used for legality and coverage screening",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "decks" / "meta"))
    args = parser.parse_args(argv)
    if not args.card_data or not Path(args.card_data).is_file():
        parser.error("--card-data (or PHASE_CARD_DATA) must name an existing card-data.json")

    resolver = CardResolver(json.loads(Path(args.card_data).read_text(encoding="utf-8")))
    phase_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SUBMODULE, check=True, capture_output=True, text=True,
    ).stdout.strip()

    for fmt in args.formats:
        feed_path = FEED_DIR / f"mtggoldfish-{fmt}.json"
        feed_bytes = feed_path.read_bytes()
        feed = json.loads(feed_bytes)
        out = Path(args.output_dir) / fmt
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, deck in enumerate(feed["decks"], start=1):
            name = f"{fmt}_{rank:02d}_{_slug(deck['name'])}.txt"
            renames: dict[str, str] = {}
            main = _normalise(_entries(deck.get("main") or []), resolver, renames)
            side = _normalise(_entries(deck.get("sideboard") or []), resolver, renames)
            text = _render(deck, fmt, feed, main, side)
            (out / name).write_text(text, encoding="utf-8")
            screen = screen_deck(main, fmt, resolver)
            records.append({
                "file": name,
                "name": deck["name"],
                "meta_rank": rank,
                "colors": deck.get("colors"),
                "tags": deck.get("tags"),
                "deck_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalization": renames,
                **screen,
            })
            flag = "ok" if screen["eligible"] else "INELIGIBLE"
            print(
                f"{fmt:9s} {rank:2d} {deck['name']:28s} {flag:10s} "
                f"implemented={screen['implemented_copy_share']:.2f} "
                f"illegal={screen['illegal_cards']} missing={screen['missing_cards']}"
            )
        manifest = {
            "schema_version": 1,
            "format": fmt,
            "source": feed.get("source"),
            "feed_id": feed.get("id"),
            "feed_updated": feed.get("updated"),
            "feed_path": feed_path.relative_to(REPO_ROOT).as_posix(),
            "feed_sha256": hashlib.sha256(feed_bytes).hexdigest(),
            "phase_rs_commit": phase_commit,
            "card_data_sha256": hashlib.sha256(Path(args.card_data).read_bytes()).hexdigest(),
            "scope": (
                "Factual metagame deck selections from MTGGoldfish as vendored by phase-rs; "
                "legality and coverage are screened against the engine card database."
            ),
            "decks": records,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

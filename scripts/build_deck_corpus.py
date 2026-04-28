#!/usr/bin/env python
"""Bulk-fetch a corpus of popular EDH decks from EDHREC.

Builds a folder of plaintext decklists under ``data/decks/edhrec/`` that
can be loaded for self-play, training, or bracket benchmarking.

Usage::

    # default: top 50 commanders by EDHREC popularity
    python scripts/build_deck_corpus.py

    # custom commander list
    python scripts/build_deck_corpus.py --commanders krenko-mob-boss atraxa-praetors-voice

    # bigger / smaller
    python scripts/build_deck_corpus.py --top 100

EDHREC may not have an "average deck" JSON for every commander; misses
are logged and skipped, not fatal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.decklist_loader import DecklistLoader

# Hand-picked seed list (top EDHREC commanders, stable across years).
# Used when EDHREC's "top commanders" listing API is unavailable.
DEFAULT_COMMANDERS = [
    "atraxa-praetors-voice", "edgar-markov", "the-ur-dragon",
    "muldrotha-the-gravetide", "korvold-fae-cursed-king",
    "yuriko-the-tigers-shadow", "krenko-mob-boss", "lathril-blade-of-the-elves",
    "meren-of-clan-nel-toth", "kenrith-the-returned-king",
    "wilhelt-the-rotcleaver", "kaalia-of-the-vast", "azusa-lost-but-seeking",
    "narset-enlightened-master", "talrand-sky-summoner",
    "kess-dissident-mage", "sliver-overlord", "alela-artful-provocateur",
    "isshin-two-heavens-as-one", "miirym-sentinel-wyrm",
    "chulane-teller-of-tales", "tatyova-benthic-druid", "go-shintai-of-life-s-origin",
    "lathliss-dragon-queen", "krark-the-thumbless", "purphoros-god-of-the-forge",
    "marrow-gnawer", "kaalia-zenith-seeker", "saskia-the-unyielding",
    "kalamax-the-stormsire",
]


def to_text(deck) -> str:
    out: list[str] = []
    if deck.commander:
        out.append("Commander")
        for name in deck.commander:
            out.append(f"1 {name}")
        out.append("")
    out.append("Mainboard")
    for name, count in sorted(deck.mainboard.items()):
        out.append(f"{count} {name}")
    out.append("")
    return "\n".join(out)


async def fetch_one(loader: DecklistLoader, slug: str, out_dir: Path) -> bool:
    out_path = out_dir / f"{slug}.txt"
    if out_path.exists():
        return True
    try:
        deck = await loader.from_edhrec_average(slug)
    except Exception as exc:
        print(f"  [skip] {slug}: {exc}")
        return False
    out_path.write_text(to_text(deck), encoding="utf-8")
    print(f"  [ok]   {slug} -> {out_path}")
    return True


async def main(args: argparse.Namespace) -> int:
    loader = DecklistLoader()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.commanders:
        slugs = list(args.commanders)
    else:
        slugs = DEFAULT_COMMANDERS[: args.top]

    print(f"Fetching {len(slugs)} EDHREC average decks -> {out_dir}")
    ok = 0
    # Sequential with tiny pause; EDHREC isn't a high-throughput target.
    for slug in slugs:
        if await fetch_one(loader, slug, out_dir):
            ok += 1
        await asyncio.sleep(0.5)

    print(f"\nDone: {ok}/{len(slugs)} fetched")

    # Manifest for downstream tooling
    manifest = out_dir / "manifest.json"
    decks = sorted(p.name for p in out_dir.glob("*.txt"))
    manifest.write_text(json.dumps({"decks": decks}, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commanders", nargs="*",
                   help="Specific EDHREC slugs (e.g. krenko-mob-boss)")
    p.add_argument("--top", type=int, default=30,
                   help="When --commanders is omitted, pull this many from the seed list")
    p.add_argument("--out-dir", default="data/decks/edhrec")
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main(build_parser().parse_args())))

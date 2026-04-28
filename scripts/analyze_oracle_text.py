#!/usr/bin/env python
"""Scan all Scryfall oracle text and rank mechanics by frequency.

Outputs a prioritised roadmap of which abilities/keywords are worth
implementing next, by counting how often each pattern appears across
the full card pool (typically ~30k unique cards).

Categories surfaced:
  - Triggered abilities: "Whenever ...", "When ... enters", "At the beginning ..."
  - Replacement effects: "If ... would ..., instead ..."
  - Static abilities (heuristic): "Creatures you control get ...", "As long as ..."
  - Counter types: "+1/+1 counter", "loyalty counter", "stun counter", ...
  - Keywords (structured): from Scryfall's `keywords` field

Usage:
    python scripts/fetch_card_data.py            # first, populate cache
    python scripts/analyze_oracle_text.py        # then, run analysis
    python scripts/analyze_oracle_text.py --top 30 --out reports/mechanics.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORACLE_PATH = PROJECT_ROOT / "data" / "scryfall" / "oracle-cards.json"

# --- Pattern catalogue ----------------------------------------------------

TRIGGER_PATTERNS = {
    "ETB (enters the battlefield)":
        re.compile(r"\bwhen\b[^.]*?\benters\b", re.I),
    "Death trigger (dies)":
        re.compile(r"\bwhen(?:ever)?\b[^.]*?\bdies\b", re.I),
    "Attack trigger (whenever ~ attacks)":
        re.compile(r"\bwhenever\b[^.]*?\battacks\b", re.I),
    "Block trigger":
        re.compile(r"\bwhenever\b[^.]*?\bblocks\b", re.I),
    "Damage trigger":
        re.compile(r"\bwhenever\b[^.]*?\bdeals (combat )?damage\b", re.I),
    "Cast trigger":
        re.compile(r"\bwhenever (you|a player) casts?\b", re.I),
    "Upkeep trigger":
        re.compile(r"\bat the beginning of\b[^.]*?\bupkeep\b", re.I),
    "End-step trigger":
        re.compile(r"\bat the beginning of\b[^.]*?\bend step\b", re.I),
    "Draw step trigger":
        re.compile(r"\bat the beginning of\b[^.]*?\bdraw step\b", re.I),
}

REPLACEMENT_PATTERNS = {
    "Replacement (would ... instead)":
        re.compile(r"\bwould\b[^.]*?\binstead\b", re.I),
    "Skip step":
        re.compile(r"\bskip\b[^.]*?\b(turn|step|phase)\b", re.I),
    "Enters tapped":
        re.compile(r"\benters (the battlefield )?tapped\b", re.I),
    "Damage prevention":
        re.compile(r"\bprevent (all|the next|that) (combat )?damage\b", re.I),
}

STATIC_PATTERNS = {
    "Anthem (creatures you control get +X/+X)":
        re.compile(r"\bcreatures you control get \+\d+/\+\d+", re.I),
    "Cost reduction":
        re.compile(r"\bcosts? \{[^\}]+\} less to cast\b", re.I),
    "Cost increase":
        re.compile(r"\bcosts? \{[^\}]+\} more to cast\b", re.I),
    "As long as ...":
        re.compile(r"\bas long as\b", re.I),
    "Lord effect (other ~ get)":
        re.compile(r"\bother [\w\s]+ you control get\b", re.I),
}

COUNTER_PATTERN = re.compile(r"\b([+\-]?\d+/[+\-]?\d+|[a-z]+) counter", re.I)


def categorise(cards: list[dict]) -> dict[str, Counter]:
    """Return frequency counts per category."""
    out = {
        "triggers": Counter(),
        "replacements": Counter(),
        "statics": Counter(),
        "counters": Counter(),
        "keywords": Counter(),
    }

    for card in cards:
        text = (card.get("oracle_text") or "").lower()
        if not text:
            continue

        for label, pat in TRIGGER_PATTERNS.items():
            if pat.search(text):
                out["triggers"][label] += 1

        for label, pat in REPLACEMENT_PATTERNS.items():
            if pat.search(text):
                out["replacements"][label] += 1

        for label, pat in STATIC_PATTERNS.items():
            if pat.search(text):
                out["statics"][label] += 1

        for m in COUNTER_PATTERN.finditer(text):
            kind = m.group(1).lower()
            # Filter noise: keep only kinds that look like counter names
            if len(kind) <= 30:
                out["counters"][kind] += 1

        for kw in card.get("keywords", []):
            out["keywords"][kw] += 1

    return out


# --- Implementation status ------------------------------------------------
# Update this set as features land in src/engine/

IMPLEMENTED = {
    "ETB (enters the battlefield)",
    "Anthem (creatures you control get +X/+X)",
    "+1/+1",       # counter
    "loyalty",     # counter
    # add more as you implement them
}


def render_report(stats: dict[str, Counter], total: int, top: int) -> str:
    lines: list[str] = []
    lines.append(f"# Mechanic frequency report\n")
    lines.append(f"Cards scanned: **{total:,}**\n")
    lines.append(
        "Counts = number of distinct cards whose oracle text matches the pattern. "
        "Use this to prioritise which mechanics to implement next.\n"
    )

    sections = [
        ("Triggered abilities", "triggers"),
        ("Replacement effects", "replacements"),
        ("Static abilities", "statics"),
        ("Counter types (top names)", "counters"),
        ("Keyword abilities (Scryfall structured field)", "keywords"),
    ]

    for title, key in sections:
        lines.append(f"\n## {title}\n")
        lines.append("| # | Pattern | Cards | Coverage | Implemented |")
        lines.append("|--:|---------|------:|---------:|:-----------:|")
        for i, (label, count) in enumerate(stats[key].most_common(top), 1):
            pct = 100.0 * count / total
            done = "✅" if label in IMPLEMENTED else " "
            lines.append(f"| {i} | `{label}` | {count:,} | {pct:.1f}% | {done} |")

    return "\n".join(lines) + "\n"


def main(args) -> int:
    if not ORACLE_PATH.exists():
        print(f"ERROR: {ORACLE_PATH} not found.", file=sys.stderr)
        print("Run: python scripts/fetch_card_data.py", file=sys.stderr)
        return 1

    print(f"Loading {ORACLE_PATH}...")
    cards = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    print(f"  {len(cards):,} cards loaded")

    print("Scanning oracle text...")
    stats = categorise(cards)

    report = render_report(stats, len(cards), args.top)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {out_path}")
    else:
        print()
        print(report)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=20,
                   help="Top-N rows to show per category (default 20)")
    p.add_argument("--out", type=str, default=None,
                   help="Write Markdown report here instead of stdout")
    return p


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))

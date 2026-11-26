"""Acquire pinned, source-attributed Forge deck facts for development benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import httpx

REVISION = "69b963e4e519e75b3f9953ac2b57301aea6cd48b"
BASE = f"https://raw.githubusercontent.com/Card-Forge/forge/{REVISION}"
SOURCES = {
    "forge_boros_burn": ("burn", "forge-gui/res/quest/world/Kaladesh/duels/82.dck"),
    "forge_atarka_burn": ("burn", "forge-gui/res/quest/world/Kaladesh/duels/93.dck"),
    "forge_eldrazi_tron": ("big-mana", "forge-gui/res/quest/world/Aether_Revolt/duels/67.dck"),
    "forge_affinity": ("artifact-aggro", "forge-gui/res/quest/world/Magic_Origins/duels/100.dck"),
}
NAME_ALIASES = {"Ghirapur AEther Grid": "Ghirapur Aether Grid"}


def parse_forge(text: str) -> dict[str, Counter[str]]:
    sections: dict[str, Counter[str]] = {"main": Counter(), "sideboard": Counter()}
    active = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            active = section if section in sections else None
            continue
        if active is None:
            continue
        match = re.fullmatch(r"([1-9]\d*)\s+([^|]+)(?:\|.*)?", line)
        if not match:
            raise ValueError(f"Malformed Forge card row: {line!r}")
        name = match[2].strip()
        sections[active][NAME_ALIASES.get(name, name)] += int(match[1])
    if sum(sections["main"].values()) < 60:
        raise ValueError("Source deck must contain at least 60 mainboard cards")
    if sum(sections["sideboard"].values()) > 15:
        raise ValueError("Source sideboard exceeds 15 cards")
    return sections


def audit_deck(
    deck: dict[str, Counter[str]], cards: list[dict],
) -> list[dict[str, str]]:
    legal_names = {
        card["name"] for card in cards
        if card.get("legalities", {}).get("modern") == "legal"
        and card.get("layout") not in {"token", "double_faced_token", "art_series"}
    }
    return [{"name": name, "reason": "not Modern-legal in frozen oracle snapshot"}
            for name in sorted(set(deck["main"]) | set(deck["sideboard"]))
            if name not in legal_names]


def render_deck(deck: dict[str, Counter[str]], url: str) -> str:
    rows = [
        "# Historical Forge deck facts; not a current metagame or strength claim.",
        f"# Source: {url}",
        "# Source project: Card-Forge/forge, GPL-3.0; see LICENSE.forge.txt.",
    ]
    for label, section in (("Mainboard", "main"), ("Sideboard", "sideboard")):
        rows.append(label)
        rows.extend(f"{count} {name}" for name, count in deck[section].items())
    return "\n".join(rows) + "\n"


def run(output: Path, oracle: Path) -> None:
    oracle_bytes = oracle.read_bytes()
    cards = json.loads(oracle_bytes)
    acquisitions = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        license_response = client.get(f"{BASE}/LICENSE")
        license_response.raise_for_status()
        for name, (archetype, source) in SOURCES.items():
            url = f"{BASE}/{source}"
            response = client.get(url)
            response.raise_for_status()
            deck = parse_forge(response.text)
            failures = audit_deck(deck, cards)
            if failures:
                raise ValueError(f"{name}: {failures}")
            text = render_deck(deck, url)
            acquisitions.append((name, text, {
                "id": name, "archetype": archetype, "file": f"{name}.txt",
                "source_url": url, "source_revision": REVISION,
                "source_sha256": hashlib.sha256(response.content).hexdigest(),
                "deck_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "mainboard_count": sum(deck["main"].values()),
                "sideboard_count": sum(deck["sideboard"].values()),
                "modern_oracle_eligible": True,
                "native_qualification": "pending",
                "normalization": NAME_ALIASES,
            }))
    output.mkdir(parents=True, exist_ok=True)
    (output / "LICENSE.forge.txt").write_bytes(license_response.content)
    for name, text, _ in acquisitions:
        (output / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "source_project": "Card-Forge/forge",
        "source_license": "GPL-3.0", "source_revision": REVISION,
        "oracle_sha256": hashlib.sha256(oracle_bytes).hexdigest(),
        "scope": "historical factual deck selections, not current tournament recommendations",
        "decks": [record for _, _, record in acquisitions],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Acquired {len(acquisitions)} decks with Modern oracle eligibility: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data") / "decks" / "benchmark" / "sourced")
    parser.add_argument("--oracle-file", type=Path,
                        default=Path("data") / "scryfall" / "oracle-cards.json")
    args = parser.parse_args()
    run(args.output_dir, args.oracle_file)


if __name__ == "__main__":
    main()

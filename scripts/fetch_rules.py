#!/usr/bin/env python
"""Download (or refresh) the Magic: The Gathering Comprehensive Rules.

The official rules are published as a `.txt` file on
https://magic.wizards.com/en/rules. The page links to the latest revision
under a name like ``MagicCompRules YYYYMMDD.txt`` (and historically also
``CRYYYYMMDD.txt``). This script:

1. Fetches the rules landing page.
2. Scrapes the most recent ``.txt`` link (Wizards updates this on every
   set release; the link contains the effective date).
3. Downloads it into ``data/rules/`` keeping the dated filename.
4. Updates ``data/rules/latest.txt`` (a copy / symlink) so downstream
   consumers like the judge vectorstore have a stable path.

Usage::

    python scripts/fetch_rules.py            # download if missing or stale
    python scripts/fetch_rules.py --force    # always re-download

Notes
-----
* The page is plain HTML; no API key is required.
* We only follow links on Wizards-owned hosts to avoid prompt-injection
  redirecting us elsewhere.
* If the network is unavailable, the script exits non-zero without
  touching existing files.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx

RULES_PAGE_URL = "https://magic.wizards.com/en/rules"
ALLOWED_HOSTS = {"magic.wizards.com", "media.wizards.com"}
RULES_DIR = Path(__file__).resolve().parents[1] / "data" / "rules"
LATEST_LINK = RULES_DIR / "latest.txt"


def _is_allowed(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def _find_rules_url(html: str) -> Optional[str]:
    """Extract the most recent ``MagicCompRules*.txt`` href from the rules page."""
    # Most recent link wins. Wizards lists them in date order with the newest
    # on top. We accept either MagicCompRules or CR prefixes.
    pattern = re.compile(
        r'href="(?P<url>https?://[^"]*?'
        r'(?:MagicCompRules|MagicComp%20Rules|CR)[^"]*?\.txt)"',
        re.IGNORECASE,
    )
    for m in pattern.finditer(html):
        url = m.group("url")
        if _is_allowed(url):
            return url
    return None


def _filename_from_url(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    # Decode "MagicComp%20Rules%2020260417.txt" → "MagicComp Rules 20260417.txt"
    return urllib.parse.unquote(name)


def fetch_rules(force: bool = False) -> Path:
    """Fetch the latest CR text file. Returns the downloaded path."""
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        page = client.get(RULES_PAGE_URL)
        page.raise_for_status()
        rules_url = _find_rules_url(page.text)
        if not rules_url:
            print("[error] could not locate a Comprehensive Rules .txt link on "
                  f"{RULES_PAGE_URL}", file=sys.stderr)
            sys.exit(2)

        target = RULES_DIR / _filename_from_url(rules_url)
        if target.exists() and not force:
            print(f"[ok] {target.name} already present ({target.stat().st_size} bytes); "
                  "use --force to refresh")
            _refresh_latest_link(target)
            return target

        print(f"[fetch] {rules_url}")
        resp = client.get(rules_url)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"[saved] {target.relative_to(RULES_DIR.parent.parent)} "
              f"({len(resp.content)} bytes)")

    _refresh_latest_link(target)
    return target


def _refresh_latest_link(target: Path) -> None:
    """Make ``data/rules/latest.txt`` point at the most recent download.

    Uses a hard copy on Windows (no symlink permission) and a symlink on
    POSIX. Either way, downstream code can read ``data/rules/latest.txt``
    without knowing the date.
    """
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        try:
            LATEST_LINK.unlink()
        except OSError:
            pass
    try:
        LATEST_LINK.symlink_to(target.name)  # relative symlink
        return
    except (OSError, NotImplementedError):
        pass
    shutil.copyfile(target, LATEST_LINK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the latest file already exists")
    args = parser.parse_args()
    fetch_rules(force=args.force)


if __name__ == "__main__":
    main()

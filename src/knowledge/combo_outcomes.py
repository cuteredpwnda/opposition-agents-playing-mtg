"""Categorize Spellbook ``feature`` strings into structured outcome buckets.

Spellbook stores combo outputs as free-form features like
``"Infinite colorless mana"`` or ``"Near-infinite damage"``.  We bucket
those into a small ontology (``category`` + ``magnitude``) so combos
that produce the same kind of effect can be linked to a shared
``:Outcome`` node and queried by category.

Categories are intentionally coarse (a few dozen at most) — fine-grained
enough to filter ("show me all infinite-mana combos") without exploding
the graph.

Magnitude is one of: ``infinite``, ``arbitrary``, ``near_infinite``,
``finite`` (default).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Order matters: more specific patterns must come first so e.g.
# "Infinite colored mana" matches MANA before generic INFINITE.
CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("win_game", re.compile(r"\b(win\s+the\s+game|wins?\s+the\s+game)\b", re.I)),
    ("opponents_lose", re.compile(r"\b(each|all)\s+opponents?\s+(lose|loses)\b", re.I)),
    ("mill_all", re.compile(r"\bmill(s|ed)?\s+(their|all)\b", re.I)),
    ("mana", re.compile(r"\bmana\b", re.I)),
    ("damage", re.compile(r"\bdamage\b", re.I)),
    ("life_gain", re.compile(r"\b(life|lifegain)\b", re.I)),
    ("life_loss", re.compile(r"\b(loss\s+of\s+life|lose\s+life)\b", re.I)),
    ("draw", re.compile(r"\b(draw|cards?\s+(in|to)\s+hand)\b", re.I)),
    ("tokens", re.compile(r"\btokens?\b", re.I)),
    ("creatures_etb", re.compile(r"\benters?\s+the\s+battlefield\b", re.I)),
    ("creature_pump", re.compile(r"\b(power|toughness|\+1/\+1\s+counters?)\b", re.I)),
    ("untap", re.compile(r"\buntap(s|ped)?\b", re.I)),
    ("storm", re.compile(r"\bstorm\b", re.I)),
    ("mill_self", re.compile(r"\bmill(s|ed)?\b", re.I)),
    ("graveyard_loop", re.compile(r"\b(graveyard|recursion|return.*from\s+graveyard)\b", re.I)),
    ("turns", re.compile(r"\b(extra\s+turns?|additional\s+turns?)\b", re.I)),
    ("cast", re.compile(r"\b(cast|spells?)\b", re.I)),
    ("counters_proliferate", re.compile(r"\b(counters?|proliferate)\b", re.I)),
    ("search_library", re.compile(r"\b(search.*library|tutor)\b", re.I)),
]

MAGNITUDE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ``near[ -]infinite`` must be tried before plain ``infinite`` so the
    # specific pattern wins.
    ("near_infinite", re.compile(r"\bnear[\s-]?infinite\b", re.I)),
    ("arbitrary", re.compile(r"\barbitrar(y|ily)\b", re.I)),
    ("infinite", re.compile(r"\binfinite\b", re.I)),
]


@dataclass(frozen=True)
class Outcome:
    """A canonical combo outcome bucket.

    ``feature`` is the raw upstream string (preserved so we can show the
    user the precise wording).  ``category`` is the coarse class used
    for graph linking.  ``magnitude`` qualifies it.
    ``outcome_id`` is a deterministic key used as the ``:Outcome`` node
    primary key.
    """
    feature: str
    category: str
    magnitude: str

    @property
    def outcome_id(self) -> str:
        return f"{self.category}:{self.magnitude}"

    @property
    def display_name(self) -> str:
        if self.magnitude == "finite":
            return self.category.replace("_", " ").title()
        return f"{self.magnitude.replace('_', ' ').title()} {self.category.replace('_', ' ').title()}"


def classify_feature(feature: str) -> Outcome:
    """Bucket a single feature string into a structured ``Outcome``."""
    feature_str = (feature or "").strip()
    if not feature_str:
        return Outcome(feature="", category="other", magnitude="finite")

    category = "other"
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(feature_str):
            category = cat
            break

    magnitude = "finite"
    for mag, pat in MAGNITUDE_PATTERNS:
        if pat.search(feature_str):
            magnitude = mag
            break

    return Outcome(feature=feature_str, category=category, magnitude=magnitude)


def classify_features(features: Iterable[str]) -> list[Outcome]:
    """Classify a sequence of features, deduping on (category, magnitude)."""
    seen: set[tuple[str, str]] = set()
    out: list[Outcome] = []
    for feat in features:
        if not (feat or "").strip():
            continue
        oc = classify_feature(feat)
        key = (oc.category, oc.magnitude)
        if key in seen:
            continue
        seen.add(key)
        out.append(oc)
    return out


def combo_display_name(card_names: list[str], max_cards: int = 3) -> str:
    """Build a short, human-readable combo name from its component cards.

    Examples::

        ["Heliod, Sun-Crowned", "Walking Ballista"] -> "Heliod, Sun-Crowned + Walking Ballista"
        ["A","B","C","D"] -> "A + B + C + 1 more"
    """
    cleaned = [n.strip() for n in card_names if n and n.strip()]
    if not cleaned:
        return "(unnamed combo)"
    if len(cleaned) <= max_cards:
        return " + ".join(cleaned)
    head = " + ".join(cleaned[:max_cards])
    return f"{head} + {len(cleaned) - max_cards} more"

"""Bridge between our Python ``Agent`` zoo and phase-server's ``GameAction``.

phase-server sends ``legal_actions`` as a list of opaque JSON tagged-union
dicts (e.g. ``{"type": "PassPriority"}``, ``{"type": "PlayLand", "data": {...}}``).
For the first iteration we treat these as opaque to the agent: the picker
sees the list and returns the index of its choice. The chosen dict is echoed
back to the server verbatim — no shape translation needed for the round-trip.

Translation of phase-rs ``GameAction`` <-> our ``src/engine`` ``Action`` is
intentionally **out of scope** for this module. That's a separate, large
effort tracked in IMPLEMENTATION_PLAN.md (sub-task: state translator).

For now agents that need a rich Python ``GameState`` cannot decide against
phase-rs — only the ``RandomActionPicker`` and ``PreferNonPassPicker``
defaults work. ``HeuristicAgent`` / ``WorldModelAgent`` integration comes
once we have an action-type taxonomy mapped.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ActionPicker(Protocol):
    """Minimal contract a phase-rs-driving policy must satisfy."""

    name: str

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
        seat: int,
    ) -> int:
        """Return the index into ``legal_actions`` of the chosen action."""
        ...


class RandomActionPicker:
    """Uniform random choice. Deterministic when seeded."""

    name = "phase_rs_random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],  # noqa: ARG002 — unused; kept for interface symmetry
        seat: int,  # noqa: ARG002
    ) -> int:
        if not legal_actions:
            raise ValueError("RandomActionPicker received empty legal_actions")
        return self._rng.randrange(len(legal_actions))


class PreferNonPassPicker:
    """Pick the first non-``PassPriority`` action; fall back to passing.

    Cheap heuristic to avoid the trivial "pass forever" trace that uniform-
    random produces when most legal actions are ``PassPriority``. Useful for
    smoke testing that the server actually advances turn structure.
    """

    name = "phase_rs_prefer_nonpass"

    PASS_TYPES = frozenset({"PassPriority", "Pass"})

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],  # noqa: ARG002
        seat: int,  # noqa: ARG002
    ) -> int:
        if not legal_actions:
            raise ValueError("PreferNonPassPicker received empty legal_actions")
        non_pass = [
            i for i, a in enumerate(legal_actions) if a.get("type") not in self.PASS_TYPES
        ]
        if non_pass:
            return self._rng.choice(non_pass)
        return 0  # only PassPriority remains


class HeuristicActionPicker:
    """Lightweight heuristic that works on opaque phase-rs ``GameAction`` dicts.

    Priority order (highest first):

    1. ``PlayLand`` — always cast a land when allowed (one per turn limit
       is enforced server-side; if it's in ``legal_actions`` it's legal).
    2. ``CastSpell`` / ``PlaySpell`` — develop the board / curve out.
    3. ``ActivateAbility`` — only if not a tap-for-mana-style ability
       (those don't appear in ``legal_actions`` at priority anyway —
       mana abilities are auto-played by phase-rs when paying).
    4. ``DeclareAttackers`` — attack with whatever's offered; combat
       legality already filtered by the server.
    5. ``Concede`` — never picked.
    6. ``PassPriority`` / ``Pass`` — fallback.

    Within a tier, ties are broken by ``self._rng.choice`` so the picker is
    deterministic per seed but not robotic across replays.

    This is a stop-gap until the full ``Action ↔ GameAction`` translator
    lands and we can drive phase-rs with the real ``HeuristicAgent``.
    """

    name = "phase_rs_heuristic"

    PASS_TYPES = frozenset({"PassPriority", "Pass"})
    NEVER_PICK = frozenset({"Concede"})
    # Tiered preferences. Lower index = higher priority.
    PRIORITY: tuple[frozenset[str], ...] = (
        frozenset({"PlayLand"}),
        frozenset({"CastSpell", "PlaySpell", "Cast"}),
        frozenset({"ActivateAbility", "Activate"}),
        frozenset({"DeclareAttackers", "Attack"}),
    )

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],  # noqa: ARG002
        seat: int,  # noqa: ARG002
    ) -> int:
        if not legal_actions:
            raise ValueError("HeuristicActionPicker received empty legal_actions")
        types = [a.get("type", "") for a in legal_actions]
        for tier in self.PRIORITY:
            tier_idx = [i for i, t in enumerate(types) if t in tier]
            if tier_idx:
                return self._rng.choice(tier_idx)
        # Nothing of substance; avoid Concede, fall back to Pass, else any.
        non_concede = [i for i, t in enumerate(types) if t not in self.NEVER_PICK]
        pass_idx = [i for i in non_concede if types[i] in self.PASS_TYPES]
        if pass_idx:
            return pass_idx[0]
        if non_concede:
            return self._rng.choice(non_concede)
        return 0

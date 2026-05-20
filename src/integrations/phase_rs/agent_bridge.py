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

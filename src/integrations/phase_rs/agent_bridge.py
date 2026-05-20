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

import json
import logging
import random
import urllib.error
import urllib.request
from typing import Any, Protocol

from src.integrations.phase_rs.adapter import (
    engine_action_to_phase_action,
    legal_actions_to_engine_actions,
    phase_state_to_game_state,
    seat_player_id,
)

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


class OllamaActionPicker:
    """LLM-backed picker for phase-rs legal-action lists.

    This picker sends a compact state/action summary to a local Ollama model
    and expects a single integer index in response. If Ollama is unavailable
    or returns malformed output, it falls back to a deterministic random pick.
    """

    name = "phase_rs_ollama"

    def __init__(
        self,
        seed: int | None = None,
        model: str = "gemma4:e2b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._fallback = RandomActionPicker(seed=seed)
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._available = self._check_ollama_available()

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
        seat: int,
    ) -> int:
        if not legal_actions:
            raise ValueError("OllamaActionPicker received empty legal_actions")
        if not self._available:
            return self._fallback.pick(legal_actions, state, seat)

        try:
            prompt = self._build_prompt(legal_actions, state, seat)
            response = self._ollama_generate(prompt)
            idx = self._parse_index(response, len(legal_actions))
            if idx is not None:
                # Prevent accidental LLM concessions.
                if legal_actions[idx].get("type") == "Concede":
                    non_concede = [
                        i
                        for i, action in enumerate(legal_actions)
                        if action.get("type") != "Concede"
                    ]
                    if non_concede:
                        return self._rng.choice(non_concede)
                return idx
        except Exception as exc:  # pragma: no cover - network/runtime path
            logger.debug("OllamaActionPicker failed, falling back: %s", exc)

        return self._fallback.pick(legal_actions, state, seat)

    def _check_ollama_available(self) -> bool:
        req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:  # nosec B310
                payload = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in payload.get("models", [])]
                available = any(self._model in m for m in models)
                if not available:
                    logger.warning(
                        "Ollama reachable but model %r not loaded; available=%s",
                        self._model,
                        models,
                    )
                return available
        except Exception as exc:  # pragma: no cover - runtime path
            logger.warning("Ollama unavailable for phase-rs picker: %s", exc)
            return False

    def _ollama_generate(self, prompt: str) -> str:
        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
            "keep_alive": "30m",
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # nosec B310
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("response", "")).strip()

    @staticmethod
    def _parse_index(text: str, n: int) -> int | None:
        digits = ""
        for ch in text:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        if not digits:
            return None
        idx = int(digits)
        if 0 <= idx < n:
            return idx
        return None

    def _build_prompt(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
        seat: int,
    ) -> str:
        players = state.get("players") or []
        me = next((p for p in players if int(p.get("id", -1)) == seat), {})
        opp_life = [
            int(p.get("life", 0))
            for p in players
            if int(p.get("id", -1)) != seat and not bool(p.get("is_eliminated", False))
        ]
        action_lines = []
        for i, action in enumerate(legal_actions):
            atype = str(action.get("type", ""))
            action_lines.append(f"[{i}] {atype}")

        return "\n".join(
            [
                "You are an expert Magic: The Gathering player.",
                "Choose the best action index. Respond with ONLY the integer index.",
                "",
                f"turn={int(state.get('turn_number', 0))}",
                f"phase={state.get('phase', 'Unknown')}",
                f"active_player={int(state.get('active_player', -1))}",
                f"priority_player={int(state.get('priority_player', -1))}",
                f"my_life={int(me.get('life', 0))}",
                f"my_hand={len(me.get('hand', []) or [])}",
                f"my_battlefield={len(state.get('battlefield', []) or [])}",
                f"opponent_life={opp_life}",
                "",
                "legal_actions:",
                *action_lines,
            ]
        )


class AgentActionPicker:
    """Run one of our async ``MTGAgent`` implementations on phase-rs.

    The picker translates wire actions/snapshots into our ``Action`` and
    ``GameState`` objects, calls ``agent.decide_action(...)``, then maps the
    chosen action back to the original phase-rs wire dict for exact round-trip.
    """

    def __init__(self, agent: Any, name: str | None = None) -> None:
        self.agent = agent
        self.name = name or f"phase_rs_agent:{getattr(agent, 'name', type(agent).__name__)}"

    async def pick_async(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
        seat: int,
    ) -> int:
        if not legal_actions:
            raise ValueError("AgentActionPicker received empty legal_actions")

        # Keep player IDs coherent with our agent contract (player_id strings).
        desired_player_id = seat_player_id(seat)
        if getattr(self.agent, "player_id", None) != desired_player_id:
            try:
                self.agent.player_id = desired_player_id
            except Exception:
                pass

        game_state = phase_state_to_game_state(state)
        candidate_actions = legal_actions_to_engine_actions(legal_actions, seat)
        chosen = await self.agent.decide_action(game_state, candidate_actions)
        chosen_wire = engine_action_to_phase_action(chosen, legal_actions)

        for idx, raw in enumerate(legal_actions):
            if raw == chosen_wire:
                return idx

        # Should be unreachable because chosen_wire came from legal_actions,
        # but keep a deterministic fallback.
        return 0

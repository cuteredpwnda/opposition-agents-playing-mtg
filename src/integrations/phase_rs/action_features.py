"""Card-aware facts about phase-rs wire actions shared by Python pickers.

Two consumers:

* :class:`ActivationGate` -- a conservative "activation hygiene" rule used by
  the heuristic and KL-control pickers. Both previously ranked every
  ``ActivateAbility`` above passing, which in native games meant animating
  man-lands hundreds of times in upkeep, spending mana in upkeep (leaving
  lands tapped for the main phase) and cracking a self-sacrificing board wipe
  (Oblivion Stone) on an empty opposing board.
* :func:`experience_key` -- a deck-agnostic description of the chosen action
  used by :mod:`src.integrations.phase_rs.experience_graph`.

Only fields of the perspective-filtered snapshot are read; nothing here is
hidden information.
"""

from __future__ import annotations

import json
from typing import Any

ACTIVATION_POLICY = "action_hygiene_v2"

_ACTIVATION_PHASES = frozenset({"PreCombatMain", "BeginCombat", "PostCombatMain"})
_MASS_REMOVAL = frozenset({"DestroyAll", "DamageAll", "ChangeZoneAll", "SacrificeAll"})


def _object(state: dict[str, Any], object_id: Any) -> dict[str, Any] | None:
    objects = state.get("objects") or {}
    obj = objects.get(str(object_id))
    if obj is None:
        obj = objects.get(object_id)
    return obj if isinstance(obj, dict) else None


def source_object(action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    data = action.get("data") or {}
    if not isinstance(data, dict):
        return None
    for key in ("object_id", "source_id"):
        if key in data:
            return _object(state, data[key])
    return None


def activated_ability(action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    obj = source_object(action, state) or {}
    abilities = obj.get("abilities") or []
    index = (action.get("data") or {}).get("ability_index")
    if (
        isinstance(index, int) and 0 <= index < len(abilities)
        and isinstance(abilities[index], dict)
    ):
        return abilities[index]
    return {}


def _effect_type(ability: dict[str, Any]) -> str:
    effect = ability.get("effect") or {}
    return str(effect.get("type") or "") if isinstance(effect, dict) else ""


def _non_land_permanents(state: dict[str, Any], seat: int) -> tuple[int, int]:
    ours = theirs = 0
    for oid in state.get("battlefield") or []:
        obj = _object(state, oid)
        if obj is None:
            continue
        core = (obj.get("card_types") or {}).get("core_types") or []
        if "Land" in core:
            continue
        if obj.get("controller") == seat:
            ours += 1
        else:
            theirs += 1
    return ours, theirs


class ActivationGate:
    """Per-game activation hygiene.

    Applies only to ``ActivateAbility`` offered at an ordinary ``Priority``
    stop; activations offered while paying a cost (``ManaPayment`` and similar
    prompts) continue a payment and are always allowed.
    """

    def __init__(self) -> None:
        self._used: set[tuple[Any, Any, Any]] = set()

    def allows(self, action: dict[str, Any], state: dict[str, Any], seat: int) -> bool:
        if action.get("type") != "ActivateAbility":
            return True
        if (state.get("waiting_for") or {}).get("type") != "Priority":
            return True
        ability = activated_ability(action, state)
        effect = _effect_type(ability)
        if effect == "Mana":
            # Mana empties between steps and the engine prompts for payment.
            return False
        if state.get("active_player") != seat or state.get("phase") not in _ACTIVATION_PHASES:
            return False
        if self._key(action, state) in self._used:
            return False
        if effect in _MASS_REMOVAL:
            ours, theirs = _non_land_permanents(state, seat)
            return theirs > ours
        return True

    def record(self, action: dict[str, Any], state: dict[str, Any]) -> None:
        if action.get("type") == "ActivateAbility":
            self._used.add(self._key(action, state))

    @staticmethod
    def _key(action: dict[str, Any], state: dict[str, Any]) -> tuple[Any, Any, Any]:
        data = action.get("data") or {}
        return (state.get("turn_number"), data.get("source_id"), data.get("ability_index"))


def gated_indices(
    gate: ActivationGate, legal_actions: list[dict[str, Any]], state: dict[str, Any], seat: int,
) -> list[int]:
    """Indices that pass ``gate``; never empty when ``legal_actions`` is not."""
    keep = [i for i, a in enumerate(legal_actions) if gate.allows(a, state, seat)]
    return keep or list(range(len(legal_actions)))


_COLOURS = ("White", "Blue", "Black", "Red", "Green")


def mulligan_choice(
    legal_actions: list[dict[str, Any]], state: dict[str, Any], seat: int,
) -> int | None:
    """Index of Keep/Mulligan under a land-count rule, or ``None`` if not a mulligan.

    Keep a seven-card hand with two to five lands. After one mulligan, keep
    one to five lands. From the second mulligan on, always keep. This is the
    conventional land-count heuristic, not an optimised policy.
    """
    if not legal_actions or legal_actions[0].get("type") != "MulliganDecision":
        return None
    choices = {
        ((a.get("data") or {}).get("choice") or {}).get("type"): i
        for i, a in enumerate(legal_actions)
    }
    if "Keep" not in choices or "Mulligan" not in choices:
        return None
    pending = ((state.get("waiting_for") or {}).get("data") or {}).get("pending") or []
    count = next(
        (p.get("mulligan_count", 0) for p in pending if p.get("player") == seat), 0,
    )
    players = state.get("players") or []
    hand = players[seat].get("hand") if 0 <= seat < len(players) else None
    if count >= 2 or not isinstance(hand, list):
        return choices["Keep"]
    lands = sum(
        "Land" in ((_object(state, oid) or {}).get("card_types") or {}).get("core_types", [])
        for oid in hand
    )
    low = 2 if count == 0 else 1
    return choices["Keep"] if low <= lands <= 5 else choices["Mulligan"]


def _colour_of(action: dict[str, Any]) -> str | None:
    choice = ((action.get("data") or {}).get("choice")) or {}
    value = choice.get("data") if isinstance(choice, dict) else None
    return value if isinstance(value, str) else None


def useful_mana_colours(legal_actions: list[dict[str, Any]], state: dict[str, Any]) -> list[int]:
    """Restrict a ``ChooseManaColor`` prompt to colours the pending cost needs.

    Any-colour sources (Starting Town, Mana Confluence) ask for a colour while
    paying. A colour the pending spell does not need cannot pay a coloured
    shard and can leave ``CancelCast`` as the only legal action. Shards are
    matched by colour name, so hybrid and Phyrexian shards count for each of
    their colours. Returns every index when nothing better is known.
    """
    every = list(range(len(legal_actions)))
    if not legal_actions or any(a.get("type") != "ChooseManaColor" for a in legal_actions):
        return every
    cost = ((state.get("pending_cast") or {}).get("cost")) or {}
    shards = [str(s) for s in cost.get("shards") or []] if isinstance(cost, dict) else []
    needed = {colour for colour in _COLOURS for shard in shards if colour in shard}
    if not needed:
        return every
    useful = [i for i, a in enumerate(legal_actions) if _colour_of(a) in needed]
    return useful or every


# ---------------------------------------------------------------------------
# Experience keys
# ---------------------------------------------------------------------------


def _stage(state: dict[str, Any]) -> str:
    turn = state.get("turn_number")
    if not isinstance(turn, int):
        return "any"
    if turn <= 4:
        return "early"
    if turn <= 10:
        return "mid"
    return "late"


def _card_class(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "unknown"
    core = (obj.get("card_types") or {}).get("core_types") or []
    primary = next(
        (t for t in ("Land", "Creature", "Planeswalker", "Instant", "Sorcery",
                     "Artifact", "Enchantment", "Battle") if t in core),
        "Other",
    )
    cost = obj.get("mana_cost") or {}
    mv = 0
    if isinstance(cost, dict):
        mv = int(cost.get("generic") or 0) + len(cost.get("shards") or [])
    return f"{primary.lower()}_mv{min(mv, 6)}"


def experience_key(
    action: dict[str, Any], state: dict[str, Any],
) -> tuple[str, str, str, str] | None:
    """Return ``(kind, card_class, card_name, stage)`` or ``None``.

    ``card_class`` is the back-off node shared across decks; ``card_name`` is
    the specific node. Actions with no strategic content (payment steps,
    targets) return ``None`` and are neither credited nor tilted.
    """
    kind = str(action.get("type") or "")
    stage = _stage(state)
    if kind in {"PlayLand", "CastSpell"}:
        obj = source_object(action, state)
        name = str((obj or {}).get("name") or "unknown")
        return (kind, _card_class(obj), name, stage)
    if kind == "ActivateAbility":
        obj = source_object(action, state)
        effect = _effect_type(activated_ability(action, state)) or "unknown"
        index = (action.get("data") or {}).get("ability_index")
        name = f"{(obj or {}).get('name') or 'unknown'}#{index}"
        return (kind, f"effect_{effect}", name, stage)
    if kind == "DeclareAttackers":
        attacks = ((action.get("data") or {}).get("attacks")) or []
        size = len(attacks) if isinstance(attacks, list) else 0
        label = f"attackers_{min(size, 3)}"
        return (kind, label, label, stage)
    if kind == "PassPriority" and (state.get("waiting_for") or {}).get("type") == "Priority":
        phase = str(state.get("phase") or "unknown")
        return (kind, f"pass_{phase}", f"pass_{phase}", stage)
    return None


def describe(key: tuple[str, ...] | None) -> str:
    return json.dumps(list(key)) if key else ""

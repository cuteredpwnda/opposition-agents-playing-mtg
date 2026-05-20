"""Bridge helpers between phase-rs wire actions and our Python agent API.

This module provides a pragmatic translation layer:

- phase-rs `GameAction` dict -> `src.engine.game_state.Action`
- minimal phase-rs snapshot -> `src.engine.game_state.GameState`
- chosen `Action` -> original phase-rs `GameAction` dict

The conversion intentionally preserves the original wire action in
`Action.metadata["phase_rs_action"]` so round-trip submission remains exact.
"""

from __future__ import annotations

from typing import Any

from src.engine_legacy.game_state import (
    Action,
    ActionType,
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    StackItem,
    Zone,
)


PHASE_TO_ENGINE_ACTION: dict[str, ActionType] = {
    "PassPriority": ActionType.PASS_PRIORITY,
    "Pass": ActionType.PASS_PRIORITY,
    "PlayLand": ActionType.PLAY_LAND,
    "CastSpell": ActionType.CAST_SPELL,
    "PlaySpell": ActionType.CAST_SPELL,
    "Cast": ActionType.CAST_SPELL,
    "ActivateAbility": ActionType.ACTIVATE_ABILITY,
    "Activate": ActionType.ACTIVATE_ABILITY,
    "DeclareAttackers": ActionType.DECLARE_ATTACKERS,
    "Attack": ActionType.DECLARE_ATTACKERS,
    "DeclareBlockers": ActionType.DECLARE_BLOCKERS,
    "Block": ActionType.DECLARE_BLOCKERS,
    "Concede": ActionType.CONCEDE,
}

ENGINE_TO_PHASE_TYPES: dict[ActionType, tuple[str, ...]] = {
    ActionType.PASS_PRIORITY: ("PassPriority", "Pass"),
    ActionType.PLAY_LAND: ("PlayLand",),
    ActionType.CAST_SPELL: ("CastSpell", "PlaySpell", "Cast"),
    ActionType.ACTIVATE_ABILITY: ("ActivateAbility", "Activate"),
    ActionType.DECLARE_ATTACKERS: ("DeclareAttackers", "Attack"),
    ActionType.DECLARE_BLOCKERS: ("DeclareBlockers", "Block"),
    ActionType.CONCEDE: ("Concede",),
    ActionType.SPECIAL_ACTION: (),
}

PHASE_NAME_MAP: dict[str, Phase] = {
    "Untap": Phase.UNTAP,
    "Upkeep": Phase.UPKEEP,
    "Draw": Phase.DRAW,
    "PreCombatMain": Phase.MAIN_1,
    "Main1": Phase.MAIN_1,
    "BeginningOfCombat": Phase.COMBAT_BEGIN,
    "DeclareAttackers": Phase.COMBAT_ATTACKERS,
    "DeclareBlockers": Phase.COMBAT_BLOCKERS,
    "CombatDamage": Phase.COMBAT_DAMAGE,
    "EndOfCombat": Phase.COMBAT_END,
    "PostCombatMain": Phase.MAIN_2,
    "Main2": Phase.MAIN_2,
    "EndStep": Phase.END_STEP,
    "Cleanup": Phase.CLEANUP,
}

ZONE_MAP: dict[str, Zone] = {
    "library": Zone.LIBRARY,
    "hand": Zone.HAND,
    "battlefield": Zone.BATTLEFIELD,
    "graveyard": Zone.GRAVEYARD,
    "exile": Zone.EXILE,
    "stack": Zone.STACK,
    "commandzone": Zone.COMMAND_ZONE,
    "command_zone": Zone.COMMAND_ZONE,
}


def seat_player_id(seat: int) -> str:
    return f"seat{seat}"


def _extract_card_id(data: dict[str, Any]) -> str | None:
    for key in (
        "card_id",
        "cardId",
        "object_id",
        "objectId",
        "source_id",
        "sourceId",
        "spell_id",
        "spellId",
        "permanent_id",
        "permanentId",
        "attacker_id",
        "attackerId",
        "blocker_id",
        "blockerId",
    ):
        value = data.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_targets(data: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for key, value in data.items():
        key_low = str(key).lower()
        if "target" in key_low or "attacker" in key_low or "blocker" in key_low:
            if isinstance(value, list):
                targets.extend(str(v) for v in value)
            elif value is not None:
                targets.append(str(value))
    return targets


def legal_actions_to_engine_actions(
    legal_actions: list[dict[str, Any]],
    seat: int,
) -> list[Action]:
    player_id = seat_player_id(seat)
    converted: list[Action] = []
    for idx, raw in enumerate(legal_actions):
        raw_type = str(raw.get("type", ""))
        raw_data = raw.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        action = Action(
            action_type=PHASE_TO_ENGINE_ACTION.get(raw_type, ActionType.SPECIAL_ACTION),
            player_id=player_id,
            card_instance_id=_extract_card_id(data),
            targets=_extract_targets(data),
            metadata={
                "phase_rs_index": idx,
                "phase_rs_type": raw_type,
                "phase_rs_action": raw,
                "phase_rs_data": data,
            },
        )
        converted.append(action)
    return converted


def engine_action_to_phase_action(
    chosen: Action,
    legal_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    idx = chosen.metadata.get("phase_rs_index") if isinstance(chosen.metadata, dict) else None
    if isinstance(idx, int) and 0 <= idx < len(legal_actions):
        return legal_actions[idx]

    preferred_types = ENGINE_TO_PHASE_TYPES.get(chosen.action_type, ())
    for i, raw in enumerate(legal_actions):
        if raw.get("type") in preferred_types:
            if chosen.card_instance_id is None:
                return legal_actions[i]
            raw_data = raw.get("data")
            data = raw_data if isinstance(raw_data, dict) else {}
            if _extract_card_id(data) == str(chosen.card_instance_id):
                return legal_actions[i]

    # Safe fallback: pass if available, else first legal action.
    for raw in legal_actions:
        if raw.get("type") in ("PassPriority", "Pass"):
            return raw
    if not legal_actions:
        raise ValueError("No legal_actions available for round-trip conversion")
    return legal_actions[0]


def _to_zone(value: Any) -> Zone:
    text = str(value or "").strip().lower().replace(" ", "").replace("-", "_")
    return ZONE_MAP.get(text, Zone.LIBRARY)


def _to_phase(value: Any) -> Phase:
    return PHASE_NAME_MAP.get(str(value or ""), Phase.MAIN_1)


def phase_state_to_game_state(state: dict[str, Any]) -> GameState:
    players_raw = state.get("players") or []
    players: list[PlayerState] = []
    for p in players_raw:
        seat = int(p.get("id", len(players)))
        players.append(
            PlayerState(
                player_id=seat_player_id(seat),
                name=str(p.get("name") or f"Seat {seat}"),
                life_total=int(p.get("life", 20)),
            )
        )

    cards: list[CardInstance] = []
    objects = state.get("objects") or {}
    for raw_oid, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        oid = str(raw_oid)
        owner = obj.get("owner")
        controller = obj.get("controller")
        card_data = {
            "name": obj.get("name") or obj.get("card_name") or f"Object {oid}",
            "oracle_text": obj.get("oracle_text") or obj.get("rules_text") or "",
            "type_line": obj.get("type_line") or "",
            "power": str(obj.get("power")) if obj.get("power") is not None else None,
            "toughness": str(obj.get("toughness")) if obj.get("toughness") is not None else None,
            "cmc": float(obj.get("cmc", 0.0) or 0.0),
        }
        cards.append(
            CardInstance(
                instance_id=oid,
                card_data=card_data,
                zone=_to_zone(obj.get("zone")),
                owner_id=seat_player_id(int(owner)) if owner is not None else "",
                controller_id=seat_player_id(int(controller)) if controller is not None else "",
                tapped=bool(obj.get("tapped", False)),
                damage_marked=int(obj.get("damage", 0) or 0),
            )
        )

    stack_items: list[StackItem] = []
    for raw in state.get("stack") or []:
        oid = str(raw)
        obj = objects.get(oid) or objects.get(raw) or {}
        controller = obj.get("controller")
        stack_items.append(
            StackItem(
                source_card_id=oid,
                controller_id=seat_player_id(int(controller)) if controller is not None else "",
                is_spell=True,
                card_data={"name": obj.get("name") or obj.get("card_name") or oid},
            )
        )

    return GameState(
        format="phase_rs",
        turn_number=int(state.get("turn_number", 0)),
        active_player_index=int(state.get("active_player", 0)),
        priority_player_index=int(state.get("priority_player", 0)),
        phase=_to_phase(state.get("phase")),
        players=players,
        cards=cards,
        stack=stack_items,
    )

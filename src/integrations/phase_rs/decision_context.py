"""Card-aware decision context without engine journals or compiled rule ASTs."""

from __future__ import annotations

import copy
import json
from typing import Any

CONTEXT_PROJECTION = "compact_card_aware_v3"

STATE_FIELDS = (
    "turn_number", "active_player", "phase", "priority_player", "waiting_for",
    "battlefield", "stack", "exile", "command_zone", "combat",
    "has_pending_cast", "allows_cancel_cast", "lands_played_this_turn",
    "max_lands_per_turn", "day_night", "commander_damage", "commander_cast_count",
    "eliminated_players", "spells_cast_this_turn", "seat_order",
)
PLAYER_FIELDS = (
    "id", "life", "mana_pool", "hand", "graveyard", "lands_played_this_turn",
    "poison_counters", "energy", "is_eliminated", "status",
    "life_gained_this_turn", "life_lost_this_turn", "cards_drawn_this_turn",
    "can_look_at_top_of_library", "commander_color_identity",
)
OBJECT_FIELDS = (
    "name", "owner", "controller", "zone", "tapped", "face_down",
    "flipped", "transformed", "damage_marked", "dealt_deathtouch_damage",
    "attached_to", "attachments", "counters", "power", "toughness", "loyalty",
    "card_types", "mana_cost", "keywords", "color", "entered_battlefield_turn",
    "oracle_text", "rules_text",
)


def decision_context(state: dict[str, Any], seat: int) -> dict[str, Any]:
    """Project an already perspective-filtered native snapshot, without truncation.

    Hidden card placeholders and library order are omitted. Visible identities,
    object IDs, characteristics and zone links survive; compiled rules and
    historical/internal journals do not. Identical card rows share a comma-
    separated ID key and a column schema, retaining every instance identity;
    omitted flags/counts/collections have their native false/zero/empty defaults,
    while missing characteristics are unknown. This is not a complete rules state.
    """
    result = {key: state[key] for key in STATE_FIELDS if key in state}
    if "players" in state:
        players = state["players"]
        if not isinstance(players, list) or any(not isinstance(p, dict) for p in players):
            raise ValueError("Native snapshot players must be a list of objects")
        summaries = []
        for player in players:
            summary = {
                key: player[key] for key in PLAYER_FIELDS if key in player
                and player[key] is not False and player[key] is not None
                and player[key] != [] and player[key] != {}
            }
            for key in (
                "lands_played_this_turn", "poison_counters", "energy",
                "life_gained_this_turn", "life_lost_this_turn", "cards_drawn_this_turn",
            ):
                if summary.get(key) == 0:
                    summary.pop(key)
            for zone in ("hand", "library"):
                cards = player.get(zone, [])
                if not isinstance(cards, list):
                    raise ValueError(f"Native player {zone} must be a list")
                summary[f"{zone}_size"] = len(cards)
            if player.get("id") != seat:
                summary.pop("hand", None)
            summaries.append(summary)
        result["players"] = summaries
    if "objects" in state:
        objects = state["objects"]
        if not isinstance(objects, dict):
            raise ValueError("Native snapshot objects must be an ID-keyed object")
        visible = {}
        for object_id, obj in objects.items():
            if not isinstance(object_id, str) or "," in object_id:
                raise ValueError("Native card object IDs must be comma-free string keys")
            if not isinstance(obj, dict):
                raise ValueError("Native card object must be an object")
            if obj.get("name") == "Hidden Card":
                continue
            card = {
                key: obj[key] for key in OBJECT_FIELDS if key in obj
                and obj[key] is not False and obj[key] is not None
                and obj[key] != [] and obj[key] != {}
            }
            if card.get("damage_marked") == 0:
                card.pop("damage_marked")
            for key in ("card_types", "mana_cost"):
                if isinstance(card.get(key), dict):
                    card[key] = {k: v for k, v in card[key].items() if v != []}
            visible[object_id] = card
        columns = [key for key in OBJECT_FIELDS if any(key in card for card in visible.values())]
        result["object_fields"] = columns
        groups: dict[str, tuple[list[str], list[Any]]] = {}
        for object_id, card in visible.items():
            row = [card.get(key) for key in columns]
            signature = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if signature in groups:
                groups[signature][0].append(object_id)
            else:
                groups[signature] = ([object_id], row)
        result["objects"] = {",".join(ids): row for ids, row in groups.values()}
    return copy.deepcopy(result)

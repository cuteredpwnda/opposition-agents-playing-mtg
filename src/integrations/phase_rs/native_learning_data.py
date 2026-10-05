"""Real, perspective-safe native decisions for offline learning."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import torch

from src.integrations.phase_rs.adapter import (
    PHASE_TO_ENGINE_ACTION,
    phase_state_to_game_state,
    seat_player_id,
)
from src.world_model.game_tokenizer import GameTokenizer

FEATURE_VERSION = "native_observation_action_v3"
ACTION_DIM = 136


def text_vector(text: str, dim: int = 128) -> np.ndarray:
    """Stable signed token hashing, not a pretrained semantic embedding."""
    result = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"\w+|[^\w\s]", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        result[int.from_bytes(digest[:4], "little") % dim] += (
            1.0 if digest[4] & 1 else -1.0
        )
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def visible_objects(state: dict, seat: int) -> dict[str, dict]:
    result = {}
    for oid, obj in (state.get("objects") or {}).items():
        if not isinstance(obj, dict):
            raise ValueError("Native learning objects must be dictionaries")
        if not obj.get("name") or obj["name"] == "Hidden Card":
            continue
        zone = str(obj.get("zone", "")).lower()
        if zone == "library" or (
            zone == "hand" and obj.get("owner") != seat
        ) or obj.get("face_down"):
            continue
        result[str(oid)] = obj
    return result


def public_zone_count(player: dict, zone: str) -> int:
    count = player.get(f"{zone}_size")
    if count is None:
        cards = player.get(zone)
        if not isinstance(cards, list):
            raise ValueError(f"Native player lacks public {zone} count")
        count = len(cards)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"Invalid native public {zone} count: {count!r}")
    return count


def observation(state: dict, seat: int) -> tuple[dict[str, torch.Tensor], list[str]]:
    objects = visible_objects(state, seat)
    safe = {**state, "objects": objects}
    game = phase_state_to_game_state(safe)
    embeddings = {
        obj["name"]: text_vector(
            obj["name"] + " " + str(obj.get("oracle_text") or obj.get("rules_text") or "")
        ) for obj in objects.values()
    }
    tokenizer = GameTokenizer(card_embeddings=embeddings)
    features = tokenizer.encode_state(game, seat_player_id(seat))
    public_counts = {
        int(player.get("id", i)): np.array([
            public_zone_count(player, "hand") / 10,
            public_zone_count(player, "library") / 60,
        ], dtype=np.float32)
        for i, player in enumerate(state.get("players") or [])
    }
    if seat not in public_counts:
        raise ValueError("Native observation lacks the controlled player's public counts")
    features["player_features"][7:9] = public_counts[seat]
    opponents = [counts for player, counts in public_counts.items() if player != seat]
    if opponents:
        features["opponent_features"][7:9] = np.stack(opponents).mean(axis=0)
    waiting = state.get("waiting_for")
    if isinstance(waiting, dict):
        waiting = {key: waiting[key] for key in ("type", "kind", "seat") if key in waiting}
    elif not isinstance(waiting, (str, int, type(None))):
        raise ValueError("Unsupported native waiting-for representation")
    features["decision_context"] = text_vector(json.dumps({
        "waiting_for": waiting,
        "has_pending_cast": bool(state.get("has_pending_cast")),
    }, sort_keys=True))
    return {key: torch.from_numpy(value).float() for key, value in features.items()}, sorted(
        {obj["name"] for obj in objects.values()}
    )


def encode_action(action: dict, state: dict, seat: int) -> torch.Tensor:
    objects = visible_objects(state, seat)

    def semantic(value: Any, field: str = "") -> Any:
        if isinstance(value, dict):
            return {key: semantic(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [semantic(item, field) for item in value]
        is_id = isinstance(value, str) or (
            isinstance(value, int) and not isinstance(value, bool)
            and field in {"source", "card", "card_id", "object_id", "target", "targets",
                          "attacker", "attackers", "blocker", "blockers"}
        )
        if is_id and str(value) in objects:
            obj = objects[str(value)]
            return {"card": obj["name"], "zone": obj.get("zone"),
                    "controller": obj.get("controller")}
        return value

    from src.engine_legacy.game_state import ActionType

    onehot = np.zeros(8, dtype=np.float32)
    action_type = PHASE_TO_ENGINE_ACTION.get(action.get("type"), ActionType.SPECIAL_ACTION)
    onehot[list(ActionType).index(action_type)] = 1
    vector = text_vector(json.dumps(semantic(action), sort_keys=True))
    return torch.from_numpy(np.concatenate([onehot, vector]))


class NativeRecordingPicker:
    """Record exactly the observations and actions seen by one native seat."""

    def __init__(self, picker: Any):
        self.picker = picker
        self.name = f"recording:{picker.name}"
        self.records: list[dict] = []

    @property
    def last_reasoning(self):
        return getattr(self.picker, "last_reasoning", None)

    def pick(self, actions: list[dict], state: dict, seat: int) -> int:
        eligible = playable_indices(actions)
        local = self.picker.pick([actions[i] for i in eligible], state, seat)
        if not 0 <= local < len(eligible):
            raise ValueError("Native recorder received an invalid action index")
        chosen = eligible[local]
        features, names = observation(state, seat)
        self.records.append({
            "features": features, "visible_cards": names,
            "actions": torch.stack([encode_action(action, state, seat) for action in actions]),
            "legal_actions": actions, "chosen_index": chosen,
            "turn": state.get("turn_number", 0), "done": False, "reward": 0.0,
        })
        return chosen

    def finish(self, result) -> dict:
        terminal = any(event.get("event") == "game_over" for event in result.trace)
        if terminal:
            if not result.final_state or not self.records:
                raise ValueError("Terminal native game lacks observations")
            features, names = observation(result.final_state, result.our_seat)
            reward = (
                0.0 if result.winner_seat is None
                else 1.0 if result.winner_seat == result.our_seat else -1.0
            )
            self.records.append({
                "features": features, "visible_cards": names,
                "actions": torch.zeros(1, ACTION_DIM), "legal_actions": [],
                "chosen_index": 0, "done": True, "reward": reward,
                "turn": result.turns_observed,
            })
        return {
            "feature_version": FEATURE_VERSION, "records": self.records,
            "completed": terminal, "winner_seat": result.winner_seat,
            "our_seat": result.our_seat, "reason": result.reason,
            "turns": result.turns_observed,
        }


def playable_indices(actions: list[dict]) -> list[int]:
    if not actions:
        raise ValueError("Native learning picker received no legal actions")
    return [i for i, action in enumerate(actions) if action.get("type") != "Concede"] or list(
        range(len(actions))
    )

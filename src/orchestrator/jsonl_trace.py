"""JSONL action trace.

A lightweight collector that conforms to the ``SelfPlayCollector``
interface (``on_state``, ``on_action``, ``finish_game``) but writes a
plain-text JSONL file alongside the human-readable game log. One JSON
object per chosen action, with a compact view of the state at the
moment the agent had priority.

This is intended as an ML-friendly trace for offline analysis. It does
**not** replace the dense tensor encodings produced by
``SelfPlayCollector`` — those still feed the world model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..engine.game_state import Action, GameState, Zone


def _summarize_state(state: GameState, player_id: str) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the public game state from
    ``player_id``'s perspective."""
    pl = next((p for p in state.players if p.player_id == player_id), None)
    summary: dict[str, Any] = {
        "turn": state.turn_number,
        "phase": state.phase.value if hasattr(state.phase, "value") else str(state.phase),
        "active_player": state.players[state.active_player_index].player_id
            if state.players else None,
        "priority_player": player_id,
        "stack_depth": len(state.stack),
        "players": [
            {
                "id": p.player_id,
                "life": p.life_total,
                "hand": len([c for c in state.cards
                             if c.zone == Zone.HAND and c.owner_id == p.player_id]),
                "library": len([c for c in state.cards
                                if c.zone == Zone.LIBRARY and c.owner_id == p.player_id]),
                "graveyard": len([c for c in state.cards
                                  if c.zone == Zone.GRAVEYARD and c.owner_id == p.player_id]),
                "battlefield": [
                    {
                        "id": c.instance_id,
                        "name": c.name,
                        "tapped": c.tapped,
                        "controller": c.controller_id,
                    }
                    for c in state.cards
                    if c.zone == Zone.BATTLEFIELD and c.controller_id == p.player_id
                ],
            }
            for p in state.players
        ],
    }
    if pl is not None:
        summary["mana_pool"] = dict(pl.mana_pool)
        summary["hand_view"] = [
            c.name for c in state.cards
            if c.zone == Zone.HAND and c.owner_id == player_id
        ]
    return summary


def _summarize_action(action: Action) -> dict[str, Any]:
    return {
        "type": action.action_type.name if hasattr(action.action_type, "name")
                else str(action.action_type),
        "player": action.player_id,
        "card_instance_id": getattr(action, "card_instance_id", None),
        "targets": list(getattr(action, "targets", []) or []),
        "metadata": dict(getattr(action, "metadata", {}) or {}),
    }


class JsonlActionTrace:
    """Append-only JSONL trace of (state, action) pairs.

    One JSON object per agent decision. Designed to be passed as the
    ``self_play_collector=`` argument on :class:`GameRunner`.
    """

    def __init__(self, path: str | Path, *, game_id: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._game_id = game_id or self.path.stem
        self._tick = 0
        self._pending_state: dict[str, Any] | None = None

    # -- collector protocol ------------------------------------------------

    def on_state(self, game_state: GameState, player_id) -> None:
        # The priority loop passes ``priority_player_index`` (an int) here,
        # but other call sites use the string id. Normalise.
        if isinstance(player_id, int) and game_state.players:
            pid = game_state.players[player_id].player_id
        else:
            pid = str(player_id)
        self._pending_state = _summarize_state(game_state, pid)

    def on_action(
        self,
        action: Action,
        reward: float = 0.0,
        done: bool = False,
        reasoning: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> None:
        record = {
            "game_id": self._game_id,
            "tick": self._tick,
            "state": self._pending_state,
            "action": _summarize_action(action),
            "reward": reward,
            "done": done,
        }
        if reasoning is not None:
            record["reasoning"] = reasoning
        if agent_name is not None:
            record["agent"] = agent_name
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        self._tick += 1
        self._pending_state = None

    def finish_game(self, winner=None, num_turns: int = 0) -> None:
        self._fh.write(json.dumps({
            "game_id": self._game_id,
            "tick": self._tick,
            "event": "game_end",
            "winner": winner,
            "num_turns": num_turns,
        }) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlActionTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

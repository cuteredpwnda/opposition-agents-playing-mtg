"""
Human player adapter — prompts the user to choose an action.
"""

from __future__ import annotations

from src.agents.base_agent import MTGAgent
from src.engine_legacy.game_state import Action, GameState


class HumanAgent(MTGAgent):
    """Adapter that lets a human choose actions via stdin/UI."""

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        print(f"\nTurn {game_state.turn_number} — Phase: {game_state.phase.value}")
        print(f"You are player {self.player_id}")
        print("\nLegal actions:")
        for i, a in enumerate(legal_actions):
            card_name = ""
            if a.card_instance_id:
                card = next((c for c in game_state.cards if c.instance_id == a.card_instance_id), None)
                card_name = card.name if card else ""
            print(f"  [{i}] {a.action_type.value} {card_name}")
        while True:
            raw = input("Choose action index: ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(legal_actions):
                return legal_actions[int(raw)]
            print(f"Invalid. Enter 0–{len(legal_actions) - 1}.")

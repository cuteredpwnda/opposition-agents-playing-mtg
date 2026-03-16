"""
Rules engine — legal action generation and game action execution.

References:
- open-mtg (MIT): get_moves() → make_move() pattern
- mtg-python-engine (MIT): check_state_based_actions() pattern
- mtg-player (MIT): tool-based validation (get_legal_actions, execute_action)
- Forge (Java, GPL-3.0): behavioral reference for correctness
"""

from __future__ import annotations

from .game_state import Action, ActionType, GameState, Phase, Zone
from .phases import is_main_phase
from .stack import is_empty as stack_is_empty


class RulesEngine:
    """Validates and executes game actions according to MTG Comprehensive Rules."""

    def get_legal_actions(self, state: GameState, player_id: str) -> list[Action]:
        """Return all legal actions for a player given current state and priority.

        This is the core interface that agents use — adapted from open-mtg's
        ``game.get_moves()`` pattern, extended with priority and instant-speed.
        """
        actions: list[Action] = []
        player = next(p for p in state.players if p.player_id == player_id)

        # Always can pass priority
        actions.append(Action(action_type=ActionType.PASS_PRIORITY, player_id=player_id))

        # Can always concede
        actions.append(Action(action_type=ActionType.CONCEDE, player_id=player_id))

        # Sorcery-speed actions: only during own main phase with empty stack
        if (
            state.active_player.player_id == player_id
            and is_main_phase(state.phase)
            and stack_is_empty(state)
        ):
            actions.extend(self._get_sorcery_speed_actions(state, player_id))

        # Instant-speed actions: any time you have priority
        actions.extend(self._get_instant_speed_actions(state, player_id))

        return actions

    def execute_action(self, state: GameState, action: Action) -> GameState:
        """Apply an action, update state, check SBAs, queue triggers."""
        # TODO: implement per action type
        return state

    def check_state_based_actions(self, state: GameState) -> list[str]:
        """CR 704 — check and apply state-based actions.

        Returns list of events that occurred.
        """
        events: list[str] = []

        # Players at 0 or less life lose
        for player in state.players:
            if player.life_total <= 0:
                events.append(f"{player.name} loses the game (life <= 0)")
                # TODO: eliminate player

        # Creatures with lethal damage or 0 toughness die
        for card in state.cards:
            if card.zone != Zone.BATTLEFIELD or not card.is_creature():
                continue
            toughness = _parse_int(card.toughness)
            if toughness is not None and card.damage_marked >= toughness:
                events.append(f"{card.name} dies (lethal damage)")
                # TODO: move to graveyard
            if toughness is not None and toughness <= 0:
                events.append(f"{card.name} dies (0 toughness)")

        # Commander damage check (21+)
        if state.format == "commander":
            for player in state.players:
                for cmd_id, dmg in player.commander_damage_received.items():
                    if dmg >= 21:
                        events.append(
                            f"{player.name} loses (21+ commander damage from {cmd_id})"
                        )

        return events

    # -- Private helpers --

    def _get_sorcery_speed_actions(
        self, state: GameState, player_id: str
    ) -> list[Action]:
        """Actions available at sorcery speed."""
        actions: list[Action] = []
        player = next(p for p in state.players if p.player_id == player_id)
        hand = state.cards_in_zone(player_id, Zone.HAND)

        # Play a land (once per turn)
        if player.land_plays_remaining > 0:
            for card in hand:
                if card.is_land():
                    actions.append(
                        Action(
                            action_type=ActionType.PLAY_LAND,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                        )
                    )

        # Cast sorcery-speed spells
        for card in hand:
            if not card.is_land() and not card.is_instant():
                actions.append(
                    Action(
                        action_type=ActionType.CAST_SPELL,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                    )
                )

        return actions

    def _get_instant_speed_actions(
        self, state: GameState, player_id: str
    ) -> list[Action]:
        """Actions available at instant speed (whenever you have priority)."""
        actions: list[Action] = []
        hand = state.cards_in_zone(player_id, Zone.HAND)

        # Cast instants / cards with flash
        for card in hand:
            if card.is_instant():
                actions.append(
                    Action(
                        action_type=ActionType.CAST_SPELL,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                    )
                )
            elif "flash" in card.card_data.get("keywords", []):
                actions.append(
                    Action(
                        action_type=ActionType.CAST_SPELL,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                    )
                )

        # Activated abilities of permanents on battlefield
        battlefield = state.cards_in_zone(player_id, Zone.BATTLEFIELD)
        for card in battlefield:
            # TODO: parse activated abilities from oracle text
            pass

        return actions


def _parse_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None

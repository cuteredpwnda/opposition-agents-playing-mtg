"""
Reward functions for RL training.

Combines terminal (win/loss) with intermediate shaping rewards.
Reference: Section 12.2 of PLAN.md.
"""

from __future__ import annotations

from src.engine_legacy.game_state import GameState, Zone


class RewardFunction:
    """Multi-faceted reward for RL training."""

    def compute(
        self,
        prev_state: GameState,
        new_state: GameState,
        player_id: str,
    ) -> float:
        reward = 0.0

        # Terminal rewards (sparse but critical)
        if new_state.game_over:
            if new_state.winner and new_state.winner.player_id == player_id:
                return 1.0
            return -1.0

        # Get player states
        me_before = next((p for p in prev_state.players if p.player_id == player_id), None)
        me_after = next((p for p in new_state.players if p.player_id == player_id), None)
        if not me_before or not me_after:
            return 0.0

        # Life advantage (positive delta is good)
        life_delta = me_after.life_total - me_before.life_total
        reward += 0.01 * life_delta / 40.0

        # Card advantage (hand + battlefield size delta)
        reward += 0.005 * self._card_advantage_delta(
            prev_state, new_state, player_id
        )

        # Board presence (number of creatures)
        reward += 0.005 * self._board_presence_delta(
            prev_state, new_state, player_id
        )

        # Per-action cost to encourage efficiency
        reward -= 0.001

        return reward

    def _card_advantage_delta(
        self, prev: GameState, new: GameState, pid: str
    ) -> float:
        def count_cards(gs: GameState, player: str) -> int:
            hand = len([c for c in gs.cards if c.zone == Zone.HAND and c.owner_id == player])
            bf = len([c for c in gs.cards if c.zone == Zone.BATTLEFIELD and c.owner_id == player])
            return hand + bf

        my_delta = count_cards(new, pid) - count_cards(prev, pid)
        opp_delta = sum(
            count_cards(new, p.player_id) - count_cards(prev, p.player_id)
            for p in new.players
            if p.player_id != pid
        )
        return my_delta - opp_delta

    def _board_presence_delta(
        self, prev: GameState, new: GameState, pid: str
    ) -> float:
        def bf_count(gs: GameState, player: str) -> int:
            creatures = [c for c in gs.cards 
                        if c.zone == Zone.BATTLEFIELD 
                        and c.owner_id == player 
                        and c.is_creature()]
            # Count power as proxy for board presence
            return sum(
                int(c.power or 0) if isinstance(c.power, (int, str)) else 0
                for c in creatures
            )

        return bf_count(new, pid) - bf_count(prev, pid)

"""
Reward functions for RL training.

Combines terminal (win/loss) with intermediate shaping rewards.
Reference: Section 12.2 of PLAN.md.
"""

from __future__ import annotations

from src.engine.game_state import GameState


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
            if new_state.winner == player_id:
                return 1.0
            return -1.0

        me_before = prev_state.players.get(player_id)
        me_after = new_state.players.get(player_id)
        if not me_before or not me_after:
            return 0.0

        # Life advantage
        reward += 0.01 * (me_after.life - me_before.life) / 40.0

        # Card advantage (hand + battlefield size delta)
        reward += 0.005 * self._card_advantage_delta(
            prev_state, new_state, player_id
        )

        # Board presence
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
            h = len(gs.cards_in_zone.get((player, "hand"), []))
            b = len(gs.cards_in_zone.get((player, "battlefield"), []))
            return h + b

        my_delta = count_cards(new, pid) - count_cards(prev, pid)
        opp_delta = sum(
            count_cards(new, p) - count_cards(prev, p)
            for p in new.players
            if p != pid
        )
        return my_delta - opp_delta

    def _board_presence_delta(
        self, prev: GameState, new: GameState, pid: str
    ) -> float:
        def bf_count(gs: GameState, player: str) -> int:
            return len(gs.cards_in_zone.get((player, "battlefield"), []))

        return bf_count(new, pid) - bf_count(prev, pid)

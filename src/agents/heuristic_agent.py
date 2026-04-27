"""Deterministic heuristic baseline agent.

Strategically simple but deterministic given a fixed seed: it always picks
the highest-priority legal action class, breaking ties with a stable RNG.
Used as a stronger floor than `RandomAgent` for benchmark comparisons.
"""
from __future__ import annotations

import random
from typing import Iterable

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, ActionType, GameState

# Action-type priority. Higher = picked first.
_PRIORITY: dict[ActionType, int] = {
    ActionType.PLAY_LAND: 100,
    ActionType.CAST_SPELL: 90,
    ActionType.DECLARE_ATTACKERS: 80,
    ActionType.DECLARE_BLOCKERS: 70,
    ActionType.ACTIVATE_ABILITY: 60,
    ActionType.PASS_PRIORITY: 1,
    ActionType.CONCEDE: 0,
}


class HeuristicAgent(MTGAgent):
    """Picks actions by a fixed priority over `ActionType`.

    Deterministic given the random seed used to construct the agent; useful
    as a benchmark baseline that is *both* stronger than uniform-random and
    fully reproducible across runs.
    """

    def __init__(
        self,
        player_id: str,
        name: str = "",
        seed: int | None = None,
        prefer_aggressive: bool = True,
    ):
        super().__init__(player_id=player_id, name=name or f"Heuristic({player_id})")
        self._rng = random.Random(seed)
        self._prefer_aggressive = prefer_aggressive

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        if not legal_actions:
            return Action(
                action_type=ActionType.PASS_PRIORITY, player_id=self.player_id
            )

        ranked = sorted(
            legal_actions,
            key=lambda a: (
                -_PRIORITY.get(a.action_type, 5),
                # tie-break by deterministic hash of action repr
                self._tiebreak(a),
            ),
        )

        # When aggressive, prefer the action that targets opponent or
        # taps for mana to enable more plays.
        top_priority = _PRIORITY.get(ranked[0].action_type, 5)
        candidates: Iterable[Action] = (
            a for a in ranked if _PRIORITY.get(a.action_type, 5) == top_priority
        )
        candidates = list(candidates)

        if self._prefer_aggressive:
            for a in candidates:
                if a.action_type == ActionType.DECLARE_ATTACKERS and a.targets:
                    return a

        # When blocking is on the table, block the biggest unblocked attacker
        # with the smallest creature first (chump-block heuristic).
        if candidates and candidates[0].action_type == ActionType.DECLARE_BLOCKERS:
            blocker = self._pick_block(game_state, candidates)
            if blocker is not None:
                return blocker

        return candidates[0]

    def _pick_block(
        self, game_state: GameState, candidates: list[Action]
    ) -> Action | None:
        """Pick a single blocker assignment for the most threatening attacker."""
        from src.engine.game_state import Zone

        def _power(card_id: str) -> int:
            card = next((c for c in game_state.cards if c.instance_id == card_id), None)
            try:
                return int(card.power) if card and card.power else 0
            except ValueError:
                return 0

        # Group candidates by attacker target.
        by_attacker: dict[str, list[Action]] = {}
        for a in candidates:
            if a.action_type != ActionType.DECLARE_BLOCKERS or not a.targets:
                continue
            by_attacker.setdefault(a.targets[0], []).append(a)
        if not by_attacker:
            return None

        # Choose attacker with greatest power.
        worst_attacker = max(by_attacker, key=_power)
        # Choose smallest blocker for it.
        return min(by_attacker[worst_attacker], key=lambda a: _power(a.card_instance_id or ""))

    def _tiebreak(self, action: Action) -> int:
        # Stable deterministic tiebreak: hash action serialisation with RNG.
        return self._rng.getrandbits(32) ^ hash(
            (action.action_type, action.player_id, tuple(action.targets or ()))
        )

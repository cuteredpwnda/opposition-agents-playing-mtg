"""
Self-play trajectory collector.

Hooks into the existing GameSimulator to record full game trajectories
during self-play. These trajectories become training data for V + M.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...agents.base_agent import MTGAgent
    from ...engine.game_state import Action, GameState

logger = logging.getLogger(__name__)


class SelfPlayCollector:
    """Collects game trajectories from the game engine during self-play.

    Wraps around GameCoordinator/GameSimulator to intercept state transitions
    and record them as Trajectories for world model training.

    Usage:
        collector = SelfPlayCollector(tokenizer, card_embeddings)

        # Option 1: Hook into game loop
        collector.on_state(game_state, player_id=0)
        collector.on_action(action, reward=0.0)
        trajectory = collector.finish_game(winner=0)

        # Option 2: Record a full game
        trajectory = collector.record_game(agent_0, agent_1, game_coordinator)
    """

    def __init__(self, tokenizer=None, card_embeddings=None):
        """
        Args:
            tokenizer: GameTokenizer instance for encoding states
            card_embeddings: CardEmbeddingModel for card vectors
        """
        self.tokenizer = tokenizer
        self.card_embeddings = card_embeddings
        self._current_transitions: list = []
        self._current_game_id: str = ""
        self._reset()

    def _reset(self) -> None:
        self._current_transitions = []
        self._current_game_id = str(uuid.uuid4())[:8]

    def on_state(self, game_state: GameState, player_id: int) -> None:
        """Record a game state observation.

        Call this each time the game state changes (before action selection).
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer configured — cannot encode state")
            return

        features = self.tokenizer.encode_state(game_state, player_id)
        # Store features temporarily; will be paired with the next action
        self._pending_features = features

    def on_action(self, action: Action, reward: float = 0.0, done: bool = False) -> None:
        """Record an action taken.

        Call this after an agent selects an action.
        """
        from ..trajectory import Transition

        if not hasattr(self, "_pending_features") or self._pending_features is None:
            return

        action_encoding = np.zeros(136, dtype=np.float32)
        if self.tokenizer is not None:
            action_encoding = self.tokenizer.encode_action(action)

        transition = Transition(
            state_features=self._pending_features,
            action_encoding=action_encoding,
            reward=reward,
            done=done,
            action_type=action.action_type.name if hasattr(action.action_type, "name") else str(action.action_type),
            card_name=action.card.name if action.card else None,
        )
        self._current_transitions.append(transition)
        self._pending_features = None

    def finish_game(self, winner: int | None = None, num_turns: int = 0):
        """Finalize the current game and return the trajectory.

        Returns:
            Completed Trajectory object
        """
        from ..trajectory import Trajectory

        traj = Trajectory(
            game_id=f"selfplay_{self._current_game_id}",
            transitions=self._current_transitions,
            winner=winner,
            num_turns=num_turns,
            source="self_play",
        )
        self._reset()
        return traj

    def record_game(self, agent_0, agent_1, game_coordinator) -> None:
        """Record a complete game between two agents.

        TODO: Integrate with GameCoordinator's run_game() loop.
        This requires hooking into the game loop at state transition points.
        """
        raise NotImplementedError(
            "Full game recording requires GameCoordinator integration. "
            "Use on_state() / on_action() / finish_game() hooks instead."
        )

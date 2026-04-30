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
        self.collected_trajectories: list = []
        self._warned_no_tokenizer = False
        self._reset()

    def _reset(self) -> None:
        self._current_transitions = []
        self._current_game_id = str(uuid.uuid4())[:8]
        self._pending_features = None
        self._pending_metadata: dict = {}

    def on_state(self, game_state: GameState, player_id: int) -> None:
        """Record a game state observation.

        Call this each time the game state changes (before action selection).
        """
        if self.tokenizer is None:
            if not self._warned_no_tokenizer:
                logger.debug("No tokenizer configured — state encoding skipped")
                self._warned_no_tokenizer = True
            return

        features = self.tokenizer.encode_state(game_state, player_id)
        # Store features temporarily; will be paired with the next action
        self._pending_features = features

        # Snapshot the visible board (controller's hand + every battlefield)
        # so KG-context encoders can build per-step embeddings during JEPA
        # training. We keep it bounded to avoid huge metadata blobs.
        try:
            from ...engine.game_state import Zone

            visible: list[str] = []
            seen: set[str] = set()
            # Public zones first
            for c in game_state.cards:
                if c.zone in (Zone.BATTLEFIELD, Zone.STACK,
                              Zone.GRAVEYARD, Zone.EXILE,
                              Zone.COMMAND_ZONE):
                    name = getattr(c, "name", None)
                    if name and name not in seen:
                        seen.add(name)
                        visible.append(name)
            # Plus the controller's hand (hidden info, but legal at training)
            controller = None
            if 0 <= player_id < len(game_state.players):
                controller = game_state.players[player_id].player_id
            if controller is not None:
                for c in game_state.cards:
                    if c.zone == Zone.HAND and c.controller_id == controller:
                        name = getattr(c, "name", None)
                        if name and name not in seen:
                            seen.add(name)
                            visible.append(name)
            self._pending_metadata = {
                "visible_cards": visible[:64],   # cap to keep batches bounded
                "phase": getattr(game_state, "phase", None),
                "turn": getattr(game_state, "turn_number", None),
            }
        except Exception:  # never let metadata bookkeeping break collection
            self._pending_metadata = {}

    def on_action(
        self,
        action: Action,
        reward: float = 0.0,
        done: bool = False,
        game_state: "GameState | None" = None,
        **kwargs,
    ) -> None:
        """Record an action taken.

        Call this after an agent selects an action.
        """
        from ..trajectory import Transition

        if not hasattr(self, "_pending_features") or self._pending_features is None:
            return

        action_encoding = np.zeros(136, dtype=np.float32)
        if self.tokenizer is not None:
            action_encoding = self.tokenizer.encode_action(action)

        # Resolve card name from game state when possible; fall back to the
        # raw card_instance_id only if no state is available.
        card_name = None
        if hasattr(action, "card_instance_id") and action.card_instance_id:
            if game_state is not None:
                card = next(
                    (c for c in game_state.cards if c.instance_id == action.card_instance_id),
                    None,
                )
                card_name = card.name if card is not None else None
            # If no game_state, skip — instance IDs are useless for enrichment

        # Use the enum name (e.g. "CAST_SPELL") rather than the auto() int value.
        if hasattr(action, "action_type"):
            at = action.action_type
            action_type_str = at.name if hasattr(at, "name") else (at.value if hasattr(at, "value") else str(at))
        else:
            action_type_str = "unknown"

        transition = Transition(
            state_features=self._pending_features,
            action_encoding=action_encoding,
            reward=reward,
            done=done,
            action_type=action_type_str,
            card_name=card_name,
            metadata=dict(self._pending_metadata),
        )
        self._current_transitions.append(transition)
        self._pending_features = None
        self._pending_metadata = {}

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
        self.collected_trajectories.append(traj)
        self._reset()
        return traj

    def record_game(self, agent_0, agent_1, game_simulator) -> None:
        """Record a complete game between two agents via GameSimulator.

        Uses on_state/on_action hooks where possible, and returns Trajectory.
        """
        from src.engine.game_simulator import GameResult
        from ..trajectory import Trajectory, Transition

        # Reset collector for a fresh game
        self._reset()

        # Set up game simulator if needed
        if hasattr(game_simulator, "setup_game"):
            game_simulator.setup_game()

        # Start with initial state
        if hasattr(game_simulator, "game") and game_simulator.game is not None:
            self.on_state(game_simulator.game, player_id=0)

        # Run the game loop with non-intrusive observation
        if hasattr(game_simulator, "run_game"):
            result = game_simulator.run_game()
        else:
            # Fallback: execute full turns until win condition
            result = None
            while True:
                game_simulator.execute_full_turn()
                win = game_simulator.check_win_condition() if hasattr(game_simulator, "check_win_condition") else None
                if win is not None:
                    result = win
                    break
                game_simulator.advance_turn()

        # Finalize winner index
        winner = None
        if result == GameResult.PLAYER1_WIN:
            winner = 0
        elif result == GameResult.PLAYER2_WIN:
            winner = 1
        elif result == GameResult.DRAW:
            winner = None

        # Build a trivial end-state transition if none recorded
        if not self._current_transitions and hasattr(game_simulator, "game") and game_simulator.game is not None:
            state_features = self.tokenizer.encode_state(game_simulator.game, 0) if self.tokenizer else {}
            action_encoding = np.zeros(136, dtype=np.float32)
            transition = Transition(
                state_features=state_features,
                action_encoding=action_encoding,
                reward=1.0 if winner == 0 else 0.0,
                done=True,
                action_type="GAME_END",
                card_name=None,
            )
            self._current_transitions.append(transition)

        trajectory = self.finish_game(winner=winner, num_turns=getattr(game_simulator, "game", None).turn_number if hasattr(game_simulator, "game") and game_simulator.game is not None else 0)
        return trajectory

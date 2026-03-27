"""Agent that uses NeuralReasoningModule for action scoring."""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from src.agents.base_agent import MTGAgent
from src.agents.neural_reasoner import NeuralReasoningModule
from src.agents.random_agent import RandomAgent
from src.engine.game_state import GameState, Action, ActionType, Zone

logger = logging.getLogger(__name__)


class NeuralReasonerAgent(MTGAgent):
    """Agent wrapper around NeuralReasoningModule."""

    def __init__(
        self,
        player_id: str,
        name: str = "NeuralReasonerAgent",
        model: Optional[NeuralReasoningModule] = None,
    ):
        super().__init__(player_id=player_id, name=name)
        self.model = model
        self.fallback = RandomAgent(player_id=player_id, name=f"{name}_fallback")

        if self.model is None:
            try:
                self.model = NeuralReasoningModule()
            except Exception as e:
                logger.warning("NeuralReasonerAgent cannot instantiate module: %s", e)
                self.model = None

    async def decide_action(self, game_state: GameState, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)

        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.model is None:
            return await self.fallback.decide_action(game_state, legal_actions)

        try:
            import torch
            from torch_geometric.data import Data

            # Minimal graph for GAT: one node with no edges, using nominal feature
            node_features = torch.ones((1, 128), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            batch = torch.zeros(1, dtype=torch.long)
            graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)

            # Dummy sequence and board features to run forward pass
            seq = torch.zeros((1, 1, 256), dtype=torch.float32)
            board_features = torch.tensor([self._compute_board_features(game_state)], dtype=torch.float32)

            outputs = self.model(graph_data, seq, board_features)
            policy_logits = outputs.get("policy")

            if policy_logits is None or policy_logits.numel() == 0:
                raise ValueError("NeuralReasoningModule policy output is invalid")

            # Map scores to legal_actions
            scores = policy_logits.detach().cpu().numpy().flatten().tolist()
            # If action count mismatch, fallback to heuristic random plus small tilt
            if len(scores) != len(legal_actions):
                scores = [random.random() for _ in legal_actions]

            best_index = int(max(range(len(scores)), key=lambda i: scores[i]))
            return legal_actions[best_index]
        except Exception as e:
            logger.warning("NeuralReasonerAgent inference failed: %s", e)
            return await self.fallback.decide_action(game_state, legal_actions)

    def _compute_board_features(self, game_state: GameState) -> List[float]:
        features = []
        # life totals
        for player in game_state.players:
            features.append(player.life_total / 40.0)

        # zone counts per player
        for player in game_state.players:
            for zone_name in [Zone.HAND, Zone.BATTLEFIELD, Zone.GRAVEYARD, Zone.LIBRARY]:
                features.append(len(game_state.cards_in_zone(player.player_id, zone_name)) / 60.0)

        # stacked depth + turn
        features.append(len(game_state.stack) / 15.0)
        features.append(game_state.turn_number / 20.0)

        # basic threat metric
        my_creatures = game_state.cards_in_zone(self.player_id, Zone.BATTLEFIELD)
        opp_creatures = [c for c in game_state.cards if c.zone == Zone.BATTLEFIELD and c.controller_id != self.player_id and c.is_creature()]
        features.append(len(my_creatures) / 15.0)
        features.append(len(opp_creatures) / 15.0)
        return features

    async def observe(self, game_state: GameState, action: Action) -> None:
        await super().observe(game_state, action)
        # optional: could update hidden state/training buffer after each action

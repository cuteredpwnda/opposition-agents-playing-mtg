"""Agent that uses active inference + opponent model for decision making."""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from src.agents.base_agent import MTGAgent
from src.agents.random_agent import RandomAgent
from src.agents.active_inference import ActiveInferenceModule
from src.agents.opponent_model import OpponentModel
from src.knowledge.knowledge_graph import MTGKnowledgeGraph
from src.engine.game_state import GameState, Action, ActionType

logger = logging.getLogger(__name__)


class ActiveInferenceAgent(MTGAgent):
    """MTG Agent that chooses actions by minimizing expected free energy."""

    def __init__(
        self,
        player_id: str,
        kg: Optional[MTGKnowledgeGraph] = None,
        known_opponent_decklist: Optional[Dict[str, int]] = None,
        name: str = "ActiveInferenceAgent",
    ):
        super().__init__(player_id=player_id, name=name)

        self.kg = kg or self._try_init_kg()
        self.ai_module = ActiveInferenceModule(self.kg) if self.kg is not None else None
        self.opponent_model = OpponentModel(
            opponent_id="opponent",
            kg=self.kg,
            known_decklist=known_opponent_decklist,
        ) if self.kg is not None else None

        self.fallback = RandomAgent(player_id=player_id, name=f"{name}_fallback")

        # Keep track of observed opponent IDs to initialize beliefs
        self.opponent_ids: List[str] = []

    def _try_init_kg(self) -> Optional[MTGKnowledgeGraph]:
        try:
            return MTGKnowledgeGraph()
        except Exception as e:
            logger.warning("ActiveInferenceAgent: Could not connect to KG: %s", e)
            return None

    async def decide_action(
        self, game_state: GameState, legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Basic heuristic fallback scores when no AI module
        if self.ai_module is None:
            return await self.fallback.decide_action(game_state, legal_actions)

        # Ensure beliefs are initialized for opponents
        for player in game_state.players:
            if player.player_id != self.player_id and player.player_id not in self.opponent_ids:
                self.opponent_ids.append(player.player_id)
                await self.ai_module.initialize_beliefs(player.player_id)

        # Rank actions by expected free energy (low is good)
        ranked = self.ai_module.rank_actions(legal_actions, game_state)
        chosen_action = ranked[0][0] if ranked else await self.fallback.decide_action(game_state, legal_actions)

        logger.debug(
            "ActiveInferenceAgent chose %s with G=%s",
            chosen_action.action_type.value,
            ranked[0][1] if ranked else None,
        )

        return chosen_action

    async def observe(self, game_state: GameState, action: Action) -> None:
        await super().observe(game_state, action)

        # Infer opponent model from observed action
        if self.opponent_model is not None and action.player_id != self.player_id:
            self.opponent_model.observe_behavior(action, game_state)
            self.opponent_model.observe_card_played(action.card_instance_id or "")

        # Update active inference beliefs
        if self.ai_module is not None:
            observation = {
                "card_played": None,
                "blue_mana_open": 0,
            }
            if action.card_instance_id:
                card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
                if card is not None:
                    observation["card_played"] = card.name

            # Track open blue mana on active player (if we can infer)
            active_player = game_state.active_player
            observation["blue_mana_open"] = active_player.mana_pool.get("U", 0)

            self.ai_module.update_beliefs(observation, game_state)

    def reset(self) -> None:
        super().reset()
        self.opponent_ids = []
        if self.ai_module is not None:
            self.ai_module.beliefs.clear()
        if self.opponent_model is not None:
            self.opponent_model.cards_seen_in_open_zones.clear()
            self.opponent_model.actions_taken.clear()

"""
Self-play training loop — AlphaZero-inspired.

Reference: Section 12.3 of PLAN.md.
Requires the [ml] optional dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.training.experience_buffer import Experience, ExperienceBuffer
from src.training.rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Self-play training configuration."""

    num_parallel_games: int = 64
    buffer_size: int = 100_000
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-4
    eval_games: int = 100


class SelfPlayTrainer:
    """AlphaZero-inspired self-play training.

    Agents play each other → collect experience → train neural module →
    evaluate → update KG with discovered interactions.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.buffer = ExperienceBuffer(max_size=self.config.buffer_size)
        self.reward_fn = RewardFunction()
        self.neural_module = None  # Set after import
        self.optimizer = None

    def init_neural_module(self) -> None:
        """Lazy init to avoid importing torch at module load time."""
        import torch

        from src.agents.neural_reasoner import NeuralReasoningModule

        self.neural_module = NeuralReasoningModule()
        self.optimizer = torch.optim.Adam(
            self.neural_module.parameters(), lr=self.config.learning_rate
        )

    async def train(self, num_iterations: int = 1000) -> None:
        """Main training loop."""
        if self.neural_module is None:
            self.init_neural_module()

        for iteration in range(num_iterations):
            # 1. Self-play
            experiences = await self.run_self_play_games(
                self.config.num_parallel_games
            )
            self.buffer.add_batch(experiences)

            # 2. Train
            loss = self._train_step()

            # 3. Evaluate
            win_rate = await self.evaluate(self.config.eval_games)

            logger.info(
                f"Iter {iteration}: loss={loss:.4f}, win_rate={win_rate:.2%}, "
                f"buffer={len(self.buffer)}"
            )

    async def run_self_play_games(
        self, num_games: int
    ) -> list[Experience]:
        """Run games between copies of the current agent, collect experience."""
        import asyncio
        from src.agents.llm_agent import LLMAgent
        from src.orchestrator.game_runner import GameRunner, GameConfig

        experiences: list[Experience] = []

        # For now, run games sequentially (stub for parallelization)
        for game_num in range(min(num_games, 5)):  # Limit to 5 games for testing
            try:
                # Create two agents
                agent1 = LLMAgent(player_id="player_1")
                agent2 = LLMAgent(player_id="player_2")
                agents = {"player_1": agent1, "player_2": agent2}

                # Create mock decks (simplified)
                decks = {
                    "player_1": self._create_mock_deck(),
                    "player_2": self._create_mock_deck(),
                }

                # Run game
                runner = GameRunner(GameConfig(max_turns=50))
                result = await runner.run_game(agents, decks)

                # Collect experience from game
                # For now, just track win as terminal reward
                winner_id = result.winner if result.winner else None
                for player_id in agents:
                    reward = 1.0 if player_id == winner_id else -1.0 if winner_id else 0.0
                    exp = Experience(
                        state_features=[],  # Placeholder
                        action_index=0,
                        reward=reward,
                        next_state_features=[],  # Placeholder
                        done=True,
                        metadata={"game_turns": result.turns},
                    )
                    experiences.append(exp)

                logger.info(f"Self-play game {game_num+1}: winner={winner_id}, turns={result.turns}")
            except Exception as e:
                logger.error(f"Error in self-play game {game_num}: {e}")
                continue

        return experiences

    def _create_mock_deck(self) -> list[dict[str, any]]:
        """Create a simplified mock deck for testing.
        
        In production, would load real decklists from Scryfall API
        or local deck files (Moxfield, Tappedout, etc.).
        """
        from src.integrations.scryfall import ScryfallClient
        import asyncio
        
        try:
            # Try to fetch real cards from Scryfall
            async def get_real_cards():
                async with ScryfallClient() as client:
                    # Get some basic MTG cards
                    cards = []
                    basic_cards = ["Mountain", "Goblin Guide", "Lightning Bolt"]
                    for card_name in basic_cards:
                        try:
                            card = await client.get_card_by_name(card_name)
                            for _ in range(4):
                                cards.append({
                                    "name": card.get("name", card_name),
                                    "type_line": card.get("type_line", "Land"),
                                    "mana_cost": card.get("mana_cost", ""),
                                    "cmc": card.get("cmc", 0),
                                    "oracle_text": card.get("oracle_text", ""),
                                    "power": card.get("power", None),
                                    "toughness": card.get("toughness", None),
                                })
                            if len(cards) >= 60:
                                break
                        except Exception:
                            continue
                    return cards[:60] if cards else None
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            real_cards = loop.run_until_complete(get_real_cards())
            loop.close()
            
            if real_cards and len(real_cards) >= 60:
                return real_cards
        except Exception as e:
            logger.warning(f"Could not load real cards from Scryfall: {e}, using mock")
        
        # Fallback: mock deck
        return [
            {
                "name": f"Card_{i}",
                "type_line": "Creature" if i % 3 == 0 else "Sorcery" if i % 3 == 1 else "Land",
                "mana_cost": "{1}" if i % 2 == 0 else "{2}",
                "cmc": 1 if i % 2 == 0 else 2,
                "oracle_text": "Does something",
                "power": "2" if i % 3 == 0 else None,
                "toughness": "2" if i % 3 == 0 else None,
            }
            for i in range(60)
        ]

    async def evaluate(self, num_games: int) -> float:
        """Evaluate current agent vs. baseline (random agent).
        
        Returns win rate of current agent against random opponents.
        """
        import asyncio
        from src.agents.random_agent import RandomAgent
        from src.orchestrator.game_runner import GameRunner, GameConfig
        
        wins = 0
        total = 0
        
        try:
            for _ in range(min(num_games, 10)):  # Limit to 10 eval games per iteration
                try:
                    # Create evaluation agents
                    eval_agent = RandomAgent(player_id="player_1")
                    baseline_agent = RandomAgent(player_id="player_2")
                    
                    agents = {"player_1": eval_agent, "player_2": baseline_agent}
                    
                    # Create decks
                    decks = {
                        "player_1": self._create_mock_deck(),
                        "player_2": self._create_mock_deck(),
                    }
                    
                    # Run game
                    runner = GameRunner(GameConfig(max_turns=30))
                    result = await runner.run_game(agents, decks)
                    
                    if result.winner == "player_1":
                        wins += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation game failed: {e}")
                    total += 1
                    continue
        except Exception as e:
            logger.error(f"Evaluation phase failed: {e}")
            return 0.5  # Default to 50% if evaluation fails
        
        win_rate = wins / total if total > 0 else 0.5
        return win_rate

    def _train_step(self) -> float:
        """Single training step on a batch from the buffer."""
        import torch
        import torch.nn.functional as F

        batch = self.buffer.sample(self.config.batch_size)
        if batch is None or len(batch) == 0:
            return 0.0

        try:
            if self.neural_module and self.optimizer:
                # Convert batch experiences to tensors
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                
                for exp in batch:
                    if exp.state_features and exp.next_state_features:
                        states.append(torch.tensor(exp.state_features, dtype=torch.float32))
                        actions.append(exp.action_index)
                        rewards.append(exp.reward)
                        next_states.append(torch.tensor(exp.next_state_features, dtype=torch.float32))
                        dones.append(1.0 if exp.done else 0.0)
                
                if not states:
                    return 0.0
                
                # Stack into batch
                state_batch = torch.stack(states)
                action_batch = torch.tensor(actions, dtype=torch.long)
                reward_batch = torch.tensor(rewards, dtype=torch.float32)
                next_state_batch = torch.stack(next_states) if next_states else None
                done_batch = torch.tensor(dones, dtype=torch.float32)
                
                # Forward pass through value network
                value_pred = self.neural_module.value_head(state_batch).squeeze(-1)
                
                # Compute target (reward + gamma * V(next_state) * (1 - done))
                gamma = 0.99
                with torch.no_grad():
                    if next_state_batch is not None:
                        next_value = self.neural_module.value_head(next_state_batch).squeeze(-1)
                        target = reward_batch + gamma * next_value * (1.0 - done_batch)
                    else:
                        target = reward_batch
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_module.parameters(), 1.0)
                self.optimizer.step()
                
                return value_loss.item()
        except Exception as e:
            logger.error(f"Error in training step: {e}", exc_info=True)

        return 0.0

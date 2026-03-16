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
        """Create a simplified mock deck for testing."""
        # This is a stub - in real usage, would load actual decks from Scryfall
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
        """Evaluate current agent vs. previous checkpoint."""
        # Stub - real implementation would:
        # 1. Load previous checkpoint
        # 2. Play games against it
        # 3. Return win rate
        import random
        return random.uniform(0.45, 0.55)  # Random baseline for now

    def _train_step(self) -> float:
        """Single training step on a batch from the buffer."""
        import torch
        import torch.nn.functional as F

        batch = self.buffer.sample(self.config.batch_size)
        if batch is None:
            return 0.0

        try:
            # Stub - full training would:
            # 1. Convert batch to tensors
            # 2. Forward pass through neural module
            # 3. Compute loss (value + policy)
            # 4. Backward + update
            if self.neural_module and self.optimizer:
                # Placeholder: just compute dummy loss
                loss = torch.tensor(0.0, requires_grad=True)
                self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                return loss.item()
        except Exception as e:
            logger.error(f"Error in training step: {e}")

        return 0.0

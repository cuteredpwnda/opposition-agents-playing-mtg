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
        # Stub — full implementation creates parallel game instances
        return []

    async def evaluate(self, num_games: int) -> float:
        """Evaluate current agent vs. previous checkpoint."""
        # Stub
        return 0.5

    def _train_step(self) -> float:
        """Single training step on a batch from the buffer."""
        import torch
        import torch.nn.functional as F

        batch = self.buffer.sample(self.config.batch_size)
        if batch is None:
            return 0.0

        # Stub — full training computes value/policy/win losses
        return 0.0

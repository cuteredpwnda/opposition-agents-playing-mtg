"""
Experience replay buffer for RL training.

Reference: Section 12.3 of PLAN.md.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experience:
    """A single (state, action, reward, next_state) tuple."""

    state_features: list[float]
    action_index: int
    reward: float
    next_state_features: list[float]
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    """Fixed-size replay buffer with uniform random sampling."""

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def add_batch(self, experiences: list[Experience]) -> None:
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> list[Experience] | None:
        if len(self.buffer) < batch_size:
            return None
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

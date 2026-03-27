"""Training pipeline for Schmidhuber-style world model (forward dynamics)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required for Schmidhuber training") from e

from src.world_model.schmidhuber_worldmodel_adapter import SchmidhuberWorldModelAdapter
from src.world_model.trajectory import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class SchmidhuberTrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"


def train_schmidhuber(
    store: TrajectoryStore,
    config: SchmidhuberTrainingConfig | None = None,
) -> SchmidhuberWorldModelAdapter:
    config = config or SchmidhuberTrainingConfig()

    if len(store) == 0:
        raise ValueError("No trajectories found for Schmidhuber training")

    model = SchmidhuberWorldModelAdapter()
    model.fit(
        trajectories=store,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
    )

    logger.info("Schmidhuber training finished")
    return model

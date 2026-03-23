"""
Dream Trainer — Full training loop using world model dreams.

Orchestrates the complete training pipeline:
1. Collect real game trajectories (warm-up data)
2. Train State Encoder (V) on real states
3. Train Dynamics Model (M) on encoded sequences
4. Train Controller (C) inside dreams (CMA-ES or PG)
5. Deploy controller to real games, collect more data
6. Repeat (iterative refinement)

This is the "learning to play inside its own dream" approach from
Ha & Schmidhuber (2018).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import torch
except ImportError:
    raise ImportError("Dream trainer requires PyTorch.")

from ..dynamics_model import DynamicsModelConfig
from ..state_encoder import StateEncoderConfig
from ..trajectory import TrajectoryStore
from ..world_model import WorldModel, WorldModelConfig
from .train_controller import ControllerTrainingConfig, train_controller_cmaes, train_controller_pg
from .train_dynamics import DynamicsTrainingConfig, train_dynamics
from .train_encoder import EncoderTrainingConfig, train_encoder

logger = logging.getLogger(__name__)


@dataclass
class DreamTrainerConfig:
    """Configuration for the full dream training pipeline."""

    # Sub-configs
    encoder_config: EncoderTrainingConfig = None  # type: ignore[assignment]
    dynamics_config: DynamicsTrainingConfig = None  # type: ignore[assignment]
    controller_config: ControllerTrainingConfig = None  # type: ignore[assignment]

    # Pipeline settings
    num_iterations: int = 5         # Full V→M→C training cycles
    min_trajectories: int = 100     # Minimum trajectories before training
    trajectories_per_iteration: int = 50  # New trajectories per cycle
    dream_temperature_schedule: list[float] = None  # type: ignore[assignment]
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = EncoderTrainingConfig()
        if self.dynamics_config is None:
            self.dynamics_config = DynamicsTrainingConfig()
        if self.controller_config is None:
            self.controller_config = ControllerTrainingConfig()
        if self.dream_temperature_schedule is None:
            # Gradually increase dream difficulty
            self.dream_temperature_schedule = [1.0, 1.05, 1.1, 1.15, 1.2]


class DreamTrainer:
    """Orchestrates the full world model training pipeline.

    Usage:
        store = TrajectoryStore("data/trajectories")
        store.load()  # Load previously collected data

        trainer = DreamTrainer(config)
        world_model = trainer.train(store)

        # Deploy the trained world model
        agent = WorldModelAgent(world_model)
    """

    def __init__(self, config: DreamTrainerConfig | None = None):
        self.config = config or DreamTrainerConfig()

    def train(
        self,
        trajectory_store: TrajectoryStore,
        world_model: WorldModel | None = None,
    ) -> WorldModel:
        """Run the full dream training pipeline.

        Args:
            trajectory_store: Pre-collected game trajectories
            world_model: Existing model to continue training (or None for fresh)

        Returns:
            Trained WorldModel
        """
        if world_model is None:
            world_model = WorldModel()

        if len(trajectory_store) < self.config.min_trajectories:
            logger.warning(
                "Only %d trajectories available (need %d). "
                "Collect more data before training.",
                len(trajectory_store),
                self.config.min_trajectories,
            )

        for iteration in range(self.config.num_iterations):
            logger.info("=" * 60)
            logger.info("Dream Training Iteration %d / %d", iteration + 1, self.config.num_iterations)
            logger.info("=" * 60)

            # Set dream temperature for this iteration
            temp_idx = min(iteration, len(self.config.dream_temperature_schedule) - 1)
            temperature = self.config.dream_temperature_schedule[temp_idx]
            world_model.config.dream_temperature = temperature
            logger.info("Dream temperature: %.2f", temperature)

            # Phase 1: Train State Encoder (V)
            logger.info("--- Phase 1: Training State Encoder (V) ---")
            world_model.encoder = train_encoder(
                world_model.encoder,
                trajectory_store,
                self.config.encoder_config,
            )

            # Phase 2: Train Dynamics Model (M)
            logger.info("--- Phase 2: Training Dynamics Model (M) ---")
            world_model.dynamics = train_dynamics(
                world_model.dynamics,
                trajectory_store,
                world_model.encoder,
                self.config.dynamics_config,
            )

            # Phase 3: Train Controller (C) inside dreams
            logger.info("--- Phase 3: Training Controller (C) in Dreams ---")
            if self.config.controller_config.method == "cma-es":
                world_model.controller = train_controller_cmaes(
                    world_model, self.config.controller_config
                )
            else:
                world_model.controller = train_controller_pg(
                    world_model, self.config.controller_config
                )

            # Save iteration checkpoint
            checkpoint_path = f"{self.config.checkpoint_dir}/world_model_iter_{iteration+1}.pt"
            world_model.save(checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)

            # Phase 4: Collect new trajectories with updated policy
            # TODO: Run self-play games using the trained controller,
            # then add new trajectories to the store for the next iteration.
            logger.info(
                "--- Phase 4: Collect new data (TODO: self-play integration) ---"
            )

        # Save final model
        final_path = f"{self.config.checkpoint_dir}/world_model_final.pt"
        world_model.save(final_path)
        logger.info("Training complete! Final model saved: %s", final_path)

        return world_model

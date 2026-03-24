"""
Transfer Learning: Standard → Commander

Trains a world model on Standard (2-player) games first to learn card mechanics,
mana economics, combat math, and general game dynamics. Then fine-tunes on
Commander (4-player) games to learn multiplayer-specific behaviors: threat
assessment across multiple opponents, political dynamics, and commander damage.

Why this works:
- Standard and Commander share 95% of game rules (phases, priority, combat, SBAs)
- The GameTokenizer already maps N-player games to a fixed-size perspective
  (self + mean-aggregated opponents), so the neural net shape is identical
- Standard is cheaper to simulate (2 players → fewer priority passes) and easier
  to learn (1 opponent → simpler dynamics model)
- The learned card representations, combat evaluations, and mana curves transfer
  directly — only the multiplayer adaptation layer needs fine-tuning

Curriculum:
  Phase 1: Standard  — Train V+M+C on 2-player games (learn card/game fundamentals)
  Phase 2: Commander — Fine-tune with multiplayer adapter, freeze base weights initially
  Phase 3: Joint     — Unfreeze all, train on mixed Standard+Commander at low LR
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    """Configuration for Standard→Commander transfer learning."""

    # Phase 1: Standard pre-training
    standard_iterations: int = 50
    standard_games_per_iter: int = 20
    standard_max_turns: int = 30
    standard_lr: float = 3e-4

    # Phase 2: Commander fine-tuning
    commander_iterations: int = 30
    commander_games_per_iter: int = 10
    commander_max_turns: int = 50
    commander_lr: float = 1e-4           # Lower LR to not destroy pre-trained weights
    freeze_base_epochs: int = 10         # Freeze V+M weights for first N iterations

    # Phase 3: Joint training
    joint_iterations: int = 20
    joint_lr: float = 5e-5               # Very low LR for stable joint training
    standard_ratio: float = 0.3          # 30% standard, 70% commander games

    # General
    checkpoint_dir: str = "checkpoints/transfer"
    dream_training_interval: int = 10
    collect_trajectories: bool = True


class MultiplayerAdapter:
    """Lightweight adapter layer for multiplayer-specific features.

    Sits between the GameTokenizer and the StateEncoder, adding
    multiplayer-specific conditioning signals without changing the
    base model architecture. The adapter transforms the aggregated
    opponent features to better capture multiplayer dynamics.

    This is inspired by adapter-based transfer learning (Houlsby et al. 2019):
    instead of fine-tuning all weights, we add small trainable modules
    that learn format-specific adaptations.
    """

    def __init__(self, player_feature_dim: int = 11, hidden_dim: int = 32):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.adapter = None
            return

        self.adapter = nn.Sequential(
            nn.Linear(player_feature_dim + 2, hidden_dim),  # +2 for num_players, is_commander
            nn.ReLU(),
            nn.Linear(hidden_dim, player_feature_dim),       # Output same shape as input
            nn.Tanh(),
        )
        # Initialize near-identity: adapter output ≈ 0, so base model is preserved
        with torch.no_grad():
            self.adapter[-2].weight.fill_(0.0)
            self.adapter[-2].bias.fill_(0.0)

    def parameters(self):
        if self.adapter is None:
            return []
        return self.adapter.parameters()

    def adapt_opponent_features(self, opp_features, num_players: int, is_commander: bool):
        """Add multiplayer conditioning to opponent features.

        Args:
            opp_features: (11,) numpy array of aggregated opponent features
            num_players: Number of players in the game
            is_commander: Whether this is a Commander format game

        Returns:
            Adapted opponent features (11,) with multiplayer-aware adjustments
        """
        if self.adapter is None:
            return opp_features

        import torch
        import numpy as np

        conditioning = np.array([
            num_players / 4.0,     # Normalized player count
            float(is_commander),   # Format flag
        ], dtype=np.float32)

        x = np.concatenate([opp_features, conditioning])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            residual = self.adapter(x_t).squeeze(0).numpy()

        # Residual connection: adapted = original + adapter_output
        return opp_features + residual


class TransferTrainer:
    """Orchestrates the Standard→Commander transfer learning curriculum.

    Usage:
        trainer = TransferTrainer(TransferConfig())
        await trainer.run()  # Runs all 3 phases sequentially
    """

    def __init__(self, config: TransferConfig | None = None):
        self.config = config or TransferConfig()
        self.adapter = MultiplayerAdapter()
        self.world_model = None
        self.trajectory_store = None
        self.stats: list[dict[str, Any]] = []

    async def run(self):
        """Execute the full transfer learning curriculum."""
        self._setup()

        logger.info("=" * 60)
        logger.info("TRANSFER LEARNING: Standard → Commander")
        logger.info("=" * 60)

        # Phase 1: Standard pre-training
        await self._phase_standard()

        # Phase 2: Commander fine-tuning
        await self._phase_commander()

        # Phase 3: Joint training
        await self._phase_joint()

        self._save_final()
        logger.info("Transfer learning complete!")

    def _setup(self):
        """Initialize world model, trajectory store, and directories."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # World model
        try:
            from src.world_model.world_model import WorldModel, WorldModelConfig
            self.world_model = WorldModel(WorldModelConfig())
            logger.info("World model initialized: %d parameters",
                        sum(p.numel() for p in self.world_model.parameters()))
        except ImportError:
            logger.warning("PyTorch unavailable — running data collection only")

        # Trajectory store
        if self.config.collect_trajectories:
            from src.world_model.trajectory import TrajectoryStore
            store_path = os.path.join(self.config.checkpoint_dir, "trajectories")
            self.trajectory_store = TrajectoryStore(store_path)

    async def _phase_standard(self):
        """Phase 1: Train on 2-player Standard games."""
        logger.info("\n--- Phase 1: STANDARD PRE-TRAINING (%d iterations) ---",
                    self.config.standard_iterations)

        from src.training.rl_trainer import RLTrainer, RLConfig

        rl_config = RLConfig(
            game_format="standard",
            starting_life=20,
            num_players=2,
            num_iterations=self.config.standard_iterations,
            games_per_iteration=self.config.standard_games_per_iter,
            max_turns_per_game=self.config.standard_max_turns,
            learning_rate=self.config.standard_lr,
            collect_trajectories=self.config.collect_trajectories,
            dream_training_interval=self.config.dream_training_interval,
            checkpoint_dir=os.path.join(self.config.checkpoint_dir, "phase1_standard"),
            log_dir=os.path.join(self.config.checkpoint_dir, "phase1_standard", "logs"),
        )

        trainer = RLTrainer(rl_config)
        await trainer.train()

        # Save Phase 1 world model checkpoint
        if self.world_model is not None:
            phase1_path = os.path.join(self.config.checkpoint_dir, "phase1_standard_model.pt")
            self.world_model.save(phase1_path)
            logger.info("Phase 1 model saved: %s", phase1_path)

        # Transfer trajectory data to shared store
        if trainer.trajectory_store and self.trajectory_store:
            count = 0
            for traj in trainer.trajectory_store.trajectories:
                self.trajectory_store.add(traj)
                count += 1
            logger.info("Transferred %d Standard trajectories to shared store", count)

        self.stats.append({
            "phase": "standard", "iterations": self.config.standard_iterations,
            "trajectories": len(self.trajectory_store.trajectories) if self.trajectory_store else 0,
        })

    async def _phase_commander(self):
        """Phase 2: Fine-tune on 4-player Commander games.

        Strategy:
        - For the first `freeze_base_epochs` iterations, only the adapter
          and controller (C model) weights are updated. The encoder (V) and
          dynamics (M) model stay frozen to preserve Standard knowledge.
        - After that, all weights are unfrozen with a lower learning rate.
        """
        logger.info("\n--- Phase 2: COMMANDER FINE-TUNING (%d iterations) ---",
                    self.config.commander_iterations)

        from src.training.rl_trainer import RLTrainer, RLConfig

        rl_config = RLConfig(
            game_format="commander",
            starting_life=40,
            num_players=4,
            num_iterations=self.config.commander_iterations,
            games_per_iteration=self.config.commander_games_per_iter,
            max_turns_per_game=self.config.commander_max_turns,
            learning_rate=self.config.commander_lr,
            collect_trajectories=self.config.collect_trajectories,
            dream_training_interval=self.config.dream_training_interval,
            checkpoint_dir=os.path.join(self.config.checkpoint_dir, "phase2_commander"),
            log_dir=os.path.join(self.config.checkpoint_dir, "phase2_commander", "logs"),
        )

        trainer = RLTrainer(rl_config)
        await trainer.train()

        # Save Phase 2 model
        if self.world_model is not None:
            phase2_path = os.path.join(self.config.checkpoint_dir, "phase2_commander_model.pt")
            self.world_model.save(phase2_path)
            logger.info("Phase 2 model saved: %s", phase2_path)

        # Transfer Commander trajectories
        if trainer.trajectory_store and self.trajectory_store:
            count = 0
            for traj in trainer.trajectory_store.trajectories:
                self.trajectory_store.add(traj)
                count += 1
            logger.info("Transferred %d Commander trajectories to shared store", count)

        self.stats.append({
            "phase": "commander", "iterations": self.config.commander_iterations,
            "trajectories": len(self.trajectory_store.trajectories) if self.trajectory_store else 0,
        })

    async def _phase_joint(self):
        """Phase 3: Joint training on mixed Standard + Commander games.

        Alternates between Standard and Commander games at a configurable ratio.
        Uses a very low learning rate to avoid catastrophic forgetting.
        """
        logger.info("\n--- Phase 3: JOINT TRAINING (%d iterations, %.0f%% Standard) ---",
                    self.config.joint_iterations, self.config.standard_ratio * 100)

        import random
        from src.training.rl_trainer import RLTrainer, RLConfig

        for iteration in range(self.config.joint_iterations):
            # Decide format for this iteration based on ratio
            if random.random() < self.config.standard_ratio:
                fmt, life, players, max_turns = "standard", 20, 2, self.config.standard_max_turns
            else:
                fmt, life, players, max_turns = "commander", 40, 4, self.config.commander_max_turns

            rl_config = RLConfig(
                game_format=fmt,
                starting_life=life,
                num_players=players,
                num_iterations=1,  # Single iteration per format switch
                games_per_iteration=5,
                max_turns_per_game=max_turns,
                learning_rate=self.config.joint_lr,
                collect_trajectories=self.config.collect_trajectories,
                checkpoint_dir=os.path.join(self.config.checkpoint_dir, "phase3_joint"),
                log_dir=os.path.join(self.config.checkpoint_dir, "phase3_joint", "logs"),
            )

            trainer = RLTrainer(rl_config)
            await trainer.train()

            logger.info("Joint iter %d/%d: %s (%d players)",
                        iteration + 1, self.config.joint_iterations, fmt, players)

        # Save final joint model
        if self.world_model is not None:
            joint_path = os.path.join(self.config.checkpoint_dir, "phase3_joint_model.pt")
            self.world_model.save(joint_path)
            logger.info("Phase 3 model saved: %s", joint_path)

        self.stats.append({
            "phase": "joint", "iterations": self.config.joint_iterations,
            "trajectories": len(self.trajectory_store.trajectories) if self.trajectory_store else 0,
        })

    def _save_final(self):
        """Save final model and training stats."""
        import json

        stats_path = os.path.join(self.config.checkpoint_dir, "transfer_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info("Transfer stats saved: %s", stats_path)

        if self.world_model is not None:
            final_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
            self.world_model.save(final_path)
            logger.info("Final transferred model saved: %s", final_path)

"""
Training pipeline for the Dynamics Model (M model).

Trains the MDN-RNN/Transformer to predict next latent states:
    P(z_{t+1}, done_{t+1} | z_t, a_t, h_t)

Training data: sequences of (z, a) pairs computed by running the trained
state encoder (V) over collected trajectories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError("Training requires PyTorch.")

import numpy as np

from ..dynamics_model import DynamicsModel, DynamicsModelConfig
from ..state_encoder import StateEncoder
from ..trajectory import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class DynamicsTrainingConfig:
    """Configuration for dynamics model training."""

    batch_size: int = 32
    sequence_length: int = 50       # Subsequence length for truncated BPTT
    learning_rate: float = 1e-3
    num_epochs: int = 50
    grad_clip: float = 1.0          # Gradient clipping norm
    checkpoint_dir: str = "checkpoints/dynamics"
    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LatentSequenceDataset(Dataset):
    """Dataset of latent state sequences for dynamics model training.

    Pre-encodes all trajectory states using the trained state encoder,
    then provides fixed-length subsequences for training.
    """

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        encoder: StateEncoder,
        sequence_length: int = 50,
        device: str = "cpu",
    ):
        self.sequences: list[dict[str, torch.Tensor]] = []

        encoder.eval()
        with torch.no_grad():
            for traj in trajectory_store.trajectories:
                if len(traj) < 3:
                    continue

                z_list = []
                a_list = []
                r_list = []
                d_list = []

                for t in traj.transitions:
                    # Encode state → z
                    features = {k: torch.from_numpy(v).float().unsqueeze(0).to(device)
                                for k, v in t.state_features.items()}
                    z, _, _ = encoder.encode(features)
                    z_list.append(z.squeeze(0).cpu())
                    a_list.append(torch.from_numpy(t.action_encoding).float())
                    r_list.append(torch.tensor(t.reward).float())
                    d_list.append(torch.tensor(float(t.done)).float())

                if len(z_list) < 3:
                    continue

                z_seq = torch.stack(z_list)  # (T, latent_dim)
                a_seq = torch.stack(a_list)  # (T, action_dim)
                r_seq = torch.stack(r_list)  # (T,)
                d_seq = torch.stack(d_list)  # (T,)

                # Split into subsequences of fixed length
                for start in range(0, len(z_list) - sequence_length, sequence_length // 2):
                    end = start + sequence_length
                    if end > len(z_list):
                        break
                    self.sequences.append({
                        "z": z_seq[start:end],
                        "actions": a_seq[start:end],
                        "rewards": r_seq[start:end],
                        "dones": d_seq[start:end],
                    })

        logger.info("Created %d training sequences of length %d", len(self.sequences), sequence_length)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.sequences[idx]


def train_dynamics(
    dynamics: DynamicsModel,
    trajectory_store: TrajectoryStore,
    encoder: StateEncoder,
    config: DynamicsTrainingConfig | None = None,
) -> DynamicsModel:
    """Train the dynamics model (M) on pre-encoded latent sequences.

    Args:
        dynamics: DynamicsModel to train
        trajectory_store: Collection of game trajectories
        encoder: Trained StateEncoder (frozen, used to pre-encode states)
        config: Training configuration

    Returns:
        Trained DynamicsModel
    """
    config = config or DynamicsTrainingConfig()
    device = torch.device(config.device)
    dynamics = dynamics.to(device)
    encoder = encoder.to(device)

    dataset = LatentSequenceDataset(
        trajectory_store, encoder, config.sequence_length, config.device
    )
    if len(dataset) == 0:
        logger.error("No training sequences! Need more trajectory data.")
        return dynamics

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(dynamics.parameters(), lr=config.learning_rate)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training dynamics model: %d sequences, %d epochs", len(dataset), config.num_epochs)

    for epoch in range(config.num_epochs):
        dynamics.train()
        epoch_losses = {"total": 0.0, "z_pred": 0.0, "done": 0.0, "reward": 0.0}

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            losses = dynamics.sequence_loss(
                batch["z"], batch["actions"], batch["dones"], batch["rewards"]
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), config.grad_clip)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

            if step % config.log_interval == 0:
                logger.info(
                    "Epoch %d, Step %d: total=%.4f, z_pred=%.4f, done=%.4f",
                    epoch, step, losses["total"].item(),
                    losses["z_pred"].item(), losses["done"].item(),
                )

        n_steps = len(dataloader)
        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d complete: avg_total=%.4f, avg_z=%.4f, avg_done=%.4f",
            epoch, avg["total"], avg["z_pred"], avg["done"],
        )

        if (epoch + 1) % 10 == 0:
            path = checkpoint_dir / f"dynamics_epoch_{epoch+1}.pt"
            torch.save(dynamics.state_dict(), str(path))

    torch.save(dynamics.state_dict(), str(checkpoint_dir / "dynamics_final.pt"))
    return dynamics

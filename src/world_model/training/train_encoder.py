"""
Training pipeline for the State Encoder (V model).

Trains the VAE to compress game states into latent vectors by:
1. Loading trajectories from TrajectoryStore
2. Encoding states with GameTokenizer
3. Training encoder/decoder with reconstruction + KL loss

The trained encoder should capture meaningful game state structure:
- Similar board positions → nearby latent vectors
- Game-relevant features preserved (who's winning, threat density, etc.)
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
    raise ImportError("Training requires PyTorch. Install with: pip install torch")

import numpy as np

from ..state_encoder import StateEncoder, StateEncoderConfig
from ..trajectory import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class EncoderTrainingConfig:
    """Configuration for state encoder training."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    kl_warmup_epochs: int = 10      # Gradually increase KL weight
    max_kl_weight: float = 0.001    # Maximum β for β-VAE
    checkpoint_dir: str = "checkpoints/encoder"
    log_interval: int = 100         # Steps between log messages
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Early-stop when avg total loss stays below this threshold for
    # ``early_stop_patience`` consecutive epochs.  Mostly defends
    # against degenerate cases where the loss converges to ~0 in
    # epoch 1 and the next 99 epochs are wasted compute.  Threshold
    # is slightly above the printed-precision floor (``%.4f``).
    early_stop_loss: float = 1e-3
    early_stop_patience: int = 3


class StateDataset(Dataset):
    """Dataset of tokenized game states from trajectories."""

    def __init__(self, trajectory_store: TrajectoryStore):
        self.samples: list[dict[str, np.ndarray]] = []
        for traj in trajectory_store.trajectories:
            for transition in traj.transitions:
                self.samples.append(transition.state_features)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        features = self.samples[idx]
        return {k: torch.from_numpy(v).float() for k, v in features.items()}


def train_encoder(
    encoder: StateEncoder,
    trajectory_store: TrajectoryStore,
    config: EncoderTrainingConfig | None = None,
) -> StateEncoder:
    """Train the state encoder (V model) on collected trajectories.

    Args:
        encoder: StateEncoder model to train
        trajectory_store: Collection of game trajectories
        config: Training configuration

    Returns:
        Trained StateEncoder
    """
    config = config or EncoderTrainingConfig()
    device = torch.device(config.device)
    encoder = encoder.to(device)

    dataset = StateDataset(trajectory_store)
    if len(dataset) == 0:
        logger.error("No training data! Collect trajectories first.")
        return encoder

    # Custom collate to handle dict of tensors
    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = Adam(encoder.parameters(), lr=config.learning_rate)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training state encoder: %d samples, %d epochs", len(dataset), config.num_epochs
    )

    converged_epochs = 0
    for epoch in range(config.num_epochs):
        encoder.train()
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}

        # KL warmup: linearly increase β over warmup period
        if epoch < config.kl_warmup_epochs:
            encoder.config.kl_weight = config.max_kl_weight * (epoch / config.kl_warmup_epochs)
        else:
            encoder.config.kl_weight = config.max_kl_weight

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            z, mu, logvar = encoder.encode(batch)
            reconstructed = encoder.decode(z)
            losses = encoder.loss(batch, reconstructed, mu, logvar)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

            if step % config.log_interval == 0:
                logger.info(
                    "Epoch %d, Step %d: total=%.4f, recon=%.4f, kl=%.4f",
                    epoch, step, losses["total"].item(),
                    losses["reconstruction"].item(), losses["kl"].item(),
                )

        # Log epoch summary
        n_steps = len(dataloader)
        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d complete: avg_total=%.4f, avg_recon=%.4f, avg_kl=%.4f",
            epoch, avg["total"], avg["reconstruction"], avg["kl"],
        )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            path = checkpoint_dir / f"encoder_epoch_{epoch+1}.pt"
            torch.save(encoder.state_dict(), str(path))
            logger.info("Saved checkpoint: %s", path)

        # Early-stop on degenerate near-zero loss to avoid wasting
        # compute when there's effectively nothing left to learn.
        if avg["total"] < config.early_stop_loss:
            converged_epochs += 1
            if converged_epochs >= config.early_stop_patience:
                logger.info(
                    "Early stop at epoch %d: avg_total=%.6f below %.6f for %d epochs",
                    epoch, avg["total"], config.early_stop_loss, converged_epochs,
                )
                break
        else:
            converged_epochs = 0

    # Save final model
    torch.save(encoder.state_dict(), str(checkpoint_dir / "encoder_final.pt"))
    return encoder

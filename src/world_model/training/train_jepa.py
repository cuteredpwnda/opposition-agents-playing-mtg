"""
Training pipeline for the JEPA Predictor.

Trains the JEPAPredictor on transition pairs (s_t, a_t, s_{t+1}) using
the LeWM-style two-term loss:

    total = MSE(ẑ_{t+1}, sg(z_{t+1})) + β · KL(N(μ, σ²) || N(0, I))

When KG context embeddings are available (via KGContextEncoder), they are
fused into the state encoding before the latent bottleneck, providing
structural semantic priors from the knowledge graph.

This function trains the encoder (V) and predictor jointly — the encoder
learns representations that are *predictable*, and the predictor learns
the dynamics of those representations.

Reference: Maes et al. (2026) "LeWorldModel: Stable End-to-End JEPA from Pixels"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError("JEPA training requires PyTorch.")

import numpy as np

from ..card_embeddings import CardEmbeddingModel
from ..jepa_predictor import JEPAPredictor, JEPAPredictorConfig
from ..kg_encoder import KGContextEncoder, KGContextEncoderConfig
from ..state_encoder import StateEncoder, StateEncoderConfig
from ..trajectory import TrajectoryStore
from ..world_model import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class JEPATrainingConfig:
    """Configuration for JEPA predictor training."""

    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 80
    beta_warmup_epochs: int = 10    # Linearly ramp β from 0 to jepa_beta
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/jepa"
    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransitionPairDataset(Dataset):
    """Dataset of consecutive (s_t, a_t, s_{t+1}) transition pairs.

    Each sample provides the raw feature dicts and action encoding for one
    step so the JEPA can encode both states and compare predictions to
    reality.

    When *card_names_extractor* is provided, card names are extracted from
    each transition's metadata and stored alongside the features so the
    KGContextEncoder can build contextual embeddings during training.
    """

    def __init__(self, trajectory_store: TrajectoryStore):
        self.pairs: list[dict] = []

        for traj in trajectory_store.trajectories:
            transitions = traj.transitions
            for i in range(len(transitions) - 1):
                t_now = transitions[i]
                t_next = transitions[i + 1]
                self.pairs.append({
                    "features_t": t_now.state_features,       # dict[str, np.ndarray]
                    "action_t": t_now.action_encoding,        # np.ndarray (136,)
                    "features_next": t_next.state_features,
                    "card_name_t": t_now.card_name,           # str | None
                    "metadata_t": t_now.metadata or {},
                    "metadata_next": t_next.metadata or {},
                })

        logger.info("Built %d transition pairs from %d trajectories",
                     len(self.pairs), len(trajectory_store))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "features_t": {k: torch.from_numpy(v).float()
                           for k, v in pair["features_t"].items()},
            "action_t": torch.from_numpy(pair["action_t"]).float(),
            "features_next": {k: torch.from_numpy(v).float()
                              for k, v in pair["features_next"].items()},
            "card_names": pair["metadata_t"].get("visible_cards", []),
            "card_names_next": pair["metadata_next"].get("visible_cards", []),
        }


def _collate_transition_pairs(batch: list[dict]) -> dict:
    """Custom collate: stack feature dicts, keep card name lists."""
    keys_t = batch[0]["features_t"].keys()
    keys_next = batch[0]["features_next"].keys()
    return {
        "features_t": {k: torch.stack([b["features_t"][k] for b in batch]) for k in keys_t},
        "action_t": torch.stack([b["action_t"] for b in batch]),
        "features_next": {k: torch.stack([b["features_next"][k] for b in batch]) for k in keys_next},
        "card_names": [b["card_names"] for b in batch],
        "card_names_next": [b["card_names_next"] for b in batch],
    }


def train_jepa(
    world_model: WorldModel,
    trajectory_store: TrajectoryStore,
    config: JEPATrainingConfig | None = None,
    kg_encoder: Optional[KGContextEncoder] = None,
) -> WorldModel:
    """Train the JEPA predictor (and encoder) on transition pairs.

    This is the dual-input JEPA training loop.  It trains the StateEncoder
    and JEPAPredictor jointly so that:
      - The encoder learns representations that are *predictable in latent space*.
      - The predictor learns the dynamics of those representations.
      - When a KGContextEncoder is provided, KG semantic context is fused
        before the bottleneck at every step.

    Args:
        world_model: WorldModel with ``use_jepa=True`` (must have a
                     ``jepa_predictor`` attribute).
        trajectory_store: Collected game trajectories.
        config: Training hypers.
        kg_encoder: Optional KGContextEncoder for dual-input path.
                    When None, only game-state features are used (single-input).

    Returns:
        The world model with updated encoder + predictor weights.
    """
    config = config or JEPATrainingConfig()
    device = torch.device(config.device)

    if world_model.jepa_predictor is None:
        raise RuntimeError(
            "world_model.jepa_predictor is None — "
            "set WorldModelConfig(use_jepa=True) when creating the model."
        )

    world_model = world_model.to(device)
    if kg_encoder is not None:
        kg_encoder = kg_encoder.to(device)

    dataset = TransitionPairDataset(trajectory_store)
    if len(dataset) == 0:
        logger.error("No transition pairs! Collect trajectories first.")
        return world_model

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate_transition_pairs,
    )

    # We optimise the encoder and predictor jointly.
    # The dynamics / controller are NOT updated here.
    params = list(world_model.encoder.parameters()) + list(world_model.jepa_predictor.parameters())
    if kg_encoder is not None:
        params += list(kg_encoder.parameters())
    optimizer = Adam(params, lr=config.learning_rate)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "JEPA training: %d pairs, %d epochs, β=%.3f, KG=%s",
        len(dataset),
        config.num_epochs,
        world_model.jepa_predictor.config.jepa_beta,
        "enabled" if kg_encoder is not None else "disabled",
    )

    original_beta = world_model.jepa_predictor.config.jepa_beta

    for epoch in range(config.num_epochs):
        world_model.train()
        if kg_encoder is not None:
            kg_encoder.train()

        epoch_losses = {"total": 0.0, "prediction": 0.0, "kl": 0.0}

        # β warmup
        if epoch < config.beta_warmup_epochs:
            world_model.jepa_predictor.config.jepa_beta = (
                original_beta * (epoch + 1) / config.beta_warmup_epochs
            )
        else:
            world_model.jepa_predictor.config.jepa_beta = original_beta

        for step, batch in enumerate(dataloader):
            features_t = {k: v.to(device) for k, v in batch["features_t"].items()}
            features_next = {k: v.to(device) for k, v in batch["features_next"].items()}
            action_t = batch["action_t"].to(device)

            # Optional KG context
            kg_t = kg_next = None
            if kg_encoder is not None:
                card_names = batch["card_names"]
                card_names_next = batch["card_names_next"]
                # Only compute KG embeddings when we have card name data
                if any(len(names) > 0 for names in card_names):
                    kg_t = kg_encoder(card_names, device=device)
                    kg_next = kg_encoder(card_names_next, device=device)

            losses = world_model.jepa_training_step(
                features_t, action_t, features_next,
                kg_embedding_t=kg_t,
                kg_embedding_next=kg_next,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if step % config.log_interval == 0:
                logger.info(
                    "Epoch %d Step %d: total=%.4f pred=%.4f kl=%.4f",
                    epoch, step,
                    losses["total"].item(),
                    losses["prediction"].item(),
                    losses["kl"].item(),
                )

        n_steps = max(len(dataloader), 1)
        avg = {k: v / n_steps for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d done: avg_total=%.4f avg_pred=%.4f avg_kl=%.4f",
            epoch, avg["total"], avg["prediction"], avg["kl"],
        )

        if (epoch + 1) % 10 == 0:
            path = checkpoint_dir / f"jepa_epoch_{epoch+1}.pt"
            torch.save({
                "encoder": world_model.encoder.state_dict(),
                "predictor": world_model.jepa_predictor.state_dict(),
                "kg_encoder": kg_encoder.state_dict() if kg_encoder else None,
            }, str(path))
            logger.info("Checkpoint: %s", path)

    # Restore β
    world_model.jepa_predictor.config.jepa_beta = original_beta

    # Save final
    final_path = checkpoint_dir / "jepa_final.pt"
    torch.save({
        "encoder": world_model.encoder.state_dict(),
        "predictor": world_model.jepa_predictor.state_dict(),
        "kg_encoder": kg_encoder.state_dict() if kg_encoder else None,
    }, str(final_path))
    logger.info("JEPA training complete → %s", final_path)

    return world_model

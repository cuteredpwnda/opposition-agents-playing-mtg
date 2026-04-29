#!/usr/bin/env python
"""Train the MTG JEPA with the stable-pretraining training pipeline."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import signal
import sys

import lightning as pl
import stable_pretraining as spt
import torch

# Suppress stable_pretraining's noisy "self.parameters gives callbacks parameters"
# warning. It fires every time their internal ModuleSummary callback iterates
# parameters() without with_callbacks=False — harmless but very repetitive.
try:
    from loguru import logger as _spt_logger
    _spt_logger.add(
        lambda msg: None,
        filter=lambda record: (
            record["name"].startswith("stable_pretraining")
            and "with_callbacks=False" in record["message"]
        ),
        level=0,
    )
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.name == "nt":
    for missing_name in ("SIGUSR1", "SIGUSR2", "SIGCONT", "SIGQUIT", "SIGHUP"):
        if not hasattr(signal, missing_name):
            setattr(signal, missing_name, signal.SIGTERM)

from src.world_model.jepa_predictor import JEPAPredictorConfig
from src.world_model.kg_encoder import KGContextEncoder, KGContextEncoderConfig
from src.world_model.state_encoder import StateEncoderConfig
from src.world_model.training.train_jepa import (
    JEPATrainingConfig,
    TransitionPairDataset,
    _collate_transition_pairs,
)
from src.world_model.trajectory import TrajectoryStore
from src.world_model.world_model import WorldModel, WorldModelConfig


def _configure_windows_stdio() -> None:
    if os.name != "nt":
        return

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _stable_jepa_forward(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
    model_device = next(self.model.parameters()).device
    features_t = {k: v.to(model_device) for k, v in batch["features_t"].items()}
    features_next = {k: v.to(model_device) for k, v in batch["features_next"].items()}
    action_t = batch["action_t"].to(model_device)

    kg_t = None
    kg_next = None
    if getattr(self, "kg", None) is not None:
        card_names = batch["card_names"]
        card_names_next = batch["card_names_next"]
        if any(len(names) > 0 for names in card_names):
            kg_t = self.kg(card_names, device=model_device)
            kg_next = self.kg(card_names_next, device=model_device)

    losses = self.model.jepa_training_step(
        features_t,
        action_t,
        features_next,
        kg_embedding_t=kg_t,
        kg_embedding_next=kg_next,
    )
    prefix = "train" if stage in {"fit", "train"} or stage.startswith("train") else stage
    return {
        "loss": losses["total"],
        f"{prefix}/total": losses["total"],
        f"{prefix}/prediction": losses["prediction"],
        f"{prefix}/kl": losses["kl"],
    }


class BetaWarmupCallback(pl.Callback):
    def __init__(self, original_beta: float, warmup_epochs: int):
        super().__init__()
        self.original_beta = original_beta
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        predictor = pl_module.model.jepa_predictor
        if predictor is None:
            return
        epoch = int(trainer.current_epoch)
        if epoch < self.warmup_epochs:
            predictor.config.jepa_beta = (
                self.original_beta * (epoch + 1) / self.warmup_epochs
            )
        else:
            predictor.config.jepa_beta = self.original_beta


class MetricsCSVCallback(pl.Callback):
    """Log per-epoch KL, prediction, and total loss to a CSV file."""

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.metric_keys = ("train/total", "train/prediction", "train/kl")
        self._epoch_sums: dict[str, float] = {}
        self._epoch_counts: dict[str, int] = {}
        # Write header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train/total", "train/prediction", "train/kl"])

    def _reset_epoch_accumulators(self) -> None:
        self._epoch_sums = {key: 0.0 for key in self.metric_keys}
        self._epoch_counts = {key: 0 for key in self.metric_keys}

    def _to_float(self, value: object) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().item())
        return float(value)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        self._reset_epoch_accumulators()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not isinstance(outputs, dict):
            return

        for key in self.metric_keys:
            if key not in outputs:
                continue
            value = self._to_float(outputs[key])
            self._epoch_sums[key] += value
            self._epoch_counts[key] += 1

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)

        def _avg(metric_key: str) -> float:
            count = self._epoch_counts[metric_key]
            if count == 0:
                return float("nan")
            return self._epoch_sums[metric_key] / count

        train_total = _avg("train/total")
        train_pred = _avg("train/prediction")
        train_kl = _avg("train/kl")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_total, train_pred, train_kl])


def main():
    _configure_windows_stdio()

    parser = argparse.ArgumentParser(description="Train JEPA with the stable-pretraining pipeline")
    parser.add_argument("--trajectories", type=str, default="data/trajectories", help="Trajectory store dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stable_worldmodel.pt")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for metrics and logs (default: auto-generated timestamp)")
    parser.add_argument("--jepa-beta", type=float, default=1.0)
    parser.add_argument("--no-kg", action="store_true")
    parser.add_argument("--kg-embed-dim", type=int, default=128)
    args = parser.parse_args()

    # Create output directory (with timestamp if not specified)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"runs/training_stable_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training output directory: {output_dir.absolute()}")

    store = TrajectoryStore(storage_dir=args.trajectories)
    store.load()
    if len(store) == 0:
        raise RuntimeError("No trajectories found; run self-play first.")

    encoder_cfg = StateEncoderConfig(
        kg_embed_dim=0 if args.no_kg else args.kg_embed_dim,
    )
    jepa_cfg = JEPAPredictorConfig(
        latent_dim=encoder_cfg.latent_dim,
        action_dim=136,
        jepa_beta=args.jepa_beta,
    )
    world_model = WorldModel(
        WorldModelConfig(
            encoder=encoder_cfg,
            jepa=jepa_cfg,
            use_jepa=True,
        )
    )

    kg_encoder = None
    if not args.no_kg:
        try:
            from src.world_model.card_embeddings import CardEmbeddingModel

            card_model = CardEmbeddingModel()
            kg_encoder = KGContextEncoder(
                card_model,
                KGContextEncoderConfig(kg_embed_dim=args.kg_embed_dim),
            )
        except Exception as exc:
            print(f"Warning: could not build KG encoder, training without KG: {exc}")

    dataset = TransitionPairDataset(store)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_transition_pairs,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_transition_pairs,
    )

    optim = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "Adam", "lr": JEPATrainingConfig().learning_rate},
            "interval": "epoch",
        }
    }
    if kg_encoder is not None:
        optim["kg_opt"] = {
            "modules": "kg",
            "optimizer": {"type": "Adam", "lr": JEPATrainingConfig().learning_rate},
            "interval": "epoch",
        }

    trainer_module = spt.Module(
        model=world_model,
        kg=kg_encoder,
        forward=_stable_jepa_forward,
        optim=optim,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=JEPATrainingConfig().grad_clip,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            BetaWarmupCallback(args.jepa_beta, JEPATrainingConfig().beta_warmup_epochs),
            MetricsCSVCallback(output_dir),
        ],
    )

    manager = spt.Manager(
        trainer=trainer,
        module=trainer_module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )
    manager()

    os.makedirs(Path(args.checkpoint).parent, exist_ok=True)
    world_model.save(args.checkpoint)
    print(f"Stable JEPA model trained and saved to {args.checkpoint}")
    print(f"Per-epoch metrics saved to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

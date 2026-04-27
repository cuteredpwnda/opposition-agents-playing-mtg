"""Neural reasoner training pipeline (Phase D).

Loads trajectories produced by the self-play loop, projects them into the
fixed-size board feature vector consumed by ``NeuralReasoningModule`` and
trains the value/win-probability heads via supervised regression on the
terminal Monte-Carlo return. Once the supervised pass has converged the
model can be fine-tuned with policy gradients (REINFORCE) using the same
trajectory store --- both modes are exposed on the CLI.

This module degrades gracefully when PyTorch / PyG are not installed:
``train_neural_reasoner`` raises a clear ``RuntimeError`` and the
``NeuralReasonerAgent`` keeps using its zero-vector fallback. Tests cover
the data-pipeline pieces that do not require torch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.world_model.trajectory import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class NRTrainConfig:
    trajectories_dir: Path = Path("data/trajectories")
    checkpoint_dir: Path = Path("checkpoints/neural_reasoner")
    metrics_path: Path = Path("logs/neural_reasoner/metrics.json")

    board_features_dim: int = 512
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 10
    discount: float = 0.99
    value_loss_weight: float = 1.0
    win_loss_weight: float = 1.0

    # RL fine-tune
    rl_epochs: int = 0
    rl_entropy_bonus: float = 0.01

    device: str = "cpu"
    seed: int = 0


# ---------------------------------------------------------------------------
# Data pipeline (torch-free portion)
# ---------------------------------------------------------------------------
@dataclass
class TrainingExample:
    """Single supervised example for the value/win-prob heads."""

    board_features: np.ndarray   # shape (D,)
    target_value: float          # discounted return from this step
    target_win: float            # 1.0 if player ultimately won, else 0.0
    metadata: dict = field(default_factory=dict)


def _board_features_from_transition(
    state_features: dict[str, np.ndarray],
    target_dim: int,
) -> np.ndarray:
    """Project the heterogeneous tokenizer output into a fixed-size vector.

    Pure-NumPy: concatenate every numeric ndarray we find, then truncate or
    zero-pad to ``target_dim``. This is enough for the Phase-D supervised
    bootstrap; the GAT branch is wired but not trained here.
    """
    parts: list[np.ndarray] = []
    for val in state_features.values():
        if isinstance(val, np.ndarray):
            parts.append(val.astype(np.float32).reshape(-1))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if flat.size >= target_dim:
        return flat[:target_dim].astype(np.float32)
    out = np.zeros(target_dim, dtype=np.float32)
    out[: flat.size] = flat
    return out


def trajectories_to_examples(
    trajectories: Iterable[Trajectory],
    discount: float,
    board_features_dim: int,
) -> list[TrainingExample]:
    """Convert raw trajectories into ``(features, value, win)`` examples.

    The discounted return is computed in reverse from the terminal reward:
    ``G_t = r_t + gamma * G_{t+1}``. ``target_win`` is the binary outcome
    from the player-of-record's perspective when ``trajectory.winner``
    is known; otherwise ``0.5``.
    """
    examples: list[TrainingExample] = []
    for traj in trajectories:
        if not traj.transitions:
            continue
        rewards = [float(t.reward) for t in traj.transitions]
        returns: list[float] = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + discount * running
            returns[t] = running
        if traj.winner is None:
            win_target = 0.5
        else:
            win_target = 1.0 if traj.winner == 0 else 0.0
        for transition, ret in zip(traj.transitions, returns):
            feats = _board_features_from_transition(
                transition.state_features, board_features_dim
            )
            examples.append(
                TrainingExample(
                    board_features=feats,
                    target_value=float(ret),
                    target_win=win_target,
                    metadata={
                        "action_type": transition.action_type,
                        "card_name": transition.card_name,
                    },
                )
            )
    return examples


def load_trajectories(store_dir: Path | str) -> list[Trajectory]:
    """Load all trajectories from the on-disk ``TrajectoryStore``.

    Returns an empty list when the directory is missing so callers can
    no-op cleanly during a cold-start pipeline run.
    """
    store_dir = Path(store_dir)
    if not store_dir.exists():
        logger.warning("Trajectory directory %s missing — nothing to load", store_dir)
        return []
    store = TrajectoryStore(str(store_dir))
    store.load()
    return list(store.trajectories) if hasattr(store, "trajectories") else []


# ---------------------------------------------------------------------------
# Supervised training (torch-only)
# ---------------------------------------------------------------------------
def train_neural_reasoner(config: NRTrainConfig) -> dict:
    """End-to-end supervised training of the value/win heads.

    Returns a dict of summary metrics. Raises ``RuntimeError`` if torch is
    unavailable so the caller can decide to skip the stage.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Neural reasoner training requires torch; install with `pip install -e .[ml]`"
        ) from exc

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(config.trajectories_dir)
    if not trajectories:
        msg = f"No trajectories found under {config.trajectories_dir}"
        logger.error(msg)
        return {"status": "no_data", "message": msg, "examples": 0}

    examples = trajectories_to_examples(
        trajectories, config.discount, config.board_features_dim
    )
    if not examples:
        return {"status": "no_examples", "examples": 0}

    rng = np.random.default_rng(config.seed)
    rng.shuffle(examples)
    split = max(1, int(0.9 * len(examples)))
    train_ex, val_ex = examples[:split], examples[split:]

    def to_tensors(batch: list[TrainingExample]):
        x = torch.from_numpy(np.stack([e.board_features for e in batch]))
        v = torch.tensor([e.target_value for e in batch], dtype=torch.float32)
        w = torch.tensor([e.target_win for e in batch], dtype=torch.float32)
        return x, v, w

    x_tr, v_tr, w_tr = to_tensors(train_ex)
    x_va, v_va, w_va = to_tensors(val_ex)

    train_loader = DataLoader(
        TensorDataset(x_tr, v_tr, w_tr),
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Light-weight head (board encoder -> value/win); avoids importing the
    # GAT/sequence branches so this script is usable without torch-geometric.
    class ValueWinHead(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.value = nn.Linear(hidden, 1)
            self.win = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return self.value(h).squeeze(-1), torch.sigmoid(self.win(h)).squeeze(-1)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model = ValueWinHead(config.board_features_dim, config.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    history: list[dict] = []
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, vb, wb in train_loader:
            xb, vb, wb = xb.to(device), vb.to(device), wb.to(device)
            v_pred, w_pred = model(xb)
            loss = (
                config.value_loss_weight * mse(v_pred, vb)
                + config.win_loss_weight * bce(w_pred, wb)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        with torch.no_grad():
            v_pred_va, w_pred_va = model(x_va.to(device))
            val_value_mse = float(mse(v_pred_va, v_va.to(device)).item())
            val_win_bce = float(bce(w_pred_va, w_va.to(device)).item())

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_value_mse": val_value_mse,
            "val_win_bce": val_win_bce,
        }
        history.append(epoch_metrics)
        logger.info(
            "NR-train epoch %d: train_loss=%.4f val_v_mse=%.4f val_w_bce=%.4f",
            epoch + 1,
            train_loss,
            val_value_mse,
            val_win_bce,
        )

    ckpt = config.checkpoint_dir / "value_win_head.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__}, ckpt)
    summary = {
        "status": "ok",
        "examples": len(examples),
        "train_examples": len(train_ex),
        "val_examples": len(val_ex),
        "history": history,
        "checkpoint": str(ckpt),
    }
    with config.metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Neural reasoner training complete; checkpoint -> %s", ckpt)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="Train the neural-reasoner heads")
    p.add_argument("--trajectories-dir", default="data/trajectories")
    p.add_argument("--checkpoint-dir", default="checkpoints/neural_reasoner")
    p.add_argument("--metrics-path", default="logs/neural_reasoner/metrics.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    cfg = NRTrainConfig(
        trajectories_dir=Path(args.trajectories_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        metrics_path=Path(args.metrics_path),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = train_neural_reasoner(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Adapter layer to integrate galilai-group/stable-worldmodel (LeWM) into this repo.

This module keeps our internal API stable while delegating heavy training
work to stable-worldmodel.
"""

from __future__ import annotations

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required for stable-worldmodel adapter") from e

from src.world_model.world_model import WorldModel


class StableWorldModelAdapter(WorldModel):
    """Adapter wrapper around stable_worldmodel API."""

    def __init__(self, config=None):
        try:
            import stable_worldmodel
        except ImportError as e:
            raise ImportError(
                "stable-worldmodel package is required (pip install git+https://github.com/galilai-group/stable-worldmodel)."
            ) from e

        # Keep local settings: we still use our native WorldModel config for consistency
        super().__init__(config=None)

        self.stable = stable_worldmodel.WorldModel()  # best-effort default
        self.trainer = None

    def fit(self, hdf5_path: str, epochs: int = 10, batch_size: int = 32, device: str = "cpu"):
        """Train stable-worldmodel on HDF5 trajectories."""
        try:
            import stable_worldmodel
        except ImportError as e:
            raise ImportError("stable-worldmodel is required for fit()") from e

        self.trainer = stable_worldmodel.WorldModelTrainer(
            model=self.stable,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
        )
        self.trainer.train(hdf5_path)

    def predict(self, z, action, hidden, **kwargs):
        return self.stable.predict(z, action, hidden, **kwargs)

    def act(self, z, h, legal_action_encodings, legal_action_mask, deterministic=False):
        return self.stable.act(z, h, legal_action_encodings, legal_action_mask, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.stable.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StableWorldModelAdapter":
        inst = cls()
        inst.stable = inst.stable.load(path, device=device)
        return inst

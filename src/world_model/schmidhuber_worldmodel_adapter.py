"""Adapter for a lightweight Schmidhuber-style world model path."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("PyTorch is required for Schmidhuber world model adapter") from e

from src.world_model.world_model import WorldModel


class SchmidhuberWorldModelAdapter(WorldModel):
    """Simple wrapper implementing same interface used by WorldModelAgent."""

    def __init__(self, config=None):
        super().__init__(config or None)
        c = self.config

        # Minimal forward model: simple MLP from (z, a) -> z' + done
        self.forward_model = nn.Sequential(
            nn.Linear(c.encoder.latent_dim + c.controller.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, c.encoder.latent_dim + 1),
        )

    def fit(self, trajectories, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3, device: str = "cpu"):
        """Train on a trajectory store (can be TrajectoryStore or HDF5 file)."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Minimal dataset from trajectories:
        training_data = []
        for traj in trajectories.trajectories:
            for i in range(len(traj.transitions) - 1):
                z = torch.from_numpy(traj.transitions[i].state_features["latent"]).float() if "latent" in traj.transitions[i].state_features else None
                a = torch.from_numpy(traj.transitions[i].action_encoding).float()
                z_next = torch.from_numpy(traj.transitions[i + 1].state_features.get("latent", traj.transitions[i].state_features.get("player_features"))).float()
                done = torch.tensor([float(traj.transitions[i + 1].done)])
                if z is None:
                    continue
                training_data.append((z, a, torch.cat([z_next, done], dim=-1)))

        if not training_data:
            raise ValueError("No training data found in trajectories")

        # simple training loop
        for epoch in range(epochs):
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i : i + batch_size]
                zs = torch.stack([x[0] for x in batch], dim=0).to(device)
                actions = torch.stack([x[1] for x in batch], dim=0).to(device)
                targets = torch.stack([x[2] for x in batch], dim=0).to(device)

                inp = torch.cat([zs, actions], dim=-1)
                pred = self.forward_model(inp)
                loss = F.mse_loss(pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, z, action, hidden=None, **kwargs):
        inp = torch.cat([z, action], dim=-1)
        out = self.forward_model(inp)
        z_next = out[:, : z.size(-1)]
        done_prob = torch.sigmoid(out[:, -1])
        return z_next, hidden, done_prob

    def act(self, z, h, legal_action_encodings, legal_action_mask, deterministic=False):
        # Use a simple controller: choose random legal action if deterministic False.
        if deterministic:
            return 0, torch.tensor([0.0])
        legal_idxs = torch.nonzero(legal_action_mask[0]).view(-1)
        idx = legal_idxs[0].item() if len(legal_idxs) else 0
        return idx, torch.tensor([0.0])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SchmidhuberWorldModelAdapter":
        inst = cls()
        inst.load_state_dict(torch.load(path, map_location=device))
        inst.to(device)
        return inst

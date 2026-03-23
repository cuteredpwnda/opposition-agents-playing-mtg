"""
Game trajectory storage.

Stores sequences of (state, action, reward, done) tuples from actual
or simulated games. Trajectories are the training data for V + M.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Transition:
    """A single game state transition."""

    state_features: dict[str, np.ndarray]   # GameTokenizer output
    action_encoding: np.ndarray             # Encoded action vector
    reward: float                            # Shaped reward signal
    done: bool                               # Game over?
    action_type: str                         # e.g., "CAST_SPELL"
    card_name: str | None = None             # Card involved, if any
    metadata: dict = field(default_factory=dict)  # Extra info (phase, turn, etc.)


@dataclass
class Trajectory:
    """A complete game trajectory (one full game)."""

    game_id: str
    transitions: list[Transition] = field(default_factory=list)
    winner: int | None = None                # Player index who won
    num_turns: int = 0
    source: str = "unknown"                  # "self_play", "17lands", "mtga_logs"
    metadata: dict = field(default_factory=dict)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def states(self) -> list[dict[str, np.ndarray]]:
        return [t.state_features for t in self.transitions]

    @property
    def actions(self) -> list[np.ndarray]:
        return [t.action_encoding for t in self.transitions]

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.transitions]


class TrajectoryStore:
    """Stores and manages a collection of game trajectories.

    Supports saving/loading to disk in a directory structure:
        data/trajectories/
        ├── metadata.json       # Index of all trajectories
        ├── game_001.npz        # Numpy arrays for one game
        ├── game_002.npz
        └── ...
    """

    def __init__(self, storage_dir: str = "data/trajectories"):
        self.storage_dir = Path(storage_dir)
        self.trajectories: list[Trajectory] = []
        self._index: dict[str, int] = {}  # game_id → index in list

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the store."""
        if trajectory.game_id in self._index:
            return  # Duplicate
        self._index[trajectory.game_id] = len(self.trajectories)
        self.trajectories.append(trajectory)

    def get(self, game_id: str) -> Trajectory | None:
        """Retrieve a trajectory by game ID."""
        idx = self._index.get(game_id)
        return self.trajectories[idx] if idx is not None else None

    def sample_batch(self, batch_size: int, min_length: int = 5) -> list[Trajectory]:
        """Sample a random batch of trajectories."""
        eligible = [t for t in self.trajectories if len(t) >= min_length]
        if not eligible:
            return []
        indices = np.random.choice(len(eligible), size=min(batch_size, len(eligible)), replace=False)
        return [eligible[i] for i in indices]

    def __len__(self) -> int:
        return len(self.trajectories)

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        """Save all trajectories to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata index
        index = []
        for traj in self.trajectories:
            # Save numpy arrays for this trajectory
            arrays = {}
            for i, t in enumerate(traj.transitions):
                for key, arr in t.state_features.items():
                    arrays[f"state_{i}_{key}"] = arr
                arrays[f"action_{i}"] = t.action_encoding
                arrays[f"reward_{i}"] = np.array(t.reward)
                arrays[f"done_{i}"] = np.array(t.done)

            npz_path = self.storage_dir / f"{traj.game_id}.npz"
            np.savez_compressed(str(npz_path), **arrays)

            index.append({
                "game_id": traj.game_id,
                "num_transitions": len(traj),
                "winner": traj.winner,
                "num_turns": traj.num_turns,
                "source": traj.source,
            })

        with open(self.storage_dir / "metadata.json", "w") as f:
            json.dump(index, f, indent=2)

    def load(self) -> None:
        """Load trajectories from disk."""
        metadata_path = self.storage_dir / "metadata.json"
        if not metadata_path.exists():
            return

        with open(metadata_path) as f:
            index = json.load(f)

        for entry in index:
            game_id = entry["game_id"]
            npz_path = self.storage_dir / f"{game_id}.npz"
            if not npz_path.exists():
                continue

            data = np.load(str(npz_path), allow_pickle=True)

            traj = Trajectory(
                game_id=game_id,
                winner=entry.get("winner"),
                num_turns=entry.get("num_turns", 0),
                source=entry.get("source", "unknown"),
            )

            # Reconstruct transitions
            i = 0
            while f"action_{i}" in data:
                state_features = {}
                for key in data:
                    prefix = f"state_{i}_"
                    if key.startswith(prefix):
                        feat_name = key[len(prefix):]
                        state_features[feat_name] = data[key]

                transition = Transition(
                    state_features=state_features,
                    action_encoding=data[f"action_{i}"],
                    reward=float(data[f"reward_{i}"]),
                    done=bool(data[f"done_{i}"]),
                    action_type="unknown",  # Not persisted in npz
                )
                traj.add(transition)
                i += 1

            self.add(traj)

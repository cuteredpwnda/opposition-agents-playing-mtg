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
        """Save all trajectories to disk.

        Uses a *dense* per-feature layout: for each trajectory we write
        one npz file containing one entry per state-feature key (stacked
        along axis 0 across transitions) plus stacked ``actions``,
        ``rewards`` and ``dones`` arrays.  When transitions disagree on a
        feature's shape we fall back to an ``object`` array.

        The previous "per-transition key" format (``state_{i}_{name}``)
        scaled the npz key count linearly with the number of transitions,
        making load times unusable (>200s per game).  Old files written
        in that legacy format are still readable via :meth:`load`.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        index = []
        for traj in self.trajectories:
            n = len(traj.transitions)
            arrays: dict[str, np.ndarray] = {}

            # Collect feature keys (union across transitions).
            feat_keys: set[str] = set()
            for t in traj.transitions:
                feat_keys.update(t.state_features.keys())

            for key in feat_keys:
                values = [t.state_features.get(key) for t in traj.transitions]
                shapes = {None if v is None else v.shape for v in values}
                if len(shapes) == 1 and None not in shapes:
                    arrays[f"state__{key}"] = np.stack(values, axis=0)
                else:
                    arrays[f"state__{key}"] = np.array(values, dtype=object)

            arrays["actions"] = np.stack([t.action_encoding for t in traj.transitions], axis=0)
            arrays["rewards"] = np.array([t.reward for t in traj.transitions], dtype=np.float32)
            arrays["dones"] = np.array([t.done for t in traj.transitions], dtype=bool)
            arrays["_format"] = np.array("dense_v1")
            arrays["_num_transitions"] = np.array(n)

            npz_path = self.storage_dir / f"{traj.game_id}.npz"
            np.savez_compressed(str(npz_path), **arrays)

            index.append({
                "game_id": traj.game_id,
                "num_transitions": n,
                "winner": traj.winner,
                "num_turns": traj.num_turns,
                "source": traj.source,
            })

        with open(self.storage_dir / "metadata.json", "w") as f:
            json.dump(index, f, indent=2)

    def load(self) -> None:
        """Load trajectories from disk."""
        import logging
        import time as _time
        _log = logging.getLogger(__name__)
        metadata_path = self.storage_dir / "metadata.json"
        if not metadata_path.exists():
            return

        with open(metadata_path) as f:
            index = json.load(f)

        _log.info("Loading %d trajectories from %s ...", len(index), self.storage_dir)

        for n, entry in enumerate(index, 1):
            game_id = entry["game_id"]
            npz_path = self.storage_dir / f"{game_id}.npz"
            if not npz_path.exists():
                continue
            _t0 = _time.time()
            data = np.load(str(npz_path), allow_pickle=True)
            files_set = set(data.files)

            traj = Trajectory(
                game_id=game_id,
                winner=entry.get("winner"),
                num_turns=entry.get("num_turns", 0),
                source=entry.get("source", "unknown"),
            )

            if "_format" in files_set and str(data["_format"]) == "dense_v1":
                # Fast path: per-feature stacked arrays.
                num = int(data["_num_transitions"])
                actions = data["actions"]
                rewards = data["rewards"]
                dones = data["dones"]
                feat_arrays = {
                    key[len("state__"):]: data[key]
                    for key in data.files
                    if key.startswith("state__")
                }
                for i in range(num):
                    state_features = {k: v[i] for k, v in feat_arrays.items()}
                    traj.add(Transition(
                        state_features=state_features,
                        action_encoding=actions[i],
                        reward=float(rewards[i]),
                        done=bool(dones[i]),
                        action_type="unknown",
                    ))
                self.add(traj)
                _log.info("  [%d/%d] %s: %d transitions in %.1fs (dense)",
                          n, len(index), game_id, num, _time.time() - _t0)
                continue

            # Legacy path: per-transition keys (slow, kept for old files).
            # Pre-bucket keys by transition index in a single pass over the
            # key list (O(K) instead of O(N*K) where N = #transitions and
            # K = #npz keys).  Per-transition key counts can reach 70k+
            # making the naive nested scan minutes-slow.
            state_keys_by_idx: dict[int, list[tuple[str, str]]] = {}
            for key in data.files:
                if key.startswith("state_"):
                    rest = key[len("state_"):]
                    sep = rest.find("_")
                    if sep <= 0:
                        continue
                    try:
                        idx = int(rest[:sep])
                    except ValueError:
                        continue
                    feat_name = rest[sep + 1:]
                    state_keys_by_idx.setdefault(idx, []).append((key, feat_name))

            i = 0
            while f"action_{i}" in files_set:
                state_features = {}
                for key, feat_name in state_keys_by_idx.get(i, ()):
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
            _log.info("  [%d/%d] %s: %d transitions in %.1fs (legacy)",
                      n, len(index), game_id, i, _time.time() - _t0)

    def save_hdf5(self, output_path: str) -> None:
        """Export trajectories into HDF5 for external world-model trainers."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required to save HDF5 trajectories") from e

        with h5py.File(output_path, "w") as h5f:
            h5f.attrs["num_trajectories"] = len(self.trajectories)
            for t_idx, traj in enumerate(self.trajectories):
                g_traj = h5f.create_group(str(t_idx))
                g_traj.attrs["game_id"] = traj.game_id
                g_traj.attrs["winner"] = traj.winner if traj.winner is not None else -1
                g_traj.attrs["num_turns"] = traj.num_turns
                g_traj.attrs["source"] = traj.source

                for step_idx, transition in enumerate(traj.transitions):
                    g_step = g_traj.create_group(str(step_idx))
                    state_grp = g_step.create_group("state")
                    for feat_name, feat_arr in transition.state_features.items():
                        state_grp.create_dataset(feat_name, data=feat_arr, compression="gzip")

                    g_step.create_dataset("action", data=transition.action_encoding, compression="gzip")
                    g_step.attrs["reward"] = transition.reward
                    g_step.attrs["done"] = bool(transition.done)
                    g_step.attrs["action_type"] = transition.action_type
                    g_step.attrs["card_name"] = transition.card_name if transition.card_name is not None else ""
                    g_step.attrs["metadata"] = json.dumps(transition.metadata)

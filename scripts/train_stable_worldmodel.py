#!/usr/bin/env python
"""Train using galilai-group/stable-worldmodel with HDF5 trajectory input."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.world_model.trajectory import TrajectoryStore
from src.world_model.stable_worldmodel_adapter import StableWorldModelAdapter


def main():
    parser = argparse.ArgumentParser(description="Train stable-worldmodel from trajectories")
    parser.add_argument("--trajectories", type=str, default="data/trajectories", help="Trajectory store dir")
    parser.add_argument("--hdf5", type=str, default="data/trajectories/world_model_train.h5", help="Output HDF5 path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stable_worldmodel.pt")
    args = parser.parse_args()

    store = TrajectoryStore(storage_dir=args.trajectories)
    store.load()
    if len(store) == 0:
        raise RuntimeError("No trajectories found; run self-play first.")

    os.makedirs(Path(args.hdf5).parent, exist_ok=True)
    store.save_hdf5(args.hdf5)

    model = StableWorldModelAdapter()
    model.fit(args.hdf5, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    os.makedirs(Path(args.checkpoint).parent, exist_ok=True)
    model.save(args.checkpoint)
    print(f"Stable-worldmodel trained and saved to {args.checkpoint}")


if __name__ == "__main__":
    main()

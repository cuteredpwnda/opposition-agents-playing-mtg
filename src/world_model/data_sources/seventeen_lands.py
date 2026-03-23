"""
17Lands data source connector.

Downloads and parses public game replay data from 17lands.com.
17Lands tracks MTG Arena limited (draft) games with detailed per-pick
and per-game-action data.

Public datasets: https://www.17lands.com/public_datasets

Provides millions of game replays with:
- Draft picks and pack contents
- Game actions (land plays, spells cast, attacks, blocks)
- Win/loss outcomes
- Card performance statistics
"""

from __future__ import annotations

import csv
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SeventeenLandsConfig:
    """Configuration for 17Lands data import."""

    data_dir: str = "data/17lands"
    # Public dataset URLs (user must download manually due to ToS)
    # https://www.17lands.com/public_datasets
    game_data_file: str = "game_data_public.csv"
    draft_data_file: str = "draft_data_public.csv"
    replay_data_dir: str = "replay_data"
    set_code: str | None = None  # Filter to specific set, e.g., "MKM"


class SeventeenLandsSource:
    """Imports game trajectories from 17Lands public datasets.

    17Lands provides CSV files with per-game statistics and (for some data)
    per-action replay logs.

    Usage:
        source = SeventeenLandsSource(config)
        for trajectory in source.iter_trajectories():
            trajectory_store.add(trajectory)
    """

    def __init__(self, config: SeventeenLandsConfig | None = None):
        self.config = config or SeventeenLandsConfig()
        self.data_dir = Path(self.config.data_dir)

    def is_available(self) -> bool:
        """Check if 17Lands data has been downloaded."""
        return (self.data_dir / self.config.game_data_file).exists()

    def iter_game_records(self) -> Iterator[dict]:
        """Iterate over game-level records from the public CSV.

        Each record contains aggregate stats per game:
        - draft_id, game_number, on_play, won
        - For each card in pool: deck (in deck?), drawn, opening_hand,
          sideboard, turns_in_hand, etc.

        Yields:
            dict with game data fields
        """
        csv_path = self.data_dir / self.config.game_data_file
        if not csv_path.exists():
            logger.warning(
                "17Lands game data not found at %s. "
                "Download from https://www.17lands.com/public_datasets",
                csv_path,
            )
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.config.set_code and row.get("expansion") != self.config.set_code:
                    continue
                yield row

    def iter_trajectories(self):
        """Convert 17Lands game records into Trajectory objects.

        Note: The public game_data CSV provides per-game aggregate stats,
        not per-action replays. For full action-by-action trajectories,
        the replay_data (if available) would be needed.

        For now, this creates simplified trajectories from the aggregate data,
        which can still be useful for training the state encoder and card
        embeddings.

        Yields:
            Trajectory objects
        """
        # Import here to avoid circular dependency
        from ..trajectory import Trajectory, Transition

        for i, record in enumerate(self.iter_game_records()):
            game_id = f"17lands_{record.get('draft_id', i)}_{record.get('game_number', 0)}"
            won = record.get("won") == "True"

            traj = Trajectory(
                game_id=game_id,
                winner=0 if won else 1,
                source="17lands",
                metadata={
                    "expansion": record.get("expansion", ""),
                    "event_type": record.get("event_type", ""),
                    "on_play": record.get("on_play") == "True",
                    "num_turns": int(record.get("num_turns", 0)),
                },
            )
            traj.num_turns = int(record.get("num_turns", 0))

            # Extract card-level data from columns
            # 17Lands CSV has columns like "deck_<cardname>", "drawn_<cardname>", etc.
            # TODO: Parse these into per-turn state snapshots
            # For now, create a single transition with aggregate features

            yield traj

    def get_card_statistics(self) -> dict[str, dict]:
        """Extract per-card win rate and usage statistics.

        Useful for building gameplay-based card embeddings.

        Returns:
            Dict mapping card_name → {games_in_deck, games_drawn, win_rate, ...}
        """
        stats: dict[str, dict] = {}

        for record in self.iter_game_records():
            won = record.get("won") == "True"

            # Scan columns for card-specific data
            for key, value in record.items():
                if key.startswith("deck_"):
                    card_name = key[5:]  # Remove "deck_" prefix
                    if card_name not in stats:
                        stats[card_name] = {
                            "games_in_deck": 0,
                            "games_drawn": 0,
                            "wins_in_deck": 0,
                            "wins_drawn": 0,
                        }
                    in_deck = int(value) > 0 if value else False
                    if in_deck:
                        stats[card_name]["games_in_deck"] += 1
                        if won:
                            stats[card_name]["wins_in_deck"] += 1

                elif key.startswith("drawn_"):
                    card_name = key[6:]
                    if card_name in stats:
                        was_drawn = int(value) > 0 if value else False
                        if was_drawn:
                            stats[card_name]["games_drawn"] += 1
                            if won:
                                stats[card_name]["wins_drawn"] += 1

        # Compute win rates
        for card_name, s in stats.items():
            if s["games_in_deck"] > 0:
                s["win_rate_in_deck"] = s["wins_in_deck"] / s["games_in_deck"]
            if s["games_drawn"] > 0:
                s["win_rate_drawn"] = s["wins_drawn"] / s["games_drawn"]

        return stats

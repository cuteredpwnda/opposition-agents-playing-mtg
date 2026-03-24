"""
MTG Arena log parser.

Parses detailed game logs from MTG Arena's output_log.txt to extract
game trajectories. MTGA writes extensive JSON-format logs during gameplay.

Log location: %APPDATA%/../LocalLow/Wizards Of The Coast/MTGA/output_log.txt

Note: MTGA logs are from the local player's perspective and may not contain
full opponent information.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Default MTGA log path on Windows
_DEFAULT_MTGA_LOG = os.path.expandvars(
    r"%LOCALAPPDATA%Low\Wizards Of The Coast\MTGA\output_log.txt"
)


@dataclass
class MTGALogConfig:
    """Configuration for MTGA log parsing."""

    log_path: str = _DEFAULT_MTGA_LOG
    output_dir: str = "data/mtga_logs"


class MTGALogParser:
    """Parses MTG Arena output_log.txt for game replay data.

    MTGA logs contain JSON messages for various game events:
    - GRE_TO_CLIENT messages: Game state updates
    - CLIENT_TO_GRE messages: Player actions
    - MatchBegin/MatchEnd: Game metadata

    Usage:
        parser = MTGALogParser(config)

        if parser.is_available():
            for game_data in parser.iter_games():
                print(f"Game {game_data['match_id']}: {'won' if game_data['won'] else 'lost'}")

        trajectories = list(parser.iter_trajectories())
    """

    def __init__(self, config: MTGALogConfig | None = None):
        self.config = config or MTGALogConfig()

    def is_available(self) -> bool:
        """Check if MTGA log file exists."""
        return Path(self.config.log_path).exists()

    def iter_raw_messages(self) -> Iterator[dict]:
        """Iterate over JSON messages from the log file.

        MTGA logs interleave plain text and JSON objects. This method
        extracts the JSON messages.

        Yields:
            Parsed JSON dicts from the log
        """
        log_path = Path(self.config.log_path)
        if not log_path.exists():
            logger.warning("MTGA log not found at %s", log_path)
            return

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            buffer = ""
            brace_depth = 0
            in_json = False

            for line in f:
                if not in_json:
                    # Look for start of JSON object
                    idx = line.find("{")
                    if idx >= 0:
                        in_json = True
                        buffer = line[idx:]
                        brace_depth = buffer.count("{") - buffer.count("}")
                else:
                    buffer += line
                    brace_depth += line.count("{") - line.count("}")

                if in_json and brace_depth <= 0:
                    try:
                        data = json.loads(buffer)
                        yield data
                    except json.JSONDecodeError:
                        pass
                    buffer = ""
                    in_json = False
                    brace_depth = 0

    def iter_games(self) -> Iterator[dict]:
        """Group raw messages into distinct game sessions.

        Yields:
            Dict with match_id, game states, actions, and outcome
        """
        current_game: dict | None = None

        for msg in self.iter_raw_messages():
            # Detect game start
            if "matchGameRoomStateChangedEvent" in str(msg) or \
               msg.get("type") == "GRE_TO_CLIENT":

                # Check for match creation
                payload = msg.get("payload", msg)
                if "matchId" in payload and current_game is None:
                    current_game = {
                        "match_id": payload["matchId"],
                        "states": [],
                        "actions": [],
                        "won": None,
                    }

                # Collect game state messages
                if current_game is not None:
                    gre_type = payload.get("type", "")

                    if gre_type == "GREMessageType_GameStateMessage":
                        current_game["states"].append(payload)
                    elif gre_type == "GREMessageType_ActionsAvailableReq":
                        current_game["actions"].append(payload)

            # Detect game end
            if current_game and ("matchGameRoomStateChangedEvent" in str(msg)):
                result = msg.get("matchGameRoomStateChangedEvent", {}).get(
                    "gameRoomInfo", {}
                ).get("finalMatchResult", {})
                if result:
                    results = result.get("resultList", [])
                    for r in results:
                        if r.get("scope") == "MatchScope_Game":
                            current_game["won"] = r.get("winningTeamId") == r.get(
                                "teamId", -1
                            )

                    yield current_game
                    current_game = None

    def iter_trajectories(self):
        """Convert parsed games into Trajectory objects.

        TODO: Map MTGA game state messages to our GameState format,
        then tokenize and create proper Transition objects.

        Yields:
            Trajectory objects (simplified for now)
        """
        from ..trajectory import Trajectory, Transition

        for i, game_data in enumerate(self.iter_games()):
            traj = Trajectory(
                game_id=f"mtga_{game_data.get('match_id', i)}",
                winner=0 if game_data.get("won") else 1,
                num_turns=len(game_data.get("states", [])),
                source="mtga_logs",
                metadata={
                    "num_state_messages": len(game_data.get("states", [])),
                    "num_action_messages": len(game_data.get("actions", [])),
                },
            )

            # Convert MTGA game_data into minimal placeholder transitions.
            # Full conversion requires mapping MTGA internal IDs to structured GameState,
            # which is beyond this initial parser.
            action_dim = 136
            for idx, state_msg in enumerate(game_data.get("states", [])):
                transition = Transition(
                    state_features={
                        "raw_state": np.array([idx], dtype=np.float32),
                        "payload_length": np.array([len(str(state_msg))], dtype=np.float32),
                    },
                    action_encoding=np.zeros(action_dim, dtype=np.float32),
                    reward=0.0,
                    done=False,
                    action_type="MTGA_STATE",
                    metadata={"state_index": idx},
                )
                traj.add(transition)

            if len(traj.transitions) > 0:
                traj.transitions[-1].done = True
                yield traj

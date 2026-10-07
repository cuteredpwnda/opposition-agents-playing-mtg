"""
Comprehensive game logging for MTG agents.

Logs all game events with timestamps, decisions, and reasoning.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional
from enum import Enum

class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    DECISION = "DECISION"
    ACTION = "ACTION"
    RESULT = "RESULT"
    ERROR = "ERROR"


class GameLogger:
    """Comprehensive game event logger."""
    
    def __init__(self, game_id: str, verbose: bool = True):
        """Initialize logger.
        
        Args:
            game_id: ID of the game being logged
            verbose: Whether to print to console
        """
        self.game_id = game_id
        self.verbose = verbose
        self.logs: list[dict] = []
        self.start_time = datetime.now()
    
    def log(self,
            level: LogLevel,
            message: str,
            player: Optional[str] = None,
            turn: Optional[int] = None,
            phase: Optional[str] = None,
            data: Optional[dict] = None) -> None:
        """Log an event.
        
        Args:
            level: Log severity level
            message: Log message
            player: Player ID if applicable
            turn: Turn number if applicable
            phase: Game phase if applicable
            data: Additional data dict
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "game_id": self.game_id,
            "message": message,
            "player": player,
            "turn": turn,
            "phase": phase,
            "data": data or {}
        }
        
        self.logs.append(entry)
        
        if self.verbose:
            self._print_log(entry)
    
    def _print_log(self, entry: dict) -> None:
        """Pretty-print log entry."""
        timestamp = entry["timestamp"].split("T")[1][:8]
        level = entry["level"]
        
        # Color codes
        colors = {
            "DEBUG": "\033[90m",      # Gray
            "INFO": "\033[94m",       # Blue
            "DECISION": "\033[93m",   # Yellow
            "ACTION": "\033[92m",     # Green
            "RESULT": "\033[95m",     # Magenta
            "ERROR": "\033[91m"       # Red
        }
        reset = "\033[0m"
        
        color = colors.get(level, reset)
        
        # Build prefix
        prefix = f"[{timestamp}] [{level:<8}]"
        if entry["turn"]:
            prefix += f" [T{entry['turn']}]"
        if entry["phase"]:
            prefix += f" [{entry['phase']}]"
        if entry["player"]:
            prefix += f" {entry['player']}"
        
        print(f"{color}{prefix}: {entry['message']}{reset}")
        
        # Print additional data if present
        if entry["data"]:
            for key, value in entry["data"].items():
                data_str = str(value)
                if len(data_str) > 100:
                    data_str = data_str[:97] + "..."
                print(f"{color}    {key}: {data_str}{reset}")
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, msg, **kwargs)
    
    def decision(self, msg: str, **kwargs) -> None:
        """Log agent decision."""
        self.log(LogLevel.DECISION, msg, **kwargs)
    
    def action(self, msg: str, **kwargs) -> None:
        """Log game action."""
        self.log(LogLevel.ACTION, msg, **kwargs)
    
    def result(self, msg: str, **kwargs) -> None:
        """Log game result."""
        self.log(LogLevel.RESULT, msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log error."""
        self.log(LogLevel.ERROR, msg, **kwargs)
    
    def get_summary(self) -> str:
        """Get text summary of all logs."""
        summary = f"\n{'='*70}\n"
        summary += f"GAME LOG SUMMARY: {self.game_id}\n"
        summary += f"{'='*70}\n"
        summary += f"Total Events: {len(self.logs)}\n"
        summary += f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f}s\n\n"
        
        # Count by level
        level_counts = {}
        for entry in self.logs:
            level = entry["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        summary += "Events by Type:\n"
        for level, count in sorted(level_counts.items()):
            summary += f"  {level}: {count}\n"
        
        return summary
    
    def get_decision_chain(self) -> str:
        """Get text of all agent decisions in order."""
        decisions = [e for e in self.logs if e["level"] == "DECISION"]
        
        if not decisions:
            return "No decisions recorded"
        
        chain = f"\n{'='*70}\n"
        chain += "AGENT DECISION CHAIN\n"
        chain += f"{'='*70}\n"
        
        for i, decision in enumerate(decisions, 1):
            chain += f"\n{i}. Turn {decision['turn']} - {decision['player']} ({decision['phase']})\n"
            chain += f"   {decision['message']}\n"
            if decision['data']:
                for key, value in decision['data'].items():
                    chain += f"   - {key}: {value}\n"
        
        return chain

"""
Legacy Python rules engine.

⚠️  DEPRECATED: This module contains the original Python implementation of Magic's
comprehensive rules. It is maintained for backward compatibility and testing only.

For new code, use the phase-rs Rust engine instead:
  from src.integrations.phase_rs import run_game, run_game_sync

The Python engine is no longer used for gameplay but remains available for:
  - Unit testing of individual mechanics
  - Reference/validation during phase-rs development
  - Educational purposes

Migration path:
  - from src.engine.X → from src.engine_legacy.X (if needed)
  - Prefer src.integrations.phase_rs for new game orchestration
  - Use src.agents with phase-rs games

Last updated: 2026-05-20
"""


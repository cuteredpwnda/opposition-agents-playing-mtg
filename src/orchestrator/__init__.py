"""
Legacy Python game orchestration.

⚠️  DEPRECATED: This module contains the original Python implementation of
game runner and priority loop logic. It is maintained for backward compatibility
and testing only.

For new code, use the phase-rs bridge instead:
  from src.integrations.phase_rs import run_game, run_game_sync

The Python orchestrator is no longer used for gameplay but remains available for:
  - Legacy example scripts (see examples_legacy/)
  - Validation and comparison during phase-rs development
  - Educational purposes

Migration path:
  - from src.orchestrator.X → from src.orchestrator_legacy.X (if needed)
  - Prefer src.integrations.phase_rs.runner.run_game for new games
  - Use scripts/phase_rs_rollout_sweep.py for benchmarks

Last updated: 2026-05-20
"""


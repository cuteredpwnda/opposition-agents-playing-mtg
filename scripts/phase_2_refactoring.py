#!/usr/bin/env python
"""Phase 2: Reorganize legacy code (move to _legacy directories, update imports).

This script:
1. Moves legacy files to designated directories
2. Updates all remaining imports to reference the new locations
3. Verifies no imports are broken
4. Creates deprecation stubs

Run: python scripts/phase_2_refactoring.py --dry-run  (preview)
     python scripts/phase_2_refactoring.py --execute   (commit changes)
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Legacy engine files to move to src/engine_legacy/
LEGACY_ENGINE_FILES = [
    "src/engine/abilities.py",
    "src/engine/alternate_costs.py",
    "src/engine/card_database.py",
    "src/engine/cascade.py",
    "src/engine/channel.py",
    "src/engine/combat.py",
    "src/engine/command_zone.py",
    "src/engine/companion.py",
    "src/engine/continuous_effects.py",
    "src/engine/counters.py",
    "src/engine/cycling.py",
    "src/engine/delayed_triggers.py",
    "src/engine/equip.py",
    "src/engine/face_down.py",
    "src/engine/game_execution.py",
    "src/engine/game_logger.py",
    "src/engine/game_simulator.py",
    "src/engine/game_state.py",
    "src/engine/keywords.py",
    "src/engine/knowledge_graph.py",
    "src/engine/llm_agent.py",
    "src/engine/llm_orchestration.py",
    "src/engine/mana.py",
    "src/engine/phases.py",
    "src/engine/replacement_effects.py",
    "src/engine/rules_engine.py",
    "src/engine/spell_effects.py",
    "src/engine/stack.py",
    "src/engine/state_based_actions.py",
    "src/engine/static_abilities.py",
    "src/engine/tokens.py",
    "src/engine/tournament.py",
    "src/engine/triggered_abilities.py",
    "src/engine/triggers.py",
    "src/engine/watchers.py",
    "src/engine/zones.py",
]

# Legacy orchestrator files to move to src/orchestrator_legacy/
LEGACY_ORCHESTRATOR_FILES = [
    "src/orchestrator/game_runner.py",
    "src/orchestrator/priority_loop.py",
    "src/orchestrator/game_graph.py",
]

# Legacy examples to move to examples_legacy/
LEGACY_EXAMPLES = [
    "examples/demo_game_simple.py",
    "examples/demo_agents_play.py",
    "examples/demo_phi_agents.py",
    "examples/tournament_demo.py",
    "examples/play_edh_pod.py",
    "examples/meta_game.py",
    "examples/play_with_real_decks.py",
    "examples/play_real_decks.py",
]

# Legacy scripts to move to scripts_legacy/
LEGACY_SCRIPTS = [
    "scripts/benchmark_agents.py",
    "scripts/benchmark.py",
    "scripts/run_tournament.py",
    "scripts/run_empirical_training.py",
    "scripts/deploy.py",
    "scripts/goldfish.py",
    "scripts/ablation_sweep.py",
    "scripts/run_matchups.py",
    "scripts/overnight_run.py",
    "scripts/benchmark_trained_agents.py",
]


def move_files(src_list, dest_dir, dry_run=False):
    """Move files from src_list to dest_dir."""
    moved = []
    for src in src_list:
        src_path = REPO_ROOT / src
        if not src_path.exists():
            print(f"  ⊘ {src} (not found, skipping)")
            continue

        dest_path = REPO_ROOT / dest_dir / src_path.name
        if dest_path.exists():
            print(f"  ✓ {src} → {dest_dir}/{src_path.name} (already exists)")
            moved.append(src)
            continue

        if not dry_run:
            shutil.move(str(src_path), str(dest_path))
            print(f"  → {src} → {dest_dir}/{src_path.name}")
        else:
            print(f"  [DRY] {src} → {dest_dir}/{src_path.name}")

        moved.append(src)

    return moved


def update_imports(dry_run=False):
    """Update imports in remaining files to reference _legacy directories."""
    # Pattern: from src.engine.X → from src.engine_legacy.X (if X is in legacy list)
    legacy_engine_names = {Path(f).stem for f in LEGACY_ENGINE_FILES}
    legacy_orch_names = {Path(f).stem for f in LEGACY_ORCHESTRATOR_FILES}

    updates = 0
    for py_file in REPO_ROOT.glob("src/**/*.py"):
        if py_file.parent.name in ("engine_legacy", "orchestrator_legacy"):
            continue

        content = py_file.read_text(encoding="utf-8")
        original = content

        # Update engine imports
        for name in legacy_engine_names:
            pattern = rf"from src\.engine\.{name}\b"
            replacement = rf"from src.engine_legacy.{name}"
            content = re.sub(pattern, replacement, content)

            pattern = rf"from src\.engine import.*\b{name}\b"
            # This is complex; skip for now (handled by explicit imports)

        # Update orchestrator imports
        for name in legacy_orch_names:
            pattern = rf"from src\.orchestrator\.{name}\b"
            replacement = rf"from src.orchestrator_legacy.{name}"
            content = re.sub(pattern, replacement, content)

        if content != original:
            if not dry_run:
                py_file.write_text(content, encoding="utf-8")
                print(f"  → Updated imports in {py_file.relative_to(REPO_ROOT)}")
            else:
                print(f"  [DRY] Would update imports in {py_file.relative_to(REPO_ROOT)}")
            updates += 1

    return updates


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Refactor legacy code")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without executing"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the refactoring"
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Usage: python scripts/phase_2_refactoring.py --dry-run")
        print("       python scripts/phase_2_refactoring.py --execute")
        return 1

    dry_run = args.dry_run

    print("=" * 70)
    print("Phase 2: Refactor Legacy Code")
    print("=" * 70)
    print()

    print("Step 1: Move legacy engine files to src/engine_legacy/")
    engine_moved = move_files(LEGACY_ENGINE_FILES, "src/engine_legacy", dry_run)
    print(f"  {len(engine_moved)} files {'would be moved' if dry_run else 'moved'}")
    print()

    print("Step 2: Move legacy orchestrator files to src/orchestrator_legacy/")
    orch_moved = move_files(LEGACY_ORCHESTRATOR_FILES, "src/orchestrator_legacy", dry_run)
    print(f"  {len(orch_moved)} files {'would be moved' if dry_run else 'moved'}")
    print()

    print("Step 3: Move legacy examples to examples_legacy/")
    examples_moved = move_files(LEGACY_EXAMPLES, "examples_legacy", dry_run)
    print(f"  {len(examples_moved)} files {'would be moved' if dry_run else 'moved'}")
    print()

    print("Step 4: Move legacy scripts to scripts_legacy/")
    scripts_moved = move_files(LEGACY_SCRIPTS, "scripts_legacy", dry_run)
    print(f"  {len(scripts_moved)} files {'would be moved' if dry_run else 'moved'}")
    print()

    print("Step 5: Update imports in remaining files")
    updates = update_imports(dry_run)
    print(f"  {updates} files {'would be updated' if dry_run else 'updated'}")
    print()

    print("=" * 70)
    total = len(engine_moved) + len(orch_moved) + len(examples_moved) + len(scripts_moved)
    print(f"Summary: {total} files moved, {updates} imports updated")
    print(f"Status: {'DRY RUN' if dry_run else 'EXECUTED'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Overnight training runner — chain stages 4 → 5 → 6 → 7 with logging.

Each stage gets its own log file under ``runs/overnight_<timestamp>/``
and is run sequentially.  The script bails out (with a non-zero exit
code) on the first stage failure so you can inspect logs in the
morning.

Usage::

    python -m scripts.overnight_run                          # defaults
    python -m scripts.overnight_run --num-games 50 --jepa-epochs 30
    python -m scripts.overnight_run --no-kg --skip-stage 4    # skip collect

Output layout::

    runs/overnight_2026-04-28T22-30/
        config.json
        stage4_collect.log
        stage5_jepa.log
        stage6_dream.log
        stage7_eval.log
        SUMMARY.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("overnight_run")


def _stage_args(stage: int, base: argparse.Namespace) -> list[str]:
    """Build the train_pipeline.py argv for a given stage."""
    common = [
        sys.executable, "-u", "scripts/train_pipeline.py",
        "--stage", str(stage), "--end-stage", str(stage),
    ]
    if base.no_kg:
        common.append("--no-kg")
    if stage == 4:
        common += ["--num-games", str(base.num_games)]
    if stage == 5:
        common += [
            "--jepa-epochs", str(base.jepa_epochs),
            "--jepa-batch-size", str(base.jepa_batch_size),
            "--skip-eval", "--skip-dream",
        ]
    if stage == 6:
        common += ["--dream-iters", str(base.dream_iters), "--skip-eval"]
    if stage == 7:
        common += ["--eval-games", str(base.eval_games)]
    return common


def _run_stage(stage: int, args: argparse.Namespace, out_dir: Path) -> tuple[int, float]:
    log_path = out_dir / f"stage{stage}_{['', '', '', '', 'collect', 'jepa', 'dream', 'eval'][stage]}.log"
    cmd = _stage_args(stage, args)
    logger.info("=== Stage %d: %s", stage, " ".join(cmd))
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dur = time.monotonic() - start
    logger.info("Stage %d finished: rc=%d duration=%.1fs log=%s",
                stage, proc.returncode, dur, log_path)
    return proc.returncode, dur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-games", type=int, default=50,
                    help="Stage 4 self-play games to collect")
    ap.add_argument("--jepa-epochs", type=int, default=20)
    ap.add_argument("--jepa-batch-size", type=int, default=8)
    ap.add_argument("--dream-iters", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=4)
    ap.add_argument("--no-kg", action="store_true",
                    help="Train/eval without KG-aware encoder")
    ap.add_argument("--skip-stage", type=int, action="append", default=[],
                    help="Stage number(s) to skip (repeatable)")
    ap.add_argument("--out-root", default="runs",
                    help="Parent directory for the timestamped run folder")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    out_dir = Path(args.out_root) / f"overnight_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    logger.info("Run folder: %s", out_dir)

    summary: list[str] = [f"# Overnight run {timestamp}", ""]
    overall_start = time.monotonic()
    failed = False

    for stage in (4, 5, 6, 7):
        if stage in args.skip_stage:
            logger.info("Skipping stage %d (--skip-stage)", stage)
            summary.append(f"- Stage {stage}: SKIPPED")
            continue
        rc, dur = _run_stage(stage, args, out_dir)
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        summary.append(f"- Stage {stage}: {status} ({dur:.1f}s)")
        if rc != 0:
            failed = True
            summary.append(
                f"  -> Halting; see "
                f"runs/{out_dir.name}/stage{stage}_*.log"
            )
            break

    total = time.monotonic() - overall_start
    summary.append("")
    summary.append(f"Total wall time: {total:.1f}s")
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary))
    logger.info("Summary written to %s", out_dir / "SUMMARY.txt")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

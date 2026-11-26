"""Resumable multi-deck campaign over agents x decks x native-AI difficulty.

Each *cell* is one (arm, our deck, opponent deck, difficulty) combination run
as a ``run_phase_rs_ablation.py`` subprocess with ``--games-per-cell`` games.
Cells run in a worker pool; a cell with a ``summary.json`` is complete and is
skipped on rerun, so an interrupted campaign resumes where it stopped.

Arms are named presets (``--arms``). Arms ending in ``_learn`` append evidence
to the campaign's shared experience graph (``<campaign>/graph``); ``_frozen``
arms read it without writing. Cells are shuffled with ``--seed`` so learning
arms see decks in a random order rather than one deck at a time.

``collect`` writes ``results.jsonl`` (one row per game) for
``scripts/analyze_phase_rs_campaign.py``.

Example (Standard round-robin against native VeryEasy)::

    python scripts/run_phase_rs_campaign.py run --name standard_rr \\
        --format Standard --our-decks "data/decks/meta/standard/*.txt" \\
        --arms random heuristic kl kl_graph_learn --games-per-cell 2 --workers 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION = REPO_ROOT / "scripts" / "run_phase_rs_ablation.py"

KL = ["--picker", "kl_control", "--objective", "kl"]
HEURISTIC = ["--picker", "heuristic"]
ARMS: dict[str, list[str]] = {
    "random": ["--picker", "random"],
    "heuristic": HEURISTIC,
    "heuristic_legacy": [*HEURISTIC, "--activation-policy", "legacy"],
    "kl": KL,
    "kl_legacy": [*KL, "--activation-policy", "legacy"],
    "kl_graph_learn": [*KL, "--experience-mode", "learn", "--exploration", "thompson"],
    "kl_graph_novelty_learn": [*KL, "--experience-mode", "learn", "--exploration", "novelty"],
    "kl_graph_greedy_learn": [*KL, "--experience-mode", "learn", "--exploration", "greedy"],
    "kl_graph_frozen": [*KL, "--experience-mode", "read", "--exploration", "greedy"],
    "heuristic_graph_learn": [
        *HEURISTIC, "--experience-mode", "learn", "--exploration", "thompson",
    ],
    "heuristic_graph_frozen": [*HEURISTIC, "--experience-mode", "read", "--exploration", "greedy"],
}


@dataclass(frozen=True)
class Cell:
    arm: str
    our_deck: str
    opp_deck: str
    difficulty: str
    seed: int

    @property
    def cell_id(self) -> str:
        return "__".join(
            [self.arm, Path(self.our_deck).stem, Path(self.opp_deck).stem, self.difficulty]
        )


def _decks(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no decks match {pattern!r}")
    return [str(Path(p).resolve().relative_to(REPO_ROOT)) for p in paths]


def _matchup_seed(base: int, our: str, opp: str, difficulty: str) -> int:
    # Arms share picker seeds per matchup (common random numbers on our side;
    # the native shuffle and AI RNG are not seeded by the bridge).
    digest = hashlib.sha256(f"{base}|{our}|{opp}|{difficulty}".encode()).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def build_cells(args: argparse.Namespace) -> list[Cell]:
    ours = _decks(args.our_decks)
    opps = _decks(args.opp_decks or args.our_decks)
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    cells = [
        Cell(arm, our, opp, difficulty, _matchup_seed(args.seed, our, opp, difficulty))
        for difficulty in args.difficulties
        for our in ours
        for opp in opps
        if args.include_mirrors or Path(our).stem != Path(opp).stem
        for arm in args.arms
    ]
    random.Random(args.seed).shuffle(cells)
    return cells


def _cell_dir(root: Path, cell: Cell) -> Path:
    return root / "cells" / cell.cell_id


def _complete(cell_dir: Path) -> bool:
    return any(cell_dir.glob("*/summary.json"))


def _run_cell(
    cell: Cell, root: Path, args: argparse.Namespace, uri: str,
) -> tuple[Cell, int, float]:
    cell_dir = _cell_dir(root, cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ABLATION), *ARMS[cell.arm],
        "--uri", uri,
        "--games", str(args.games_per_cell),
        "--seed", str(cell.seed),
        "--format", args.format,
        "--our-deck-file", cell.our_deck,
        "--ai-deck-file", cell.opp_deck,
        "--ai-difficulty", cell.difficulty,
        "--max-game-seconds", str(args.max_game_seconds),
        "--output-dir", str(cell_dir),
    ]
    if "--experience-mode" in ARMS[cell.arm]:
        graph_root = Path(args.graph_root) if args.graph_root else root / "graph"
        if args.graph_per_arm:
            graph_root = graph_root / cell.arm
        command += ["--experience-root", str(graph_root)]
    started = time.perf_counter()
    with (cell_dir / "stdout.log").open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        try:
            code = subprocess.run(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.cell_timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -1
            log.write("cell timeout\n")
    return cell, code, time.perf_counter() - started


def run(args: argparse.Namespace) -> int:
    root = Path(args.output_root) / args.name
    root.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args)
    manifest_path = root / "campaign.json"
    manifest = {
        "schema": "phase_rs_campaign_v1",
        "created": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (Python 3.10)
        "argv": sys.argv[1:],
        "config": vars(args),
        "arms": {arm: ARMS[arm] for arm in args.arms},
        "cells": len(cells),
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip(),
        "seed_scope": "picker seeds only; native shuffle and AI RNG are not seeded",
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending = [cell for cell in cells if not _complete(_cell_dir(root, cell))]
    print(f"{len(cells)} cells, {len(cells) - len(pending)} already complete", flush=True)
    uris = args.uri
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_cell, cell, root, args, uris[i % len(uris)])
            for i, cell in enumerate(pending)
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            cell, code, elapsed = future.result()
            ok = _complete(_cell_dir(root, cell))
            failures += not ok
            print(
                f"[{done}/{len(pending)}] {cell.cell_id}: exit={code} "
                f"{'ok' if ok else 'FAILED'} {elapsed:.0f}s",
                flush=True,
            )
    collect(root)
    return 1 if failures else 0


def collect(root: Path) -> Path:
    """Flatten every completed cell into ``results.jsonl``."""
    manifest = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    rows = []
    for cell_dir in sorted((root / "cells").iterdir()):
        arm, our, opp, difficulty = cell_dir.name.split("__")
        for summary in sorted(cell_dir.glob("*/summary.json")):
            run_dir = summary.parent
            finished = summary.stat().st_mtime
            for line in (run_dir / "games.jsonl").read_text(encoding="utf-8").splitlines():
                game = json.loads(line)
                rows.append({
                    "arm": arm, "our_deck": our, "opp_deck": opp, "difficulty": difficulty,
                    "outcome": game["outcome"], "reason": game["reason"],
                    "turns": game["turns"], "actions": game["actions"],
                    "elapsed_sec": game["elapsed_sec"], "seed": game["seed"],
                    "cell_finished": finished, "run_dir": run_dir.name,
                    "format": manifest["config"]["format"],
                })
    out = root / "results.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"collected {len(rows)} games -> {out}", flush=True)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--output-root", default="runs/campaigns")
    run_parser.add_argument("--format", default="Standard")
    run_parser.add_argument("--our-decks", required=True, help="Glob for our deck files.")
    run_parser.add_argument("--opp-decks", help="Glob for opponent decks (default: --our-decks).")
    run_parser.add_argument("--include-mirrors", action="store_true")
    run_parser.add_argument("--arms", nargs="+", required=True)
    run_parser.add_argument("--difficulties", nargs="+", default=["VeryEasy"],
                            choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"])
    run_parser.add_argument("--games-per-cell", type=int, default=2)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--uri", nargs="+", default=["ws://127.0.0.1:9474/ws"])
    run_parser.add_argument("--seed", type=int, default=2026)
    run_parser.add_argument("--max-game-seconds", type=float, default=600)
    run_parser.add_argument("--cell-timeout", type=float, default=3600)
    run_parser.add_argument(
        "--graph-root",
        help="Experience-graph directory (default <campaign>/graph); point frozen arms "
             "at a previous campaign's graph to evaluate what it learned.",
    )
    run_parser.add_argument(
        "--graph-per-arm", action="store_true",
        help="Give each experience arm its own graph (exploration-mode comparisons).",
    )
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "collect":
        collect(Path(args.root))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

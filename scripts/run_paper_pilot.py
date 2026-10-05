"""Bounded, sequential feasibility campaign; not a publication-strength benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.integrations.phase_rs import PhaseServerProcess
from src.integrations.phase_rs.server_process import DEFAULT_SUBMODULE, REPO_ROOT, _binary_name
from src.integrations.scryfall_bulk import file_sha256

ARMS = ("random", "heuristic", "kl", "efe_infogain", "efe_ambiguity")
COMMANDER_DECKS = (
    "krenko-mob-boss_core.txt", "atraxa-praetors-voice_core.txt",
    "meren-of-clan-nel-toth_core.txt", "urza-lord-high-artificer_core.txt",
)
MODERN_DIVERSE_DECKS = (
    Path("data") / "decks" / "modern" / "modern_mono_red_burn.txt",
    Path("data") / "decks" / "benchmark" / "modern_green_stompy.txt",
    Path("data") / "decks" / "benchmark" / "modern_azorius_control.txt",
)


@dataclass(frozen=True)
class Matchup:
    host: Path
    opponents: tuple[Path, ...]

    def record(self) -> dict[str, object]:
        return {"host": str(self.host), "opponents": [str(p) for p in self.opponents]}


def build_matchups(args: argparse.Namespace) -> list[Matchup]:
    pool = getattr(args, "deck_pool", "default")
    custom = getattr(args, "deck_files", None)
    mode = getattr(args, "matchups", "mirrors")
    if args.format == "Commander":
        if pool != "default" or custom or mode != "mirrors":
            raise ValueError("Deck pools and matchup modes currently require --format Modern")
        paths = [REPO_ROOT / "data" / "decks" / "edh" / p for p in COMMANDER_DECKS]
        return [Matchup(paths[0], tuple(paths[1:]))]
    paths = (
        [Path(p).resolve() for p in custom] if custom
        else [REPO_ROOT / p for p in MODERN_DIVERSE_DECKS] if pool == "modern-diverse"
        else [REPO_ROOT / MODERN_DIVERSE_DECKS[0]]
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Deck pool contains duplicate paths")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Deck file does not exist: {path}")
    if mode == "round-robin":
        return [Matchup(host, (opponent,)) for host in paths for opponent in paths]
    return [Matchup(path, (path,)) for path in paths]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3, help="Replicates per arm and matchup")
    parser.add_argument("--format", choices=["Modern", "Commander"], default="Modern")
    decks = parser.add_mutually_exclusive_group()
    decks.add_argument("--deck-pool", choices=["default", "modern-diverse"], default="default")
    decks.add_argument("--deck-files", nargs="+", type=Path, help="Explicit Modern deck pool")
    parser.add_argument("--matchups", choices=["mirrors", "round-robin"], default="mirrors",
                        help="Round-robin includes mirrors and both host/opponent orientations")
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        help="Eligible policy subset; Commander supports random/heuristic only")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "paper_pilot")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if (
        args.games < 1 or not math.isfinite(args.budget_seconds)
        or not 0 < args.budget_seconds <= 1800
    ):
        raise ValueError("games must be positive and budget-seconds must be in (0, 1800]")
    matchups = build_matchups(args)
    arms = getattr(args, "arms", None) or (
        ARMS if args.format == "Modern" else ("random", "heuristic")
    )
    if len(set(arms)) != len(arms) or (
        args.format == "Commander" and set(arms) - {"random", "heuristic"}
    ):
        raise ValueError("Policy subset must be unique and supported by the chosen format")
    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    data_paths = [
        DEFAULT_SUBMODULE / "data" / "card-data.json",
        REPO_ROOT / "data" / "scryfall" / "oracle-cards.json",
        REPO_ROOT / "data" / "scryfall" / "rulings.json",
        REPO_ROOT / "data" / "scryfall" / "by_name.json",
    ]
    snapshots = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in data_paths}
    source_manifest = REPO_ROOT / "data" / "scryfall" / "snapshot_manifest.json"
    if not source_manifest.exists():
        raise ValueError("Refresh Scryfall first to create snapshot_manifest.json")
    source = json.loads(source_manifest.read_text("utf-8"))
    for path in data_paths[1:]:
        if source["files"][path.name]["sha256"] != snapshots[str(path.relative_to(REPO_ROOT))]:
            raise ValueError(f"Snapshot manifest mismatch: {path.name}")
    if source["files"]["by_name.json"]["oracle_sha256"] != source["files"]["oracle-cards.json"][
        "sha256"
    ]:
        raise ValueError("Name index does not derive from the frozen oracle snapshot")
    binary = DEFAULT_SUBMODULE / "target" / "release" / _binary_name()
    if not binary.exists():
        raise ValueError("Build the pinned release server before running a paper pilot")
    binary_hash = file_sha256(binary)
    source_paths = [
        *(REPO_ROOT / "src").rglob("*.py"),
        REPO_ROOT / "scripts" / "run_phase_rs_ablation.py",
        REPO_ROOT / "scripts" / "run_paper_pilot.py",
        REPO_ROOT / "src" / "integrations" / "phase_rs" / "format_defaults.json",
    ]
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sorted(source_paths)
    }
    deck_paths = sorted({path for m in matchups for path in (m.host, *m.opponents)})
    deck_hashes = {
        str(path): file_sha256(path) for path in deck_paths
    }
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 3, "purpose": "feasibility; not powered comparative evidence",
        "arguments": vars(args),
        "matchups": [m.record() for m in matchups], "arms": list(arms),
        "data_sha256": snapshots, "scryfall_snapshot": source,
        "source_sha256": source_hashes, "server_binary_sha256": binary_hash,
        "deck_file_sha256": deck_hashes,
        "python": sys.version,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("websockets", "numpy", "httpx")
        },
        "limits": [
            "Python picker seeds only; no paired native RNG replay or controlled seat rotation",
            "KL horizon planning versus one-step EFE; policy-stack comparison",
            "Only one Python-controlled seat; opponents are native phase-ai",
            "Tev1/chat/KG/JEPA omitted until their respective scientific gates pass",
            "Pilot timing is exploratory; hardware isolation and warm-up are not controlled",
        ],
    }, indent=2, default=str), encoding="utf-8")
    deadline = time.monotonic() + args.budget_seconds
    cells = []
    schedule = [
        (index, matchup_index, arm) for index in range(args.games)
        for matchup_index in range(len(matchups)) for arm in arms
    ]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="paper-pilot-db-") as temporary:
        with PhaseServerProcess(
            port=port, binary=binary,
            extra_env={"PHASE_GAMES_DB": str(Path(temporary) / "games.db"),
                                 "RUST_LOG": "warn"},
        ) as server, (output / "campaign.log").open("w", encoding="utf-8") as log:
            if server._adopted:  # noqa: SLF001
                raise ValueError("Pilot requires an owned server; refusing an adopted instance")
            for index, matchup_index, arm in schedule:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                matchup = matchups[matchup_index]
                cell_dir = output / f"{index:03d}_m{matchup_index:03d}_{arm}"
                cli = [
                    sys.executable, "-m", "scripts.run_phase_rs_ablation",
                    "--games", "1", "--format", args.format, "--ai-difficulty", "VeryEasy",
                    "--seed", str(args.seed + index), "--uri", server.uri,
                    "--max-retries", "1", "--max-actions", "1500",
                    "--max-game-seconds", str(min(240, remaining)),
                    "--max-message-mib", "16", "--stream-timeout", "60",
                    "--output-dir", str(cell_dir),
                ]
                cli += ["--our-deck-file", str(matchup.host)]
                for path in matchup.opponents:
                    cli += ["--ai-deck-file", str(path)]
                cli += (
                    ["--picker", "kl_control", "--objective", arm]
                    if arm in {"kl", "efe_infogain", "efe_ambiguity"}
                    else ["--picker", arm]
                )
                print(f"pilot {len(cells) + 1}/{len(schedule)}: matchup {matchup_index} {arm}",
                      flush=True)
                try:
                    process = subprocess.run(
                        cli, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        timeout=remaining, check=False,
                    )
                    status = "finished" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "campaign_timeout"
                    log.write(f"\nCampaign wall-clock budget exhausted while running {arm}\n")
                    process = None
                summaries = (
                    [directory / "summary.json" for directory in cell_dir.iterdir()
                     if directory.is_dir() and (directory / "summary.json").exists()]
                    if cell_dir.exists() else []
                )
                if len(summaries) > 1:
                    raise ValueError(f"Ambiguous pilot output in {cell_dir}")
                result_summary = json.loads(summaries[0].read_text("utf-8")) if summaries else None
                if status == "finished" and (
                    result_summary is None or result_summary.get("completed_games") != 1
                ):
                    status = "invalid_output"
                    log.write(f"\n{arm}: successful exit without one verified completed game\n")
                cell = {
                    "arm": arm, "seed": args.seed + index, "status": status,
                    "matchup_index": matchup_index, "matchup": matchup.record(),
                    "returncode": process.returncode if process is not None else None,
                    "summary": result_summary,
                }
                cells.append(cell)
                with (output / "cells.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(cell) + "\n")
                if status == "campaign_timeout":
                    break
    frozen_files = {
        path: snapshots[str(path.relative_to(REPO_ROOT))] for path in data_paths
    }
    frozen_files.update({REPO_ROOT / name: digest for name, digest in source_hashes.items()})
    frozen_files.update({Path(name): digest for name, digest in deck_hashes.items()})
    frozen_files[binary] = binary_hash
    changed_inputs = [
        str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        for path, digest in frozen_files.items()
        if not path.is_file() or file_sha256(path) != digest
    ]
    unchanged = not changed_inputs
    summary = {
        "purpose": "functional pilot, not a powered strength comparison",
        "format": args.format, "scheduled_cells": len(schedule), "attempted_cells": len(cells),
        "unstarted_cells": len(schedule) - len(cells),
        "completed_games": sum((c["summary"] or {}).get("completed_games", 0) for c in cells),
        "incomplete_cells": sum(c["status"] != "finished" for c in cells),
        "input_freeze_unchanged": unchanged,
        "invalid_reason": None if unchanged else "input_freeze_changed",
        "changed_inputs": changed_inputs,
        "cells": cells,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pilot outputs: {output}", flush=True)
    if not unchanged:
        print("Input freeze changed; this pilot is invalid for comparative claims.", flush=True)
    return 0 if unchanged and len(cells) == len(schedule) and not summary["incomplete_cells"] else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

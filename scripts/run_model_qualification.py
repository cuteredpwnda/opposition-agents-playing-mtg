"""Bounded native model-stack feasibility checks, not a strength comparison."""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from src.integrations.phase_rs import PhaseServerProcess
from src.integrations.phase_rs.decision2_picker import add_decision2_arguments
from src.integrations.phase_rs.server_process import REPO_ROOT
from src.integrations.scryfall_bulk import file_sha256


def model_snapshot(schedule: list[tuple[str, list[str]]]) -> dict[str, dict]:
    names = sorted({
        options[index + 1]
        for _, options in schedule for index, option in enumerate(options[:-1])
        if option in {"--ollama-model", "--tev1-model"}
    })
    if not names:
        return {}
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = {row["name"]: row for row in json.load(response)["models"]}
    snapshot = {}
    for name in names:
        if name not in tags:
            raise ValueError(f"Qualification model is not installed: {name}")
        request = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            metadata = json.load(response)
        snapshot[name] = {
            "digest": tags[name]["digest"],
            "parameters": metadata.get("parameters", ""),
            "details": metadata.get("details", {}),
        }
    return snapshot


def conditions(
    checkpoint: Path, models: list[str], tev1_model: str = "tev1:0.8b",
) -> list[tuple[str, list[str]]]:
    rows = [("random", ["--picker", "random"]),
            ("heuristic", ["--picker", "heuristic"])]
    rows.extend((f"chat_{index}", ["--picker", "ollama", "--ollama-model", model])
                for index, model in enumerate(models))
    rows.append(("tev1", ["--picker", "tev1", "--tev1-model", tev1_model,
                         "--tev1-timeout", "30"]))
    for mode in ("direct", "dream_search"):
        rows.append((f"world_model_{mode}", [
            "--picker", "agent:world_model", "--agent-checkpoint", str(checkpoint),
            "--agent-mode", mode, "--agent-deterministic",
            "--dream-rollouts", "2", "--dream-depth", "3",
        ]))
    rows.append(("fusion", [
        "--picker", "agent:fusion", "--agent-checkpoint", str(checkpoint),
        "--ollama-model", models[0], "--dream-rollouts", "2", "--dream-depth", "3",
    ]))
    for objective in ("kl", "efe_infogain", "efe_ambiguity"):
        rows.append((objective, ["--picker", "kl_control", "--objective", objective]))
    return rows


def run(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_file():
        raise ValueError(f"Explicit checkpoint missing: {args.checkpoint}")
    if not args.models or len(set(args.models)) != len(args.models):
        raise ValueError("Supply at least one unique installed chat model")
    if (
        not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 1800
        or not math.isfinite(args.condition_seconds) or args.condition_seconds <= 0
    ):
        raise ValueError("Campaign budget must be in (0, 1800]; condition budget positive")
    if args.games < 1:
        raise ValueError("games must be positive")
    schedule = conditions(args.checkpoint, args.models, args.tev1_model)
    if args.include_decision2:
        schedule.append(("decision2", [
            "--picker", "decision2", "--decision2-model", args.decision2_model,
            "--decision2-revision", args.decision2_revision,
            "--decision2-candidates", str(args.decision2_candidates),
            "--decision2-device", args.decision2_device,
            "--decision2-threads", str(args.decision2_threads),
        ]))
    if args.only:
        selected = set(args.only)
        if len(selected) != len(args.only) or selected - {name for name, _ in schedule}:
            raise ValueError("--only must name unique conditions from the configured schedule")
        schedule = [(name, options) for name, options in schedule if name in selected]
    models = model_snapshot(schedule)
    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    deck = REPO_ROOT / "data" / "decks" / "modern" / "modern_mono_red_burn.txt"
    frozen = {
        path: file_sha256(path) for path in (
            args.checkpoint, deck, *(REPO_ROOT / "src").rglob("*.py"),
            REPO_ROOT / "scripts" / "run_phase_rs_ablation.py",
            REPO_ROOT / "scripts" / "run_model_qualification.py",
        )
    }
    manifest = {
        "purpose": "feasibility only, not qualified native training or causal ablation",
        "arguments": vars(args), "schedule": schedule,
        "model_snapshot": models,
        "file_sha256": {str(p): digest for p, digest in frozen.items()},
        "blocked_combinations": [
            "Tev1+world-model critic/prior: no qualified integration",
            "KG/induced-memory ablations: runtime grounding and component activation unqualified",
            "JEPA-driven dreaming: current dreaming uses recurrent dynamics, not JEPA head",
            "objective-only KL/EFE comparison: planning budgets differ",
        ],
        "limitations": [
            "Archived legacy checkpoint, not native-trained dynamics",
            "Permissive checkpoint loader and chat/fusion fallback need activation audit",
            "Native RNG not seeded; no matched-context comparison",
            "Per-game time limit is cooperative; subprocess deadline bounds stuck inference",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="model-pilot-db-") as temporary:
        with PhaseServerProcess(
            port=port, extra_env={"PHASE_GAMES_DB": str(Path(temporary) / "games.db"),
                                 "RUST_LOG": "warn"},
        ) as server:
            if server._adopted:  # noqa: SLF001
                raise ValueError("Qualification requires an owned private server")
            rows = run_conditions(args, schedule, output, deck, server.uri)
    changed_models = model_snapshot(schedule) != models
    changed = [str(p) for p, digest in frozen.items()
               if not p.is_file() or file_sha256(p) != digest]
    summary = {
        "scheduled_conditions": len(schedule), "attempted_conditions": len(rows),
        "unstarted_conditions": len(schedule) - len(rows),
        "completed_games": sum((row["summary"] or {}).get("completed_games", 0)
                               for row in rows),
        "failed_conditions": sum(row["status"] != "finished" for row in rows),
        "changed_inputs": changed, "conditions": rows,
        "model_snapshot_changed": changed_models,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Qualification outputs: {output}", flush=True)
    return int(bool(changed or changed_models or len(rows) != len(schedule)
                    or summary["failed_conditions"]))


def run_conditions(args, schedule, output: Path, deck: Path, uri: str) -> list[dict]:
    deadline = time.monotonic() + args.budget_seconds
    rows = []
    for name, options in schedule:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        cell = output / name
        cell.mkdir()
        limit = min(remaining, args.condition_seconds)
        cli = [
            sys.executable, "-m", "scripts.run_phase_rs_ablation", *options,
            "--our-deck-file", str(deck), "--ai-deck-file", str(deck),
            "--format", "Modern", "--ai-difficulty", args.ai_difficulty,
            "--games", str(args.games),
            "--seed", str(args.seed), "--max-retries", "1", "--max-actions", "1500",
            "--max-game-seconds", str(max(1, (limit - 15) / args.games)),
            "--stream-timeout", "45", "--output-dir", str(cell), "--uri", uri,
        ]
        print(f"qualification {len(rows)+1}/{len(schedule)}: {name}", flush=True)
        started = time.monotonic()
        with (cell / "process.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    cli, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                    timeout=limit, check=False,
                )
                status = "finished" if process.returncode == 0 else "failed"
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                status, returncode = "subprocess_timeout", None
        summaries = list(cell.glob("*/summary.json"))
        if len(summaries) > 1:
            raise ValueError(f"Ambiguous condition output: {cell}")
        summary = json.loads(summaries[0].read_text("utf-8")) if summaries else None
        if status == "finished" and (
            not summary or summary.get("completed_games") != args.games
        ):
            status = "invalid_output"
        row = {
            "condition": name, "options": options, "status": status,
            "returncode": returncode, "elapsed_sec": time.monotonic() - started,
            "summary": summary,
        }
        rows.append(row)
        with (output / "conditions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["llama3.2:1b", "gemma4:e2b"])
    parser.add_argument("--tev1-model", default="tev1:0.8b",
                        help="Installed decision-model tag, including explicit context aliases")
    parser.add_argument("--include-decision2", action="store_true",
                        help="Add the pinned local Decision 2.0 classifier condition")
    add_decision2_arguments(parser)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--games", type=int, default=1,
                        help="Replicates per condition within the shared condition deadline")
    parser.add_argument("--ai-difficulty", default="VeryEasy",
                        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard", "CEDH"])
    parser.add_argument("--only", nargs="+", help="Re-run selected named conditions")
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--condition-seconds", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "model_qualification")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

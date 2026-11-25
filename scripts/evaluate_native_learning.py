"""Resumable evaluation of frozen native-learning checkpoints and graph controls."""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

from scripts.run_native_learning import (
    CURATED_TEST_DECK,
    TEST_DECK,
)
from scripts.run_phase_rs_ablation import GameRow, _outcome, _summarize
from src.integrations.phase_rs import (
    PhaseServerConfig,
    PhaseServerProcess,
    load_deck_data,
    run_game_sync,
)
from src.integrations.phase_rs.native_learning_data import FEATURE_VERSION
from src.integrations.phase_rs.server_process import REPO_ROOT
from src.integrations.scryfall_bulk import file_sha256
from src.world_model.native_learning import NativeLearnedPicker, shuffled_graph_identity

logger = logging.getLogger(__name__)
DECK_STRATA = (("azorius", TEST_DECK), ("hardened_scales", CURATED_TEST_DECK))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_campaign(campaign: Path) -> tuple[dict, dict]:
    manifest_path = campaign / "manifest.json"
    summary_path = campaign / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("Campaign requires manifest.json and summary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_development_pipeline":
        raise ValueError("Only a completed native-learning campaign can be evaluated")
    if summary.get("changed_inputs") != []:
        raise ValueError("Campaign recorded changed frozen inputs")
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Campaign feature version is incompatible with this evaluator")
    changed = [
        path for path, expected in manifest["file_sha256"].items()
        if not Path(path).is_file() or file_sha256(Path(path)) != expected
    ]
    if changed:
        raise ValueError(f"Campaign frozen inputs have changed: {changed}")
    return manifest, summary


def graph_arms(campaign: Path, training_seed: int, picker_seed: int) -> dict:
    curated = torch.load(
        campaign / "curated_graph.pt", map_location="cpu", weights_only=False,
    )
    induced = torch.load(
        campaign / f"seed_{training_seed}" / "graph.pt",
        map_location="cpu", weights_only=False,
    )
    combined = torch.load(
        campaign / f"seed_{training_seed}" / "curated_plus_induced_graph.pt",
        map_location="cpu", weights_only=False,
    )
    return {
        "none": (None, False),
        "curated": (curated, True),
        "induced": (induced, True),
        "curated_plus_induced": (combined, True),
        "identity_shuffled": (
            shuffled_graph_identity(combined, picker_seed + training_seed + 20000), True,
        ),
    }


def attempt_paths(output: Path, arm: str, game_index: int) -> list[Path]:
    return sorted(
        output.glob(f"{arm}_game{game_index:03d}_attempt*.json"),
        key=lambda path: int(path.stem.rsplit("attempt", 1)[1]),
    )


def campaign_server_binary(manifest: dict) -> Path:
    from src.integrations.phase_rs.server_process import _binary_name

    candidates = [
        Path(path) for path in manifest["file_sha256"]
        if Path(path).name == _binary_name()
    ]
    if len(candidates) != 1:
        raise ValueError("Campaign must identify exactly one frozen phase-server binary")
    return candidates[0]


def summarize_attempts(output: Path, arms: set[str]) -> dict:
    summaries = {}
    for arm in sorted(arms):
        rows = []
        for path in sorted(output.glob(f"{arm}_game*_attempt*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(GameRow(**payload["row"]))
        if rows:
            summaries[arm] = _summarize(rows)
    return summaries


def evaluate(args: argparse.Namespace) -> Path:
    campaign = args.campaign.resolve()
    manifest, _ = load_campaign(campaign)
    configured_seeds = manifest["arguments"]["training_seeds"]
    seeds = args.training_seeds or configured_seeds
    if not set(seeds) <= set(configured_seeds):
        raise ValueError("Requested training seeds are not present in this campaign")
    if args.games_per_cell < 1 or args.threads < 1 or args.max_actions < 1:
        raise ValueError("Game, thread and action budgets must be positive")
    if not math.isfinite(args.max_game_seconds) or args.max_game_seconds <= 0:
        raise ValueError("max-game-seconds must be finite and positive")
    conditions = (
        "none", "curated", "induced", "curated_plus_induced", "identity_shuffled",
    )
    available_arms = {
        f"seed{seed}_{weights}_{condition}_{stratum}"
        for seed in seeds for weights in ("initial", "trained")
        for condition in conditions for stratum, _ in DECK_STRATA
    }
    if args.arms and not set(args.arms) <= available_arms:
        raise ValueError(f"Unknown evaluation arm(s): {sorted(set(args.arms) - available_arms)}")

    output = args.resume_dir.resolve() if args.resume_dir else (
        args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=args.resume_dir is not None)
    torch.set_num_threads(args.threads)
    hashes = {
        str(path): file_sha256(path)
        for path in (
            campaign / "manifest.json", campaign / "curated_graph.pt",
            *(campaign / f"seed_{seed}" / filename for seed in seeds
              for filename in ("initial.pt", "trained.pt", "graph.pt",
                               "curated_plus_induced_graph.pt")),
            Path(__file__), TEST_DECK, CURATED_TEST_DECK,
            args.opponent_deck,
        )
    }
    schedule = {
        "campaign": str(campaign),
        "campaign_manifest_sha256": hashes[str(campaign / "manifest.json")],
        "feature_version": FEATURE_VERSION,
        "training_seeds": seeds,
        "games_per_cell": args.games_per_cell,
        "picker_seed": args.seed,
        "max_game_seconds": args.max_game_seconds,
        "max_actions": args.max_actions,
        "selection": args.selection,
        "arms": sorted(args.arms) if args.arms else "all",
        "ai_difficulty": args.ai_difficulty,
        "opponent_deck": str(args.opponent_deck),
        "input_sha256": hashes,
    }
    schedule_path = output / "schedule.json"
    if schedule_path.exists():
        old = json.loads(schedule_path.read_text(encoding="utf-8"))
        if old != schedule:
            raise ValueError("Resume schedule does not match frozen inputs or arguments")
    else:
        write_json(schedule_path, schedule)
    progress_path = output / "progress.json"
    if not progress_path.exists():
        write_json(progress_path, {"status": "in_progress", "completed_attempts": 0})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    completed_attempts = 0
    with tempfile.TemporaryDirectory(prefix="native-learning-eval-db-") as temporary:
        with PhaseServerProcess(
            port=port, binary=campaign_server_binary(manifest),
            extra_env={
                "PHASE_GAMES_DB": str(Path(temporary) / "games.db"),
                "RUST_LOG": "warn",
            },
        ) as server:
            if server._adopted:  # noqa: SLF001
                raise ValueError("Evaluation requires an owned private phase server")
            for training_seed in seeds:
                controls = graph_arms(campaign, training_seed, args.seed)
                for weights in ("initial", "trained"):
                    checkpoint = campaign / f"seed_{training_seed}" / f"{weights}.pt"
                    for condition, (graph, enabled) in controls.items():
                        for stratum, deck_path in DECK_STRATA:
                            arm = f"seed{training_seed}_{weights}_{condition}_{stratum}"
                            if args.arms and arm not in args.arms:
                                continue
                            for game_index in range(1, args.games_per_cell + 1):
                                paths = attempt_paths(output, arm, game_index)
                                if paths:
                                    last = json.loads(paths[-1].read_text(encoding="utf-8"))
                                    if (
                                        not args.retry_incomplete
                                        or last["row"]["outcome"] != "incomplete"
                                    ):
                                        continue
                                    attempt = int(paths[-1].stem.rsplit("attempt", 1)[1]) + 1
                                else:
                                    attempt = 1
                                game_seed = args.seed + 10000 + game_index - 1
                                if attempt > 1:
                                    game_seed += (attempt - 1) * 1_000_000
                                picker = NativeLearnedPicker(
                                    checkpoint, enabled, seed=game_seed,
                                    selection=args.selection, graph_override=graph,
                                )
                                started = time.monotonic()
                                result = run_game_sync(
                                    deck=load_deck_data(deck_path), picker=picker,
                                    config=PhaseServerConfig(
                                        uri=server.uri, stream_timeout_s=45,
                                        expected_server_commit=manifest["phase_rs_commit"],
                                    ),
                                    ai_difficulty=args.ai_difficulty,
                                    ai_decks=[load_deck_data(args.opponent_deck)],
                                    format_name="Modern", max_actions=args.max_actions,
                                    max_game_seconds=args.max_game_seconds,
                                )
                                row = GameRow(
                                    game_index=game_index, winner_seat=result.winner_seat,
                                    our_seat=result.our_seat, reason=result.reason,
                                    turns=result.turns_observed, actions=result.actions_sent,
                                    seed=game_seed, objective=None, outcome=_outcome(result),
                                    attempts=1, elapsed_sec=time.monotonic() - started,
                                    decision_latencies_sec=[
                                        event["decision_time_sec"] for event in result.trace
                                        if "decision_time_sec" in event
                                    ],
                                )
                                result_path = output / (
                                    f"{arm}_game{game_index:03d}_attempt{attempt:03d}.json"
                                )
                                write_json(result_path, {
                                    "row": vars(row), "trace": result.trace,
                                    "attempt": attempt, "arm": arm, "deck_path": str(deck_path),
                                    "graph_condition": condition,
                                    "graph_sha256": picker.graph_digest,
                                    "checkpoint_sha256": hashes[str(checkpoint)],
                                })
                                completed_attempts += 1
                                write_json(progress_path, {
                                    "status": "in_progress",
                                    "completed_attempts": completed_attempts,
                                    "last_result": result_path.name,
                                })
                                logger.info(
                                    "%s game %d attempt %d: %s",
                                    arm, game_index, attempt, result.reason,
                                )
                                write_json(
                                    output / "evaluation_summary.json",
                                    summarize_attempts(output, {
                                        f"seed{seed}_{weight}_{control}_{deck}"
                                        for seed in seeds
                                        for weight in ("initial", "trained")
                                        for control in controls
                                        for deck, _ in DECK_STRATA
                                    }),
                                )
    changed = [
        path for path, expected in hashes.items()
        if not Path(path).is_file() or file_sha256(Path(path)) != expected
    ]
    if changed:
        raise ValueError(f"Evaluation inputs changed during execution: {changed}")
    write_json(progress_path, {
        "status": "completed",
        "completed_attempts": len(list(output.glob("*_attempt*.json"))),
        "evaluation_summary": "evaluation_summary.json",
        "changed_inputs": changed,
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument("--games-per-cell", type=int, default=1)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--ai-difficulty", choices=["VeryEasy", "Easy", "Medium"],
                        default="VeryEasy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-game-seconds", type=float, default=300)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--opponent-deck", type=Path,
                        default=REPO_ROOT / "data" / "decks" / "modern"
                        / "modern_mono_red_burn.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs") / "native_learning_evaluations")
    parser.add_argument("--resume-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(evaluate(parse_args()), flush=True)

"""Native collection -> induced graph -> JEPA/dynamics/imitation -> held-out play.

This is a development learning study against a fixed native AI, not self-play
promotion, causal combo discovery or the complete confirmatory RQ1--RQ5 study.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

from scripts.run_phase_rs_ablation import GameRow, _outcome, _revision, _summarize
from src.integrations.phase_rs import (
    HeuristicActionPicker,
    PhaseServerConfig,
    PhaseServerProcess,
    RandomActionPicker,
    load_deck_data,
    run_game_sync,
)
from src.integrations.phase_rs.client import PROTOCOL_VERSION
from src.integrations.phase_rs.native_learning_data import FEATURE_VERSION, NativeRecordingPicker
from src.integrations.phase_rs.server_process import (
    DEFAULT_SUBMODULE,
    REPO_ROOT,
    _resolve_binary,
)
from src.integrations.scryfall_bulk import file_sha256
from src.world_model.native_learning import (
    NativeLearnedPicker,
    combine_graph_sources,
    curated_combo_graph,
    diagnostics,
    induce_graph,
    make_model,
    parameter_digest,
    save_bundle,
    shuffled_graph_identity,
    train_components,
    train_graph,
)

logger = logging.getLogger(__name__)
BENCHMARK = REPO_ROOT / "data" / "decks" / "benchmark"
TRAIN_DECKS = [
    REPO_ROOT / "data" / "decks" / "modern" / "modern_mono_red_burn.txt",
    BENCHMARK / "modern_green_stompy.txt",
]
TEST_DECK = BENCHMARK / "modern_azorius_control.txt"
CURATED_TEST_DECK = REPO_ROOT / "data" / "decks" / "modern" / "modern_hardened_scales.txt"
COMBO_SOURCE = REPO_ROOT / "data" / "combos.json"
COMPONENTS = ("encoder", "jepa_predictor", "dynamics", "controller")
SERVER_DATA_FILES = (
    "card-data.json", "draft-pools.json", "combo-table.json", "ai-commander-decks.json",
)


def server_data_dir() -> Path:
    configured = os.environ.get("PHASE_DATA_DIR")
    path = Path(configured).expanduser() if configured else Path("data")
    return (path if path.is_absolute() else DEFAULT_SUBMODULE / path).resolve()


def snapshot_server_data(data_dir: Path) -> dict[str, str | None]:
    if not (data_dir / "card-data.json").is_file():
        raise ValueError(f"Native data directory lacks card-data.json: {data_dir}")
    return {
        name: file_sha256(data_dir / name) if (data_dir / name).is_file() else None
        for name in SERVER_DATA_FILES
    }


def changed_server_data(data_dir: Path, expected: dict[str, str | None]) -> list[str]:
    if set(expected) != set(SERVER_DATA_FILES) or not expected["card-data.json"]:
        raise ValueError("Campaign must record all native data inputs and the card-data hash")
    changed = []
    for name, digest in expected.items():
        path = data_dir / name
        if (file_sha256(path) if path.is_file() else None) != digest:
            changed.append(str(path))
    return changed


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def collect_game(
    output: Path, uri: str, game_id: str, split: str, deck_path: Path,
    producer: str, seed: int, args: argparse.Namespace, server_commit: str,
) -> dict:
    picker = (
        HeuristicActionPicker(seed) if producer == "heuristic" else RandomActionPicker(seed)
    )
    recorder = NativeRecordingPicker(picker)
    deck = load_deck_data(deck_path)
    opponent = load_deck_data(args.opponent_deck)
    started = time.monotonic()
    result = run_game_sync(
        deck=deck, picker=recorder,
        config=PhaseServerConfig(
            uri=uri, stream_timeout_s=45, expected_server_commit=server_commit,
        ),
        ai_difficulty=args.ai_difficulty, ai_decks=[opponent], format_name="Modern",
        max_actions=args.max_actions, max_game_seconds=args.max_game_seconds,
    )
    game = recorder.finish(result)
    identity = next(
        (event for event in result.trace if event.get("event") == "game_started"), None,
    )
    if identity is None:
        raise ValueError("Native game result lacks the server handshake identity")
    game.update({
        "game_id": game_id, "split": split, "producer": producer, "picker_seed": seed,
        "deck_path": str(deck_path), "deck_sha256": file_sha256(deck_path),
        "elapsed_sec": time.monotonic() - started,
        "server_version": identity["server_version"],
        "server_build_commit": identity["server_build_commit"],
    })
    path = output / "dataset" / f"{game_id}.pt"
    torch.save(game, path)
    game["source_sha256"] = file_sha256(path)
    summary = {key: value for key, value in game.items() if key != "records"}
    summary["decision_count"] = len(game["records"]) - int(game["completed"])
    with (output / "games.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    logger.info("%s %s: %s, decisions=%d", split, game_id, result.reason,
                summary["decision_count"])
    return game


def evaluate(
    output: Path, uri: str, seeds: list[int], args: argparse.Namespace, server_commit: str,
) -> dict:
    summaries = {}
    curated_bundle = torch.load(
        output / "curated_graph.pt", map_location="cpu", weights_only=False,
    )
    for seed in seeds:
        induced_bundle = torch.load(
            output / f"seed_{seed}" / "graph.pt",
            map_location="cpu", weights_only=False,
        )
        combined_bundle = torch.load(
            output / f"seed_{seed}" / "curated_plus_induced_graph.pt",
            map_location="cpu", weights_only=False,
        )
        shuffled_bundle = shuffled_graph_identity(combined_bundle, args.seed + seed + 20000)
        graph_conditions = (
            ("none", None, False),
            ("curated", curated_bundle, True),
            ("induced", induced_bundle, True),
            ("curated_plus_induced", combined_bundle, True),
            ("identity_shuffled", shuffled_bundle, True),
        )
        for weights in ("initial", "trained"):
            for condition, graph_override, graph_enabled in graph_conditions:
                for stratum, deck_path in (
                    ("azorius", TEST_DECK), ("hardened_scales", CURATED_TEST_DECK),
                ):
                    arm = f"seed{seed}_{weights}_{condition}_{stratum}"
                    rows = []
                    for i in range(args.eval_games):
                        game_seed = args.seed + 10000 + i
                        picker = NativeLearnedPicker(
                            output / f"seed_{seed}" / f"{weights}.pt", graph_enabled,
                            seed=game_seed, selection=args.selection,
                            graph_override=graph_override,
                        )
                        started = time.monotonic()
                        result = run_game_sync(
                            deck=load_deck_data(deck_path), picker=picker,
                            config=PhaseServerConfig(
                                uri=uri, stream_timeout_s=45,
                                expected_server_commit=server_commit,
                            ),
                            ai_difficulty=args.ai_difficulty,
                            ai_decks=[load_deck_data(args.opponent_deck)], format_name="Modern",
                            max_actions=args.max_actions, max_game_seconds=args.max_game_seconds,
                        )
                        row = GameRow(
                            game_index=i + 1, winner_seat=result.winner_seat,
                            our_seat=result.our_seat, reason=result.reason,
                            turns=result.turns_observed, actions=result.actions_sent,
                            seed=game_seed, objective=None, outcome=_outcome(result),
                            attempts=1, elapsed_sec=time.monotonic() - started,
                            decision_latencies_sec=[
                                event["decision_time_sec"] for event in result.trace
                                if "decision_time_sec" in event
                            ],
                        )
                        rows.append(row)
                        write_json(output / "evaluation" / f"{arm}_game{i+1:03d}.json", {
                            "row": vars(row), "trace": result.trace,
                            "deck_path": str(deck_path),
                            "graph_condition": condition,
                            "graph_sha256": picker.graph_digest,
                        })
                        logger.info(
                            "%s game %d/%d: %s", arm, i + 1, args.eval_games, result.reason,
                        )
                        summaries[arm] = _summarize(rows)
                        write_json(output / "evaluation_summary.json", summaries)
    return summaries


def run(args: argparse.Namespace) -> Path:
    counts = (args.train_games, args.validation_games, args.test_games,
              args.epochs, args.graph_epochs, args.eval_games, args.threads, args.max_actions)
    if any(count < 1 for count in counts):
        raise ValueError("All native learning budgets must be positive")
    if not math.isfinite(args.max_game_seconds) or args.max_game_seconds <= 0:
        raise ValueError("max-game-seconds must be finite and positive")
    if any(count < 2 for count in (args.train_games, args.validation_games, args.test_games)):
        raise ValueError("Each native collection split requires at least two games")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("Independent training seeds must be unique")
    env_binary = os.environ.get("PHASE_RS_SERVER_BINARY")
    binary = Path(env_binary) if env_binary else _resolve_binary(DEFAULT_SUBMODULE)
    if binary is None or not binary.is_file():
        raise ValueError("Build phase-server before native learning so its binary can be frozen")
    binary = binary.resolve()
    server_commit = _revision(DEFAULT_SUBMODULE)
    data_dir = server_data_dir()
    data_snapshot = snapshot_server_data(data_dir)
    torch.set_num_threads(args.threads)
    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for folder in ("dataset", "evaluation"):
        (output / folder).mkdir(parents=True, exist_ok=False)
    frozen = {
        path: file_sha256(path) for path in (
            *TRAIN_DECKS, TEST_DECK, CURATED_TEST_DECK, COMBO_SOURCE, args.opponent_deck,
            binary,
            DEFAULT_SUBMODULE / "data-files.json",
            *(REPO_ROOT / "src").rglob("*.py"),
            REPO_ROOT / "scripts" / "run_native_learning.py",
            REPO_ROOT / "scripts" / "run_phase_rs_ablation.py",
        )
    }
    frozen.update({
        data_dir / name: digest for name, digest in data_snapshot.items() if digest is not None
    })
    write_json(output / "manifest.json", {
        "purpose": "end-to-end native offline development training; not confirmatory strength",
        "arguments": vars(args), "feature_version": FEATURE_VERSION,
        "repository_commit": _revision(REPO_ROOT), "phase_rs_commit": server_commit,
        "phase_rs_commit_scope": "checkout; actual server build is checked and recorded per game",
        "protocol_version": PROTOCOL_VERSION, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu",
        "file_sha256": {str(path): digest for path, digest in frozen.items()},
        "server_data": {"directory": str(data_dir), "files": data_snapshot},
        "seed_scope": "Python producer and training only; native engine RNG is not seeded",
        "opponent_deck": str(args.opponent_deck),
        "training_objective": {
            "target_encoder": "EMA, momentum 0.99",
            "latent_std_floor": 0.05, "variance_weight": 10, "kl_weight": 0.001,
            "controller": "non-conceding producer imitation",
        },
        "splits": {
            "train": [str(path) for path in TRAIN_DECKS],
            "validation": "independent games of training families; never optimized",
            "test": str(TEST_DECK),
        },
        "limitations": [
            "One controlled seat against fixed native AI with privileged internal information",
            "Hashed text features, compressed tokenizer and observed decision-to-decision targets",
            "Induced battlefield evidence is association, not causal synergy or rules truth",
            "Controller learns producer imitation, not reward-maximizing self-play promotion",
            "Curated combos are an external source control and not play-induced evidence",
            "Native shuffles are unseeded; family holdout still has shared staple cards",
            "No recurrent dream controller training or powered RQ1--RQ5 results",
        ],
    })
    logger.info("Native learning output: %s", output)
    write_json(output / "progress.json", {"stage": "collection", "output": str(output)})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="native-learning-db-") as temporary:
        with PhaseServerProcess(
            port=port, binary=binary, no_data_download=True, bind_address="127.0.0.1",
            extra_env={
                "PHASE_DATA_DIR": str(data_dir),
                "PHASE_GAMES_DB": str(Path(temporary) / "games.db"),
                "RUST_LOG": "warn",
            },
        ) as server:
            if server._adopted:  # noqa: SLF001
                raise ValueError("Native learning requires an owned private server")
            games = []
            offset = 0
            for split, count in (
                ("train", args.train_games), ("validation", args.validation_games),
                ("test", args.test_games),
            ):
                for i in range(count):
                    deck = TEST_DECK if split == "test" else TRAIN_DECKS[i % len(TRAIN_DECKS)]
                    game = collect_game(
                        output, server.uri, f"{split}_{i:04d}", split, deck,
                        "heuristic" if (i // len(TRAIN_DECKS)) % 2 == 0 else "random",
                        args.seed + offset + i, args, server_commit,
                    )
                    games.append(game)
                offset += count
            by_split = {split: [game for game in games if game["split"] == split]
                        for split in ("train", "validation", "test")}
            collection = {
                split: {"scheduled": len(selected),
                        "completed": sum(game["completed"] for game in selected)}
                for split, selected in by_split.items()
            }
            write_json(output / "collection_summary.json", collection)
            for split, selected in by_split.items():
                completed = sum(game["completed"] for game in selected)
                if completed < max(2, len(selected) * 0.95):
                    write_json(output / "summary.json", {
                        "status": "failed_collection_gate", "collection": collection,
                        "failed_split": split, "eligible_for_strength_claims": False,
                    })
                    raise ValueError(
                        f"{split} collection gate failed: "
                        f"{completed}/{len(selected)} terminal games"
                    )
            graph = induce_graph(by_split["train"], output / "induced_graph.json")
            write_json(output / "progress.json", {"stage": "training"})
            curated_source = curated_combo_graph(COMBO_SOURCE, "Modern")
            write_json(output / "curated_graph.json", curated_source)
            curated_bundle, curated_losses = train_graph(
                curated_source, args.seed, args.graph_epochs,
            )
            torch.save(curated_bundle, output / "curated_graph.pt")
            write_json(output / "curated_graph_metrics.json", {
                "source_sha256": curated_source["source_sha256"],
                "format": curated_source["format"],
                "combo_count": len(curated_source["combos"]),
                "edge_count": len(curated_source["edges"]),
                "node_count": len(curated_source["names"]),
                "training_loss": curated_losses,
            })
            metrics = {}
            for seed in args.training_seeds:
                folder = output / f"seed_{seed}"
                folder.mkdir()
                graph_bundle, graph_losses = train_graph(graph, seed, args.graph_epochs)
                torch.save(graph_bundle, folder / "graph.pt")
                combined_graph = combine_graph_sources(curated_source, graph)
                write_json(folder / "curated_plus_induced_graph.json", combined_graph)
                combined_bundle, combined_losses = train_graph(
                    combined_graph, seed, args.graph_epochs,
                )
                torch.save(combined_bundle, folder / "curated_plus_induced_graph.pt")
                write_json(folder / "curated_plus_induced_graph_metrics.json", {
                    "loss": combined_losses,
                    "relation": combined_graph["relation"],
                    "edge_count": len(combined_graph["edges"]),
                    "training_game_ids": combined_graph["training_game_ids"],
                })
                model = make_model(seed)
                before = {key: parameter_digest(getattr(model, key)) for key in COMPONENTS}
                metadata = {
                    "training_seed": seed,
                    "training_game_ids": graph["training_game_ids"],
                    "graph_sha256": file_sha256(output / "induced_graph.json"),
                }
                save_bundle(folder / "initial.pt", model, graph_bundle, {
                    **metadata, "weight_training": "none", "graph_training": "training_games",
                })
                initial = diagnostics(model, by_split["validation"], graph_bundle)
                history = train_components(model, by_split["train"], graph_bundle, args.epochs)
                after = {key: parameter_digest(getattr(model, key)) for key in COMPONENTS}
                if any(before[key] == after[key] for key in COMPONENTS):
                    raise ValueError("A requested native model component did not update")
                save_bundle(folder / "trained.pt", model, graph_bundle, {
                    **metadata, "weight_training": "native_JEPA_recurrent_imitation",
                })
                metrics[str(seed)] = {
                    "initial_validation": initial,
                    "trained_validation": diagnostics(model, by_split["validation"], graph_bundle),
                    "trained_test": diagnostics(model, by_split["test"], graph_bundle),
                    "component_updates": {key: before[key] != after[key] for key in COMPONENTS},
                    "graph_loss": graph_losses, "training_history": history,
                }
                write_json(output / "training_metrics.json", metrics)
                logger.info("Training seed %d complete: all four model components updated", seed)
                write_json(output / "progress.json", {
                    "stage": "training", "completed_training_seeds": list(metrics),
                })
            write_json(output / "progress.json", {"stage": "evaluation"})
            evaluation = evaluate(
                output, server.uri,
                args.training_seeds if args.evaluate_all_seeds else args.training_seeds[:1],
                args, server_commit,
            )
    changed = sorted({
        *(str(path) for path, digest in frozen.items()
          if not path.is_file() or file_sha256(path) != digest),
        *changed_server_data(data_dir, data_snapshot),
    })
    if changed:
        raise ValueError(f"Native learning inputs changed during execution: {changed}")
    write_json(output / "summary.json", {
        "status": "completed_development_pipeline", "changed_inputs": changed,
        "training_seeds": args.training_seeds,
        "collection": {
            split: {"scheduled": len(selected),
                    "completed": sum(game["completed"] for game in selected)}
            for split, selected in by_split.items()
        },
        "graph_edges": len(graph["edges"]), "graph_evidence": len(graph["evidence"]),
        "curated_graph_edges": len(curated_source["edges"]),
        "curated_graph_combos": len(curated_source["combos"]),
        "curated_plus_induced_edges": len(combined_graph["edges"]),
        "evaluation": evaluation,
        "eligible_for_strength_claims": False,
    })
    write_json(output / "progress.json", {"stage": "completed"})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-games", type=int, default=64)
    parser.add_argument("--validation-games", type=int, default=16)
    parser.add_argument("--test-games", type=int, default=8)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--graph-epochs", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--evaluate-all-seeds", action="store_true")
    parser.add_argument("--selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--ai-difficulty", default="VeryEasy",
                        choices=["VeryEasy", "Easy", "Medium"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-game-seconds", type=float, default=300)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--opponent-deck", type=Path, default=TRAIN_DECKS[0])
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "native_learning")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(parse_args()), flush=True)

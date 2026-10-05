#!/usr/bin/env python
"""Ablation runner: our phase-rs picker policies vs phase-rs built-in AI.

This keeps the rules backend and opponent policy inside phase-rs, while our
contribution is the decision layer on the Python side (LLM picker, future
KG-aware pickers, etc.).

Examples:

  python -m scripts.run_phase_rs_ablation --help
  python -m scripts.run_phase_rs_ablation --games 4 --picker kl_control \
      --objectives kl efe_infogain efe_ambiguity --format Modern --ai-deck "Blue Control" \
      --our-deck-file data/decks/modern/modern_mono_red_burn.txt --autostart
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.agents import make_agent
from src.integrations.phase_rs import (
    STARTER_DECK_NAMES,
    AgentActionPicker,
    HeuristicActionPicker,
    KLControlActionPicker,
    OllamaActionPicker,
    PhaseServerConfig,
    PreferNonPassPicker,
    RandomActionPicker,
    load_deck_data,
    run_game_sync,
)
from src.integrations.phase_rs.client import PROTOCOL_VERSION
from src.integrations.phase_rs.decision2_picker import (
    Decision2ActionPicker,
    add_decision2_arguments,
    make_decision2_picker,
)
from src.integrations.phase_rs.kl_control_picker import OBJECTIVES
from src.integrations.phase_rs.runner import GameRunResult
from src.integrations.phase_rs.server_process import DEFAULT_SUBMODULE, REPO_ROOT
from src.integrations.phase_rs.tev1_picker import add_tev1_arguments, make_tev1_picker


@dataclass
class GameRow:
    game_index: int
    winner_seat: int | None
    our_seat: int
    reason: str
    turns: int
    actions: int
    seed: int
    objective: str | None
    outcome: str
    attempts: int
    elapsed_sec: float
    decision_latencies_sec: list[float] = field(default_factory=list)
    candidate_coverages: list[float] = field(default_factory=list)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _outcome(result: GameRunResult) -> str:
    # A missing winner alone is not evidence of a rules-engine draw.
    if not any(event.get("event") == "game_over" for event in result.trace):
        return "incomplete"
    if result.winner_seat is None:
        return "draw"
    return "win" if result.winner_seat == result.our_seat else "loss"


def _summarize(rows: list[GameRow]) -> dict:
    wins = sum(row.outcome == "win" for row in rows)
    losses = sum(row.outcome == "loss" for row in rows)
    draws = sum(row.outcome == "draw" for row in rows)
    completed = wins + losses + draws
    interval = None
    if completed:
        z = 1.959963984540054
        p = wins / completed
        denominator = 1 + z * z / completed
        center = (p + z * z / (2 * completed)) / denominator
        radius = z * math.sqrt(
            p * (1 - p) / completed + z * z / (4 * completed * completed)
        ) / denominator
        interval = [max(0.0, center - radius), min(1.0, center + radius)]
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row.reason] = reason_counts.get(row.reason, 0) + 1
    latencies = [value for row in rows for value in row.decision_latencies_sec]
    coverages = [value for row in rows for value in row.candidate_coverages]
    return {
        "games": len(rows),
        "completed_games": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "incomplete_games": len(rows) - completed,
        "completion_rate": completed / len(rows) if rows else None,
        "win_rate": wins / completed if completed else None,
        "win_rate_ci95": interval,
        "reason_counts": reason_counts,
        "decision_count": len(latencies),
        "decision_latency_sec": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "candidate_coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        "shortlisted_decisions": sum(value < 1 for value in coverages),
    }


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _make_picker(args: argparse.Namespace, seed: int):
    if args.picker == "random":
        return RandomActionPicker(seed=seed)
    if args.picker == "prefer-nonpass":
        return PreferNonPassPicker(seed=seed)
    if args.picker == "heuristic":
        return HeuristicActionPicker(seed=seed)
    if args.picker == "tev1":
        return make_tev1_picker(args, seed)
    if args.picker == "decision2":
        return make_decision2_picker(args, seed)
    if args.picker == "native_learned":
        from src.world_model.native_learning import NativeLearnedPicker

        if not args.native_checkpoint:
            raise ValueError("--picker native_learned requires --native-checkpoint")
        return NativeLearnedPicker(
            Path(args.native_checkpoint), args.native_graph == "induced", seed=seed,
            selection=getattr(args, "native_selection", "sample"),
        )
    if args.picker == "kl_control":
        return KLControlActionPicker(
            seed=seed,
            objective=args.objective,
            horizon=args.horizon,
            beta=args.beta,
            rollouts=args.rollouts,
            engine=args.planner,
        )
    if args.picker == "ollama":
        return OllamaActionPicker(
            seed=seed,
            model=args.ollama_model,
            base_url=args.ollama_url,
        )
    if args.picker.startswith("agent:"):
        agent_name = args.picker.split(":", 1)[1].strip()
        if not agent_name:
            raise ValueError("agent picker requires a name, e.g. agent:heuristic")
        # seat id is corrected by AgentActionPicker at runtime.
        options = {}
        checkpoint = getattr(args, "agent_checkpoint", None)
        if checkpoint:
            if agent_name not in {"world_model", "fusion", "llm_fusion"}:
                raise ValueError("--agent-checkpoint requires a world-model or fusion agent")
            if not Path(checkpoint).is_file():
                raise ValueError(f"Checkpoint does not exist: {checkpoint}")
            options["checkpoint"] = checkpoint
        if agent_name == "world_model":
            options.update(
                mode=args.agent_mode, device=args.agent_device,
                deterministic=args.agent_deterministic,
                dream_rollouts=args.dream_rollouts, dream_depth=args.dream_depth,
            )
        elif agent_name in {"fusion", "llm_fusion"}:
            options.update(llm_model=args.ollama_model, dream_rollouts=args.dream_rollouts,
                           dream_depth=args.dream_depth)
        agent = make_agent(agent_name, player_id="seat0", seed=seed, **options)
        return AgentActionPicker(agent=agent, name=f"phase_rs_agent:{agent_name}")
    raise ValueError(f"unsupported picker: {args.picker}")


def _load_our_deck(args: argparse.Namespace) -> dict:
    if args.our_deck_file:
        return load_deck_data(args.our_deck_file)
    return {"main_deck": [], "sideboard": [], "commander": []}


def run(args: argparse.Namespace) -> int:
    if args.games < 1 or args.max_retries < 1 or args.max_actions < 1:
        raise ValueError("games, max-retries and max-actions must be positive")
    if args.dream_rollouts < 1 or args.dream_depth < 1:
        raise ValueError("dream-rollouts and dream-depth must be positive")
    if args.agent_checkpoint and not Path(args.agent_checkpoint).is_file():
        raise ValueError(f"Checkpoint does not exist: {args.agent_checkpoint}")
    if not 1 <= args.max_message_mib <= 64:
        raise ValueError("max-message-mib must be between 1 and 64")
    if args.max_turns is not None and args.max_turns < 1:
        raise ValueError("max-turns must be positive")
    if args.max_game_seconds is not None and (
        not math.isfinite(args.max_game_seconds) or args.max_game_seconds <= 0
    ):
        raise ValueError("max-game-seconds must be finite and positive")
    if args.objectives and args.picker != "kl_control":
        raise ValueError("--objectives requires --picker kl_control")
    if not args.ai_deck_file and args.ai_deck not in STARTER_DECK_NAMES:
        raise SystemExit(
            f"--ai-deck must be one of {STARTER_DECK_NAMES}, got {args.ai_deck!r}"
        )

    deck = _load_our_deck(args)
    if not deck.get("main_deck"):
        raise ValueError("Supply --our-deck-file with a nonempty main deck for evaluation")
    ai_decks = (
        [load_deck_data(path) for path in args.ai_deck_file]
        if args.ai_deck_file else None
    )
    objectives = (
        list(dict.fromkeys(args.objectives or [args.objective]))
        if args.picker == "kl_control" else [None]
    )
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "config": vars(args),
        "objectives": objectives,
        "repository_commit": _revision(REPO_ROOT),
        "agent_checkpoint_sha256": (
            hashlib.sha256(Path(args.agent_checkpoint).read_bytes()).hexdigest()
            if args.agent_checkpoint else None
        ),
        "native_checkpoint_sha256": (
            hashlib.sha256(Path(args.native_checkpoint).read_bytes()).hexdigest()
            if getattr(args, "native_checkpoint", None) else None
        ),
        "agent_qualification": (
            "Archived checkpoint/native semantics and component activation require auditing"
            if args.picker.startswith("agent:") else None
        ),
        "phase_rs_commit": _revision(DEFAULT_SUBMODULE),
        "protocol_version": PROTOCOL_VERSION,
        "deck_sha256": hashlib.sha256(
            json.dumps(deck, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "ai_decks_sha256": (
            [hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
             for d in ai_decks] if ai_decks is not None else None
        ),
        "player_count": 2 if ai_decks is None else len(ai_decks) + 1,
        "seed_scope": "Python picker only; engine shuffle and AI RNG are not seeded",
        "objective_comparison": (
            "KL uses horizon planning; legacy EFE arms use one-step scoring"
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows: list[GameRow] = []
    games_path = out_dir / "games.jsonl"
    games_path.write_text("", encoding="utf-8")
    schedule = [(i, objective) for i in range(args.games) for objective in objectives]
    for run_index, (i, objective) in enumerate(schedule, start=1):
        game_seed = args.seed + i
        picker_args = argparse.Namespace(**vars(args))
        if objective is not None:
            picker_args.objective = objective
        started = time.perf_counter()
        for attempt in range(1, args.max_retries + 1):
            picker = _make_picker(picker_args, seed=game_seed)
            if isinstance(picker, Decision2ActionPicker):
                picker.prepare()
            cfg = PhaseServerConfig(
                uri=args.uri, stream_timeout_s=args.stream_timeout,
                max_message_bytes=args.max_message_mib * 1024 * 1024,
            )
            result = run_game_sync(
                deck=deck,
                picker=picker,
                config=cfg,
                ai_difficulty=args.ai_difficulty,
                ai_deck_name=args.ai_deck,
                max_actions=args.max_actions,
                autostart_server=args.autostart,
                format_name=args.format,
                ai_decks=ai_decks,
                max_game_seconds=args.max_game_seconds,
                max_turns=args.max_turns,
            )
            if result.trace and result.reason == "stream_timeout":
                trace_dir = out_dir / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / f"game_{run_index:04d}_attempt_{attempt:02d}.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in result.trace) + "\n",
                    encoding="utf-8",
                )
            # If succeeded or not a transient failure, stop retrying.
            if result.reason != "stream_timeout" or attempt == args.max_retries:
                break
            print(f"[retry {attempt}/{args.max_retries}] game {run_index}: {result.reason}")
        row = GameRow(
            game_index=i + 1,
            winner_seat=result.winner_seat,
            our_seat=result.our_seat,
            reason=result.reason,
            turns=result.turns_observed,
            actions=result.actions_sent,
            seed=game_seed,
            objective=objective,
            outcome=_outcome(result),
            attempts=attempt,
            elapsed_sec=time.perf_counter() - started,
            decision_latencies_sec=[
                float(event["decision_time_sec"])
                for event in result.trace
                if "decision_time_sec" in event
            ],
            candidate_coverages=[
                float(event["picker_reasoning"]["candidate_coverage"])
                for event in result.trace
                if isinstance(event.get("picker_reasoning"), dict)
                and "candidate_coverage" in event["picker_reasoning"]
            ],
        )
        rows.append(row)
        with games_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row)) + "\n")

        if result.trace:
            trace_dir = out_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"game_{run_index:04d}.jsonl").write_text(
                "\n".join(json.dumps(evt) for evt in result.trace) + "\n",
                encoding="utf-8",
            )

        print(
            f"game {run_index}/{len(schedule)} [{objective or args.picker}]: {row.outcome} "
            f"(reason={row.reason}, winner={result.winner_seat}, turns={result.turns_observed})"
        )

    summary = {
        **_summarize(rows),
        "schema_version": 2,
        "by_objective": {
            objective: _summarize([row for row in rows if row.objective == objective])
            for objective in objectives if objective is not None
        },
        "picker": args.picker,
        "ai_difficulty": args.ai_difficulty,
        "ai_deck": args.ai_deck if ai_decks is None else None,
        "autostart": args.autostart,
        "seed": args.seed,
        "player_count": 2 if ai_decks is None else len(ai_decks) + 1,
        "ai_deck_files": args.ai_deck_file,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== phase-rs ablation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"output: {out_dir}")

    return 1 if summary["incomplete_games"] else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES,
        help="Interleave multiple objective arms with the same picker seed schedule.",
    )
    parser.add_argument(
        "--picker",
        help=(
            "Policy for our seat: random | prefer-nonpass | heuristic | "
            "kl_control | tev1 | decision2 | native_learned | ollama | agent:<name> "
            "(e.g. agent:heuristic, "
            "agent:world_model, agent:fusion)"
        ),
        default="random",
    )
    parser.add_argument(
        "--objective",
        choices=["kl", "efe_infogain", "efe_ambiguity"],
        default="kl",
        help=(
            "Objective for --picker kl_control. 'kl' is belief-space KL control "
            "(the one intended for play); the other two are the legacy expected-"
            "free-energy factorisations, kept as ablation arms."
        ),
    )
    parser.add_argument("--horizon", type=int, default=3, help="KL-control planning depth H.")
    parser.add_argument("--beta", type=float, default=4.0, help="Bounded-rationality precision.")
    parser.add_argument("--rollouts", type=int, default=48, help="MPPI sample count K.")
    parser.add_argument(
        "--planner", choices=["mppi", "tree"], default="mppi", help="KL-control planner engine."
    )
    parser.add_argument("--ollama-model", default="gemma4:e2b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--agent-checkpoint", default=None,
                        help="Explicit checkpoint for agent:world_model or agent:fusion")
    parser.add_argument("--native-checkpoint", default=None,
                        help="Strict complete bundle from scripts.run_native_learning")
    parser.add_argument("--native-graph", choices=["none", "induced"], default="induced")
    parser.add_argument("--native-selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--agent-mode", choices=["direct", "dream_search"], default="direct")
    parser.add_argument("--agent-device", default="cpu")
    parser.add_argument("--agent-deterministic", action="store_true")
    parser.add_argument("--dream-rollouts", type=int, default=8)
    parser.add_argument("--dream-depth", type=int, default=10)
    add_tev1_arguments(parser)
    add_decision2_arguments(parser)
    parser.add_argument(
        "--ai-difficulty",
        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
        default="Medium",
    )
    parser.add_argument("--ai-deck", default="Red Deck Wins")
    parser.add_argument(
        "--ai-deck-file", action="append", default=None,
        help="Explicit deck for the next AI seat; repeat three times for a Commander pod.",
    )
    parser.add_argument("--our-deck-file", default=None)
    parser.add_argument("--format", default=None, help="Engine format (default: Standard).")
    parser.add_argument("--uri", default="ws://127.0.0.1:9374/ws")
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for the next message before ending with reason=stream_timeout",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--max-game-seconds", type=float, default=None)
    parser.add_argument("--max-message-mib", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Maximum attempts per scheduled game, including the first attempt.",
    )
    parser.add_argument("--output-dir", default="runs/phase_rs_ablation")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

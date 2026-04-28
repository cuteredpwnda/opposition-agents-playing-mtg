"""Ablation sweep over world-model sizes and LLM sizes.

Modes
-----
- ``--sweep wm_size`` — sweep ``WORLD_MODEL_PRESETS``; for each preset
  build a ``WorldModel`` and play ``--eval-games`` round-robin matches
  against a baseline.  No real training is performed by this stub; the
  full pipeline-driven training run is described in
  :doc:`/docs/FULL_VISION_ROADMAP`.
- ``--sweep llm_size`` — sweep entries of ``LLM_REGISTRY``; play each
  LLM-backed agent against a baseline ``HeuristicAgent``.
- ``--sweep cross`` — Cartesian product of the two.

Outputs ``runs/ablation/<timestamp>/{config.json, results.csv,
summary.md}``.

Run with::

    python scripts/ablation_sweep.py --sweep wm_size --eval-games 20
    python scripts/ablation_sweep.py --sweep llm_size --eval-games 20
    python scripts/ablation_sweep.py --sweep cross --eval-games 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Heavy imports (numpy / torch via the world_model package) are deferred
# into ``run()`` so that ``--help`` works on machines without the ML
# dependencies installed.
from src.utils.seeding import derive_seed, set_global_seed


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """One row of the ablation grid."""

    sweep: str
    wm_preset: Optional[str]
    llm_name: Optional[str]
    games_played: int
    wins: int
    losses: int
    draws: int
    avg_turns: float
    avg_decision_ms: float
    notes: str = ""

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0


# ---------------------------------------------------------------------------
# Stubbed evaluation runner
# ---------------------------------------------------------------------------


def evaluate_world_model(preset: "WorldModelPreset", *,
                         eval_games: int, seed: int) -> CellResult:
    """Stub: build the model, count parameters, do a smoke forward pass.

    A full evaluation calls ``BenchmarkSuite`` from ``src.training`` once
    the world-model agent is wired to the controller checkpoints; that
    integration is tracked in ``docs/FULL_VISION_ROADMAP.md`` (item 1 of
    "Outstanding Work").
    """
    set_global_seed(seed)
    try:
        import torch  # noqa: F401
        from src.world_model.world_model import WorldModel

        cfg = preset.build_config()
        wm = WorldModel(cfg)
        n_params = sum(p.numel() for p in wm.parameters())
        notes = f"params={n_params:,}"
    except Exception as exc:  # pragma: no cover - smoke-only stub
        notes = f"build_failed: {exc}"

    # NOTE: Real round-robin via BenchmarkSuite goes here once the
    # WorldModelAgent <-> checkpoint loading path is finalised.
    return CellResult(
        sweep="wm_size",
        wm_preset=preset.name,
        llm_name=None,
        games_played=eval_games,
        wins=0,
        losses=0,
        draws=eval_games,
        avg_turns=0.0,
        avg_decision_ms=0.0,
        notes=notes,
    )


def evaluate_llm(spec: "LLMSpec", *, eval_games: int, seed: int) -> CellResult:
    """Stub: probe Ollama availability and record metadata.

    The full version instantiates ``OllamaAgent(model=spec.ollama_tag)``
    and runs it through ``BenchmarkSuite`` against a fixed
    ``HeuristicAgent``.
    """
    set_global_seed(seed)
    notes = f"params_b={spec.params_billion} vram_gb={spec.vram_gb_estimate}"
    return CellResult(
        sweep="llm_size",
        wm_preset=None,
        llm_name=spec.name,
        games_played=eval_games,
        wins=0,
        losses=0,
        draws=eval_games,
        avg_turns=0.0,
        avg_decision_ms=0.0,
        notes=notes,
    )


def evaluate_cross(preset: "WorldModelPreset", spec: "LLMSpec", *,
                   eval_games: int, seed: int) -> CellResult:
    set_global_seed(seed)
    notes = (f"wm_params_m={preset.approx_params_million} "
             f"llm_params_b={spec.params_billion}")
    return CellResult(
        sweep="cross",
        wm_preset=preset.name,
        llm_name=spec.name,
        games_played=eval_games,
        wins=0,
        losses=0,
        draws=eval_games,
        avg_turns=0.0,
        avg_decision_ms=0.0,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_results(out_dir: Path, sweep: str, results: list[CellResult],
                  config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    csv_path = out_dir / "results.csv"
    if results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys())
                                    + ["win_rate"])
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row["win_rate"] = round(r.win_rate, 3)
                writer.writerow(row)

    summary_path = out_dir / "summary.md"
    lines = [f"# Ablation sweep: {sweep}", ""]
    for r in results:
        tag = r.wm_preset or r.llm_name or "?"
        if r.sweep == "cross":
            tag = f"{r.wm_preset} x {r.llm_name}"
        lines.append(
            f"- **{tag}**: {r.wins}W/{r.losses}L/{r.draws}D "
            f"({r.win_rate:.0%}) — {r.notes}"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def run(args) -> int:
    # Deferred heavy imports.
    from src.agents.llm_registry import LLM_REGISTRY, list_llms
    from src.world_model.presets import WORLD_MODEL_PRESETS, list_presets

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / timestamp

    presets = (
        list_presets(max_vram_gb=args.max_vram_gb)
        if args.sweep in {"wm_size", "cross"}
        else []
    )
    llms = (
        list_llms(max_vram_gb=args.max_vram_gb)
        if args.sweep in {"llm_size", "cross"}
        else []
    )

    results: list[CellResult] = []
    base_seed = args.seed

    if args.sweep == "wm_size":
        for i, preset in enumerate(presets):
            results.append(evaluate_world_model(
                preset,
                eval_games=args.eval_games,
                seed=derive_seed(base_seed, "wm", preset.name, i),
            ))
    elif args.sweep == "llm_size":
        for i, spec in enumerate(llms):
            results.append(evaluate_llm(
                spec,
                eval_games=args.eval_games,
                seed=derive_seed(base_seed, "llm", spec.name, i),
            ))
    elif args.sweep == "cross":
        for i, (preset, spec) in enumerate(product(presets, llms)):
            results.append(evaluate_cross(
                preset, spec,
                eval_games=args.eval_games,
                seed=derive_seed(base_seed, preset.name, spec.name, i),
            ))
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(f"Unknown sweep: {args.sweep}")

    config = {
        "sweep": args.sweep,
        "eval_games": args.eval_games,
        "seed": args.seed,
        "max_vram_gb": args.max_vram_gb,
        "presets": list(WORLD_MODEL_PRESETS),
        "llms": list(LLM_REGISTRY),
    }

    write_results(out_dir, args.sweep, results, config)
    print(f"Wrote {len(results)} results to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", choices=["wm_size", "llm_size", "cross"],
                   required=True)
    p.add_argument("--eval-games", type=int, default=10,
                   help="Round-robin games per cell.")
    p.add_argument("--seed", type=int, default=20260428,
                   help="Master RNG seed.")
    p.add_argument("--max-vram-gb", type=float, default=None,
                   help="Skip cells whose static VRAM estimate exceeds this.")
    p.add_argument("--out-root", type=str, default="runs/ablation",
                   help="Output root directory (timestamped subdir created).")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

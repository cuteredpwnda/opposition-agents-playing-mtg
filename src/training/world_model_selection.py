"""Utilities for selecting world-model checkpoints during training.

The selector uses an active-inference-inspired score that trades off:

- pragmatic value: empirical win rate against baselines
- epistemic value: uncertainty reduction / robustness across opponents
- computational cost: latency and game length

Lower expected free energy is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass(slots=True)
class BenchmarkCell:
    """One checkpoint-vs-baseline measurement."""

    baseline: str
    win_rate: float
    avg_turns: float
    avg_elapsed_sec: float
    games: int


@dataclass(slots=True)
class WorldModelCandidate:
    """Aggregate evidence for one checkpoint candidate."""

    checkpoint: str
    cells: list[BenchmarkCell] = field(default_factory=list)
    training_loss: float | None = None
    validation_loss: float | None = None

    def mean_win_rate(self) -> float:
        return mean(cell.win_rate for cell in self.cells) if self.cells else 0.0

    def win_rate_stdev(self) -> float:
        if len(self.cells) < 2:
            return 0.0
        return pstdev(cell.win_rate for cell in self.cells)

    def mean_elapsed_sec(self) -> float:
        return mean(cell.avg_elapsed_sec for cell in self.cells) if self.cells else 0.0

    def mean_turns(self) -> float:
        return mean(cell.avg_turns for cell in self.cells) if self.cells else 0.0


@dataclass(slots=True)
class ActiveInferenceSelectionConfig:
    """Weights for the checkpoint selector.

    Lower expected free energy is preferred.
    """

    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.6
    latency_weight: float = 0.15
    horizon_weight: float = 0.05
    loss_weight: float = 0.25


@dataclass(slots=True)
class SelectionResult:
    checkpoint: str
    expected_free_energy: float
    pragmatic_value: float
    epistemic_value: float
    latency_cost: float
    horizon_cost: float
    loss_cost: float


def score_candidate(
    candidate: WorldModelCandidate,
    config: ActiveInferenceSelectionConfig | None = None,
) -> SelectionResult:
    """Compute an active-inference-style selection score.

    Pragmatic value rewards strong benchmark performance.
    Epistemic value rewards robustness across baselines while penalizing
    high variance across benchmark cells.
    Latency / horizon / validation losses act as risk terms.
    """
    cfg = config or ActiveInferenceSelectionConfig()

    pragmatic_value = candidate.mean_win_rate()
    epistemic_value = max(0.0, pragmatic_value - candidate.win_rate_stdev())
    latency_cost = candidate.mean_elapsed_sec()
    horizon_cost = candidate.mean_turns()
    loss_cost = 0.0
    if candidate.validation_loss is not None:
        loss_cost = candidate.validation_loss
    elif candidate.training_loss is not None:
        loss_cost = candidate.training_loss

    expected_free_energy = (
        -cfg.pragmatic_weight * pragmatic_value
        -cfg.epistemic_weight * epistemic_value
        + cfg.latency_weight * latency_cost
        + cfg.horizon_weight * horizon_cost
        + cfg.loss_weight * loss_cost
    )

    return SelectionResult(
        checkpoint=candidate.checkpoint,
        expected_free_energy=expected_free_energy,
        pragmatic_value=pragmatic_value,
        epistemic_value=epistemic_value,
        latency_cost=latency_cost,
        horizon_cost=horizon_cost,
        loss_cost=loss_cost,
    )


def select_best_candidate(
    candidates: list[WorldModelCandidate],
    config: ActiveInferenceSelectionConfig | None = None,
) -> SelectionResult:
    """Return the candidate with the lowest expected free energy."""
    if not candidates:
        raise ValueError("No world-model candidates supplied")

    scored = [score_candidate(candidate, config=config) for candidate in candidates]
    return min(scored, key=lambda item: item.expected_free_energy)

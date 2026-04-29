from src.training.world_model_selection import (
    ActiveInferenceSelectionConfig,
    BenchmarkCell,
    WorldModelCandidate,
    score_candidate,
    select_best_candidate,
)


def test_selector_prefers_stronger_checkpoint():
    strong = WorldModelCandidate(
        checkpoint="strong.pt",
        cells=[
            BenchmarkCell("heuristic", 0.65, 7.0, 1.8, 16),
            BenchmarkCell("random", 0.95, 6.0, 1.6, 16),
        ],
        validation_loss=0.22,
    )
    weak = WorldModelCandidate(
        checkpoint="weak.pt",
        cells=[
            BenchmarkCell("heuristic", 0.45, 7.5, 1.7, 16),
            BenchmarkCell("random", 0.80, 6.5, 1.5, 16),
        ],
        validation_loss=0.24,
    )

    best = select_best_candidate([weak, strong])

    assert best.checkpoint == "strong.pt"


def test_selector_penalizes_latency_when_strength_is_close():
    fast = WorldModelCandidate(
        checkpoint="fast.pt",
        cells=[BenchmarkCell("heuristic", 0.55, 6.0, 1.0, 16)],
        validation_loss=0.20,
    )
    slow = WorldModelCandidate(
        checkpoint="slow.pt",
        cells=[BenchmarkCell("heuristic", 0.56, 6.0, 20.0, 16)],
        validation_loss=0.20,
    )

    cfg = ActiveInferenceSelectionConfig(latency_weight=0.3)
    best = select_best_candidate([slow, fast], config=cfg)

    assert best.checkpoint == "fast.pt"


def test_score_candidate_exposes_components():
    candidate = WorldModelCandidate(
        checkpoint="model.pt",
        cells=[
            BenchmarkCell("heuristic", 0.50, 7.0, 2.0, 16),
            BenchmarkCell("random", 0.90, 5.0, 1.5, 16),
        ],
        training_loss=0.3,
    )

    result = score_candidate(candidate)

    assert result.checkpoint == "model.pt"
    assert result.pragmatic_value > 0.0
    assert result.epistemic_value >= 0.0

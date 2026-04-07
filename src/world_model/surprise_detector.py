"""
Surprise detection — monitors world model prediction errors and triggers
KG queries when the model encounters unexpected game transitions.

When the JEPA predictor's MSE between predicted z_{t+1} and actual z_{t+1}
exceeds a threshold, the model was "surprised". This means the game did
something the model didn't expect — likely due to a card interaction
the model hasn't learned yet.

The surprise detector:
1. Computes per-transition prediction error in latent space
2. Identifies high-surprise transitions (above threshold)
3. Queries the KG for the cards involved to find the missed interaction
4. Logs surprise events for human review and KG gap analysis
5. Optionally upweights high-surprise transitions for retraining

This implements Phase A.4 of IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.knowledge.knowledge_graph import MTGKnowledgeGraph
    from src.world_model.trajectory import Trajectory, Transition

logger = logging.getLogger(__name__)


@dataclass
class SurpriseEvent:
    """A single high-surprise transition."""

    game_id: str
    transition_idx: int
    prediction_error: float
    action_type: str
    card_name: str | None
    kg_context: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class SurpriseConfig:
    """Configuration for surprise detection."""

    error_threshold: float = 2.0       # MSE threshold for "surprise"
    top_k_surprises: int = 20          # Max surprises to log per batch
    upweight_factor: float = 3.0       # Training weight multiplier for surprise transitions
    min_transitions: int = 50          # Min transitions to compute threshold adaptively


class SurpriseDetector:
    """Detects surprising game transitions and queries the KG for explanations.

    Usage:
        detector = SurpriseDetector(world_model, kg)
        surprises = await detector.analyze_trajectory(trajectory)
        prioritized = detector.get_retraining_weights(trajectory, surprises)
    """

    def __init__(
        self,
        world_model: Any = None,
        kg: MTGKnowledgeGraph | None = None,
        config: SurpriseConfig | None = None,
    ):
        self.world_model = world_model
        self.kg = kg
        self.config = config or SurpriseConfig()
        self._surprise_log: list[SurpriseEvent] = []

    def compute_prediction_errors(
        self, trajectory: Trajectory
    ) -> list[float]:
        """Compute per-transition prediction error using the JEPA predictor.

        For each (z_t, a_t) → z_{t+1}, computes MSE(ẑ_{t+1}, z_{t+1}).
        If the world model or PyTorch is unavailable, returns empty list.

        Returns:
            List of prediction errors (one per consecutive transition pair).
        """
        if self.world_model is None or len(trajectory.transitions) < 2:
            return []

        try:
            import torch
        except ImportError:
            return []

        errors: list[float] = []

        # Encode all states and actions in the trajectory
        with torch.no_grad():
            for i in range(len(trajectory.transitions) - 1):
                t_curr = trajectory.transitions[i]
                t_next = trajectory.transitions[i + 1]

                # Skip if state features are missing
                if not isinstance(t_curr.state_features, dict) or not isinstance(
                    t_next.state_features, dict
                ):
                    errors.append(0.0)
                    continue

                try:
                    # Flatten state features into tensors
                    curr_flat = self._flatten_features(t_curr.state_features)
                    next_flat = self._flatten_features(t_next.state_features)
                    action_enc = torch.tensor(
                        t_curr.action_encoding, dtype=torch.float32
                    ).unsqueeze(0)

                    # Encode current and next states
                    z_curr, _ = self.world_model.encode(curr_flat)
                    z_next, _ = self.world_model.encode(next_flat)

                    # Predict next state from current
                    if hasattr(self.world_model, "jepa") and self.world_model.jepa is not None:
                        z_hat_next = self.world_model.jepa(z_curr, action_enc)
                    elif hasattr(self.world_model, "predict"):
                        z_hat_next, _, _ = self.world_model.predict(z_curr, action_enc)
                    else:
                        errors.append(0.0)
                        continue

                    mse = torch.nn.functional.mse_loss(z_hat_next, z_next).item()
                    errors.append(mse)
                except Exception:
                    errors.append(0.0)

        return errors

    async def analyze_trajectory(
        self, trajectory: Trajectory
    ) -> list[SurpriseEvent]:
        """Find high-surprise transitions and look up KG context.

        Returns list of SurpriseEvent objects for transitions where
        prediction error exceeds the threshold.
        """
        errors = self.compute_prediction_errors(trajectory)
        if not errors:
            return []

        # Adaptive threshold: mean + 2*std if enough data, else fixed
        error_array = np.array(errors)
        if len(errors) >= self.config.min_transitions:
            threshold = float(error_array.mean() + 2.0 * error_array.std())
        else:
            threshold = self.config.error_threshold

        surprises: list[SurpriseEvent] = []

        # Find indices above threshold
        for idx, err in enumerate(errors):
            if err < threshold:
                continue

            transition = trajectory.transitions[idx]
            event = SurpriseEvent(
                game_id=trajectory.game_id,
                transition_idx=idx,
                prediction_error=err,
                action_type=transition.action_type,
                card_name=transition.card_name,
            )

            # Query KG for context on the surprising card
            if self.kg is not None and transition.card_name:
                try:
                    context = await self.kg.subgraph_context(
                        transition.card_name, depth=2
                    )
                    event.kg_context = context
                    event.explanation = (
                        f"High surprise (MSE={err:.3f}) on {transition.action_type} "
                        f"of '{transition.card_name}' — KG neighbors: "
                        f"{context.get('entities', [])[:5]}"
                    )
                except Exception as e:
                    event.explanation = (
                        f"High surprise (MSE={err:.3f}) on {transition.action_type} "
                        f"of '{transition.card_name}' — KG query failed: {e}"
                    )
            else:
                event.explanation = (
                    f"High surprise (MSE={err:.3f}) on {transition.action_type}"
                )

            surprises.append(event)
            logger.info("Surprise detected: %s", event.explanation)

        # Keep top-k
        surprises.sort(key=lambda s: s.prediction_error, reverse=True)
        surprises = surprises[: self.config.top_k_surprises]

        self._surprise_log.extend(surprises)
        return surprises

    def get_retraining_weights(
        self, trajectory: Trajectory, surprises: list[SurpriseEvent]
    ) -> list[float]:
        """Get per-transition training weights — upweight high-surprise transitions.

        Returns a weight for each transition (default 1.0, surprise transitions
        get upweight_factor).
        """
        weights = [1.0] * len(trajectory.transitions)
        surprise_indices = {s.transition_idx for s in surprises}
        for idx in surprise_indices:
            if idx < len(weights):
                weights[idx] = self.config.upweight_factor
        return weights

    @property
    def surprise_log(self) -> list[SurpriseEvent]:
        """All surprises detected so far (across multiple trajectories)."""
        return list(self._surprise_log)

    def get_surprise_summary(self) -> dict[str, Any]:
        """Get a summary of all detected surprises for reporting."""
        if not self._surprise_log:
            return {"total_surprises": 0}

        errors = [s.prediction_error for s in self._surprise_log]
        card_counts: dict[str, int] = {}
        for s in self._surprise_log:
            if s.card_name:
                card_counts[s.card_name] = card_counts.get(s.card_name, 0) + 1

        return {
            "total_surprises": len(self._surprise_log),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "most_surprising_cards": sorted(
                card_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def _flatten_features(self, features: dict[str, np.ndarray]) -> Any:
        """Flatten a state feature dict into a tensor the world model can encode."""
        import torch

        flat = []
        for v in features.values():
            flat.extend(np.asarray(v).flatten().tolist())
        return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

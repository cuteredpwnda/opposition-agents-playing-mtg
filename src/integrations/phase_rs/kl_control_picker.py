"""Belief-space KL-control picker for phase-rs.

Drives a phase-rs seat with the canonical discrete Active Inference
objective (:mod:`src.agents.kl_control`): control cost against a passive
prior plus expected terminal potential on the *plant* state, planned
closed-loop over a belief about the opponent's hidden cards.

Three objectives are selectable so the structural comparison from
Kaufmann (2026) can be run inside MTG rather than only on toy POMDPs:

``kl``
    Canonical belief-space KL control. The default; the only one intended
    for play.
``efe_infogain``
    Legacy factorisation with an additive information-gain bonus.
    Expected failure mode: noisy-TV capture (scry/surveil loops).
``efe_ambiguity``
    Legacy factorisation with a likelihood-entropy penalty.
    Expected failure mode: scotophobia (refusing variance).
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.agents.kl_control import (
    BeliefSpaceKLControl,
    BeliefState,
    KLControlConfig,
    LegacyEFE,
    PreferenceConfig,
    PreferencePotential,
    gibbs_policy,
)
from src.agents.mtg_transition import (
    ActionForecast,
    CombatForecast,
    MTGTransitionModel,
    action_kind,
    default_belief,
    forecast_action,
    latent_from_phase_rs,
    passive_prior,
)
from src.integrations.phase_rs.action_features import (
    ACTIVATION_POLICY,
    ActivationGate,
    experience_key,
    mulligan_choice,
    useful_mana_colours,
)
from src.integrations.phase_rs.experience_graph import ExperienceSession, centred

logger = logging.getLogger(__name__)

OBJECTIVES = ("kl", "efe_infogain", "efe_ambiguity")


class KLControlActionPicker:
    """Phase-rs ``ActionPicker`` backed by belief-space KL control."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        objective: str = "kl",
        horizon: int = 3,
        beta: float = 4.0,
        rollouts: int = 48,
        engine: str = "mppi",
        preference: PreferenceConfig | None = None,
        sample_policy: bool = False,
        experience: ExperienceSession | None = None,
        experience_weight: float = 6.0,
        activation_policy: str = "hygiene",
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if activation_policy not in ("hygiene", "legacy"):
            raise ValueError("activation_policy must be hygiene or legacy")
        self.objective = objective
        self.name = f"phase_rs_klcontrol:{objective}"
        self.sample_policy = sample_policy
        self.experience = experience
        self.experience_weight = experience_weight
        self.activation_policy = activation_policy
        self._gate = ActivationGate() if activation_policy == "hygiene" else None

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._transition = MTGTransitionModel()
        self._preference = PreferencePotential(preference)
        self._planner = BeliefSpaceKLControl(
            transition=self._transition,
            preference=self._preference,
            config=KLControlConfig(
                horizon=horizon,
                beta=beta,
                rollouts=rollouts,
                engine=engine,
                seed=seed,
            ),
        )
        self._legacy = LegacyEFE(self._preference)
        self._belief: BeliefState = default_belief()
        self.last_reasoning: dict[str, Any] = {}

    # -- ActionPicker protocol -------------------------------------------

    def pick(
        self,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
        seat: int,
    ) -> int:
        if not legal_actions:
            raise ValueError("KLControlActionPicker received empty legal_actions")
        if self._gate is not None:
            # The analytic model has no notion of hand quality, so the
            # mulligan follows the shared land-count rule.
            mulligan = mulligan_choice(legal_actions, state, seat)
            if mulligan is not None:
                self.last_reasoning = {
                    "rule": "mulligan_land_count", "chosen_index": mulligan,
                    "chosen_type": "MulliganDecision",
                    "activation_policy": ACTIVATION_POLICY,
                }
                return mulligan
        if len(legal_actions) == 1:
            self.last_reasoning = {
                "forced": True, "chosen_index": 0,
                "chosen_type": str(legal_actions[0].get("type", "")),
            }
            return 0

        latent = latent_from_phase_rs(state, seat)
        if any(action_kind(action) == "cancel_cast" for action in legal_actions):
            # CancelCast is only offered while this seat has a pending
            # announcement, including cost-payment prompts whose wire data
            # does not expose ``pending_cast`` directly.
            latent = latent.replace(our_pending_cast=True)
        candidates, index_map = self._filter_candidates(legal_actions, state, seat, self._gate)
        projected_candidates = [
            action if self._gate is None and action_kind(action) in {"attack", "block"}
            else forecast_action(action, state, seat)
            for action in candidates
        ]
        keys = [experience_key(action, state) for action in candidates]
        tilt = None
        if self.experience is not None:
            # The shared graph tilts the reference measure q(a): deviating from
            # what has historically won costs -ln q(a) in the KL objective.
            utilities = centred(self.experience.scorer.utilities(keys))
            tilt = [math.exp(self.experience_weight * u) - 1.0 for u in utilities]
        prior = np.array(passive_prior(projected_candidates, tilt), dtype=float)

        self._belief = self._observe(state, seat)

        if self.objective == "kl":
            result = self._planner.plan(latent, projected_candidates, self._belief, prior)
            costs = result.total_cost
            self.last_reasoning = result.as_trace()
        else:
            if self.objective == "efe_infogain":
                costs = self._legacy.information_gain(
                    latent, projected_candidates, self._belief, self._transition
                )
            else:
                costs = self._legacy.ambiguity(
                    latent, projected_candidates, self._belief, self._transition
                )
            self.last_reasoning = {
                "objective": self.objective,
                "G": [round(float(c), 4) for c in costs],
                "belief_entropy_nats": self._belief.entropy(),
            }

        if self.sample_policy:
            policy = gibbs_policy(costs, self._planner.config.beta, prior)
            local = int(self._np_rng.choice(len(projected_candidates), p=policy))
        else:
            local = int(np.argmin(costs))

        chosen = index_map[local]
        if self._gate is not None:
            self._gate.record(legal_actions[chosen], state)
        if self.experience is not None:
            self.experience.note(keys[local])
            self.last_reasoning.update(self.experience.trace())
        self.last_reasoning["chosen_index"] = chosen
        self.last_reasoning["chosen_type"] = str(legal_actions[chosen].get("type", ""))
        self.last_reasoning["transition_model"] = (
            "analytic_combat_v5" if self._gate is not None else "analytic_pending_cast_v4"
        )
        self.last_reasoning["activation_policy"] = (
            ACTIVATION_POLICY if self._gate is not None else "legacy_ungated"
        )
        self.last_reasoning["pending_cast"] = latent.our_pending_cast
        self.last_reasoning["fixed_damage_forecasts"] = sum(
            isinstance(action, ActionForecast) for action in projected_candidates
        )
        self.last_reasoning["combat_forecasts"] = sum(
            isinstance(action, CombatForecast) for action in projected_candidates
        )
        return chosen

    # -- internals -------------------------------------------------------

    @staticmethod
    def _filter_candidates(
        legal_actions: Sequence[dict[str, Any]],
        state: dict[str, Any] | None = None,
        seat: int | None = None,
        gate: ActivationGate | None = None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Drop ``Concede`` and gated activations unless nothing else remains.

        The passive prior already prices Concede at ~0, but removing it keeps a
        sampled policy from ever hitting it through numerical noise. The
        analytic transition model credits every activation with board gain, so
        without the activation gate it animated man-lands in upkeep and
        cracked self-destructive board wipes.
        """
        keep = [
            i for i, a in enumerate(legal_actions)
            if action_kind(a) != "concede"
            and (gate is None or state is None or seat is None or gate.allows(a, state, seat))
        ]
        if gate is not None and state is not None:
            useful = set(useful_mana_colours(legal_actions, state))
            keep = [i for i in keep if i in useful]
        if not keep:
            keep = [i for i, a in enumerate(legal_actions) if action_kind(a) != "concede"]
        if not keep:
            keep = list(range(len(legal_actions)))
        return [legal_actions[i] for i in keep], keep

    @staticmethod
    def _observe(state: dict[str, Any], seat: int) -> BeliefState:
        """Condition the default belief on the current public snapshot.

        The only signal used is how many lands opponents have untapped.
        A snapshot is a current-state observation, not fresh independent
        evidence at every priority pass, so it is applied once to the
        prior rather than accumulated across decisions. This is the
        sensory channel; it never enters the objective.
        """
        belief = default_belief()
        untapped = KLControlActionPicker._opponent_untapped(state, seat)
        if untapped <= 0:
            return belief.update(
                lambda h: 0.4 if h in ("holds_counterspell", "holds_removal") else 1.3
            )
        if untapped >= 2:
            return belief.update(
                lambda h: 1.4 if h in ("holds_counterspell", "holds_removal") else 0.8
            )
        return belief

    @staticmethod
    def _opponent_untapped(state: dict[str, Any], seat: int) -> int:
        objects = state.get("objects") or {}
        count = 0
        for oid in state.get("battlefield") or []:
            obj = objects.get(str(oid)) or objects.get(oid)
            if not isinstance(obj, dict):
                continue
            if obj.get("controller") == seat:
                continue
            type_line = str(obj.get("type_line") or "").lower()
            if "land" in type_line and not obj.get("tapped", False):
                count += 1
        return count

    def observe_outcome(self, outcome: str, provenance: dict[str, Any] | None = None) -> bool:
        """Credit this game's choices to the shared experience graph."""
        return self.experience.finish(outcome, provenance) if self.experience else False

    def reset(self) -> None:
        self._belief = default_belief()
        self._gate = ActivationGate() if self.activation_policy == "hygiene" else None
        self.last_reasoning = {}

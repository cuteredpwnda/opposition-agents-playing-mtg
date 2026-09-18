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
import random
from typing import Any, Sequence

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
    MTGTransitionModel,
    action_kind,
    default_belief,
    latent_from_phase_rs,
    passive_prior,
)

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
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.name = f"phase_rs_klcontrol:{objective}"
        self.sample_policy = sample_policy

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
        if len(legal_actions) == 1:
            return 0

        latent = latent_from_phase_rs(state, seat)
        candidates, index_map = self._filter_candidates(legal_actions)
        prior = np.array(passive_prior(candidates), dtype=float)

        self._belief = self._observe(state)

        if self.objective == "kl":
            result = self._planner.plan(latent, candidates, self._belief, prior)
            costs = result.total_cost
            self.last_reasoning = result.as_trace()
        else:
            if self.objective == "efe_infogain":
                costs = self._legacy.information_gain(
                    latent, candidates, self._belief, self._transition
                )
            else:
                costs = self._legacy.ambiguity(
                    latent, candidates, self._belief, self._transition
                )
            self.last_reasoning = {
                "objective": self.objective,
                "G": [round(float(c), 4) for c in costs],
                "belief_entropy_nats": self._belief.entropy(),
            }

        if self.sample_policy:
            policy = gibbs_policy(costs, self._planner.config.beta, prior)
            local = int(self._np_rng.choice(len(candidates), p=policy))
        else:
            local = int(np.argmin(costs))

        chosen = index_map[local]
        self.last_reasoning["chosen_index"] = chosen
        self.last_reasoning["chosen_type"] = str(legal_actions[chosen].get("type", ""))
        return chosen

    # -- internals -------------------------------------------------------

    @staticmethod
    def _filter_candidates(
        legal_actions: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Drop ``Concede`` unless it is the only option.

        The passive prior already prices it at ~0, but removing it keeps a
        sampled policy from ever hitting it through numerical noise.
        """
        keep = [i for i, a in enumerate(legal_actions) if action_kind(a) != "concede"]
        if not keep:
            keep = list(range(len(legal_actions)))
        return [legal_actions[i] for i in keep], keep

    def _observe(self, state: dict[str, Any]) -> BeliefState:
        """Bayes-update the belief from what the snapshot reveals.

        Only public signals are used: how much mana the opponent has left
        untapped and how large their graveyard is. This is the sensory
        channel; it never enters the objective.
        """
        untapped = self._opponent_untapped(state)
        if untapped <= 0:
            return self._belief.update(
                lambda h: 0.4 if h in ("holds_counterspell", "holds_removal") else 1.3
            )
        if untapped >= 2:
            return self._belief.update(
                lambda h: 1.4 if h in ("holds_counterspell", "holds_removal") else 0.8
            )
        return self._belief

    @staticmethod
    def _opponent_untapped(state: dict[str, Any]) -> int:
        objects = state.get("objects") or {}
        active = state.get("active_player")
        count = 0
        for oid in state.get("battlefield") or []:
            obj = objects.get(str(oid)) or objects.get(oid)
            if not isinstance(obj, dict):
                continue
            if obj.get("controller") == active:
                continue
            type_line = str(obj.get("type_line") or "").lower()
            if "land" in type_line and not obj.get("tapped", False):
                count += 1
        return count

    def reset(self) -> None:
        self._belief = default_belief()
        self.last_reasoning = {}

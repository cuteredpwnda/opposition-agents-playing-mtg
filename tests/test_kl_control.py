"""Tests for belief-space KL control.

These assert the *structural* properties the formulation is supposed to
have — the ones that distinguish it from the two legacy EFE
factorisations — rather than specific action choices, which depend on
the transition model's tuning.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.kl_control import (
    BeliefSpaceKLControl,
    BeliefState,
    KLControlConfig,
    LatentState,
    LegacyEFE,
    PreferencePotential,
    gibbs_policy,
    softmin,
)
from src.agents.mtg_transition import (
    MTGTransitionModel,
    default_belief,
    latent_from_phase_rs,
    passive_prior,
)

ACTIONS = [
    {"type": "PlayLand"},
    {"type": "CastSpell"},
    {"type": "DeclareAttackers"},
    {"type": "PassPriority"},
]


def _state() -> LatentState:
    return LatentState(
        our_life=18,
        opponent_life=12,
        our_board_power=4,
        opponent_board_power=3,
        our_permanents=3,
        opponent_permanents=2,
        our_hand_size=4,
        our_library_size=40,
        turn=5,
    )


def _planner(**cfg) -> BeliefSpaceKLControl:
    return BeliefSpaceKLControl(
        transition=MTGTransitionModel(),
        preference=PreferencePotential(),
        config=KLControlConfig(rollouts=16, seed=11, **cfg),
    )


def test_softmin_recovers_min_at_high_precision():
    v = np.array([1.0, 0.25, 3.0])
    # The uniform reference measure contributes -ln(1/n)/beta, which vanishes
    # as beta grows; at beta=1e6 that residual is ~1.1e-6.
    assert softmin(v, beta=1e6) == pytest.approx(0.25, abs=1e-5)


def test_softmin_approaches_prior_average_at_low_precision():
    v = np.array([0.0, 2.0])
    # beta -> 0 is the zero-deliberation limit: the operator returns the
    # prior-weighted mean, not the minimum.
    assert softmin(v, beta=1e-6) == pytest.approx(1.0, abs=1e-3)


def test_gibbs_policy_is_normalised_and_prefers_low_cost():
    costs = np.array([0.0, 5.0, 10.0])
    p = gibbs_policy(costs, beta=2.0)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_control_cost_penalises_deviation_from_passive_prior():
    planner = _planner()
    prior = np.array([0.7, 0.2, 0.09, 0.01])
    cost = planner._control_cost(prior)
    # -ln q is monotonically decreasing in q: the likelier action is cheaper.
    assert cost[0] < cost[1] < cost[2] < cost[3]


def test_objective_decomposes_into_control_cost_plus_risk():
    planner = _planner()
    result = planner.plan(_state(), ACTIONS, default_belief(), np.array(passive_prior(ACTIONS)))
    np.testing.assert_allclose(
        result.total_cost, result.control_cost + result.risk, rtol=1e-9
    )


def test_no_epistemic_bonus_is_added_to_the_objective():
    """A belief change must not move the objective on its own.

    Sharpening the belief without changing the plant state leaves the
    control cost untouched and only moves risk through the expectation.
    This is the property that rules out noisy-TV capture: there is no term
    that pays for information as such.
    """
    planner = _planner()
    state = _state()
    prior = np.array(passive_prior(ACTIONS))

    flat = BeliefState.uniform(["holds_nothing"] * 1)
    sharp = BeliefState(["holds_nothing"], np.array([1.0]))

    a = planner.plan(state, ACTIONS, flat, prior)
    b = planner.plan(state, ACTIONS, sharp, prior)
    np.testing.assert_allclose(a.control_cost, b.control_cost, rtol=1e-12)


def test_belief_entropy_is_reported_but_not_scored():
    planner = _planner()
    result = planner.plan(_state(), ACTIONS, default_belief(), None)
    assert result.belief_entropy > 0.0
    assert "belief_entropy_nats" in result.as_trace()


def test_tree_engine_produces_a_legal_choice():
    planner = _planner(engine="tree", horizon=2)
    result = planner.plan(_state(), ACTIONS, default_belief(), None)
    assert 0 <= result.index < len(ACTIONS)
    assert result.engine == "tree"


def test_planner_is_deterministic_under_a_fixed_seed():
    a = _planner().plan(_state(), ACTIONS, default_belief(), None)
    b = _planner().plan(_state(), ACTIONS, default_belief(), None)
    assert a.index == b.index
    np.testing.assert_allclose(a.total_cost, b.total_cost)


def test_preference_potential_prefers_winning():
    V = PreferencePotential()
    losing = _state().replace(we_are_eliminated=True)
    winning = _state().replace(opponent_eliminated=True)
    assert V(winning) < V(_state()) < V(losing)


def test_belief_update_is_bayesian():
    belief = default_belief()
    posterior = belief.update(lambda h: 3.0 if h == "holds_counterspell" else 1.0)
    idx = posterior.hypotheses.index("holds_counterspell")
    assert posterior.weights[idx] > belief.weights[idx]
    assert posterior.weights.sum() == pytest.approx(1.0)
    assert posterior.entropy() < belief.entropy()


def test_legacy_factorisations_are_available_and_differ():
    legacy = LegacyEFE()
    transition = MTGTransitionModel()
    belief = default_belief()
    ig = legacy.information_gain(_state(), ACTIONS, belief, transition)
    amb = legacy.ambiguity(_state(), ACTIONS, belief, transition)
    assert ig.shape == amb.shape == (len(ACTIONS),)
    assert not np.allclose(ig, amb)


def test_latent_projection_from_phase_rs_snapshot():
    snapshot = {
        "turn_number": 7,
        "players": [
            {"id": 0, "life": 15, "hand": [1, 2], "library": list(range(30))},
            {"id": 1, "life": 9, "hand": [], "library": list(range(28))},
        ],
        "battlefield": [100, 200],
        "objects": {
            "100": {"controller": 0, "power": 3, "type_line": "Creature"},
            "200": {"controller": 1, "power": 5, "type_line": "Creature"},
        },
    }
    latent = latent_from_phase_rs(snapshot, our_seat=0)
    assert latent.our_life == 15
    assert latent.opponent_life == 9
    assert latent.our_board_power == 3
    assert latent.opponent_board_power == 5
    assert latent.our_hand_size == 2
    assert latent.turn == 7


def test_passive_prior_never_recommends_conceding():
    actions = ACTIONS + [{"type": "Concede"}]
    q = passive_prior(actions)
    assert q[-1] < min(q[:-1])
    assert sum(q) == pytest.approx(1.0)

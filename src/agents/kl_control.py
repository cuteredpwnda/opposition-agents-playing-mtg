"""Belief-space KL control — the canonical discrete Active Inference objective.

Background
----------
Kaufmann (2026), *Active Inference is Optimal Control*, shows that
continuous-time Active Inference on path space **is** finite-horizon
path-integral stochastic MPC, and that its canonical discretisation is
classical KL control on belief space:

.. math::

    G(\\pi) \\;=\\; \\underbrace{D_{KL}\\!\\big(Q(\\eta\\mid\\pi)\\,\\|\\,P_0(\\eta)\\big)}_{\\text{control cost}}
                 \\;+\\; \\underbrace{\\mathbb{E}_{Q(\\eta\\mid\\pi)}\\!\\big[V(\\eta)\\big]}_{\\text{risk}}

where :math:`V(\\eta) = -\\ln p(\\eta)` is a preference potential over the
**latent plant state** — never over observations.

Three consequences drive the design of this module:

1. **No additive information-gain bonus** (the "Extant EFE 1"
   factorisation). It is not needed and it is actively harmful: an
   unconstrained epistemic bonus is captured by noisy-TV distractors.
2. **No ambiguity / likelihood-entropy penalty** (the "Extant EFE 2"
   factorisation). Penalising sensor entropy produces scotophobia —
   fleeing informative-but-noisy regions of the state space.
3. **Exploration must come from closed-loop planning.** Evaluating plans
   over a *belief* state with observation branching recovers Fel'dbaum's
   dual control effect: a probe is taken iff the resulting drop in
   expected terminal potential exceeds its control cost. Open-loop
   receding-horizon planning (what most discrete Active Inference
   toolboxes do) provably destroys the operational value of information.

Under finite deliberation capacity :math:`\\beta < \\infty` the hard
``min`` is replaced by the softmin (free-energy) operator and policy
selection follows the bounded-rational Gibbs distribution
:math:`p_\\beta(\\pi) \\propto q(\\pi)\\,e^{-\\beta G(\\pi)}`.

Two planners are provided:

``tree``
    Closed-loop belief-tree dynamic programming with explicit observation
    branching. Exact for small horizons; this is the reference semantics.
``mppi``
    Model Predictive Path Integral. Samples :math:`K` action sequences
    from the passive prior, scores each rollout by
    :math:`S_k = \\sum_\\tau c(a_\\tau) + V(\\eta_T)`, and returns the
    exponentially-tilted first-action distribution
    :math:`w_k \\propto e^{-\\beta S_k}`. This is the scalable path-integral
    form and is what runs against phase-rs.

The two legacy factorisations are kept in :class:`LegacyEFE` so the
comparison in the paper can be reproduced inside MTG rather than only on
toy POMDPs.

Only numpy is required; the world model is optional and injected.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Natural log of a very small number, used as a floor so a zero-probability
# action contributes a large-but-finite control cost instead of +inf.
_LOG_FLOOR = -30.0


# ---------------------------------------------------------------------------
# Latent state features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentState:
    """Engine-agnostic projection of the plant state :math:`\\eta`.

    Deliberately small and *physical*: these are properties of the game
    itself, not of our sensor channel. The preference potential is defined
    over exactly these fields, which is what keeps the objective free of
    transducer wireheading.
    """

    our_life: int
    opponent_life: int
    our_board_power: int
    opponent_board_power: int
    our_permanents: int
    opponent_permanents: int
    our_hand_size: int
    our_library_size: int
    turn: int
    we_are_eliminated: bool = False
    opponent_eliminated: bool = False

    def replace(self, **kwargs: Any) -> "LatentState":
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


@dataclass
class PreferenceConfig:
    """Weights of the preference potential :math:`V(\\eta) = -\\ln p(\\eta)`.

    Each weight converts one physical quantity into nats of surprisal.
    Signs are chosen so that *lower* :math:`V` means *more preferred*.
    """

    win_potential: float = -12.0
    loss_potential: float = 12.0
    life_differential: float = 0.09
    board_differential: float = 0.16
    permanent_differential: float = 0.10
    card_advantage: float = 0.12
    # Mild pressure to close the game out rather than durdle forever.
    turn_pressure: float = 0.02
    # Losing the library is a loss condition; make it visible before it fires.
    decking_pressure: float = 0.30


class PreferencePotential:
    """:math:`V(\\eta) = -\\ln p(\\eta)` over :class:`LatentState`.

    This is the *only* place goals are encoded. Keeping it a single
    function of the plant state (rather than of observations) is what
    Kaufmann's Proposition 1 requires, and it is what makes the
    exploration behaviour fall out of closed-loop planning instead of a
    hand-tuned curiosity term.
    """

    def __init__(self, config: PreferenceConfig | None = None) -> None:
        self.config = config or PreferenceConfig()

    def __call__(self, state: LatentState) -> float:
        c = self.config
        if state.opponent_eliminated and not state.we_are_eliminated:
            return c.win_potential
        if state.we_are_eliminated:
            return c.loss_potential

        v = 0.0
        v -= c.life_differential * (state.our_life - state.opponent_life)
        v -= c.board_differential * (
            state.our_board_power - state.opponent_board_power
        )
        v -= c.permanent_differential * (
            state.our_permanents - state.opponent_permanents
        )
        v -= c.card_advantage * state.our_hand_size
        v += c.turn_pressure * state.turn
        if state.our_library_size <= 5:
            v += c.decking_pressure * (6 - state.our_library_size)
        return float(v)


# ---------------------------------------------------------------------------
# Belief state over hidden information
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """Categorical belief over hidden-information hypotheses.

    Each hypothesis is an opaque label (an opponent archetype, a
    "holds removal" / "holds counterspell" flag combination, a sampled
    hand composition) carrying a weight. The planner only needs three
    operations: entropy, Bayes update, and expectation of a state-valued
    function, so the hypothesis representation stays free-form.
    """

    hypotheses: list[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def uniform(cls, hypotheses: Sequence[str]) -> "BeliefState":
        n = max(1, len(hypotheses))
        return cls(list(hypotheses), np.full(n, 1.0 / n))

    def normalise(self) -> None:
        total = float(self.weights.sum())
        if total <= 0.0 or not math.isfinite(total):
            self.weights = np.full(len(self.weights), 1.0 / max(1, len(self.weights)))
        else:
            self.weights = self.weights / total

    def entropy(self) -> float:
        """Shannon entropy in nats. Reported for diagnostics only.

        It is deliberately *not* added to the objective — that is exactly
        the surrogate term this formulation removes.
        """
        w = np.clip(self.weights, 1e-12, 1.0)
        return float(-(w * np.log(w)).sum())

    def update(self, likelihood: Callable[[str], float]) -> "BeliefState":
        """Bayes update: ``w_i <- w_i * p(o | h_i)``, renormalised."""
        if not self.hypotheses:
            return self
        lik = np.array(
            [max(1e-12, float(likelihood(h))) for h in self.hypotheses], dtype=float
        )
        posterior = BeliefState(list(self.hypotheses), self.weights * lik)
        posterior.normalise()
        return posterior

    def expectation(self, f: Callable[[str], float]) -> float:
        if not self.hypotheses:
            return 0.0
        vals = np.array([float(f(h)) for h in self.hypotheses], dtype=float)
        return float((self.weights * vals).sum())


# ---------------------------------------------------------------------------
# Generative model interface
# ---------------------------------------------------------------------------


class TransitionModel(Protocol):
    """Predicts the plant state after an action, per hidden hypothesis.

    Implementations may be a hand-written forward model, a phase-rs
    lookahead, or the JEPA world model decoded back into
    :class:`LatentState` coordinates.
    """

    def step(
        self, state: LatentState, action: Any, hypothesis: str
    ) -> LatentState: ...

    def observation_likelihood(
        self, state: LatentState, action: Any, hypothesis: str
    ) -> float:
        """``p(o | eta, a, h)`` for the observation actually expected.

        Used only to branch the belief tree; it never enters the
        objective.
        """


# ---------------------------------------------------------------------------
# Planner configuration + result
# ---------------------------------------------------------------------------


@dataclass
class KLControlConfig:
    """Planner hyperparameters.

    ``beta`` is the bounded-rationality precision: the deliberation budget.
    ``beta -> inf`` recovers the deterministic Bellman minimiser;
    small ``beta`` recovers probability matching against the passive prior.
    """

    horizon: int = 3
    beta: float = 4.0
    rollouts: int = 64
    engine: str = "mppi"  # "mppi" | "tree"
    # Per-step actuation cost floor. Keeps "do nothing" strictly cheaper
    # than "do something pointless" without encoding any preference.
    move_cost: float = 0.1
    discount: float = 1.0
    seed: int | None = None


@dataclass
class PlanResult:
    """Chosen action plus the full decomposition of the objective."""

    index: int
    action: Any
    policy: np.ndarray
    total_cost: np.ndarray
    control_cost: np.ndarray
    risk: np.ndarray
    belief_entropy: float
    engine: str

    @property
    def expected_free_energy(self) -> float:
        return float(self.total_cost[self.index])

    def as_trace(self, top_k: int = 5) -> dict[str, Any]:
        order = np.argsort(self.total_cost)[:top_k]
        return {
            "objective": "belief_space_kl_control",
            "engine": self.engine,
            "chosen_index": int(self.index),
            "G": float(self.total_cost[self.index]),
            "control_cost": float(self.control_cost[self.index]),
            "risk": float(self.risk[self.index]),
            "belief_entropy_nats": self.belief_entropy,
            "policy": [round(float(p), 4) for p in self.policy],
            "top_candidates": [
                {
                    "index": int(i),
                    "G": round(float(self.total_cost[i]), 4),
                    "control_cost": round(float(self.control_cost[i]), 4),
                    "risk": round(float(self.risk[i]), 4),
                }
                for i in order
            ],
        }


def softmin(values: np.ndarray, beta: float, prior: np.ndarray | None = None) -> float:
    """Bounded-rational free-energy operator ``-1/beta * ln sum q e^{-beta V}``.

    Recovers ``min(values)`` as ``beta -> inf``.
    """
    if values.size == 0:
        return 0.0
    if not math.isfinite(beta) or beta <= 0:
        return float(values.min())
    q = np.full(values.size, 1.0 / values.size) if prior is None else prior
    shift = float(values.min())
    z = float((q * np.exp(-beta * (values - shift))).sum())
    if z <= 0.0:
        return shift
    return shift - math.log(z) / beta


def gibbs_policy(
    costs: np.ndarray, beta: float, prior: np.ndarray | None = None
) -> np.ndarray:
    """``p_beta(a) ∝ q(a) exp(-beta * G(a))`` — Eq. (14)."""
    if costs.size == 0:
        return costs
    q = np.full(costs.size, 1.0 / costs.size) if prior is None else np.asarray(prior)
    logits = -beta * (costs - float(costs.min())) + np.log(np.clip(q, 1e-12, None))
    logits -= logits.max()
    p = np.exp(logits)
    total = p.sum()
    return p / total if total > 0 else np.full(costs.size, 1.0 / costs.size)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


class BeliefSpaceKLControl:
    """Closed-loop KL control over belief space.

    The passive prior ``q(a)`` plays a load-bearing role: it *is* the
    reference measure the control cost is charged against. Handing it the
    heuristic/KG/LLM prior therefore gives those signals a principled home
    (a thermodynamic price on deviating from them) instead of the ad-hoc
    additive weighting they had previously.
    """

    def __init__(
        self,
        transition: TransitionModel,
        preference: PreferencePotential | None = None,
        config: KLControlConfig | None = None,
    ) -> None:
        self.transition = transition
        self.preference = preference or PreferencePotential()
        self.config = config or KLControlConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # -- public API ------------------------------------------------------

    def plan(
        self,
        state: LatentState,
        actions: Sequence[Any],
        belief: BeliefState,
        passive_prior: np.ndarray | None = None,
    ) -> PlanResult:
        if not actions:
            raise ValueError("BeliefSpaceKLControl.plan requires at least one action")

        prior = self._normalised_prior(passive_prior, len(actions))
        control = self._control_cost(prior)

        if self.config.engine == "tree":
            risk = self._risk_tree(state, actions, belief)
        else:
            risk = self._risk_mppi(state, actions, belief, prior)

        total = control + risk
        policy = gibbs_policy(total, self.config.beta, prior)
        index = int(np.argmin(total))
        return PlanResult(
            index=index,
            action=actions[index],
            policy=policy,
            total_cost=total,
            control_cost=control,
            risk=risk,
            belief_entropy=belief.entropy(),
            engine=self.config.engine,
        )

    # -- cost terms ------------------------------------------------------

    def _normalised_prior(
        self, passive_prior: np.ndarray | None, n: int
    ) -> np.ndarray:
        if passive_prior is None:
            return np.full(n, 1.0 / n)
        q = np.asarray(passive_prior, dtype=float).reshape(-1)
        if q.size != n:
            logger.debug("passive prior size %d != %d actions; using uniform", q.size, n)
            return np.full(n, 1.0 / n)
        q = np.clip(q, 0.0, None)
        total = q.sum()
        return q / total if total > 0 else np.full(n, 1.0 / n)

    def _control_cost(self, prior: np.ndarray) -> np.ndarray:
        """``C_action(a) = -ln q(a) + c_move``.

        With deterministic transitions the path-space divergence
        ``D_KL(Q(eta|pi) || P0(eta))`` collapses onto the action-space
        divergence against the passive prior, which for a deterministic
        choice of ``a`` is ``-ln q(a)`` up to a policy-independent
        constant. ``c_move`` is the additive stage cost used in the
        reference POMDP benchmarks.
        """
        logq = np.maximum(np.log(np.clip(prior, 1e-12, None)), _LOG_FLOOR)
        return -logq + self.config.move_cost

    # -- risk: path-integral (MPPI) --------------------------------------

    def _risk_mppi(
        self,
        state: LatentState,
        actions: Sequence[Any],
        belief: BeliefState,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Expected terminal potential under exponentially-tilted rollouts.

        For each candidate first action we sample ``K`` continuations from
        the passive prior, roll the transition model forward ``H`` steps
        under a hypothesis drawn from the belief, and take the
        path-integral (soft) aggregate of terminal potentials. Sampling
        the hypothesis *per rollout* is what makes the estimate an
        expectation over the belief rather than over a point estimate.
        """
        cfg = self.config
        n = len(actions)
        risk = np.zeros(n)
        hyp_idx = self._sample_hypotheses(belief, cfg.rollouts)

        for i, first_action in enumerate(actions):
            scores = np.empty(cfg.rollouts)
            for k in range(cfg.rollouts):
                h = hyp_idx[k]
                s = self.transition.step(state, first_action, h)
                acc = self.preference(s)
                discount = cfg.discount
                for _ in range(max(0, cfg.horizon - 1)):
                    j = int(self._rng.choice(n, p=prior))
                    s = self.transition.step(s, actions[j], h)
                    acc += discount * self.preference(s)
                    discount *= cfg.discount
                scores[k] = acc / max(1, cfg.horizon)
            # Path-integral aggregation: the free-energy operator, not a
            # plain mean. beta -> inf gives the best achievable
            # continuation, beta -> 0 the passive average.
            risk[i] = softmin(scores, cfg.beta)
        return risk

    def _sample_hypotheses(self, belief: BeliefState, k: int) -> list[str]:
        if not belief.hypotheses:
            return [""] * k
        idx = self._rng.choice(len(belief.hypotheses), size=k, p=belief.weights)
        return [belief.hypotheses[int(i)] for i in idx]

    # -- risk: closed-loop belief tree -----------------------------------

    def _risk_tree(
        self,
        state: LatentState,
        actions: Sequence[Any],
        belief: BeliefState,
    ) -> np.ndarray:
        """Exact closed-loop DP over the belief tree (Eq. 15).

        Observation branching is what creates directed exploration: an
        action whose successor beliefs are *sharper* unlocks cheaper
        continuations, so it wins on terminal potential without any
        epistemic bonus being added anywhere.
        """
        return np.array(
            [
                self._tree_value(state, a, belief, depth=self.config.horizon)
                for a in actions
            ]
        )

    def _tree_value(
        self, state: LatentState, action: Any, belief: BeliefState, depth: int
    ) -> float:
        expected = 0.0
        for h, w in zip(belief.hypotheses or [""], belief.weights if belief.hypotheses else [1.0]):
            successor = self.transition.step(state, action, h)
            value = self.preference(successor)
            if depth > 1:
                posterior = belief.update(
                    lambda cand, s=state, a=action: self.transition.observation_likelihood(
                        s, a, cand
                    )
                )
                child = [
                    self._tree_value(successor, a2, posterior, depth - 1)
                    for a2 in self._candidate_continuations()
                ]
                if child:
                    value += self.config.discount * softmin(
                        np.array(child), self.config.beta
                    )
            expected += float(w) * value
        return expected / max(1, self.config.horizon)

    def _candidate_continuations(self) -> list[Any]:
        """Continuation set for tree search.

        Overridden by callers that can enumerate legal actions at depth;
        the default keeps the tree to a single "do nothing" continuation
        so the exact planner degrades to a one-step lookahead rather than
        fabricating illegal lines.
        """
        return [None]


# ---------------------------------------------------------------------------
# Legacy factorisations — kept for the ablation, not for play
# ---------------------------------------------------------------------------


class LegacyEFE:
    """The two extant discrete EFE factorisations, for controlled comparison.

    Neither is used by default. They exist so the MTG ablation can
    reproduce the structural failure modes reported on toy POMDPs:

    ``information_gain``
        ``G = -epistemic - pragmatic``. Susceptible to noisy-TV capture:
        in MTG this shows up as compulsive scry/surveil/cantrip loops that
        gather information with no bearing on the win condition.
    ``ambiguity``
        ``G = risk + E[H[p(o|eta)]]``. Susceptible to scotophobia: the
        agent avoids high-variance lines (combat with unknown blockers,
        casting into open mana) purely because outcomes are uncertain.
    """

    def __init__(self, preference: PreferencePotential | None = None) -> None:
        self.preference = preference or PreferencePotential()

    def information_gain(
        self,
        state: LatentState,
        actions: Sequence[Any],
        belief: BeliefState,
        transition: TransitionModel,
    ) -> np.ndarray:
        out = np.zeros(len(actions))
        h_prior = belief.entropy()
        for i, a in enumerate(actions):
            posterior = belief.update(
                lambda cand, s=state, act=a: transition.observation_likelihood(
                    s, act, cand
                )
            )
            epistemic = h_prior - posterior.entropy()
            pragmatic = -belief.expectation(
                lambda h, s=state, act=a: self.preference(transition.step(s, act, h))
            )
            out[i] = -(epistemic + pragmatic)
        return out

    def ambiguity(
        self,
        state: LatentState,
        actions: Sequence[Any],
        belief: BeliefState,
        transition: TransitionModel,
    ) -> np.ndarray:
        out = np.zeros(len(actions))
        for i, a in enumerate(actions):
            risk = belief.expectation(
                lambda h, s=state, act=a: self.preference(transition.step(s, act, h))
            )
            likelihoods = np.array(
                [
                    max(1e-12, transition.observation_likelihood(state, a, h))
                    for h in (belief.hypotheses or [""])
                ]
            )
            likelihoods = likelihoods / likelihoods.sum()
            ambiguity = float(-(likelihoods * np.log(likelihoods)).sum())
            out[i] = risk + ambiguity
        return out

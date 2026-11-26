# Active Inference as Belief-Space KL Control

**Status:** implemented, tested, not yet evaluated in live games (blocked on a local `phase-server` build).
**Code:** `src/agents/kl_control.py`, `src/agents/mtg_transition.py`, `src/integrations/phase_rs/kl_control_picker.py`
**Tests:** `tests/test_kl_control.py` (14 tests)
**Source result:** Kaufmann (2026), *Active Inference is Optimal Control*

This document is for review. It explains what was replaced, why, what the new
objective actually is, what is honest about the implementation, and what is
still missing.

---

## 1. What was there before, and why it was wrong

`src/agents/active_inference.py` claimed to minimise expected free energy. In
practice it did this:

```python
def compute_expected_free_energy(self, action, game_state) -> float:
    epistemic = self._compute_epistemic_value(action, game_state)
    pragmatic = self._compute_pragmatic_value(action, game_state)
    return -(epistemic + pragmatic)

def _compute_epistemic_value(self, action, game_state) -> float:
    text = card.oracle_text.lower()
    if any(kw in text for kw in ["look at", "reveal", "target player discards"]):
        return 0.4
    if "draw" in text:
        return 0.2
    return 0.0
```

That is a keyword lookup table with the word "epistemic" attached to it. There
is no belief, no expectation, no generative model, no horizon, and the
constants (0.4, 0.2, 0.35) are unmotivated. The `OpponentBelief` dataclass
existed but `update_beliefs` only popped played cards out of a dict and nudged
a counterspell probability by a hard-coded 0.15.

The separate `LLMFusionAgent` then combined four signals additively:

```
score(a) = 0.15 * heuristic + 0.15 * kg + 0.40 * world_model + 0.30 * log p_LLM
```

Four free parameters, four quantities in incompatible units, no principle for
setting them. Empirically it underperformed its own components
(31.3% win rate vs. 56.3% for the KG heuristic alone — see the paper,
Table `tab:ablation-1v1`). That result is what motivated the rewrite.

---

## 2. The theoretical result we are implementing

Kaufmann (2026) establishes an equivalence the active-inference literature had
backwards.

**Continuous time.** Let $Q^u$ be the path measure induced by control $u$ over a
planning horizon, $P_0$ the uncontrolled (passive) path measure, and
$V(\eta) = -\ln p(\eta)$ a preference potential over **latent plant state**
$\eta$. The Gibbs tilt $P_V \propto e^{-V(\eta_T)}P_0$ is the target. Path-form
expected free energy is just

$$G[u] = D_{KL}\big(Q^u \,\|\, P_V\big)$$

and by Girsanov's theorem this equals, up to policy-independent constants, a
finite-horizon stochastic MPC objective with quadratic control cost and terminal
potential $V$. **Minimising EFE *is* path-integral control.** It is not a
generalisation of optimal control; it is optimal control.

**Discrete time.** Time-slicing gives belief-space KL control:

$$G(\pi) = \underbrace{D_{KL}\big(Q(\eta\mid\pi) \,\|\, P_0(\eta)\big)}_{\text{control cost}} + \underbrace{\mathbb{E}_{Q(\eta\mid\pi)}\big[V(\eta)\big]}_{\text{risk}}$$

**What is absent, and why that matters.** There is no additive information-gain
bonus and no ambiguity (likelihood-entropy) penalty. Measurement likelihoods
belong to the observation filtration that drives the belief estimator; they are
not part of the plant objective.

Exploration is not missing. It is Fel'dbaum's dual control effect (1960):
because $\pi$ is evaluated **closed-loop** over belief states with observation
branching, an action that sharpens the posterior unlocks cheaper continuations
and therefore lowers $\mathbb{E}[V(\eta_T)]$ *by itself*. A probe is taken iff
the drop in expected terminal potential exceeds its control cost. No coefficient
encodes that trade-off; it falls out.

The historical claim that KL control cannot explore came from benchmarking it
**open-loop** — scoring fixed action sequences with no sensory conditioning,
which mathematically destroys the value of information. Most discrete
active-inference toolboxes (including `pymdp`) still plan that way.

---

## 3. What we built

### 3.1 `src/agents/kl_control.py` — the objective and the planners

| Component | Role |
|---|---|
| `LatentState` | The plant coordinates $\eta$. Life totals, board power, permanent counts, hand/library size, turn, two terminal flags. Deliberately small and *physical*. |
| `PreferencePotential` | $V(\eta) = -\ln p(\eta)$. The **only** place goals are encoded. |
| `BeliefState` | Categorical over hidden-information hypotheses. `update()` (Bayes), `entropy()` (diagnostic), `expectation()`. |
| `TransitionModel` (Protocol) | $p(\eta' \mid \eta, a, h)$ plus an observation likelihood used *only* to branch the belief tree. |
| `BeliefSpaceKLControl` | The planner. Two engines. |
| `LegacyEFE` | The two extant EFE factorisations, for ablation only. |
| `softmin`, `gibbs_policy` | Bounded-rationality operators. |

**Control cost.** With deterministic transitions the path-space divergence
collapses onto the action-space divergence against the passive prior, so for a
deterministic choice of $a$:

```python
C_action(a) = -ln q(a) + c_move
```

`c_move` (default 0.1 nats) is the additive stage cost used in the reference
POMDP benchmarks. It keeps "pass" strictly cheaper than "do something
pointless" without encoding any preference.

**Two planner engines:**

- **`tree`** — exact closed-loop dynamic programming over the belief tree with
  observation branching. Reference semantics. Exponential in horizon.
- **`mppi`** — Model Predictive Path Integral. For each candidate first action
  and each hidden hypothesis, sample continuations from the passive prior, roll
  each forward $H$ steps, and aggregate horizon-averaged potentials with the
  softmin operator. Hypotheses are then averaged under the belief. Linear in
  $K \cdot H$. **Default.** This is the same algorithm that runs thousands of
  parallel rollouts on GPUs in robotics.

The softmin ranges only over *our* sampled continuations. The rollout budget is
stratified across hypotheses and their values are belief-weighted. Applying the
softmin across hypotheses as well (the pre-Nov 25 implementation) let the
planner effectively choose the opponent's hidden cards, an optimism bias rather
than an expectation. MPPI does not update the belief inside a rollout, so it
plans open-loop with respect to hidden information.

**Bounded rationality.** Deliberation is not free. The hard `min` is replaced by
the softmin (free-energy) operator and policy selection follows

$$p_\beta(\pi) \propto q(\pi)\,e^{-\beta G(\pi)}$$

$\beta \to \infty$ recovers the deterministic Bellman minimiser; $\beta \to 0$
recovers the passive prior. One scalar, with an operational meaning (how much
compute per priority window), replaces four tuned weights.

### 3.2 `src/agents/mtg_transition.py` — the generative model

Projects a phase-rs snapshot onto `LatentState` and supplies the forward model.

The hypothesis set is coarse on purpose — the planner only needs enough
resolution for a probe to change a downstream decision:

```
holds_removal | holds_counterspell | holds_threat | holds_nothing
```

The hypothesis enters exactly where hidden information actually bites: a
counterspell blanks a cast, removal blanks a board commitment or absorbs an
attack, a hidden threat accelerates the opponent's clock. **That coupling is the
entire mechanism by which information becomes operationally valuable**, and
therefore the entire mechanism by which the planner decides to probe. If you
change nothing else, change this: it is the most consequential 40 lines.

`observation_likelihood` encodes how diagnostic an action is — casting into
open mana is informative about `holds_counterspell`, attacking into untapped
blockers about `holds_removal`. It affects the objective only *indirectly*, via
sharper posteriors unlocking cheaper continuations.

**The passive prior** ($P_0$) is the reference measure. It encodes "what a game
of Magic normally looks like", **not** "what is good":

```python
_PASSIVE_WEIGHTS = {"land": 6.0, "cast": 4.0, "attack": 3.0, "block": 3.0,
                    "activate": 2.0, "pass": 1.0, "other": 1.0, "concede": 1e-6}
```

`passive_prior(actions, extra_scores)` lets heuristic / KG / LLM signals tilt
it. **This is where the four fusion weights went.** Those signals now enter as a
thermodynamic price on deviating from an informed default ($-\ln q(a)$) rather
than as an additive bonus with a tuned coefficient.

### 3.3 `src/integrations/phase_rs/kl_control_picker.py` — the driver

Implements the phase-rs `ActionPicker` protocol. Filters `Concede` unless it is
the only legal option. Conditions the default belief on **public signals only**
(untapped lands controlled by opponents of the picker's seat), which is the
sensory channel and never enters the objective. The snapshot is applied once per
decision rather than accumulated, because repeated priority passes re-observe the
same state rather than supplying independent evidence.

Selects one of three objectives:

| `--objective` | What it is | Expected MTG failure mode |
|---|---|---|
| `kl` | Belief-space KL control. **The only one intended for play.** | — |
| `efe_infogain` | $G = -(\text{epistemic} + \text{pragmatic})$ | **Noisy-TV capture**: compulsive scry/surveil/cantrip loops buying information with no bearing on the win condition. |
| `efe_ambiguity` | $G = \text{risk} + \mathbb{E}[H[p(o\mid\eta)]]$ | **Scotophobia**: refusing profitable attacks into possible blockers, or casts into open mana, purely because the outcome distribution is wide. |

All three share belief, transition model, potential, prior and seed. They differ
in **nothing but the scoring function**, which is what makes the ablation clean.

### 3.4 Action hygiene (`action_hygiene_v2`)

An agent audit of the v374 traces found that the analytic model credits every
`ActivateAbility` with board gain. The KL picker therefore made 1,430 upkeep
activations over 60 games. One two-game probe animated Inkmoth Nexus 269 times
in upkeep, and Oblivion Stone's `DestroyAll` was cracked during our own upkeep.
The action-type heuristic had the same flaw: it ranked any activation above
passing. Across 10 Hardened Scales games it made 608 upkeep activations, spending
mana in upkeep and leaving lands tapped for the main phase.

`src/integrations/phase_rs/action_features.py::ActivationGate` is shared by
both pickers. It applies only to `ActivateAbility` at an ordinary `Priority`
stop; activations inside payment prompts continue the payment and are always
allowed. The gate rejects:

- mana abilities, because the engine prompts for payment and mana empties
  between steps;
- activations outside our own pre-combat main, beginning of combat or
  post-combat main;
- a second activation of the same (source, ability) in the same turn;
- `DestroyAll`, `DamageAll`, `ChangeZoneAll` and `SacrificeAll` unless the
  opponent controls more non-land permanents than we do.

The gate never empties the candidate set. `--activation-policy legacy`
reproduces the ungated pre-fix arms, so the fix can be measured as an ablation.
Traces record `activation_policy`.

**Payment hygiene (v2).** The first Standard campaign attempt exposed a
second shared flaw:
- Any-colour sources such as Starting Town prompt `ChooseManaColor` during
  payment. Both pickers chose the colour at random, so a Jund deck often
  produced White or Blue mana. The pending cast could then not be paid, and
  `CancelCast` became the only legal action.
- In target prompts, the heuristic's random fallback also sampled
  `CancelCast` among the `ChooseTarget` options.

Version 2 adds `useful_mana_colours`, which restricts colour prompts to colours
that appear in `pending_cast.cost.shards`, including hybrid and Phyrexian
shards. The heuristic also never chooses `CancelCast` unless it is the only
legal action. The policy tag is `action_hygiene_v2`; `legacy` disables all of
this.

**Mulligans.** The heuristic's random fallback also handled
`MulliganDecision`. In nine calibration games it took 13 mulligans and kept
only 9 hands. Under `hygiene`, both pickers use
`action_features.mulligan_choice`:

| Mulligans taken | Keep with this many lands |
|---|---|
| none (7 cards) | 2–5 |
| one | 1–5 |
| two or more | any |

Under the legacy policy, the heuristic still mulligans at random and KL keeps
every hand.

**Combat forecasts (`analytic_combat_v5`).** The attack step used to deal
`0.6 × our_board_power` regardless of which creatures were declared. Every
`DeclareAttackers` option therefore scored the same, and the argmin
tie-break chose index 0, the empty attack. Blocks had the same problem. In 24
Standard calibration games the KL picker declared no attackers in all 128
attack steps and won 2 games.

`mtg_transition.CombatForecast` now resolves each declaration against the
visible board:
- **Attacks:** each attacker, largest first, meets the strongest remaining
  untapped opposing creature that can legally block it. Flying is respected
  (only flying or reach creatures can block it), and so are deathtouch,
  trample and indestructible.
- **Blocks:** each declaration is evaluated against `combat.attackers`.
  Unblocked damage replaces the generic opponent clock for that step.

On the same 12 matchups, KL then attacked in 53 of 62 declarations and won
10 of 24 games. The forecast is still pessimistic and ignores combat tricks,
first strike and the defender's own risk. It is enabled only under `hygiene`,
so `kl_legacy` keeps the `analytic_pending_cast_v4` model as an ablation.

### 3.5 Explore to learn: the shared experience graph

`src/integrations/phase_rs/experience_graph.py` is the cross-game,
cross-agent memory. It is append-only, with one JSONL shard per writer
process, so parallel workers need no lock. A record is one game with its outcome
(draw = ½) and the deduplicated `experience_key`s of the choices made:
`(kind, card_class, card, stage)`, for example
`CastSpell / creature_mv2 / Heartfire Hero / early`.

Credit is first-visit Monte Carlo. Each node holds a Beta posterior, and child
nodes borrow `m = 4` pseudo-observations from their parent:
(kind, stage) → (kind, card class, stage) → (kind, class, card, stage).
Evidence about "two-drop creatures early" therefore transfers to unseen cards
and across decks.

The KL picker consumes the graph only through the **reference measure**:

$$
q(a) \;\propto\; q_0(a)\,\exp\!\big(\lambda\,(u_a - \bar u)\big), \qquad \lambda = 6,
$$

so deviating from historically winning choices costs $-\ln q(a)$ nats. No
term is added to the objective, and the invariant in §5 still holds. With an
attached graph, the heuristic picker breaks ties *within* a priority tier by
$u_a$ instead of by a coin flip. Evidence written by KL games therefore changes
heuristic play, which is the cross-agent sharing claim.

$u_a$ depends on `--exploration`:

| Mode | $u_a$ | Role |
|---|---|---|
| `greedy` | posterior mean | exploitation only |
| `thompson` (default) | one draw $\theta_a \sim \mathrm{Beta}(\alpha_a, \beta_a)$ | posterior sampling |
| `novelty` | mean $+ \;w\cdot I(\theta_a; o)$ | Active Inference parameter-novelty term |

$I(\theta;o) = H[\mathrm{Bern}(\bar\theta)] - \mathbb{E}_\theta H[\mathrm{Bern}(\theta)]$
has a closed form in digamma functions. This is exploration over **model
parameters**, not hidden game state. It is reducible, decays as roughly $1/n$,
and cannot be captured by a noisy TV the way `efe_infogain` can. That is why it
is allowed here, while the in-game objective keeps no epistemic bonus.
`--experience-mode read` evaluates a frozen graph; `learn` also appends to it.

Known limits:
- Credit is confounded by deck: a card that appears only in strong decks looks
  good.
- The stage bucket is coarse.
- The prior tilt is applied to every objective, but only `kl` uses the prior,
  so the legacy EFE arms are unaffected.

---

## 4. Defaults

| Parameter | `KLControlConfig` | Picker | Meaning |
|---|---|---|---|
| `horizon` | 3 | 3 | Planning depth $H$ |
| `beta` | 4.0 | 4.0 | Bounded-rationality precision |
| `rollouts` | 64 | 48 | MPPI sample count $K$ |
| `engine` | `mppi` | `mppi` | Planner |
| `move_cost` | 0.1 | — | Additive stage cost (nats) |
| `discount` | 1.0 | — | Per-step discount on the potential |
| `seed` | `None` | — | Threads both RNGs; required for determinism |

`PreferenceConfig` weights (nats per physical unit):

```
win_potential         -12.0      loss_potential         +12.0
life_differential       0.09     board_differential       0.16
permanent_differential  0.10     card_advantage           0.12
turn_pressure           0.02     decking_pressure         0.30
```

These are hand-set and unvalidated. They are the obvious thing to fit once win
rates exist.

---

## 5. The invariant under test

The single most important test in `tests/test_kl_control.py`:

```python
def test_no_epistemic_bonus_is_added_to_the_objective():
    """A belief change must not move the objective on its own."""
    flat  = BeliefState.uniform(["holds_nothing"])
    sharp = BeliefState(["holds_nothing"], np.array([1.0]))
    a = planner.plan(state, ACTIONS, flat,  prior)
    b = planner.plan(state, ACTIONS, sharp, prior)
    np.testing.assert_allclose(a.control_cost, b.control_cost, rtol=1e-12)
```

This is the structural property distinguishing belief-space KL control from
`efe_infogain`, and it is the one a future refactor is most likely to break —
someone will "improve" the agent by adding an exploration bonus.

Other tests assert: the objective decomposes exactly into control cost + risk;
softmin recovers `min` as $\beta\to\infty$ and the prior mean as $\beta\to 0$;
control cost is monotonically decreasing in $q(a)$; belief updates are Bayesian
and reduce entropy; the planner is deterministic under a fixed seed; the passive
prior never recommends conceding.

Two of these caught **my own** wrong expectations rather than code bugs
(the $\beta\to 0$ limit is the prior *mean*, not the minimum; the uniform
reference measure contributes $-\ln(1/n)/\beta$ at finite $\beta$).

---

## 6. How to run it

```powershell
# Single game, KL control
.\.venv\Scripts\python.exe scripts\run_phase_rs_ablation.py `
    --picker kl_control --objective kl --games 8 --autostart

# The three-arm objective ablation
foreach ($obj in @("kl","efe_infogain","efe_ambiguity")) {
  .\.venv\Scripts\python.exe scripts\run_phase_rs_ablation.py `
      --picker kl_control --objective $obj --games 20 --seed 7 --autostart `
      --output-dir runs/aif_objective_ablation
}

# Planner variants
--planner tree --horizon 2          # exact closed-loop DP
--planner mppi --rollouts 128       # more path-integral samples
--beta 0.5                          # near-prior-following
--beta 50                           # near-deterministic
```

**Status (November 2026):** the signed v374 `phase-server` is used in practice;
the local-build note below is historical.

```powershell
# Deck-diverse campaign with the shared graph (see docs/EXPERIMENTS.md)
.\.venv\Scripts\python.exe scripts\run_phase_rs_campaign.py run --name standard_rr `
    --format Standard --our-decks "data/decks/meta/standard/*.txt" `
    --arms random heuristic kl kl_graph_learn --games-per-cell 3 --workers 6
.\.venv\Scripts\python.exe scripts\analyze_phase_rs_campaign.py runs\campaigns\standard_rr
```

**Previously blocked**: `phase-server` does not build locally — the MSVC toolset
is installed but `link.exe` is not on `PATH`. Run the cargo build from a
Developer Command Prompt or wrap it in `vcvars64.bat`. Nothing in the Python
layer needs a live server to import or be tested.

---

## 7. What is honest about this, and what is not done

**Honest:**

- The objective is implemented as stated. There is no hidden epistemic term.
- The three ablation arms genuinely differ only in the scoring function.
- `LegacyEFE` implements the two published factorisations faithfully, not straw
  men. They are the objectives most discrete active-inference code actually uses.

**Not done / caveats:**

1. **Only a small, weak-opponent evaluation exists.** The v374 four-deck
   Modern benchmark is described in `docs/EXPERIMENTS.md`: five games per
   arm against VeryEasy Azorius Control, with every confidence interval
   overlapping. It shows that all three objectives complete games. It does not
   rank them, and the §6 pathology instrumentation has not been run.
2. **The transition model is analytic, not learned.** A path-integral controller
   is only useful because rollouts are cheap, so calling back into the rules
   engine $|A| \times K \times H$ times per decision is not viable. The intended
   replacement is a JEPA decoder behind the same `TransitionModel` protocol.
   Until then, plan quality is bounded by a ~60-line hand-written forward model.
   While a cast or activation is pending, every non-cancel action is modelled
   as continuing it. This includes confirming payment with `PassPriority`, mana
   abilities, convoke and `PayCost` selection. Scoring those actions as
   independent passes or activations made `CancelCast` look cheaper and caused
   cast/cancel loops on v374.
3. **The `tree` planner's continuation set is degenerate.** `_candidate_continuations()`
   returns `[None]`, so exact closed-loop DP currently degrades to a one-step
   lookahead. It does not fabricate illegal lines, but it also does not exercise
   the deep belief branching that produces the most interesting dual-control
   behaviour. MPPI does not branch on observations either, so **no planner that
   currently runs in games exhibits the dual-control exploration effect**; the
   claim that exploration falls out of closed-loop planning is a property of the
   formulation, not yet of this implementation.
4. **Preference weights are unfitted.**
5. **The belief is coarse** (4 hypotheses, no decklist conditioning). The
   existing `OpponentModel` particle filter is not yet wired in.
6. **We adopt the theory, we do not re-derive it.** Our contribution on this
   axis is the domain transfer plus the ablation design.

---

## 8. Is this "modern agentic programming"? (ReAct, skills, tools)

Asked during review; worth recording because it is a real architectural
decision rather than an omission.

**No, and deliberately not — for the game-playing agent.**

ReAct-style agents (reason → call tool → observe → repeat) exist to solve a
specific problem: an open-ended environment where the action space is *not*
enumerable and the model must decide what information to gather and which
capability to invoke. Tool/skill registries are the answer to "the space of
things you could do is huge and unstructured."

Our decision problem is the exact opposite shape:

- The action space is **enumerated by the engine on every priority window**.
- Legality is **decided externally** and is not the agent's problem.
- The output is **an index into a list**.

Putting an LLM reasoning loop in that position adds latency and hallucination
surface for no gain, and it is precisely what the earlier `LLMFusionAgent`
result showed: the most "agentic" component was the weakest player at
114 s/game.

The architectural claim is narrower and stronger than "use ReAct":

> Put the LLM in the **prior**, not in the control loop.
> Put a control-theoretic objective in the loop.

That is what $q(a)$ in §3.2 is for. Language-model knowledge enters as a
reference measure over legal actions; deviating from it costs $-\ln q(a)$ nats.
It is a principled home for a fuzzy signal, and it degrades gracefully — if the
LLM is unavailable, $q$ falls back to the structural prior and the objective is
unchanged in form.

**Where the modern agentic stack *does* fit in this repo:**

| Pattern | Where it belongs here | Status |
|---|---|---|
| Multi-agent generate → validate → repair | Ontology engineering (MASEO-style): propose axioms, check with reasoner/SHACL/CQ-SPARQL, repair, human-gate | Validator cascade built (`scripts/validate_ontology.py`); generation loop not built |
| Tools in a toolbox | The validator *is* this — rdflib, HermiT, pySHACL, SPARQL as independent checkers behind a uniform stage interface | Built, deterministic orchestration (no LLM controller) |
| RAG / retrieval | Rules lookup for the judge; retrieval-scoped ontology extension per competency question | Partially built |
| ReAct | Rules Q&A over the Comprehensive Rules, where the query *is* open-ended | Not built |

So: tools yes, toolbox yes, deterministic orchestration yes — ReAct only where
the action space is genuinely open. In the game loop it is not, and pretending
otherwise would be cargo-culting the pattern.

---

## 9. Reading order for review

1. `src/agents/kl_control.py` — module docstring first, then
   `BeliefSpaceKLControl.plan`, then `_control_cost` and `_risk_mppi`.
2. `src/agents/mtg_transition.py` — `MTGTransitionModel.step` and
   `observation_likelihood`. This is where the domain assumptions live.
3. `tests/test_kl_control.py` — `test_no_epistemic_bonus_is_added_to_the_objective`.
4. `paper/opposition_agents_mtg.tex` §`sec:aif` for the derivation and
   §`sec:eval-aif-ablation` for the experiment design.
5. `IMPLEMENTATION_PLAN.md` queue items **K1** (build blocker) and **K2**
   (the ablation).

## References

- Kaufmann, Rafael (2026). *Active Inference is Optimal Control.* SSRN preprint,
  [doi:10.2139/ssrn.7504418](https://doi.org/10.2139/ssrn.7504418) — the result
  this implements; shared citation key `kaufmann2026aifcontrol` in
  [references.bib](../paper/references.bib).
- Fel'dbaum, A. A. (1960). *Dual Control Theory I–II.* — where exploration actually comes from.
- Kappen (2005), Todorov (2009) — path-integral control / linearly solvable MDPs.
- Williams et al. (2017) — MPPI, the algorithm `_risk_mppi` implements.
- Millidge et al. (2021), *Whence the Expected Free Energy?* — the two factorisations only coincide under a condition hand-authored preferences never satisfy.

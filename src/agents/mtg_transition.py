"""MTG generative model for :mod:`src.agents.kl_control`.

Supplies the two things the belief-space KL-control planner needs and does
not know how to build itself:

* a projection from an engine snapshot onto :class:`LatentState`
  (the plant coordinates :math:`\\eta`), and
* a transition model :math:`p(\\eta' \\mid \\eta, a, h)` plus an
  observation likelihood used *only* to branch the belief tree.

The transition model is deliberately a fast analytic approximation rather
than a call back into the rules engine: the planner evaluates it
``|A| x K x H`` times per decision, and a path-integral controller is
useful precisely because the rollouts are cheap. A learned JEPA decoder
can be swapped in behind the same protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.agents.kl_control import BeliefState, LatentState

# Hidden-information hypotheses. Coarse on purpose: the planner only needs
# enough resolution for a probe to change the downstream decision.
OPPONENT_HYPOTHESES: tuple[str, ...] = (
    "holds_removal",
    "holds_counterspell",
    "holds_threat",
    "holds_nothing",
)


def default_belief(prior: dict[str, float] | None = None) -> BeliefState:
    belief = BeliefState.uniform(OPPONENT_HYPOTHESES)
    if prior:
        belief = belief.update(lambda h: prior.get(h, 1.0))
    return belief


@dataclass(frozen=True)
class ActionForecast:
    """Local analytic features; never submitted in place of a native action."""

    wire: dict[str, Any]
    damage: int
    target_is_ours: bool
    target_is_player: bool
    removed_power: int = 0
    removes_permanent: bool = False


@dataclass(frozen=True)
class CombatForecast:
    """Visible-board outcome of one attack or block declaration.

    Every declaration used to share one board-power forecast, so the empty
    declaration tied with every attack and won the argmin tie-break.
    """

    wire: dict[str, Any]
    damage_to_opponent: int = 0
    damage_to_us: int = 0
    our_lost_power: int = 0
    our_lost_permanents: int = 0
    opponent_lost_power: int = 0
    opponent_lost_permanents: int = 0


def _keywords(obj: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for keyword in obj.get("keywords") or []:
        if isinstance(keyword, str):
            names.add(keyword)
        elif isinstance(keyword, dict):
            names.update(str(k) for k in keyword)
            if isinstance(keyword.get("type"), str):
                names.add(keyword["type"])
    return names


def _is_creature(obj: dict[str, Any]) -> bool:
    return "Creature" in ((obj.get("card_types") or {}).get("core_types") or [])


def _kills(source: dict[str, Any], victim: dict[str, Any], damage: int) -> bool:
    if damage <= 0 or "Indestructible" in _keywords(victim):
        return False
    if "Deathtouch" in _keywords(source):
        return True
    return damage >= _int(victim.get("toughness")) - _int(victim.get("damage_marked"))


def _can_block(blocker: dict[str, Any], attacker: dict[str, Any]) -> bool:
    if "Flying" in _keywords(attacker):
        return bool({"Flying", "Reach"} & _keywords(blocker))
    return True


def _forecast_attack(
    action: dict[str, Any], state: dict[str, Any], seat: int,
) -> CombatForecast | None:
    objects = state.get("objects") or {}
    attackers = []
    for entry in (action.get("data") or {}).get("attacks") or []:
        if not isinstance(entry, list) or len(entry) != 2:
            return None
        obj = objects.get(str(entry[0]))
        if not isinstance(obj, dict):
            return None
        target = entry[1] if isinstance(entry[1], dict) else {}
        attackers.append((obj, target.get("type") == "Player"))
    blockers = sorted(
        (
            obj for oid in state.get("battlefield") or []
            if isinstance(obj := objects.get(str(oid)), dict)
            and obj.get("controller") != seat and _is_creature(obj) and not obj.get("tapped")
        ),
        key=lambda o: _int(o.get("power")), reverse=True,
    )
    # Pessimistic: each attacker, largest first, meets the strongest remaining
    # legal blocker. This ignores combat tricks and the defender's own risk.
    dealt = lost_power = lost = killed_power = killed = 0
    by_power = sorted(attackers, key=lambda a: _int(a[0].get("power")), reverse=True)
    for attacker, at_player in by_power:
        power = _int(attacker.get("power"))
        blocker = next((b for b in blockers if _can_block(b, attacker)), None)
        if blocker is None:
            dealt += power if at_player else 0
            continue
        blockers.remove(blocker)
        if "Trample" in _keywords(attacker) and at_player:
            dealt += max(0, power - _int(blocker.get("toughness")))
        if _kills(blocker, attacker, _int(blocker.get("power"))):
            lost_power += power
            lost += 1
        if _kills(attacker, blocker, power):
            killed_power += _int(blocker.get("power"))
            killed += 1
    return CombatForecast(
        action, damage_to_opponent=dealt, our_lost_power=lost_power, our_lost_permanents=lost,
        opponent_lost_power=killed_power, opponent_lost_permanents=killed,
    )


def _forecast_block(
    action: dict[str, Any], state: dict[str, Any], seat: int,
) -> CombatForecast | None:
    objects = state.get("objects") or {}
    incoming: dict[int, tuple[dict[str, Any], bool]] = {}
    for entry in (state.get("combat") or {}).get("attackers") or []:
        if not isinstance(entry, dict) or entry.get("defending_player") != seat:
            continue
        obj = objects.get(str(entry.get("object_id")))
        if not isinstance(obj, dict):
            return None
        target = entry.get("attack_target") or {}
        incoming[entry["object_id"]] = (obj, target.get("type") == "Player")
    blocks: dict[int, list[dict[str, Any]]] = {}
    for entry in (action.get("data") or {}).get("assignments") or []:
        if not isinstance(entry, list) or len(entry) != 2 or entry[1] not in incoming:
            return None
        blocker = objects.get(str(entry[0]))
        if not isinstance(blocker, dict):
            return None
        blocks.setdefault(entry[1], []).append(blocker)
    taken = lost_power = lost = killed_power = killed = 0
    for attacker_id, (attacker, at_player) in incoming.items():
        power = _int(attacker.get("power"))
        assigned = blocks.get(attacker_id, [])
        if not assigned:
            taken += power if at_player else 0
            continue
        remaining = power
        for blocker in sorted(assigned, key=lambda b: _int(b.get("toughness"))):
            needed = 1 if "Deathtouch" in _keywords(attacker) else max(
                1, _int(blocker.get("toughness")) - _int(blocker.get("damage_marked")),
            )
            if remaining >= needed and _kills(attacker, blocker, remaining):
                lost_power += _int(blocker.get("power"))
                lost += 1
            remaining = max(0, remaining - needed)
        if "Trample" in _keywords(attacker) and at_player:
            taken += remaining
        block_damage = sum(_int(b.get("power")) for b in assigned)
        if any(_kills(b, attacker, block_damage) for b in assigned):
            killed_power += power
            killed += 1
    return CombatForecast(
        action, damage_to_us=taken, our_lost_power=lost_power, our_lost_permanents=lost,
        opponent_lost_power=killed_power, opponent_lost_permanents=killed,
    )


def forecast_action(
    action: dict[str, Any], state: dict[str, Any], seat: int,
) -> dict[str, Any] | ActionForecast | CombatForecast:
    """Resolve conservative combat and two-player fixed-damage target forecasts."""
    if action.get("type") == "DeclareAttackers":
        return _forecast_attack(action, state, seat) or action
    if action.get("type") == "DeclareBlockers":
        return _forecast_block(action, state, seat) or action
    if action.get("type") not in {"ChooseTarget", "SelectTargets"}:
        return action
    # Opponent life is a minimum aggregate, not a target-specific multiplayer value.
    if len(state.get("players") or []) != 2:
        return action
    waiting_data = (state.get("waiting_for") or {}).get("data") or {}
    if waiting_data.get("player") != seat:
        return action
    pending = waiting_data.get("pending_cast") or state.get("pending_cast") or {}
    ability = pending.get("ability") or {}
    if ability.get("sub_ability") or ability.get("condition"):
        return action
    effect = ability.get("effect") or {}
    amount = effect.get("amount") or {}
    if effect.get("type") != "DealDamage" or amount.get("type") != "Fixed":
        return action
    damage = amount.get("value")
    if type(damage) is not int or damage < 0:
        return action
    data = action.get("data") or {}
    targets = data.get("targets") if action.get("type") == "SelectTargets" else [data.get("target")]
    if not isinstance(targets, list) or len(targets) != 1:
        return action
    target = targets[0]
    if not isinstance(target, dict):
        return action
    if "Player" in target:
        return ActionForecast(action, damage, target["Player"] == seat, True)
    if "Object" not in target:
        return action
    objects = state.get("objects") or {}
    obj = objects.get(str(target["Object"])) or objects.get(target["Object"])
    if not isinstance(obj, dict) or type(obj.get("toughness")) is not int:
        return action
    keywords = obj.get("keywords") or []
    if not all(isinstance(keyword, str) for keyword in keywords):
        return action
    lethal = (
        damage >= obj["toughness"] - _int(obj.get("damage_marked"))
        and "Indestructible" not in keywords
    )
    return ActionForecast(
        action, damage, obj.get("controller") == seat, False,
        removed_power=_int(obj.get("power")) if lethal else 0,
        removes_permanent=lethal,
    )


# ---------------------------------------------------------------------------
# Snapshot -> LatentState
# ---------------------------------------------------------------------------


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def latent_from_phase_rs(state: dict[str, Any], our_seat: int) -> LatentState:
    """Project a phase-rs ``StateUpdate.state`` snapshot onto plant coordinates."""
    players = state.get("players") or []
    me: dict[str, Any] = {}
    opponents: list[dict[str, Any]] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        if _int(p.get("id"), -1) == our_seat:
            me = p
        else:
            opponents.append(p)

    objects = state.get("objects") or {}
    battlefield = state.get("battlefield") or []

    our_power = opp_power = 0
    our_perms = opp_perms = 0
    for oid in battlefield:
        obj = objects.get(str(oid)) or objects.get(oid)
        if not isinstance(obj, dict):
            continue
        controller = _int(obj.get("controller"), -1)
        power = _int(obj.get("power"), 0)
        if controller == our_seat:
            our_perms += 1
            our_power += power
        else:
            opp_perms += 1
            opp_power += power

    opp_life = min(
        (_int(o.get("life"), 20) for o in opponents), default=20
    )
    waiting = state.get("waiting_for") or {}
    waiting_data = waiting.get("data") or {}
    pending = waiting_data.get("pending_cast") or state.get("pending_cast")
    return LatentState(
        our_life=_int(me.get("life"), 20),
        opponent_life=opp_life,
        our_board_power=our_power,
        opponent_board_power=opp_power,
        our_permanents=our_perms,
        opponent_permanents=opp_perms,
        our_hand_size=len(me.get("hand") or []),
        our_library_size=len(me.get("library") or []),
        turn=_int(state.get("turn_number"), 0),
        we_are_eliminated=bool(me.get("is_eliminated", False)),
        opponent_eliminated=bool(
            opponents and all(o.get("is_eliminated", False) for o in opponents)
        ),
        our_pending_cast=bool(pending and waiting_data.get("player") == our_seat),
    )


# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

_LAND = frozenset({"PlayLand"})
_CAST = frozenset({"CastSpell", "PlaySpell", "Cast"})
_ACTIVATE = frozenset({"ActivateAbility", "Activate"})
_ATTACK = frozenset({"DeclareAttackers", "Attack"})
_BLOCK = frozenset({"DeclareBlockers", "Block"})
_PASS = frozenset({"PassPriority", "Pass"})
_CONCEDE = frozenset({"Concede"})
_CAST_CHOICE = frozenset({
    "SelectModes", "SelectTargets", "ChooseTarget", "ChooseX", "OrderCostReductions",
    "DecideOptionalCost", "ChooseAlternativeCast", "ChooseCastingVariant",
})
_CANCEL_CAST = frozenset({"CancelCast"})


def action_kind(action: Any) -> str:
    """Normalise a phase-rs ``GameAction`` dict (or ``None``) to a coarse kind."""
    if action is None:
        return "pass"
    if isinstance(action, (ActionForecast, CombatForecast)):
        action = action.wire
    raw = action.get("type", "") if isinstance(action, dict) else str(action)
    if raw in _LAND:
        return "land"
    if raw in _CAST:
        return "cast"
    if raw in _CAST_CHOICE:
        return "cast_choice"
    if raw in _CANCEL_CAST:
        return "cancel_cast"
    if raw in _ACTIVATE:
        return "activate"
    if raw in _ATTACK:
        return "attack"
    if raw in _BLOCK:
        return "block"
    if raw in _CONCEDE:
        return "concede"
    if raw in _PASS:
        return "pass"
    return "other"


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------


@dataclass
class MTGTransitionConfig:
    """Per-action-kind expected effect sizes, in plant units."""

    cast_board_gain: int = 2
    cast_permanent_gain: int = 1
    activate_board_gain: int = 1
    attack_damage_fraction: float = 0.6
    opponent_clock: int = 2
    # How much of our attack a hypothesised blocker/removal absorbs.
    removal_absorption: float = 0.5
    counter_absorption: float = 1.0
    threat_extra_clock: int = 2


class MTGTransitionModel:
    """Analytic one-step forward model over :class:`LatentState`.

    The hypothesis enters exactly where hidden information actually bites:
    a counterspell blanks a cast, removal blanks a board commitment, and a
    hidden threat accelerates the opponent's clock. That coupling is what
    makes information about the opponent's hand *operationally* valuable,
    which in turn is what makes the closed-loop planner probe.

    Casting subdecisions forecast completion of the pending spell as one
    macro-step, not resolution of an individual mode/target announcement.
    The forecast is still generic: it does not distinguish card effects or
    targets, and must not be treated as a rules-exact dynamics model.
    """

    def __init__(self, config: MTGTransitionConfig | None = None) -> None:
        self.config = config or MTGTransitionConfig()

    def step(self, state: LatentState, action: Any, hypothesis: str) -> LatentState:
        c = self.config
        kind = action_kind(action)
        if state.our_pending_cast and kind not in {"cancel_cast", "concede"}:
            # While an announcement is pending, every other legal action
            # (mana abilities, convoke taps, cost-card selection, confirming
            # payment via PassPriority) continues it. Scoring those as
            # independent activations/passes that pay the opponent clock made
            # cancellation look strictly cheaper and caused cast/cancel loops.
            kind = "cast_choice"
        if kind == "cancel_cast":
            return state.replace(our_pending_cast=False)
        if kind == "cast_choice" and not state.our_pending_cast:
            # Rollouts reuse the root candidate set. A pending spell's
            # completion forecast must be credited at most once.
            return state

        our_power = state.our_board_power
        our_perms = state.our_permanents
        our_hand = state.our_hand_size
        opp_life = state.opponent_life
        our_life = state.our_life
        opp_power = state.opponent_board_power
        opp_perms = state.opponent_permanents

        if kind == "land":
            our_perms += 1
            our_hand = max(0, our_hand - 1)
        elif kind in {"cast", "cast_choice"}:
            our_hand = max(0, our_hand - 1)
            if isinstance(action, ActionForecast):
                if hypothesis != "holds_counterspell":
                    if action.target_is_player:
                        if action.target_is_ours:
                            our_life = max(0, our_life - action.damage)
                        else:
                            opp_life = max(0, opp_life - action.damage)
                    elif action.target_is_ours:
                        our_power = max(0, our_power - action.removed_power)
                        our_perms = max(0, our_perms - int(action.removes_permanent))
                    else:
                        opp_power = max(0, opp_power - action.removed_power)
                        opp_perms = max(0, opp_perms - int(action.removes_permanent))
            else:
                gain = c.cast_board_gain
                if hypothesis == "holds_counterspell":
                    gain = int(gain * (1.0 - c.counter_absorption))
                elif hypothesis == "holds_removal":
                    gain = int(gain * (1.0 - c.removal_absorption))
                our_power += gain
                our_perms += c.cast_permanent_gain if gain > 0 else 0
        elif kind == "activate":
            our_power += c.activate_board_gain
        elif kind in {"attack", "block"} and isinstance(action, CombatForecast):
            dealt = action.damage_to_opponent
            if kind == "attack" and hypothesis == "holds_removal":
                dealt = int(dealt * (1.0 - c.removal_absorption))
            opp_life = max(0, opp_life - dealt)
            our_life = max(0, our_life - action.damage_to_us)
            our_power = max(0, our_power - action.our_lost_power)
            our_perms = max(0, our_perms - action.our_lost_permanents)
            opp_power = max(0, opp_power - action.opponent_lost_power)
            opp_perms = max(0, opp_perms - action.opponent_lost_permanents)
        elif kind == "attack":
            dealt = int(state.our_board_power * c.attack_damage_fraction)
            if hypothesis == "holds_removal":
                dealt = int(dealt * (1.0 - c.removal_absorption))
            opp_life = max(0, opp_life - dealt)
        elif kind == "concede":
            return state.replace(we_are_eliminated=True)

        # Opponent's answer-back over the same step. A block forecast already
        # carries this combat's incoming damage.
        clock = c.opponent_clock + (
            c.threat_extra_clock if hypothesis == "holds_threat" else 0
        )
        resolved_combat = kind == "block" and isinstance(action, CombatForecast)
        incoming = (
            clock if state.opponent_board_power > 0 and kind != "cast_choice"
            and not resolved_combat else 0
        )
        our_life = max(0, our_life - incoming)

        return state.replace(
            our_life=our_life,
            opponent_life=opp_life,
            our_board_power=our_power,
            our_permanents=our_perms,
            our_hand_size=our_hand,
            opponent_board_power=opp_power,
            opponent_permanents=opp_perms,
            turn=state.turn + (1 if kind == "pass" else 0),
            we_are_eliminated=our_life <= 0,
            opponent_eliminated=opp_life <= 0,
            our_pending_cast=False if kind == "cast_choice" else state.our_pending_cast,
        )

    def observation_likelihood(
        self, state: LatentState, action: Any, hypothesis: str
    ) -> float:
        """How diagnostic this action is about ``hypothesis``.

        Actions that force the opponent to reveal or respond (casting into
        open mana, attacking into untapped blockers) carry a
        hypothesis-dependent likelihood; inert actions do not. This is the
        *only* channel through which exploration can pay off, and it
        affects the objective only indirectly, via sharper posteriors
        unlocking cheaper continuations.
        """
        kind = action_kind(action)
        opponent_can_respond = state.opponent_permanents > 0

        if kind == "cast" and opponent_can_respond:
            return {
                "holds_counterspell": 1.6,
                "holds_removal": 1.2,
                "holds_threat": 0.9,
                "holds_nothing": 0.7,
            }.get(hypothesis, 1.0)
        if kind == "attack" and state.opponent_permanents > 0:
            return {
                "holds_removal": 1.4,
                "holds_threat": 1.1,
                "holds_counterspell": 0.9,
                "holds_nothing": 0.8,
            }.get(hypothesis, 1.0)
        return 1.0


# ---------------------------------------------------------------------------
# Passive prior
# ---------------------------------------------------------------------------

# Reference measure P0 over action kinds. The control cost is charged
# against this, so it encodes "what a game of Magic normally looks like",
# not "what is good" — preferences live solely in the potential V.
_PASSIVE_WEIGHTS: dict[str, float] = {
    "land": 6.0,
    "cast": 4.0,
    "cast_choice": 4.0,
    "cancel_cast": 1.0,
    "attack": 3.0,
    "block": 3.0,
    "activate": 2.0,
    "pass": 1.0,
    "other": 1.0,
    "concede": 1e-6,
}


def passive_prior(
    actions: Sequence[Any], extra_scores: Sequence[float] | None = None
) -> list[float]:
    """Build ``q(a)`` over the legal set.

    ``extra_scores`` lets a heuristic / KG / LLM signal tilt the reference
    measure. Because the planner charges ``-ln q(a)``, those signals become
    a thermodynamic price on deviating from an informed default rather than
    an unprincipled additive bonus.
    """
    weights = [_PASSIVE_WEIGHTS.get(action_kind(a), 1.0) for a in actions]
    if extra_scores is not None and len(extra_scores) == len(actions):
        weights = [w * max(1e-6, 1.0 + float(s)) for w, s in zip(weights, extra_scores)]
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1.0 / len(actions)] * len(actions)

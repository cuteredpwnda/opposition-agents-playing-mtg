"""Activation hygiene gate, shared experience graph and their picker wiring."""

from __future__ import annotations

import math

import pytest

from src.integrations.phase_rs.action_features import ActivationGate, experience_key
from src.integrations.phase_rs.agent_bridge import HeuristicActionPicker
from src.integrations.phase_rs.experience_graph import (
    ExperienceGraph,
    ExperienceScorer,
    ExperienceSession,
    beta_bernoulli_information_gain,
)
from src.integrations.phase_rs.kl_control_picker import KLControlActionPicker

PASS = {"type": "PassPriority"}


def _state(*, phase="PreCombatMain", active=0, waiting="Priority", turn=3, objects=None,
           battlefield=None):
    return {
        "phase": phase, "active_player": active, "turn_number": turn,
        "waiting_for": {"type": waiting, "data": {"player": 0}},
        "objects": objects or {}, "battlefield": battlefield or [],
        "players": [{"life": 20}, {"life": 20}],
    }


def _ability(effect: str) -> dict:
    return {"abilities": [{"effect": {"type": effect}, "cost": {}}]}


def _activate(source=10) -> dict:
    return {"type": "ActivateAbility", "data": {"source_id": source, "ability_index": 0}}


def _obj(name, controller, core, effect="Pump"):
    return {"name": name, "controller": controller,
            "card_types": {"core_types": core}, **_ability(effect)}


def test_gate_blocks_mana_upkeep_repeats_and_self_defeating_wipes():
    gate = ActivationGate()
    animate = {"10": _obj("Inkmoth Nexus", 0, ["Land"], "Animate")}
    assert not gate.allows(_activate(), _state(phase="Upkeep", objects=animate), 0)
    assert not gate.allows(_activate(), _state(active=1, objects=animate), 0)
    main = _state(objects=animate)
    assert gate.allows(_activate(), main, 0)
    gate.record(_activate(), main)
    assert not gate.allows(_activate(), main, 0)
    assert gate.allows(_activate(), _state(turn=4, objects=animate), 0)

    mana = {"10": _obj("Theorist's Sanctum", 0, ["Land"], "Mana")}
    assert not gate.allows(_activate(), _state(objects=mana), 0)

    stone = {
        "10": _obj("Oblivion Stone", 0, ["Artifact"], "DestroyAll"),
        "11": _obj("Bear", 1, ["Creature"]),
    }
    gate = ActivationGate()
    assert not gate.allows(_activate(), _state(objects=stone, battlefield=[10, 11]), 0)
    stone.update({"12": _obj("Bear", 1, ["Creature"])})
    assert gate.allows(_activate(), _state(objects=stone, battlefield=[10, 11, 12]), 0)


def test_gate_allows_activations_inside_payment_prompts():
    mana = {"10": _obj("Forest", 0, ["Land"], "Mana")}
    gate = ActivationGate()
    assert gate.allows(_activate(), _state(phase="Upkeep", waiting="ManaPayment", objects=mana), 0)


def test_heuristic_passes_instead_of_upkeep_activation_unless_legacy():
    state = _state(phase="Upkeep", objects={"10": _obj("Mutavault", 0, ["Land"], "Animate")})
    actions = [_activate(), PASS]
    assert HeuristicActionPicker(seed=1).pick(actions, state, 0) == 1
    assert HeuristicActionPicker(seed=1, activation_policy="legacy").pick(actions, state, 0) == 0


def _colour(colour: str) -> dict:
    return {"type": "ChooseManaColor",
            "data": {"choice": {"type": "SingleColor", "data": colour}, "count": 1}}


@pytest.mark.parametrize("make", [
    lambda: HeuristicActionPicker(seed=1),
    lambda: KLControlActionPicker(seed=2, rollouts=8),
], ids=["heuristic", "kl"])
def test_colour_prompts_pick_a_colour_the_pending_cost_needs(make):
    state = _state(waiting="ChooseManaColor")
    state["pending_cast"] = {"cost": {"type": "Cost", "shards": ["Black"], "generic": 1}}
    colours = [_colour(c) for c in ("White", "Blue", "Black", "Red", "Green")]
    picker = make()
    assert {picker.pick(colours, state, 0) for _ in range(8)} == {2}


def test_heuristic_never_cancels_a_cast_while_targeting():
    state = _state(waiting="TargetSelection")
    actions = [{"type": "ChooseTarget", "data": {"target": i}} for i in range(4)]
    actions.append({"type": "CancelCast"})
    picks = {HeuristicActionPicker(seed=s).pick(actions, state, 0) for s in range(40)}
    assert 4 not in picks
    assert HeuristicActionPicker(seed=0).pick([{"type": "CancelCast"}], state, 0) == 0


def _mulligan_state(lands: int, count: int = 0) -> tuple[list[dict], dict]:
    objects = {
        str(i): {"name": f"c{i}",
                 "card_types": {"core_types": ["Land" if i < lands else "Creature"]}}
        for i in range(7)
    }
    state = _state(waiting="MulliganDecision", objects=objects)
    state["waiting_for"]["data"] = {"pending": [{"player": 0, "mulligan_count": count}]}
    state["players"][0]["hand"] = list(range(7))
    actions = [{"type": "MulliganDecision", "data": {"choice": {"type": kind}}}
               for kind in ("Keep", "Mulligan")]
    return actions, state


@pytest.mark.parametrize(("lands", "count", "expected"), [
    (0, 0, 1), (1, 0, 1), (3, 0, 0), (6, 0, 1), (1, 1, 0), (0, 2, 0),
])
def test_both_pickers_mulligan_by_land_count(lands, count, expected):
    actions, state = _mulligan_state(lands, count)
    assert HeuristicActionPicker(seed=0).pick(actions, state, 0) == expected
    assert KLControlActionPicker(seed=0, rollouts=8).pick(actions, state, 0) == expected


def _creature(controller, power, toughness, keywords=(), tapped=False):
    return {"name": f"c{power}/{toughness}", "controller": controller, "power": power,
            "toughness": toughness, "keywords": list(keywords), "tapped": tapped,
            "card_types": {"core_types": ["Creature"]}}


def _attacks(*ids):
    return {"type": "DeclareAttackers",
            "data": {"attacks": [[i, {"type": "Player", "data": 1}] for i in ids], "bands": []}}


@pytest.mark.parametrize(("opponent", "expected"), [
    ({}, 1),                                     # open board: swing
    ({"9": _creature(1, 5, 5)}, 0),              # 2/2 into an untapped 5/5: hold
    ({"9": _creature(1, 5, 5, tapped=True)}, 1),  # blocker tapped: swing
])
def test_kl_combat_forecast_distinguishes_attack_declarations(opponent, expected):
    objects = {"1": _creature(0, 2, 2), **opponent}
    state = _state(phase="DeclareAttackers", waiting="DeclareAttackers", objects=objects,
                   battlefield=[int(k) for k in objects])
    picker = KLControlActionPicker(seed=0, rollouts=8)
    assert picker.pick([_attacks(), _attacks(1)], state, 0) == expected
    assert picker.last_reasoning["combat_forecasts"] == 2


def test_kl_combat_forecast_respects_flying():
    objects = {"1": _creature(0, 2, 2, keywords=["Flying"]), "9": _creature(1, 5, 5)}
    state = _state(phase="DeclareAttackers", waiting="DeclareAttackers", objects=objects,
                   battlefield=[1, 9])
    assert KLControlActionPicker(seed=0, rollouts=8).pick([_attacks(), _attacks(1)], state, 0) == 1

def test_kl_combat_forecast_blocks_profitably_and_legacy_keeps_old_model():
    objects = {"1": _creature(0, 4, 4), "9": _creature(1, 2, 2)}
    state = _state(phase="DeclareBlockers", active=1, waiting="DeclareBlockers",
                   objects=objects, battlefield=[1, 9])
    state["combat"] = {"attackers": [{"object_id": 9, "defending_player": 0,
                                      "attack_target": {"type": "Player", "data": 0}}]}
    actions = [{"type": "DeclareBlockers", "data": {"assignments": []}},
               {"type": "DeclareBlockers", "data": {"assignments": [[1, 9]]}}]
    assert KLControlActionPicker(seed=0, rollouts=8).pick(actions, state, 0) == 1
    legacy = KLControlActionPicker(seed=0, rollouts=8, activation_policy="legacy")
    legacy.pick(actions, state, 0)
    assert legacy.last_reasoning["combat_forecasts"] == 0


def test_kl_picker_does_not_activate_in_upkeep():
    state = _state(phase="Upkeep", objects={"10": _obj("Mutavault", 0, ["Land"], "Animate")})
    picker = KLControlActionPicker(seed=3, rollouts=8)
    assert picker.pick([_activate(), PASS], state, 0) == 1
    assert picker.last_reasoning["activation_policy"] == "action_hygiene_v2"


def test_information_gain_decays_with_evidence():
    gains = [beta_bernoulli_information_gain(1 + n / 2, 1 + n / 2) for n in (0, 4, 40, 400)]
    assert all(a > b for a, b in zip(gains, gains[1:]))
    assert gains[-1] < gains[0] / 100


def test_graph_shards_aggregate_and_back_off_to_card_class(tmp_path):
    winner = ("CastSpell", "creature_mv2", "Heartfire Hero", "early")
    loser = ("CastSpell", "creature_mv2", "Bad Bear", "early")
    a = ExperienceGraph.load(tmp_path)
    b = ExperienceGraph.load(tmp_path)
    for _ in range(6):
        a.record_game([winner], "win")
        b.record_game([loser], "loss")
    assert a.record_game([winner], "incomplete") is False
    assert len(list(tmp_path.glob("*.jsonl"))) == 2

    merged = ExperienceGraph.load(tmp_path)
    assert merged.games == 12

    def mean(key):
        alpha, beta = merged.posterior(key)
        return alpha / (alpha + beta)

    unseen = ("CastSpell", "creature_mv2", "New Card", "early")
    assert mean(winner) > mean(unseen) > mean(loser)
    assert math.isclose(mean(unseen), 0.5, abs_tol=0.05)


def test_thompson_scorer_is_seeded(tmp_path):
    graph = ExperienceGraph.load(tmp_path)
    key = ("PlayLand", "land_mv0", "Forest", "early")
    draws = [ExperienceScorer(graph, mode="thompson", seed=5).utility(key) for _ in range(2)]
    assert draws[0] == draws[1]
    with pytest.raises(ValueError):
        ExperienceScorer(graph, mode="ucb")


def test_kl_learn_session_records_and_tilts_toward_winners(tmp_path):
    objects = {
        "1": {"name": "Winner", "card_types": {"core_types": ["Creature"]},
              "mana_cost": {"generic": 1, "shards": ["G"]}},
        "2": {"name": "Loser", "card_types": {"core_types": ["Creature"]},
              "mana_cost": {"generic": 1, "shards": ["G"]}},
    }
    state = _state(objects=objects)
    cast = [{"type": "CastSpell", "data": {"object_id": i, "card_id": i}} for i in (1, 2)]
    graph = ExperienceGraph.load(tmp_path)
    for _ in range(30):
        graph.record_game([experience_key(cast[0], state)], "win")
        graph.record_game([experience_key(cast[1], state)], "loss")

    session = ExperienceSession(
        ExperienceScorer(ExperienceGraph.load(tmp_path), mode="greedy"), learn=True, agent="kl",
    )
    picker = KLControlActionPicker(seed=1, rollouts=8, experience=session)
    assert picker.pick(cast, state, 0) == 0
    assert picker.observe_outcome("win", {"seed": 1}) is True
    assert ExperienceGraph.load(tmp_path).games == 61


def test_heuristic_reads_frozen_graph_without_writing(tmp_path):
    graph = ExperienceGraph.load(tmp_path)
    state = _state(objects={"1": {"name": "Good"}, "2": {"name": "Bad"}})
    lands = [{"type": "PlayLand", "data": {"object_id": i, "card_id": i}} for i in (1, 2)]
    for _ in range(20):
        graph.record_game([experience_key(lands[1], state)], "win")
        graph.record_game([experience_key(lands[0], state)], "loss")
    session = ExperienceSession(
        ExperienceScorer(ExperienceGraph.load(tmp_path), mode="greedy"), learn=False, agent="h",
    )
    picker = HeuristicActionPicker(seed=0, experience=session)
    assert picker.pick(lands, state, 0) == 1
    assert picker.observe_outcome("win") is False
    assert ExperienceGraph.load(tmp_path).games == 40

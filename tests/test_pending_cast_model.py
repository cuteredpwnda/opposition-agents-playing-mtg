from __future__ import annotations

import pytest

from src.agents.mtg_transition import (
    MTGTransitionModel,
    action_kind,
    forecast_action,
    latent_from_phase_rs,
)
from src.integrations.phase_rs.kl_control_picker import KLControlActionPicker


def snapshot(player=0, embedded=True):
    pending = {"object_id": 43, "card_id": 43}
    state = {
        "turn_number": 6, "phase": "Upkeep",
        "players": [
            {"id": 0, "life": 10, "hand": [1, 2, 3, 4, 43], "library": list(range(51))},
            {"id": 1, "life": 12, "hand": [5], "library": list(range(52))},
        ],
        "battlefield": [10, 11],
        "objects": {"10": {"controller": 0, "power": 2},
                    "11": {"controller": 1, "power": 3}},
        "waiting_for": {"type": "ModeChoice", "data": {"player": player}},
    }
    if embedded:
        state["waiting_for"]["data"]["pending_cast"] = pending
    else:
        state["pending_cast"] = pending
    return state


@pytest.mark.parametrize("embedded", [False, True])
def test_pending_cast_is_projected_only_for_the_acting_seat(embedded):
    assert latent_from_phase_rs(snapshot(embedded=embedded), 0).our_pending_cast
    assert not latent_from_phase_rs(snapshot(player=1, embedded=embedded), 0).our_pending_cast


def test_casting_choices_and_cancel_have_distinct_physical_forecasts():
    model = MTGTransitionModel()
    state = latent_from_phase_rs(snapshot(), 0)
    cancel = model.step(state, {"type": "CancelCast"}, "holds_nothing")
    assert cancel == state.replace(our_pending_cast=False)
    choice = {"type": "SelectModes", "data": {"indices": [0]}}
    completed = model.step(state, choice, "holds_nothing")
    assert completed.our_board_power > state.our_board_power
    assert completed.our_hand_size == state.our_hand_size - 1
    assert completed.our_life == state.our_life
    assert completed.turn == state.turn
    assert not completed.our_pending_cast
    assert model.step(completed, choice, "holds_nothing") == completed


@pytest.mark.parametrize("objective", ["kl", "efe_infogain", "efe_ambiguity"])
@pytest.mark.parametrize("seed", [7, 8])
@pytest.mark.parametrize("choice", ["SelectModes", "ChooseTarget"])
def test_native_reproducer_does_not_tie_cancel_with_every_choice(objective, seed, choice):
    actions = [{"type": "CancelCast"}] + [
        {"type": choice, "data": (
            {"indices": [i]} if choice == "SelectModes" else {"target": {"Player": i}}
        )} for i in range(3)
    ]
    picker = KLControlActionPicker(seed=seed, objective=objective, rollouts=16)
    chosen = picker.pick(actions, snapshot(), 0)
    assert 1 <= chosen <= 3
    assert picker.last_reasoning["pending_cast"] is True
    assert actions[0] == {"type": "CancelCast"}


def test_cancel_remains_available_when_it_is_the_only_native_choice():
    picker = KLControlActionPicker(seed=7)
    assert picker.pick([{"type": "CancelCast"}], snapshot(), 0) == 0
    assert action_kind({"type": "CancelCast"}) == "cancel_cast"


@pytest.mark.parametrize("objective", ["kl", "efe_infogain", "efe_ambiguity"])
def test_native_fixed_damage_preserves_target_direction_and_original_index(objective):
    state = snapshot()
    state["waiting_for"]["type"] = "TargetSelection"
    pending = state["waiting_for"]["data"]["pending_cast"]
    pending["ability"] = {
        "effect": {"type": "DealDamage", "amount": {"type": "Fixed", "value": 3}},
        "sub_ability": None, "condition": None,
    }
    actions = [{"type": "ChooseTarget", "data": {"target": {"Player": i}}} for i in (0, 1)]
    actions.append({"type": "CancelCast"})
    picker = KLControlActionPicker(seed=7, objective=objective, rollouts=16)
    assert picker.pick(actions, state, 0) == 1
    latent = latent_from_phase_rs(state, 0)
    model = MTGTransitionModel()
    own = model.step(latent, forecast_action(actions[0], state, 0), "holds_nothing")
    enemy = model.step(latent, forecast_action(actions[1], state, 0), "holds_nothing")
    assert own.our_life == latent.our_life - 3
    assert enemy.opponent_life == latent.opponent_life - 3
    assert enemy.our_board_power == latent.our_board_power
    assert model.step(enemy, forecast_action(actions[1], state, 0), "holds_nothing") == enemy
    assert actions[1] == {"type": "ChooseTarget", "data": {"target": {"Player": 1}}}


def test_multiplayer_damage_does_not_reduce_the_minimum_opponents_life():
    state = snapshot()
    state["players"].append({"id": 2, "life": 40})
    state["waiting_for"]["data"]["pending_cast"]["ability"] = {
        "effect": {"type": "DealDamage", "amount": {"type": "Fixed", "value": 3}},
    }
    action = {"type": "ChooseTarget", "data": {"target": {"Player": 2}}}
    assert forecast_action(action, state, 0) is action

import asyncio
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import run_native_learning
from src.integrations.phase_rs.agent_bridge import HeuristicActionPicker
from src.integrations.phase_rs.native_learning_data import (
    NativeRecordingPicker,
    encode_action,
    observation,
)
from src.world_model.native_learning import (
    NativeLearnedPicker,
    diagnostics,
    induce_graph,
    make_model,
    parameter_digest,
    save_bundle,
    train_components,
    train_graph,
)


def state():
    return {
        "turn_number": 2, "phase": "PreCombatMain", "active_player": 0,
        "priority_player": 0,
        "players": [
            {"id": 0, "life": 20, "hand": ["42"], "library": ["99"]},
            {"id": 1, "life": 18, "hand": ["43"], "library": []},
        ],
        "objects": {
            "42": {"name": "Lightning Bolt", "zone": "Hand", "owner": 0,
                   "controller": 0, "oracle_text": "Deal 3 damage."},
            "43": {"name": "Opponent Secret", "zone": "Hand", "owner": 1},
            "99": {"name": "Library Secret", "zone": "Library", "owner": 0},
            "44": {"name": "Mountain", "zone": "Battlefield", "owner": 0, "controller": 0},
        },
        "stack": [],
    }


def game(game_id="train1", cards=None, split="train", outcome=1):
    original = state()
    actions = [{"type": "PlayLand", "data": {"source": 44}}, {"type": "PassPriority"}]
    recorder = NativeRecordingPicker(HeuristicActionPicker(7))
    recorder.pick(actions, original, 0)
    following = deepcopy(original)
    following["turn_number"] = 3
    recorder.pick(actions, following, 0)
    result = SimpleNamespace(
        trace=[{"event": "game_over"}], final_state=following, our_seat=0,
        winner_seat=0 if outcome > 0 else 1, reason="game_rules", turns_observed=3,
    )
    data = recorder.finish(result)
    for record in data["records"]:
        record["visible_cards"] = cards or ["Lightning Bolt", "Mountain"]
    data.update({
        "game_id": game_id, "split": split, "producer": "heuristic",
        "deck_sha256": "deck", "source_sha256": "source",
    })
    return data


def test_features_exclude_hidden_names_and_have_real_arrays():
    features, names = observation(state(), 0)
    assert names == ["Lightning Bolt", "Mountain"]
    assert features and all(torch.isfinite(value).all() for value in features.values())
    assert features["hand_cards"].abs().sum() > 0
    assert features["player_features"][7].item() == pytest.approx(0.1)
    assert features["player_features"][8].item() == pytest.approx(1 / 60)
    assert features["opponent_features"][7].item() == pytest.approx(0.1)
    changed = state()
    changed["objects"]["43"]["name"] = "Another Secret"
    other, _ = observation(changed, 0)
    assert all(torch.equal(features[key], other[key]) for key in features)


def test_public_counts_survive_redaction_and_invalid_counts_fail():
    original = state()
    original["players"][0].update({"library_size": 53, "hand_size": 7})
    original["players"][1].update({"library_size": 54, "hand_size": 6})
    features, names = observation(original, 0)
    assert "Library Secret" not in names and "Opponent Secret" not in names
    assert features["player_features"][7:9].tolist() == pytest.approx([0.7, 53 / 60])
    assert features["opponent_features"][7:9].tolist() == pytest.approx([0.6, 54 / 60])
    original["players"][1]["hand_size"] = -1
    with pytest.raises(ValueError, match="Invalid native public hand count"):
        observation(original, 0)


def test_native_pending_choice_context_is_encoded_without_hidden_names():
    original = state()
    features, _ = observation(original, 0)
    pending = {**original, "waiting_for": {"type": "ChooseOption", "card_name": "Secret"},
               "has_pending_cast": True}
    following, _ = observation(pending, 0)
    assert not torch.equal(features["decision_context"], following["decision_context"])
    pending["waiting_for"]["card_name"] = "Another Secret"
    safe, _ = observation(pending, 0)
    assert torch.equal(following["decision_context"], safe["decision_context"])
    model = make_model(7).eval()
    with torch.no_grad():
        before = model.encoder.encode({key: value.unsqueeze(0)
                                       for key, value in features.items()})[1]
        after = model.encoder.encode({key: value.unsqueeze(0)
                                      for key, value in following.items()})[1]
    assert not torch.equal(before, after)


def test_action_encoding_keeps_option_target_and_numeric_semantics():
    actions = [
        {"type": "ChooseOption", "data": {"option": "Cancel"}},
        {"type": "ChooseOption", "data": {"option": "Cast", "source": 42}},
        {"type": "ChooseOption", "data": {"amount": 0}},
        {"type": "ChooseOption", "data": {"amount": 1}},
    ]
    vectors = [encode_action(action, state(), 0) for action in actions]
    assert all(vector.shape == (136,) for vector in vectors)
    assert all(not torch.equal(a, b) for i, a in enumerate(vectors) for b in vectors[i+1:])
    renamed = state()
    renamed["objects"]["80"] = renamed["objects"].pop("42")
    assert torch.equal(
        vectors[1],
        encode_action({"type": "ChooseOption", "data": {"option": "Cast", "source": 80}},
                      renamed, 0),
    )


def test_recorder_retains_legal_identity_and_terminal_reward():
    data = game()
    assert data["completed"] and len(data["records"]) == 3
    assert data["records"][0]["legal_actions"][0]["type"] == "PlayLand"
    assert data["records"][-1]["done"] is True
    assert data["records"][-1]["reward"] == 1


def test_timeout_is_not_a_draw_or_training_terminal():
    recorder = NativeRecordingPicker(HeuristicActionPicker(7))
    recorder.pick([{"type": "PassPriority"}], state(), 0)
    data = recorder.finish(SimpleNamespace(
        trace=[{"event": "game_timeout"}], final_state=state(), our_seat=0,
        winner_seat=None, reason="game_timeout", turns_observed=2,
    ))
    assert not data["completed"]
    assert not any(record["done"] for record in data["records"])


def test_producer_excludes_concession_and_preserves_original_indices():
    class FirstPicker:
        name = "first"

        def pick(self, actions, state, seat):
            assert all(action["type"] != "Concede" for action in actions)
            return 0

    recorder = NativeRecordingPicker(FirstPicker())
    actions = [{"type": "Concede"}, {"type": "PassPriority"}]
    assert recorder.pick(actions, state(), 0) == 1
    assert recorder.records[0]["chosen_index"] == 1
    assert recorder.records[0]["legal_actions"] == actions


def test_graph_exposure_keeps_wins_losses_and_training_provenance(tmp_path):
    graph = induce_graph([
        game(), game("train2", outcome=-1),
    ], tmp_path / "graph.json")
    assert graph["edges"][0]["exposed_games"] == 2
    assert graph["edges"][0]["wins"] == graph["edges"][0]["losses"] == 1
    assert all(evidence["split"] == "train" for evidence in graph["evidence"])
    assert graph["relation"] == "CO_VISIBLE"
    with pytest.raises(ValueError, match="training games only"):
        induce_graph([game(split="test")], tmp_path / "leak.json")


def test_full_component_updates_strict_loading_and_graph_influence(tmp_path):
    torch.set_num_threads(2)
    games = [game(), game("train2", ["Island", "Counterspell"], outcome=-1)]
    graph = induce_graph(games, tmp_path / "graph.json")
    graph_bundle, losses = train_graph(graph, 7, 2)
    assert all(torch.isfinite(torch.tensor(losses)))
    model = make_model(7)
    keys = ("encoder", "jepa_predictor", "dynamics", "controller")
    initial = {key: parameter_digest(getattr(model, key)) for key in keys}
    history = train_components(model, games, graph_bundle, 2)
    assert len(history) == 2
    assert all(parameter_digest(getattr(model, key)) != initial[key] for key in keys)
    metrics = diagnostics(model, [game("validation", split="validation")], graph_bundle)
    assert metrics["pairs"] == 2 and metrics["recurrent_metrics"]
    assert metrics["nonforced_choice_count"] == 2
    assert metrics["nonforced_imitation_nll"] is not None
    assert metrics["latent_variance_mean"] > 0
    path = tmp_path / "trained.pt"
    save_bundle(path, model, graph_bundle, {"test": True})
    picker = NativeLearnedPicker(path)
    actions = [
        {"type": "PlayLand", "data": {"source": 44}}, {"type": "PassPriority"},
        {"type": "Concede"},
    ]
    chosen = picker.pick(actions, state(), 0)
    assert chosen in (0, 1)
    assert picker.last_reasoning["controller_invoked"]
    assert picker.last_reasoning["dynamics_invoked"]
    assert picker.last_reasoning["retrieved_cards"]
    assert picker.last_reasoning["graph_score_delta_max"] > 0
    without = NativeLearnedPicker(path, False)
    without.pick(actions, state(), 0)
    assert without.last_reasoning["graph_score_delta_max"] == 0
    first = NativeLearnedPicker(path, seed=23)
    second = NativeLearnedPicker(path, seed=23)
    sequence = [first.pick(actions, state(), 0) for _ in range(30)]
    assert sequence == [second.pick(actions, state(), 0) for _ in range(30)]
    assert set(sequence) == {0, 1}
    assert first.last_reasoning["selection"] == "sample"
    assert sum(first.last_reasoning["candidate_probabilities"]) == pytest.approx(1)
    corrupt = torch.load(path, weights_only=False)
    intact = deepcopy(corrupt)
    corrupt["component_digests"].pop("controller")
    torch.save(corrupt, path)
    with pytest.raises(ValueError, match="four component digests"):
        NativeLearnedPicker(path)
    corrupt = intact
    corrupt["state_dict"].pop("controller.policy.bias")
    torch.save(corrupt, path)
    with pytest.raises(RuntimeError):
        NativeLearnedPicker(path)


def test_training_rejects_holdout_and_complete_graph(tmp_path):
    with pytest.raises(ValueError, match="training games only"):
        train_components(make_model(7), [game(split="test")], {}, 1)
    graph = induce_graph([game()], tmp_path / "graph.json")
    with pytest.raises(ValueError, match="negative pairs"):
        train_graph(graph, 7, 1)


def test_invalid_pipeline_budgets_do_not_start_a_server(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_native_learning.run(Namespace(
            train_games=0, validation_games=1, test_games=1, epochs=1,
            graph_epochs=1, eval_games=1, threads=1, max_actions=100,
        ))


def test_retired_summary_training_and_missing_native_checkpoint_are_rejected():
    from scripts.run_phase_rs_ablation import _make_picker
    from scripts.train_pipeline import stage_4_1_phase_rs_traces

    with pytest.raises(ValueError, match="actual observation/action pairs"):
        asyncio.run(stage_4_1_phase_rs_traces())
    with pytest.raises(ValueError, match="requires --native-checkpoint"):
        _make_picker(Namespace(picker="native_learned", native_checkpoint=None), 7)

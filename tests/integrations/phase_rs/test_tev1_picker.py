from __future__ import annotations

import io
import json
import urllib.error

import pytest

from scripts import collect_phase_rs_traces, phase_rs_rollout_sweep, run_phase_rs_ablation
from src.integrations.phase_rs.agent_bridge import PickerError
from src.integrations.phase_rs.tev1_picker import MAX_REQUEST_BYTES, Tev1ActionPicker


def _answer(keys, chosen=None):
    return {"answers": {"move": {
        "choice": chosen or keys[-1],
        "probabilities": {key: 1 / len(keys) for key in keys},
        "confidence": 0.0,
    }}}


def _mock_response(monkeypatch, payload):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )


def test_systemone_request_maps_to_original_action(monkeypatch):
    calls = []

    def respond(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(_answer(["action_0", "action_2"], "action_2")).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    actions = [
        {"type": "PassPriority"},
        {"type": "Concede"},
        {"type": "CastSpell", "data": {"source": 12, "target": 34}},
    ]
    picker = Tev1ActionPicker(seed=7)
    assert picker.pick(actions, {"turn_number": 3}, 0) == 2
    request, timeout = calls[0]
    assert request.full_url == "http://localhost:11434/v1/systemone"
    assert timeout == 120
    payload = json.loads(request.data)
    assert payload["state"] == {"controlled_seat": 0, "game": {"turn_number": 3}}
    criteria = payload["questions"]["move"]["criteria"]
    assert set(criteria) == {"action_0", "action_2"}
    assert json.loads(criteria["action_2"]) == actions[2]
    assert picker.last_reasoning["probabilities"] == {"action_0": 0.5, "action_2": 0.5}
    assert picker.last_reasoning["model_invoked"] is True
    assert picker.last_reasoning["candidate_coverage"] == 1.0
    assert picker.last_reasoning["context_projection"] == "compact_card_aware_v3"
    assert picker.last_reasoning["request_bytes"] == len(request.data)


def test_oversized_visible_context_is_not_silently_truncated(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("oversized requests must be rejected before HTTP")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_call)
    state = {"objects": {"1": {"name": "Visible Card", "oracle_text": "x" * MAX_REQUEST_BYTES}}}
    picker = Tev1ActionPicker()
    with pytest.raises(PickerError, match="no state or actions were truncated") as error:
        picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], state, 0)
    assert error.value.reason == "picker_input_too_large"
    assert picker.last_reasoning["request_bytes"] > MAX_REQUEST_BYTES


def test_context_alias_and_actual_usage_are_preserved(monkeypatch):
    payload = _answer(["action_0", "action_1"])
    payload["usage"] = {"input_tokens": 2906, "output_tokens": 1}
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    picker = Tev1ActionPicker(model="tev1-mtg-8k:0.8b")
    picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert requests[0]["model"] == "tev1-mtg-8k:0.8b"
    assert picker.last_reasoning["usage"] == payload["usage"]


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_exact_request_byte_limit(monkeypatch, delta):
    calls = []

    def respond(request, **kwargs):
        calls.append(request)
        return io.BytesIO(json.dumps(_answer(["action_0", "action_1"])).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    actions = [{"type": "PassPriority"}, {"type": "PlayLand"}]
    state = {"objects": {"1": {"name": "Card", "oracle_text": "x"}}}
    picker = Tev1ActionPicker()
    picker.pick(actions, state, 0)
    overhead = picker.last_reasoning["request_bytes"] - 1
    state["objects"]["1"]["oracle_text"] = "x" * (MAX_REQUEST_BYTES - overhead + delta)
    if delta <= 0:
        picker.pick(actions, state, 0)
        assert len(calls) == 2
    else:
        with pytest.raises(PickerError) as error:
            picker.pick(actions, state, 0)
        assert error.value.reason == "picker_input_too_large"
        assert len(calls) == 1
    assert picker.last_reasoning["request_bytes"] == MAX_REQUEST_BYTES + delta


def test_http_error_includes_server_diagnostic(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://localhost", 413, "Too large", None,
            io.BytesIO(b'{"error":"request body must not exceed 64 KiB without images"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(PickerError, match="64 KiB"):
        Tev1ActionPicker().pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)


def test_shortlist_is_bounded_seeded_and_preserves_pass(monkeypatch):
    actions = [
        {"type": "CastSpell", "data": {"source": i}}
        for i in range(50)
    ] + [{"type": "PassPriority"}, {"type": "Concede"}]
    first = Tev1ActionPicker(seed=7)
    second = Tev1ActionPicker(seed=7)
    candidates, eligible = first._shortlist(actions)
    assert candidates == second._shortlist(actions)[0]
    assert len(candidates) == 24
    assert len(set(candidates)) == 24
    assert 50 in candidates and 51 not in candidates
    assert candidates != Tev1ActionPicker(seed=8)._shortlist(actions)[0]
    keys = [f"action_{i}" for i in candidates]
    _mock_response(monkeypatch, _answer(keys, "action_50"))
    # Recreate to compare the first RNG draw rather than a subsequent shortlist.
    picker = Tev1ActionPicker(seed=7)
    assert picker.pick(actions, {}, 0) == 50
    assert picker.last_reasoning["candidate_coverage"] == pytest.approx(24 / len(eligible))


@pytest.mark.parametrize("actions", [
    [{"type": "PassPriority"}],
    [{"type": "Concede"}],
    [{"type": "PlayLand"}, {"type": "Concede"}],
])
def test_forced_decision_does_not_call_model(monkeypatch, actions):
    def unexpected_call(*args, **kwargs):
        pytest.fail("forced choices must not require an available model")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_call)
    picker = Tev1ActionPicker()
    picker.last_reasoning = {"old": "stale"}
    assert picker.pick(actions, {}, 0) == 0
    assert picker.last_reasoning["forced"] is True
    assert "old" not in picker.last_reasoning


@pytest.mark.parametrize("payload", [
    {},
    {"answers": {}},
    _answer(["action_0", "action_1"], "action_99"),
    {"answers": {"move": {
        "choice": "action_0", "probabilities": {"action_0": 1.0}, "confidence": 1,
    }}},
    {"answers": {"move": {
        "choice": "action_0",
        "probabilities": {"action_0": float("nan"), "action_1": 0.5},
        "confidence": 0.5,
    }}},
    {"answers": {"move": {
        "choice": "action_0",
        "probabilities": {"action_0": 0.2, "action_1": 0.2}, "confidence": 0.5,
    }}},
    {"answers": {"move": {
        "choice": "action_0",
        "probabilities": {"action_0": 0.5, "action_1": 0.5}, "confidence": True,
    }}},
])
def test_invalid_answers_fail_explicitly(monkeypatch, payload):
    _mock_response(monkeypatch, payload)
    with pytest.raises(PickerError) as error:
        Tev1ActionPicker().pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert error.value.reason == "picker_invalid_response"


@pytest.mark.parametrize(("exception", "reason"), [
    (urllib.error.HTTPError("http://localhost", 404, "Missing model", None, None),
     "picker_http_error"),
    (urllib.error.URLError("Not reachable"), "picker_connection_error"),
    (TimeoutError("Timed out"), "picker_connection_error"),
], ids=["http", "unreachable", "timeout"])
def test_network_failure_never_falls_back(monkeypatch, exception, reason):
    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(PickerError) as error:
        Tev1ActionPicker().pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert error.value.reason == reason


@pytest.mark.parametrize("options", [
    {"max_candidates": 1}, {"max_candidates": 25},
    {"timeout_s": 0}, {"timeout_s": float("nan")},
    {"base_url": "https://remote.example"}, {"model": ""},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        Tev1ActionPicker(**options)


@pytest.mark.parametrize("module", [
    collect_phase_rs_traces, phase_rs_rollout_sweep, run_phase_rs_ablation,
])
def test_cli_factories_support_tev1(monkeypatch, module):
    selector = "--pickers" if module is phase_rs_rollout_sweep else "--picker"
    monkeypatch.setattr("sys.argv", [
        "test", selector, "tev1", "--tev1-model", "tev1:4b", "--tev1-candidates", "12",
    ])
    args = module.parse_args()
    picker = (
        module._make_picker(args, 7)
        if module is run_phase_rs_ablation else module._build_picker("tev1", 7, args)
    )
    assert isinstance(picker, Tev1ActionPicker)
    assert picker.model == "tev1:4b"
    assert picker.max_candidates == 12

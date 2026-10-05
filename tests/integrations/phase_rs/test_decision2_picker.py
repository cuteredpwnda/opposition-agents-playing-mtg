import sys
from types import SimpleNamespace

import pytest

from scripts import collect_phase_rs_traces, phase_rs_rollout_sweep, run_phase_rs_ablation
from src.integrations.phase_rs import decision2_picker
from src.integrations.phase_rs.agent_bridge import PickerError
from src.integrations.phase_rs.decision2_picker import EOS_REVISION, Decision2ActionPicker


def test_larger_option_space_preserves_original_identity_and_usage():
    calls = []

    def choose(**payload):
        calls.append(payload)
        keys = list(payload["questions"]["move"]["criteria"])
        return {"answers": {"move": {
            "choice": "action_299", "probabilities": {key: 1 / len(keys) for key in keys},
            "confidence": 0.0,
        }}, "usage": {"input_tokens": 5000, "output_tokens": 0}}

    picker = Decision2ActionPicker(seed=7)
    picker._decision_model = SimpleNamespace(max_input_tokens=16384, system_one=choose)
    actions = [{"type": "CastSpell", "data": {"source": i}} for i in range(299)]
    actions += [{"type": "PassPriority"}, {"type": "Concede"}]
    assert picker.pick(actions, {"turn_number": 3}, 0) == 299
    keys = calls[0]["questions"]["move"]["criteria"]
    assert len(keys) == 255
    assert "action_300" not in keys
    assert picker.last_reasoning["candidate_coverage"] == pytest.approx(255 / 300)
    assert picker.last_reasoning["usage"]["input_tokens"] == 5000
    assert picker.last_reasoning["model_revision"] == EOS_REVISION
    assert picker.last_reasoning["backend"] == "transformers_system_one"


def test_large_context_fails_explicitly_without_fallback():
    picker = Decision2ActionPicker()
    picker._decision_model = SimpleNamespace(
        max_input_tokens=16384,
        system_one=lambda **kwargs: {"answers": {"move": {"error": "max_length_exceeded"}}},
    )
    with pytest.raises(PickerError) as error:
        picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert error.value.reason == "picker_input_too_large"


def test_forced_decisions_do_not_load_transformers():
    picker = Decision2ActionPicker()
    assert picker.pick([{"type": "PassPriority"}], {}, 0) == 0
    assert picker._decision_model is None


def test_prepare_loads_before_play_and_keeps_picker_rng_independent(monkeypatch):
    calls = []
    model = SimpleNamespace(max_input_tokens=16384)
    monkeypatch.setattr(
        "src.integrations.phase_rs.decision2_picker._load_model",
        lambda *args: calls.append(args) or model,
    )
    picker = Decision2ActionPicker(seed=7)
    picker.prepare()
    picker.prepare()
    assert calls == [(picker.model, EOS_REVISION, "cpu", 4)]
    assert picker.model_prepare_seconds is not None
    assert picker._decision_model is model
    assert picker._rng is not Decision2ActionPicker(seed=7)._rng


def test_prepare_load_failure_is_explicit(monkeypatch):
    def fail(*args):
        raise OSError("unavailable pinned weights")

    monkeypatch.setattr("src.integrations.phase_rs.decision2_picker._load_model", fail)
    with pytest.raises(PickerError, match="unavailable pinned weights") as error:
        Decision2ActionPicker().prepare()
    assert error.value.reason == "picker_model_load_error"


def test_same_runtime_is_reused_across_replicates(monkeypatch):
    calls = []
    model = SimpleNamespace(max_input_tokens=16384)
    fake_loader = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)) or model,
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=fake_loader))
    decision2_picker._load_model.cache_clear()
    try:
        first, second = Decision2ActionPicker(seed=7), Decision2ActionPicker(seed=8)
        first.prepare()
        second.prepare()
        assert first._decision_model is second._decision_model is model
        assert len(calls) == 1
        assert calls[0][1]["revision"] == EOS_REVISION
        assert calls[0][1]["code_revision"] == EOS_REVISION
    finally:
        decision2_picker._load_model.cache_clear()


@pytest.mark.parametrize("options", [
    {"revision": "main"}, {"threads": 0}, {"max_candidates": 256},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        Decision2ActionPicker(**options)


@pytest.mark.parametrize("module", [
    collect_phase_rs_traces, phase_rs_rollout_sweep, run_phase_rs_ablation,
])
def test_all_native_cli_factories_support_decision2(module):
    args = SimpleNamespace(
        picker="decision2", decision2_model="vllm-sr/Decision-2.0-Eos-0.8B",
        decision2_revision=EOS_REVISION, decision2_candidates=128,
        decision2_device="cpu", decision2_threads=4,
    )
    if module is run_phase_rs_ablation:
        picker = module._make_picker(args, 7)
    else:
        picker = module._build_picker("decision2", 7, args)
    assert isinstance(picker, Decision2ActionPicker)
    assert picker.max_candidates == 128

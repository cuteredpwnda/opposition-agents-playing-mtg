from __future__ import annotations

from argparse import Namespace

import pytest

from src.integrations.phase_rs.liquid_d1_picker import (
    D1_MODELS,
    LiquidD1ActionPicker,
    make_liquid_d1_picker,
)


class _FakeD1:
    def system_one(self, state, questions):
        keys = list(questions["move"]["criteria"])
        return {
            "answers": {
                "move": {
                    "choice": keys[-1],
                    "probabilities": {key: 1 / len(keys) for key in keys},
                    "confidence": 0.0,
                },
            },
            "usage": {"input_tokens": 321, "output_tokens": 0},
        }


def test_liquid_d1_uses_native_system_one_contract():
    picker = LiquidD1ActionPicker(seed=7)
    picker._decision_model = _FakeD1()
    actions = [{"type": "PassPriority"}, {"type": "PlayLand"}]
    assert picker.pick(actions, {"turn_number": 2}, 0) == 1
    assert picker.last_reasoning["backend"] == "transformers_system_one"
    assert picker.last_reasoning["usage"]["output_tokens"] == 0
    assert picker.last_reasoning["model_invoked"] is True
    assert picker.last_reasoning["published_context_tokens"] == 16_384


def test_liquid_d1_defaults_to_pinned_revision():
    args = Namespace(
        liquid_d1_model="LiquidAI/d1-3B",
        liquid_d1_revision=None,
        liquid_d1_candidates=64,
        liquid_d1_device="cuda",
        liquid_d1_dtype="bfloat16",
    )
    picker = make_liquid_d1_picker(args, 17)
    assert picker.revision == D1_MODELS["LiquidAI/d1-3B"]
    assert picker.max_candidates == 64
    assert picker.device == "cuda"
    assert picker.dtype == "bfloat16"


def test_liquid_d1_rejects_unpinned_and_unknown_models():
    with pytest.raises(ValueError, match="full pinned"):
        LiquidD1ActionPicker(revision="main")
    with pytest.raises(ValueError, match="must be one of"):
        LiquidD1ActionPicker(model="some/chat-model")

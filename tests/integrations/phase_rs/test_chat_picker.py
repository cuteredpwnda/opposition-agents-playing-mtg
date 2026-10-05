import io
import json
import urllib.error

import pytest

from src.integrations.phase_rs.agent_bridge import OllamaActionPicker


def test_chat_generation_bounds_output_and_disables_thinking(monkeypatch):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        result = ({"models": [{"name": "gemma4:e2b"}]}
                  if request.full_url.endswith("/api/tags") else {"response": "1"})
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    picker = OllamaActionPicker(seed=7)
    assert picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0) == 1
    body = json.loads(requests[-1].data)
    assert body["think"] is False
    assert body["options"] == {"temperature": 0.2, "num_predict": 16}
    assert picker.last_reasoning["model_invoked"] is True
    assert "fallback_reason" not in picker.last_reasoning


def test_chat_inference_failure_records_fallback(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)

    def fail(self, prompt):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(OllamaActionPicker, "_ollama_generate", fail)
    picker = OllamaActionPicker(seed=7)
    chosen = picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert picker.last_reasoning["chosen_index"] == chosen
    assert picker.last_reasoning["fallback_reason"] == "inference_error"


def test_forced_chat_choice_skips_inference(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    picker = OllamaActionPicker()
    assert picker.pick([{"type": "PassPriority"}], {}, 0) == 0
    assert picker.last_reasoning["forced"]


@pytest.mark.parametrize("reply", ["not an index", "-1", "1.5", "1e2"])
def test_invalid_reply_and_concession_are_traceable(monkeypatch, reply):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    picker = OllamaActionPicker(seed=7)
    monkeypatch.setattr(picker, "_ollama_generate", lambda prompt: reply)
    actions = [{"type": "Concede"}, {"type": "PlayLand"}]
    picker.pick(actions, {}, 0)
    assert picker.last_reasoning["fallback_reason"] == "invalid_response"
    monkeypatch.setattr(picker, "_ollama_generate", lambda prompt: "0")
    assert picker.pick(actions, {}, 0) == 1
    assert picker.last_reasoning["fallback_reason"] == "model_concession"


@pytest.mark.parametrize(("reply", "expected"), [
    ("1", 1), ("Action [1].", 1), ("-1", None), ("1.5", None),
    ("1e2", None), ("2", None), ("Choose -1 instead of 1", None),
])
def test_chat_index_parser_preserves_sign_and_integer_semantics(reply, expected):
    assert OllamaActionPicker._parse_index(reply, 2) == expected


def test_chat_prompt_preserves_card_and_choice_semantics(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    picker = OllamaActionPicker()
    state = {"objects": {
        "42": {"name": "Lightning Bolt", "zone": "Hand", "owner": 0},
        "43": {"name": "Hidden Card", "zone": "Hand", "owner": 1},
    }}
    actions = [
        {"type": "ChooseOption", "data": {"option": "Cancel"}},
        {"type": "ChooseOption", "data": {"option": "Cast", "source": 42}},
    ]
    prompt = picker._build_prompt(actions, state, 0)
    assert "Lightning Bolt" in prompt and "Hidden Card" not in prompt
    assert '"option":"Cancel"' in prompt
    assert '"option":"Cast","source":42' in prompt
    assert picker.last_reasoning["context_projection"] == "compact_card_aware_v3"

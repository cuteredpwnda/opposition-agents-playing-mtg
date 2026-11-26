import io
import json
import urllib.error

import pytest

from src.integrations.phase_rs.agent_bridge import (
    OllamaActionPicker,
    OpenAICompatibleActionPicker,
    PickerError,
)


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

    def fail(self, prompt, eligible_indices):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(OllamaActionPicker, "_ollama_generate", fail)
    picker = OllamaActionPicker(seed=7)
    chosen = picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert picker.last_reasoning["chosen_index"] == chosen
    assert picker.last_reasoning["fallback_reason"] == "inference_error"
    assert picker.last_reasoning["model_request_attempted"] is True
    assert picker.last_reasoning["model_error_reason"] == "picker_inference_error"


def test_strict_chat_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    monkeypatch.setattr(
        OllamaActionPicker, "_ollama_generate",
        lambda self, prompt, eligible_indices: (_ for _ in ()).throw(TimeoutError("slow")),
    )
    picker = OllamaActionPicker(seed=7, failure_policy="error")
    with pytest.raises(Exception, match="slow") as error:
        picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert getattr(error.value, "reason", None) == "picker_inference_error"
    assert picker.last_reasoning["model_request_attempted"] is True


def test_strict_chat_unavailable_fails_without_random_fallback(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: False)
    picker = OllamaActionPicker(seed=7, failure_policy="error")
    with pytest.raises(PickerError) as error:
        picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert error.value.reason == "picker_model_unavailable"
    assert "fallback_reason" not in picker.last_reasoning


def test_forced_chat_choice_does_not_require_model_availability(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: False)
    picker = OllamaActionPicker(failure_policy="error")
    assert picker.pick([{"type": "PassPriority"}], {}, 0) == 0
    assert picker.last_reasoning["forced"] is True


def test_legacy_chat_unavailable_still_falls_back(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: False)
    picker = OllamaActionPicker(seed=7, failure_policy="random")
    actions = [{"type": "Concede"}, {"type": "PassPriority"}]
    assert picker.pick(actions, {}, 0) == 1
    assert picker.last_reasoning["fallback_reason"] == "model_unavailable"


def test_strict_ollama_uses_legal_nonconcession_schema(monkeypatch):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        result = ({"models": [{"name": "llama3.2:1b"}]}
                  if request.full_url.endswith("/api/tags")
                  else {"response": '{"index":2}'})
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    picker = OllamaActionPicker(model="llama3.2:1b", failure_policy="error")
    actions = [{"type": "Concede"}, {"type": "PassPriority"}, {"type": "PlayLand"}]
    assert picker.pick(actions, {}, 0) == 2
    body = json.loads(requests[-1].data)
    assert body["format"]["properties"]["index"]["enum"] == [1, 2]
    assert "JSON" in body["prompt"]
    assert picker.last_reasoning["model_invoked"] is True
    assert "fallback_reason" not in picker.last_reasoning


@pytest.mark.parametrize("reply", [
    "1", '{"index":0}', '{"index":true}', '{"index":"1"}',
    '{"index":1.0}', '{"index":1,"extra":2}', "not JSON",
])
def test_strict_chat_rejects_malformed_or_excluded_choice(monkeypatch, reply):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    monkeypatch.setattr(OllamaActionPicker, "_ollama_generate", lambda *args: reply)
    picker = OllamaActionPicker(seed=7, failure_policy="error")
    actions = [{"type": "Concede"}, {"type": "PassPriority"}, {"type": "PlayLand"}]
    with pytest.raises(PickerError) as error:
        picker.pick(actions, {}, 0)
    assert error.value.reason == "picker_invalid_response"
    assert "fallback_reason" not in picker.last_reasoning


def test_forced_chat_choice_skips_inference(monkeypatch):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    picker = OllamaActionPicker()
    assert picker.pick([{"type": "PassPriority"}], {}, 0) == 0
    assert picker.last_reasoning["forced"]


@pytest.mark.parametrize("reply", ["not an index", "-1", "1.5", "1e2"])
def test_invalid_reply_and_concession_are_traceable(monkeypatch, reply):
    monkeypatch.setattr(OllamaActionPicker, "_check_ollama_available", lambda self: True)
    picker = OllamaActionPicker(seed=7)
    monkeypatch.setattr(picker, "_ollama_generate", lambda prompt, eligible_indices: reply)
    actions = [{"type": "Concede"}, {"type": "PlayLand"}]
    picker.pick(actions, {}, 0)
    assert picker.last_reasoning["fallback_reason"] == "invalid_response"
    monkeypatch.setattr(picker, "_ollama_generate", lambda prompt, eligible_indices: "0")
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


def test_openai_compatible_picker_uses_structured_local_endpoint(monkeypatch):
    requests = []

    def respond(request, **kwargs):
        requests.append((request, kwargs))
        payload = {
            "choices": [{"message": {"content": '{"index":1}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    picker = OpenAICompatibleActionPicker(
        seed=7, model="Qwen/Qwen3-0.6B", base_url="http://localhost:8000/v1",
    )
    assert picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0) == 1
    request, kwargs = requests[0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    body = json.loads(request.data)
    assert body["response_format"]["type"] == "json_schema"
    assert body["temperature"] == 0.0
    assert kwargs["timeout"] == 120.0
    assert picker.last_reasoning["backend"] == "openai_compatible_chat_completions"
    assert picker.last_reasoning["model_invoked"] is True


@pytest.mark.parametrize("payload", [
    [],
    {},
    {"choices": [{"message": {"content": 1}}]},
    {"choices": [{"message": {"content": "1"}}]},
])
def test_vllm_rejects_malformed_response(monkeypatch, payload):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    picker = OpenAICompatibleActionPicker()
    with pytest.raises(PickerError) as error:
        picker.pick([{"type": "PassPriority"}, {"type": "PlayLand"}], {}, 0)
    assert error.value.reason == "picker_invalid_response"
    assert "fallback_reason" not in picker.last_reasoning


def test_openai_compatible_picker_rejects_remote_url():
    with pytest.raises(ValueError, match="local URL"):
        OpenAICompatibleActionPicker(base_url="https://example.com/v1")

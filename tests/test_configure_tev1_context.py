import io
import json

import pytest

from scripts.configure_tev1_context import configure


def test_alias_preserves_source_and_verifies_context(monkeypatch):
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        result = ({"status": "success"} if len(requests) == 1
                  else {"parameters": "num_ctx 8192\n"})
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    assert configure("tev1:0.8b", "tev1-mtg-8k:0.8b", 8192)["num_ctx"] == 8192
    assert requests[0]["from"] == "tev1:0.8b"
    assert requests[0]["model"] != requests[0]["from"]


@pytest.mark.parametrize("source,target,context", [
    ("a", "a", 8192), ("", "b", 8192), ("a", "b", 1024),
    ("a", " a ", 8192), ("a", "a:latest", 8192),
])
def test_invalid_alias_configuration(source, target, context):
    with pytest.raises(ValueError):
        configure(source, target, context)


def test_alias_configuration_mismatch_is_explicit(monkeypatch):
    results = iter([{"status": "success"}, {"parameters": "num_ctx 2050"}])
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(next(results)).encode()))
    with pytest.raises(ValueError, match="verification failed"):
        configure("a", "b", 8192)

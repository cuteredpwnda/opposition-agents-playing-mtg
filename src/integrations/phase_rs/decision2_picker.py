"""Pinned local Decision 2.0 inference; no vLLM server or chat generation."""

from __future__ import annotations

import argparse
import re
import time
from functools import lru_cache
from typing import Any

from src.integrations.phase_rs.tev1_picker import Tev1ActionPicker

EOS_MODEL = "vllm-sr/Decision-2.0-Eos-0.8B"
EOS_REVISION = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"


@lru_cache(maxsize=1)
def _load_model(model: str, revision: str, device: str, threads: int) -> Any:
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        model, revision=revision, code_revision=revision,
        trust_remote_code=True, device=device, threads=threads,
    )


class Decision2ActionPicker(Tev1ActionPicker):
    MAX_CANDIDATES = 255
    REQUEST_BYTE_LIMIT = None

    def __init__(
        self, seed: int | None = None, *, model: str = EOS_MODEL,
        revision: str = EOS_REVISION, max_candidates: int = 255,
        device: str = "cpu", threads: int = 4,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Decision 2.0 requires a full pinned Hub commit revision")
        if threads < 1:
            raise ValueError("Decision 2.0 threads must be positive")
        super().__init__(seed, model=model, max_candidates=max_candidates)
        self.name = f"phase_rs_decision2:{model}@{revision}"
        self.revision, self.device, self.threads = revision, device, threads
        self._decision_model: Any = None
        self.model_prepare_seconds: float | None = None

    def prepare(self) -> None:
        """Load before starting a live game; reuse the runtime across picker seeds."""
        if self._decision_model is None:
            started = time.perf_counter()
            try:
                self._decision_model = _load_model(
                    self.model, self.revision, self.device, self.threads,
                )
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self._fail(f"Decision 2.0 load failed: {exc}", "picker_model_load_error")
            self.model_prepare_seconds = time.perf_counter() - started

    def _infer(self, payload: dict[str, Any], data: bytes) -> Any:
        self.prepare()
        self.last_reasoning.update({
            "model_revision": self.revision,
            "max_input_tokens": self._decision_model.max_input_tokens,
            "backend": "transformers_system_one",
            "model_prepare_seconds": self.model_prepare_seconds,
        })
        try:
            result = self._decision_model.system_one(
                state=payload["state"], questions=payload["questions"],
            )
        except (ValueError, RuntimeError) as exc:
            self._fail(f"Decision 2.0 inference failed: {exc}", "picker_inference_error")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            self._fail("Decision 2.0 returned malformed answers", "picker_invalid_response")
        answer = result["answers"].get("move")
        if not isinstance(answer, dict):
            self._fail("Decision 2.0 returned malformed move", "picker_invalid_response")
        error = answer.get("error")
        if error:
            self._fail(
                f"Decision 2.0 returned {error}",
                "picker_input_too_large" if error == "max_length_exceeded"
                else "picker_invalid_response",
            )
        return result


def add_decision2_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision2-model", default=EOS_MODEL)
    parser.add_argument("--decision2-revision", default=EOS_REVISION)
    parser.add_argument("--decision2-candidates", type=int, choices=range(2, 256), default=255)
    parser.add_argument("--decision2-device", default="cpu")
    parser.add_argument("--decision2-threads", type=int, default=4)


def make_decision2_picker(args: argparse.Namespace, seed: int | None) -> Decision2ActionPicker:
    return Decision2ActionPicker(
        seed, model=args.decision2_model, revision=args.decision2_revision,
        max_candidates=args.decision2_candidates, device=args.decision2_device,
        threads=args.decision2_threads,
    )

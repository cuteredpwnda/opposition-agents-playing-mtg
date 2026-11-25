"""Pinned local Liquid d1 decision inference over legal-action shortlists."""

from __future__ import annotations

import argparse
import re
import time
from functools import lru_cache
from typing import Any

from src.integrations.phase_rs.tev1_picker import Tev1ActionPicker

D1_MODELS = {
    "LiquidAI/d1-omni-600M": "02b55d7076f15129e59ab3f94783f32c4b088674",
    "LiquidAI/d1-3B": "051bcc464b01b9f92942b364d9586b0ef5912432",
}


@lru_cache(maxsize=2)
def _load_model(model: str, revision: str, device: str, dtype_name: str) -> Any:
    import torch
    from transformers import AutoModel

    if dtype_name == "auto":
        if device == "cpu":
            dtype = torch.float32
        elif model.endswith("d1-omni-600M"):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        dtype = getattr(torch, dtype_name)
    loaded = AutoModel.from_pretrained(
        model,
        revision=revision,
        code_revision=revision,
        trust_remote_code=True,
        dtype=dtype,
    )
    return loaded.to(device).eval()


class LiquidD1ActionPicker(Tev1ActionPicker):
    """Use Liquid d1's native ``system_one`` head, not token generation."""

    MAX_CANDIDATES = 255
    REQUEST_BYTE_LIMIT = None

    def __init__(
        self,
        seed: int | None = None,
        *,
        model: str = "LiquidAI/d1-omni-600M",
        revision: str | None = None,
        max_candidates: int = 128,
        device: str = "cpu",
        dtype: str = "auto",
    ) -> None:
        if model not in D1_MODELS:
            raise ValueError(f"Liquid d1 model must be one of {sorted(D1_MODELS)}")
        resolved_revision = revision or D1_MODELS[model]
        if re.fullmatch(r"[0-9a-f]{40}", resolved_revision) is None:
            raise ValueError("Liquid d1 requires a full pinned Hub commit revision")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("Liquid d1 dtype must be auto, float16, bfloat16 or float32")
        super().__init__(seed, model=model, max_candidates=max_candidates)
        self.name = f"phase_rs_liquid_d1:{model}@{resolved_revision}"
        self.revision = resolved_revision
        self.device = device
        self.dtype = dtype
        self._decision_model: Any = None
        self.model_prepare_seconds: float | None = None

    def prepare(self) -> None:
        if self._decision_model is not None:
            return
        started = time.perf_counter()
        try:
            self._decision_model = _load_model(
                self.model, self.revision, self.device, self.dtype,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            self._fail(f"Liquid d1 load failed: {exc}", "picker_model_load_error")
        self.model_prepare_seconds = time.perf_counter() - started

    def _infer(self, payload: dict[str, Any], data: bytes) -> Any:
        self.prepare()
        self.last_reasoning.update({
            "model_revision": self.revision,
            "backend": "transformers_system_one",
            "device": self.device,
            "dtype": self.dtype,
            "model_prepare_seconds": self.model_prepare_seconds,
            "published_context_tokens": 16_384 if self.model.endswith("600M") else 32_768,
        })
        try:
            result = self._decision_model.system_one(
                payload["state"], payload["questions"],
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._fail(f"Liquid d1 inference failed: {exc}", "picker_inference_error")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            self._fail("Liquid d1 returned malformed answers", "picker_invalid_response")
        return result


def add_liquid_d1_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--liquid-d1-model", choices=sorted(D1_MODELS),
        default="LiquidAI/d1-omni-600M",
    )
    parser.add_argument("--liquid-d1-revision", default=None)
    parser.add_argument("--liquid-d1-candidates", type=int, choices=range(2, 256), default=128)
    parser.add_argument("--liquid-d1-device", default="cpu")
    parser.add_argument(
        "--liquid-d1-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )


def make_liquid_d1_picker(
    args: argparse.Namespace, seed: int | None,
) -> LiquidD1ActionPicker:
    return LiquidD1ActionPicker(
        seed,
        model=args.liquid_d1_model,
        revision=args.liquid_d1_revision,
        max_candidates=args.liquid_d1_candidates,
        device=args.liquid_d1_device,
        dtype=args.liquid_d1_dtype,
    )

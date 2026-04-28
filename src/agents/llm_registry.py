"""Catalogue of LLM models used by ``OllamaAgent`` and ablation sweeps.

Every entry is constrained to **<= 7 B effective parameters at inference
time** (so a quantised 8 B model with q4 weights occupying ~5 GB still
counts as "8B base"; we mark it explicitly).  The registry is consumed by
``scripts/ablation_sweep.py`` to enumerate the LLM-size axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMSpec:
    """Hardware-aware description of an Ollama-served model."""

    name: str                       # short canonical name used in CSV output
    ollama_tag: str                 # tag used in ``ollama pull <tag>``
    family: str                     # gemma | phi | qwen | llama | mistral
    params_billion: float           # advertised base-model parameter count
    quantisation: str = "q4_K_M"   # default quant served by Ollama
    vram_gb_estimate: float = 4.0   # working-set VRAM at default quant
    context_tokens: int = 4096
    notes: str = ""

    @property
    def fits_seven_billion_budget(self) -> bool:
        """True if the model is within the <= 7 B effective-size budget."""
        return self.params_billion <= 7.0


LLM_REGISTRY: dict[str, LLMSpec] = {
    "gemma2-2b": LLMSpec(
        name="gemma2-2b",
        ollama_tag="gemma2:2b",
        family="gemma",
        params_billion=2.0,
        vram_gb_estimate=2.0,
        notes="Default for low-end / CI.",
    ),
    "phi3-mini": LLMSpec(
        name="phi3-mini",
        ollama_tag="phi3:mini",
        family="phi",
        params_billion=3.8,
        vram_gb_estimate=3.0,
        notes="Strong instruction following at small size.",
    ),
    "qwen2.5-3b": LLMSpec(
        name="qwen2.5-3b",
        ollama_tag="qwen2.5:3b-instruct",
        family="qwen",
        params_billion=3.0,
        vram_gb_estimate=2.5,
    ),
    "llama3.2-3b": LLMSpec(
        name="llama3.2-3b",
        ollama_tag="llama3.2:3b-instruct",
        family="llama",
        params_billion=3.0,
        vram_gb_estimate=2.5,
    ),
    "mistral-7b": LLMSpec(
        name="mistral-7b",
        ollama_tag="mistral:7b-instruct",
        family="mistral",
        params_billion=7.0,
        vram_gb_estimate=5.0,
        notes="Top of the <=7B budget.",
    ),
    "qwen2.5-7b": LLMSpec(
        name="qwen2.5-7b",
        ollama_tag="qwen2.5:7b-instruct",
        family="qwen",
        params_billion=7.0,
        vram_gb_estimate=5.0,
    ),
}


def get_llm_spec(name: str) -> LLMSpec:
    """Look up an LLM spec by its canonical ``name``."""
    if name not in LLM_REGISTRY:
        raise KeyError(
            f"Unknown LLM '{name}'. Available: {sorted(LLM_REGISTRY)}"
        )
    return LLM_REGISTRY[name]


def list_llms(max_vram_gb: Optional[float] = None) -> list[LLMSpec]:
    """Return all registered LLMs that fit ``max_vram_gb`` (None = all)."""
    specs = [s for s in LLM_REGISTRY.values() if s.fits_seven_billion_budget]
    if max_vram_gb is not None:
        specs = [s for s in specs if s.vram_gb_estimate <= max_vram_gb]
    return sorted(specs, key=lambda s: s.params_billion)

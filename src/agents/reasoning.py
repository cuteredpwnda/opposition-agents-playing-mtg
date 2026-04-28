"""Structured agent reasoning traces.

Every agent decision can attach a ``ReasoningTrace`` describing *why*
the action was picked — the candidate set, scoring breakdown, internal
beliefs, etc.  Traces are JSON-serialisable so they slot into the
existing JSONL game log.

The contract is intentionally loose:

* Agents call ``self.set_reasoning(...)`` at the end of
  ``decide_action`` (or anywhere in the call) to record a trace.
* The priority loop reads ``agent.last_reasoning`` after
  ``decide_action`` returns and forwards it to the collector.
* ``JsonlActionTrace`` stores the trace verbatim under the ``reasoning``
  key of the action record.

Sub-agents must keep traces *small* (a few hundred bytes per turn).
Big tensors → store summary stats only (mean / std / top-k indices).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReasoningTrace:
    """Structured decision rationale for one agent action.

    Required:
        agent_kind:   short string, e.g. ``"random"`` / ``"heuristic"``
                      / ``"kg_heuristic"`` / ``"world_model"``
                      / ``"active_inference"``.
        rationale:    one human-readable sentence.

    Optional (populated where applicable):
        legal_action_count: how many legal actions were on offer.
        chosen_index:       index of the chosen action in
                            ``legal_actions``.
        scores:             per-candidate numerical scores
                            (list of floats, same order as
                            ``legal_actions``).  Truncated to first 32
                            entries to bound log size.
        top_candidates:     list of ``{"action": str, "score": float,
                            "reason": str}`` for the top-N candidates.
        beliefs:            free-form dict for agent-internal state
                            (KG hits, latent norms, EFE breakdown, …).
        timing_ms:          wall-clock cost of the decision.
    """

    agent_kind: str
    rationale: str = ""
    legal_action_count: int = 0
    chosen_index: int = 0
    scores: list[float] = field(default_factory=list)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    beliefs: dict[str, Any] = field(default_factory=dict)
    timing_ms: float | None = None

    _MAX_SCORES: int = 32

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # bound size — never let the trace dwarf the state record
        if len(d.get("scores", [])) > self._MAX_SCORES:
            d["scores"] = d["scores"][: self._MAX_SCORES]
            d["scores_truncated"] = True
        if len(d.get("top_candidates", [])) > 10:
            d["top_candidates"] = d["top_candidates"][:10]
        d.pop("_MAX_SCORES", None)
        return d


def attach_reasoning(agent: Any, trace: ReasoningTrace | dict[str, Any] | None) -> None:
    """Helper for ``MTGAgent.set_reasoning`` — accepts either a
    :class:`ReasoningTrace` or a raw dict (for ad-hoc traces from
    untrusted sub-agents)."""
    if trace is None:
        agent.last_reasoning = None
        return
    if isinstance(trace, ReasoningTrace):
        agent.last_reasoning = trace.to_dict()
    elif isinstance(trace, dict):
        agent.last_reasoning = dict(trace)
    else:  # pragma: no cover
        agent.last_reasoning = {"rationale": str(trace)}

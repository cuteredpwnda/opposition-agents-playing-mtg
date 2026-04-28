"""Summarize agent reasoning from a JSONL game log.

Usage::

    python -m scripts.inspect_reasoning runs/some_game.jsonl
    python -m scripts.inspect_reasoning runs/some_game.jsonl --player player_0
    python -m scripts.inspect_reasoning runs/some_game.jsonl --tail 20

Reads a JSONL trace produced by :class:`JsonlActionTrace` (with the
agent ``reasoning`` field added) and prints a per-tick walkthrough plus
aggregate statistics: action breakdown, reasoning fields used, KG-bias
hit rate, average expected free energy, etc.

Designed as a transparency tool — read what the agent was *thinking*
when it chose each action.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _load(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _decision_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if "action" in r and r.get("action")]


def _format_decision(r: dict[str, Any]) -> str:
    state = r.get("state") or {}
    action = r.get("action") or {}
    reasoning = r.get("reasoning") or {}
    agent = r.get("agent") or "?"
    turn = state.get("turn", "?")
    phase = state.get("phase", "?")
    pp = state.get("priority_player", "?")
    a_type = action.get("type", "?")
    a_card = action.get("card_instance_id") or ""
    rationale = reasoning.get("rationale") or "(no rationale)"
    kind = reasoning.get("agent_kind") or ""
    extras: list[str] = []
    if "scores" in reasoning and reasoning["scores"]:
        scores = reasoning["scores"]
        extras.append(f"scores[min={min(scores):.3f} max={max(scores):.3f} n={len(scores)}]")
    if "beliefs" in reasoning and reasoning["beliefs"]:
        beliefs = reasoning["beliefs"]
        # only print scalar / short beliefs
        short = {
            k: v for k, v in beliefs.items()
            if isinstance(v, (int, float, str, bool)) and len(str(v)) < 24
        }
        if short:
            extras.append("beliefs={" + ", ".join(f"{k}={v}" for k, v in short.items()) + "}")
    extras_str = "  " + " ".join(extras) if extras else ""
    return (
        f"  T{turn} {phase} pri={pp} | {agent} [{kind}] -> "
        f"{a_type}({a_card})\n"
        f"      reason: {rationale}{extras_str}"
    )


def _summary(records: list[dict[str, Any]]) -> str:
    decisions = _decision_records(records)
    n = len(decisions)
    if not n:
        return "(no decision records found)"

    action_counter: Counter[str] = Counter()
    agent_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    with_reasoning = 0
    efes: list[float] = []
    kg_active_hits = 0
    kg_total = 0

    for r in decisions:
        a = r.get("action") or {}
        action_counter[a.get("type", "?")] += 1
        agent_counter[r.get("agent") or "?"] += 1
        reason = r.get("reasoning") or {}
        if reason:
            with_reasoning += 1
            kind = reason.get("agent_kind") or "?"
            kind_counter[kind] += 1
            if kind == "active_inference":
                best = (reason.get("beliefs") or {}).get("best_efe")
                if isinstance(best, (int, float)):
                    efes.append(float(best))
            if kind == "kg_heuristic":
                kg_total += 1
                if (reason.get("beliefs") or {}).get("kg_active"):
                    kg_active_hits += 1

    out: list[str] = []
    out.append(f"Total decisions: {n}")
    out.append(f"With reasoning trace: {with_reasoning} ({100 * with_reasoning / n:.1f}%)")
    out.append(f"Action breakdown: {dict(action_counter.most_common())}")
    out.append(f"Agent breakdown: {dict(agent_counter.most_common())}")
    out.append(f"Reasoning kinds: {dict(kind_counter.most_common())}")
    if efes:
        out.append(
            f"Active-inference EFE: mean={statistics.mean(efes):.3f} "
            f"min={min(efes):.3f} max={max(efes):.3f} (n={len(efes)})"
        )
    if kg_total:
        out.append(
            f"KG-heuristic: {kg_active_hits}/{kg_total} decisions used KG re-rank "
            f"({100 * kg_active_hits / kg_total:.1f}%)"
        )

    end = next((r for r in records if r.get("event") == "game_end"), None)
    if end is not None:
        out.append(f"Game end: winner={end.get('winner')} turns={end.get('num_turns')}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="JSONL trace file")
    ap.add_argument("--player", help="Only show decisions made by this player_id")
    ap.add_argument("--agent", help="Only show decisions made by this agent name")
    ap.add_argument("--kind", help="Only show decisions of this agent_kind (e.g. active_inference)")
    ap.add_argument("--tail", type=int, default=0,
                    help="Show only the last N decisions (after filters)")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip per-decision walkthrough; print just summary")
    args = ap.parse_args()

    records = _load(args.path)
    decisions = _decision_records(records)

    if args.player:
        decisions = [r for r in decisions if (r.get("action") or {}).get("player") == args.player]
    if args.agent:
        decisions = [r for r in decisions if r.get("agent") == args.agent]
    if args.kind:
        decisions = [r for r in decisions if (r.get("reasoning") or {}).get("agent_kind") == args.kind]

    if args.tail and args.tail > 0:
        decisions = decisions[-args.tail:]

    if not args.summary_only:
        for r in decisions:
            print(_format_decision(r))
        print()
    print("=== Summary ===")
    print(_summary(records))


if __name__ == "__main__":
    main()

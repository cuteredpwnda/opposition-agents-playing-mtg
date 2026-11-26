"""Shared, append-only experience graph for cross-game and cross-agent learning.

The graph is the collective-memory layer described in ``AGENTS.md``: every
agent that plays may append provenance-tagged evidence, and any agent may read
the aggregate as a prior. Nodes form a three-level hierarchy::

    (kind, stage)  ->  (kind, card_class, stage)  ->  (kind, card_class, card, stage)

e.g. ``CastSpell/early`` -> ``CastSpell/creature_mv2/early`` ->
``CastSpell/creature_mv2/Heartfire Hero/early``. Each node carries a Beta
posterior over "a game in which this choice was made is won". Child nodes
borrow ``prior_strength`` pseudo-observations from their parent, so evidence
about a card class transfers to unseen cards of that class and across decks.

Credit assignment is first-visit Monte Carlo: each distinct node chosen in a
game receives the game outcome once (draws count half). This is a deliberately
simple, biased estimator -- it is confounded by deck strength and by which
decks happen to contain a card -- and is reported as such.

Exploration ("explore to learn") is over *model parameters*, not hidden game
state, so it cannot be captured by a noisy TV: the uncertainty it pays for is
reducible and shrinks as evidence accumulates.

``greedy``
    Posterior mean only; no exploration.
``thompson``
    Posterior (Thompson) sampling: one draw per node per decision.
``novelty``
    Posterior mean plus ``novelty_weight`` times the expected information gain
    about the node's parameter from one more outcome -- the Active Inference
    "novelty" term for a Beta-Bernoulli model.

Storage: one JSONL shard per writer under ``root``; a line is one game. Shards
are never rewritten, so concurrent workers need no lock.
"""

from __future__ import annotations

import json
import math
import os
import socket
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import digamma

SCHEMA = "phase_rs_experience_graph_v1"
EXPLORATION_MODES = ("greedy", "thompson", "novelty")

Key = tuple[str, str, str, str]


def node_chain(key: Key) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    kind, card_class, card, stage = key
    return (kind, stage), (kind, card_class, stage), (kind, card_class, card, stage)


def bernoulli_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def beta_bernoulli_information_gain(alpha: float, beta: float) -> float:
    """Mutual information between θ ~ Beta(α, β) and one Bernoulli(θ) outcome."""
    total = alpha + beta
    mean = alpha / total
    expected_entropy = -(
        mean * (digamma(alpha + 1) - digamma(total + 1))
        + (1 - mean) * (digamma(beta + 1) - digamma(total + 1))
    )
    return max(0.0, bernoulli_entropy(mean) - float(expected_entropy))


@dataclass
class ExperienceGraph:
    """Aggregated read view plus an append-only writer."""

    root: Path
    prior_strength: float = 4.0
    writer_id: str = field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    counts: dict[tuple[str, ...], list[float]] = field(default_factory=dict)
    games: int = 0

    @classmethod
    def load(cls, root: str | Path, *, prior_strength: float = 4.0) -> ExperienceGraph:
        graph = cls(Path(root), prior_strength=prior_strength)
        graph.root.mkdir(parents=True, exist_ok=True)
        for shard in sorted(graph.root.glob("*.jsonl")):
            for line in shard.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a concurrently half-written trailing line
                if record.get("schema") != SCHEMA:
                    continue
                graph._absorb(record["keys"], float(record["reward"]))
        return graph

    def _absorb(self, keys: Iterable[Sequence[str]], reward: float) -> None:
        self.games += 1
        nodes: set[tuple[str, ...]] = set()
        for key in keys:
            nodes.update(node_chain(tuple(key)))  # type: ignore[arg-type]
        for node in nodes:
            wins_losses = self.counts.setdefault(node, [0.0, 0.0])
            wins_losses[0] += reward
            wins_losses[1] += 1.0 - reward

    def posterior(self, key: Key) -> tuple[float, float]:
        """Hierarchical Beta posterior for the most specific node."""
        alpha, beta = 1.0, 1.0
        for node in node_chain(key):
            wins, losses = self.counts.get(node, (0.0, 0.0))
            mean = alpha / (alpha + beta)
            alpha = self.prior_strength * mean + wins
            beta = self.prior_strength * (1 - mean) + losses
        return alpha, beta

    def observations(self, key: Key) -> float:
        wins, losses = self.counts.get(node_chain(key)[-1], (0.0, 0.0))
        return wins + losses

    def record_game(
        self, keys: Iterable[Key], outcome: str, provenance: dict[str, Any] | None = None,
    ) -> bool:
        """Append one game; returns False for outcomes that carry no reward."""
        reward = {"win": 1.0, "loss": 0.0, "draw": 0.5}.get(outcome)
        if reward is None:
            return False
        unique = sorted({tuple(key) for key in keys})
        record = {
            "schema": SCHEMA,
            "writer": self.writer_id,
            "outcome": outcome,
            "reward": reward,
            "keys": [list(key) for key in unique],
            "provenance": provenance or {},
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f"{self.writer_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._absorb(unique, reward)
        return True


class ExperienceScorer:
    """Turn graph posteriors into per-candidate utilities for a picker."""

    def __init__(
        self,
        graph: ExperienceGraph,
        *,
        mode: str = "thompson",
        novelty_weight: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if mode not in EXPLORATION_MODES:
            raise ValueError(f"exploration mode must be one of {EXPLORATION_MODES}")
        self.graph = graph
        self.mode = mode
        self.novelty_weight = novelty_weight
        self._rng = np.random.default_rng(seed)

    def utility(self, key: Key | None) -> float | None:
        if key is None:
            return None
        alpha, beta = self.graph.posterior(key)
        if self.mode == "thompson":
            return float(self._rng.beta(alpha, beta))
        mean = alpha / (alpha + beta)
        if self.mode == "novelty":
            return mean + self.novelty_weight * beta_bernoulli_information_gain(alpha, beta)
        return mean

    def utilities(self, keys: Sequence[Key | None]) -> list[float | None]:
        return [self.utility(key) for key in keys]


def centred(values: Sequence[float | None]) -> list[float]:
    """Centre known utilities on their mean; unknown entries become 0."""
    known = [v for v in values if v is not None]
    if not known:
        return [0.0] * len(values)
    mean = sum(known) / len(known)
    return [0.0 if v is None else v - mean for v in values]


class ExperienceSession:
    """Per-game bridge between a picker and the shared graph.

    ``learn=False`` reads the graph as a frozen prior (evaluation); ``learn``
    additionally appends this game's first-visit evidence when it ends.
    """

    def __init__(self, scorer: ExperienceScorer, *, learn: bool, agent: str) -> None:
        self.scorer = scorer
        self.learn = learn
        self.agent = agent
        self.keys: list[Key] = []

    def note(self, key: Key | None) -> None:
        if key is not None:
            self.keys.append(key)

    def finish(self, outcome: str, provenance: dict[str, Any] | None = None) -> bool:
        if not self.learn:
            return False
        meta = {"agent": self.agent, "mode": self.scorer.mode, **(provenance or {})}
        return self.scorer.graph.record_game(self.keys, outcome, meta)

    def trace(self) -> dict[str, Any]:
        return {
            "experience_mode": self.scorer.mode,
            "experience_learn": self.learn,
            "experience_games": self.scorer.graph.games,
        }

"""Deck mutator — G5.

Iteratively improves a Commander deck via a (μ+λ) evolutionary / simulated-
annealing swap loop.  Starting from a deck built by :class:`DeckBuilderAgent`
(G3), each iteration:

1. Identifies the *K* worst cards by :class:`DeckEvaluator` contribution
   (G4) — the "swap candidates".
2. Scores *N* replacement cards from the pool via :class:`CardScorer` (G2).
3. Proposes a batch of *K* swaps (one swap per worst card).
4. Evaluates the mutated deck with :class:`DeckEvaluator`.
5. Accepts the mutation if it improves win rate (greedy) or with a
   temperature-scaled probability (simulated annealing).

After ``max_iterations`` the best-seen deck is returned together with a
full :class:`MutationLog`.

Usage::

    from src.agents.deck_builder.mutator import DeckMutator

    mutator = DeckMutator(
        evaluator=DeckEvaluator(
            reference_decks=[("data/decks/edh/atraxa-praetors-voice_core.txt", "heuristic")],
            games_per_pair=2,
            format="commander",
        ),
        scorer=CardScorer(),
        card_db=get_default_db(),
        max_iterations=10,
        swaps_per_iter=3,
        pool_sample=200,
        seed=42,
    )
    best_deck, log = asyncio.run(
        mutator.mutate(initial_decklist, commander_cards)
    )
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result / logging types
# ---------------------------------------------------------------------------


@dataclass
class MutationStep:
    """Record of a single mutation iteration."""
    iteration: int
    swaps_proposed: list[tuple[str, str]]  # (removed, added)
    win_rate_before: float
    win_rate_after: float
    accepted: bool
    elapsed: float = 0.0

    @property
    def delta(self) -> float:
        return self.win_rate_after - self.win_rate_before


@dataclass
class MutationLog:
    """Full history of the mutation loop."""
    steps: list[MutationStep] = field(default_factory=list)
    best_win_rate: float = 0.0
    total_elapsed: float = 0.0

    def accepted_steps(self) -> list[MutationStep]:
        return [s for s in self.steps if s.accepted]

    def summary(self) -> str:
        accepted = len(self.accepted_steps())
        lines = [
            f"Mutation log: {len(self.steps)} iterations, "
            f"{accepted} accepted, "
            f"best win rate {self.best_win_rate:.1%}, "
            f"elapsed {self.total_elapsed:.1f}s",
        ]
        for s in self.steps:
            mark = "✓" if s.accepted else "✗"
            swap_str = ", ".join(f"{r}→{a}" for r, a in s.swaps_proposed)
            lines.append(
                f"  [{mark}] iter {s.iteration:2d}  Δ={s.delta:+.3f}  [{swap_str}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------


class DeckMutator:
    """Evolutionary / simulated-annealing deck optimizer (G5).

    Args:
        evaluator:       :class:`~src.agents.deck_builder.evaluator.DeckEvaluator`
            instance to score decks.
        scorer:          :class:`~src.agents.deck_builder.scorer.CardScorer`
            instance to score candidate replacements.
        card_db:         :class:`~src.integrations.offline_card_db.OfflineCardDB`.
        max_iterations:  Maximum number of mutation attempts.
        swaps_per_iter:  Cards to swap out per iteration (K).
        pool_sample:     Candidate replacement pool size drawn randomly from
            the full card DB each iteration.
        temperature:     Initial SA temperature.  Set to 0 to use pure greedy
            hill-climbing (only accept improvements).
        cooling:         Multiplicative cooling factor applied each iteration.
        seed:            RNG seed.
    """

    def __init__(
        self,
        evaluator: "DeckEvaluator",
        scorer: "CardScorer | None" = None,
        card_db: Any = None,
        *,
        max_iterations: int = 10,
        swaps_per_iter: int = 3,
        pool_sample: int = 200,
        temperature: float = 0.05,
        cooling: float = 0.8,
        seed: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.scorer = scorer
        self.card_db = card_db
        self.max_iterations = max_iterations
        self.swaps_per_iter = swaps_per_iter
        self.pool_sample = pool_sample
        self.temperature = temperature
        self.cooling = cooling
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mutate(
        self,
        initial_decklist: "Decklist",
        commander_cards: list[dict[str, Any]],
    ) -> tuple["Decklist", MutationLog]:
        """Run the mutation loop; return ``(best_decklist, log)``."""
        from src.agents.deck_builder.constraints import (
            ConstraintSet,
            commander_color_identity,
            is_basic_land,
        )
        from src.integrations.offline_card_db import get_default_db
        from src.integrations.decklist_loader import Decklist

        if self.card_db is None:
            self.card_db = get_default_db()

        cmd_ci = commander_color_identity(commander_cards)
        constraints_template = ConstraintSet(
            commander_ci=cmd_ci, format=self.evaluator.format
        )

        log = MutationLog()
        t_global = time.perf_counter()

        current = copy.deepcopy(initial_decklist)
        current_eval = await self.evaluator.evaluate(
            current, commander_cards, self.card_db
        )
        current_wr = current_eval.win_rate

        best_deck = copy.deepcopy(current)
        log.best_win_rate = current_wr
        temp = self.temperature

        logger.info(
            "Mutation start: baseline win rate %.1f%%  (max_iter=%d, swaps=%d)",
            current_wr * 100, self.max_iterations, self.swaps_per_iter,
        )

        for iteration in range(1, self.max_iterations + 1):
            t_iter = time.perf_counter()

            # ----------------------------------------------------------------
            # 1. Pick worst cards to swap out.
            # ----------------------------------------------------------------
            # Re-evaluate contributions from last eval result.
            bottom = current_eval.bottom_contributors(
                n=self.swaps_per_iter * 3  # oversample so we avoid basics
            )
            remove_candidates = [
                name for name, _ in bottom
                if name in current.mainboard
                and not _is_basic(name)
            ][:self.swaps_per_iter]

            if not remove_candidates:
                logger.info("iter %d: no swap candidates — stopping early", iteration)
                break

            # ----------------------------------------------------------------
            # 2. Build replacement pool (random sample from card DB).
            # ----------------------------------------------------------------
            existing = set(current.mainboard.keys()) | {
                c.get("name", "") for c in commander_cards
            }
            all_names = list(self.card_db.all_names())
            self._rng.shuffle(all_names)
            pool_names = [
                n for n in all_names
                if n not in existing
            ][:self.pool_sample]

            pool_cards: list[dict[str, Any]] = []
            for name in pool_names:
                data = self.card_db.get(name)
                if data is None:
                    continue
                # Quick colour-identity filter before scoring.
                result = constraints_template.accept(data)
                if result.ok:
                    pool_cards.append(data)

            if not pool_cards:
                logger.warning("iter %d: empty replacement pool", iteration)
                break

            # ----------------------------------------------------------------
            # 3. Score replacements (use scorer if available).
            # ----------------------------------------------------------------
            if self.scorer is not None:
                scored = await self.scorer.score_batch(
                    pool_cards, list(current.mainboard.keys())
                )
                scored.sort(key=lambda x: x[1], reverse=True)
                replacements = [c for c, _ in scored[:self.swaps_per_iter]]
            else:
                replacements = pool_cards[:self.swaps_per_iter]

            # Ensure we have enough replacements.
            replacements = replacements[: len(remove_candidates)]

            # ----------------------------------------------------------------
            # 4. Build candidate deck with swaps applied.
            # ----------------------------------------------------------------
            candidate = copy.deepcopy(current)
            swaps: list[tuple[str, str]] = []
            for out_name, in_card in zip(remove_candidates, replacements):
                in_name = in_card.get("name", "?")
                count = candidate.mainboard.pop(out_name, 0)
                if count > 1:
                    candidate.mainboard[out_name] = count - 1
                candidate.mainboard[in_name] = candidate.mainboard.get(in_name, 0) + 1
                swaps.append((out_name, in_name))

            # ----------------------------------------------------------------
            # 5. Evaluate candidate deck.
            # ----------------------------------------------------------------
            cand_eval = await self.evaluator.evaluate(
                candidate, commander_cards, self.card_db
            )
            cand_wr = cand_eval.win_rate
            delta = cand_wr - current_wr

            # ----------------------------------------------------------------
            # 6. Acceptance criterion (greedy or SA).
            # ----------------------------------------------------------------
            if delta >= 0:
                accept = True
            elif temp > 0:
                accept = self._rng.random() < math.exp(delta / temp)
            else:
                accept = False

            step = MutationStep(
                iteration=iteration,
                swaps_proposed=swaps,
                win_rate_before=current_wr,
                win_rate_after=cand_wr,
                accepted=accept,
                elapsed=time.perf_counter() - t_iter,
            )
            log.steps.append(step)

            if accept:
                current = candidate
                current_eval = cand_eval
                current_wr = cand_wr
                if current_wr > log.best_win_rate:
                    best_deck = copy.deepcopy(current)
                    log.best_win_rate = current_wr

            # Cool down.
            temp *= self.cooling

            logger.info(
                "iter %d/%d  %s  Δ=%+.3f  wr_after=%.1f%%  temp=%.4f  swaps=%s",
                iteration,
                self.max_iterations,
                "✓" if accept else "✗",
                delta,
                cand_wr * 100,
                temp,
                ", ".join(f"{r}→{a}" for r, a in swaps),
            )

        log.total_elapsed = time.perf_counter() - t_global
        logger.info("Mutation done.\n%s", log.summary())
        return best_deck, log

    def mutate_sync(
        self,
        initial_decklist: "Decklist",
        commander_cards: list[dict[str, Any]],
    ) -> tuple["Decklist", MutationLog]:
        """Synchronous wrapper around :meth:`mutate`."""
        return asyncio.run(self.mutate(initial_decklist, commander_cards))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASIC_LAND_NAMES: frozenset[str] = frozenset({
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest",
})


def _is_basic(name: str) -> bool:
    return name in _BASIC_LAND_NAMES

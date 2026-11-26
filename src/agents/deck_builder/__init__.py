"""Deck-builder agent package (Phase G).

Entry point::

    from src.agents.deck_builder import DeckBuilderAgent

See ``scripts/brew_decks.py`` for the CLI runner.
"""

from src.agents.deck_builder.agent import DeckBuilderAgent
from src.agents.deck_builder.constraints import (
    ConstraintSet,
    commander_color_identity,
    validate_full_deck,
)
from src.agents.deck_builder.evaluator import DeckEvaluator, EvalResult
from src.agents.deck_builder.mutator import DeckMutator, MutationLog, MutationStep
from src.agents.deck_builder.scorer import CardScorer

__all__ = [
    "DeckBuilderAgent",
    "ConstraintSet",
    "CardScorer",
    "commander_color_identity",
    "validate_full_deck",
    "DeckEvaluator",
    "EvalResult",
    "DeckMutator",
    "MutationLog",
    "MutationStep",
]

"""Tests for the KG enrichment + surprise detection + tokenized trajectory pipeline."""

import asyncio
import pytest
import numpy as np

from src.world_model.trajectory import Trajectory, Transition, TrajectoryStore
from src.knowledge.kg_enrichment import KGEnrichment, EnrichmentConfig, EnrichmentReport
from src.world_model.surprise_detector import SurpriseDetector, SurpriseConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(action_type: str, card_name: str | None = None, reward: float = 0.0, done: bool = False):
    return Transition(
        state_features={"player_features": np.zeros(12, dtype=np.float32)},
        action_encoding=np.zeros(136, dtype=np.float32),
        reward=reward,
        done=done,
        action_type=action_type,
        card_name=card_name,
    )


def _make_trajectory(game_id: str, cards: list[str], winner: int | None = 0) -> Trajectory:
    transitions = [_make_transition("CAST_SPELL", card) for card in cards]
    if transitions:
        transitions[-1].done = True
        transitions[-1].reward = 1.0 if winner == 0 else 0.0
    return Trajectory(
        game_id=game_id,
        transitions=transitions,
        winner=winner,
        num_turns=len(transitions),
        source="test",
    )


# ---------------------------------------------------------------------------
# KG Enrichment tests
# ---------------------------------------------------------------------------

class TestKGEnrichment:
    """Tests for KGEnrichment synergy and combo discovery."""

    def test_extract_card_statistics(self):
        enrichment = KGEnrichment()
        trajs = [
            _make_trajectory("g1", ["Lightning Bolt", "Goblin Guide", "Mountain"], winner=0),
            _make_trajectory("g2", ["Lightning Bolt", "Goblin Guide", "Swamp"], winner=0),
            _make_trajectory("g3", ["Counterspell", "Island", "Brainstorm"], winner=1),
        ]

        game_cards, card_wins, card_games = enrichment._extract_card_statistics(trajs)

        assert len(game_cards) == 3
        assert card_games["Lightning Bolt"] == 2
        assert card_wins["Lightning Bolt"] == 2
        assert card_games["Counterspell"] == 1
        assert card_wins.get("Counterspell", 0) == 0  # winner=1, not player 0

    def test_discover_synergies(self):
        config = EnrichmentConfig(min_co_occurrence=2, min_synergy_lift=1.0)
        enrichment = KGEnrichment(config=config)

        # Create games where "Bolt + Guide" always win together
        game_cards = [
            ({"Lightning Bolt", "Goblin Guide"}, True),
            ({"Lightning Bolt", "Goblin Guide"}, True),
            ({"Lightning Bolt", "Goblin Guide"}, True),
            ({"Counterspell", "Island"}, False),
            ({"Counterspell", "Island"}, False),
        ]
        card_wins = {"Lightning Bolt": 3, "Goblin Guide": 3}
        card_games = {"Lightning Bolt": 3, "Goblin Guide": 3, "Counterspell": 2, "Island": 2}

        from collections import Counter
        synergies = enrichment._discover_synergies(
            game_cards, Counter(card_wins), Counter(card_games)
        )

        assert len(synergies) >= 1
        bolt_guide = [s for s in synergies if "Goblin Guide" in (s["card_a"], s["card_b"])
                      and "Lightning Bolt" in (s["card_a"], s["card_b"])]
        assert len(bolt_guide) == 1
        assert bolt_guide[0]["lift"] >= 1.0
        assert bolt_guide[0]["co_occurrences"] == 3

    def test_discover_combos(self):
        config = EnrichmentConfig(min_combo_co_occurrence=2, max_combo_size=3)
        enrichment = KGEnrichment(config=config)

        game_cards = [
            ({"Devoted Druid", "Vizier of Remedies", "Walking Ballista"}, True),
            ({"Devoted Druid", "Vizier of Remedies", "Walking Ballista"}, True),
            ({"Some Card", "Another Card"}, True),
        ]

        combos = enrichment._discover_combos(game_cards)

        # Should find the 3-card combo appearing 2 times in wins
        three_card = [c for c in combos if c["size"] == 3]
        assert len(three_card) >= 1
        assert set(three_card[0]["cards"]) == {"Devoted Druid", "Vizier of Remedies", "Walking Ballista"}

    @pytest.mark.asyncio
    async def test_full_enrichment_offline(self):
        """Test full enrichment pipeline without a KG connection."""
        config = EnrichmentConfig(min_co_occurrence=2, min_synergy_lift=1.0)
        enrichment = KGEnrichment(kg=None, config=config)

        store = TrajectoryStore("data/test_trajectories")
        store.trajectories = [
            _make_trajectory(f"g{i}", ["Bolt", "Goblin"], winner=0)
            for i in range(5)
        ]
        store.trajectories.extend([
            _make_trajectory(f"loss{i}", ["Island", "Counterspell"], winner=1)
            for i in range(3)
        ])

        report = await enrichment.enrich_from_trajectories(store)

        assert isinstance(report, EnrichmentReport)
        assert report.games_analyzed == 8
        assert report.synergies_proposed >= 1
        # No KG connection, so nothing written
        assert report.synergies_written == 0


# ---------------------------------------------------------------------------
# Surprise Detector tests
# ---------------------------------------------------------------------------

class TestSurpriseDetector:
    """Tests for the surprise detection module."""

    def test_compute_errors_no_model(self):
        detector = SurpriseDetector(world_model=None)
        traj = _make_trajectory("g1", ["A", "B", "C"])
        errors = detector.compute_prediction_errors(traj)
        assert errors == []

    def test_retraining_weights(self):
        detector = SurpriseDetector()
        traj = _make_trajectory("g1", ["A", "B", "C", "D"])

        from src.world_model.surprise_detector import SurpriseEvent
        surprises = [
            SurpriseEvent(
                game_id="g1",
                transition_idx=1,
                prediction_error=5.0,
                action_type="CAST_SPELL",
                card_name="B",
            ),
        ]

        weights = detector.get_retraining_weights(traj, surprises)
        assert len(weights) == 4
        assert weights[0] == 1.0
        assert weights[1] == 3.0  # upweight_factor default
        assert weights[2] == 1.0

    def test_surprise_summary_empty(self):
        detector = SurpriseDetector()
        summary = detector.get_surprise_summary()
        assert summary["total_surprises"] == 0


# ---------------------------------------------------------------------------
# Tokenized Trajectory Integration
# ---------------------------------------------------------------------------

class TestTokenizedTrajectoryIntegration:
    """Tests that SelfPlayCollector produces real tokenized states."""

    def test_collector_with_tokenizer(self):
        from src.world_model.data_sources.self_play_collector import SelfPlayCollector
        from src.world_model.game_tokenizer import GameTokenizer
        from src.engine.game_state import (
            GameState, PlayerState, CardInstance, Phase, Zone,
            Action, ActionType,
        )

        tokenizer = GameTokenizer()
        collector = SelfPlayCollector(tokenizer=tokenizer)

        # Build a minimal game state
        players = [
            PlayerState(player_id="p0", name="P0", life_total=20),
            PlayerState(player_id="p1", name="P1", life_total=20),
        ]
        cards = [
            CardInstance(
                instance_id="p0_0",
                card_data={"name": "Mountain", "type_line": "Land", "mana_cost": ""},
                zone=Zone.HAND,
                owner_id="p0",
                controller_id="p0",
            ),
        ]
        game_state = GameState(
            format="standard",
            turn_number=1,
            active_player_index=0,
            priority_player_index=0,
            phase=Phase.MAIN_1,
            players=players,
            cards=cards,
        )

        collector.on_state(game_state, player_id="p0")

        # Record an action
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id="p0",
            card_instance_id="p0_0",
            metadata={"card_name": "Mountain"},
        )
        collector.on_action(action, reward=0.0)

        # Verify the transition has real features
        assert len(collector._current_transitions) == 1
        trans = collector._current_transitions[0]

        # State features should be a dict of numpy arrays, not empty
        assert isinstance(trans.state_features, dict)
        assert "player_features" in trans.state_features
        assert len(trans.state_features["player_features"]) >= 10  # player vitals + mana

        # Action encoding should be non-zero (PLAY_LAND one-hot + card embed)
        assert trans.action_encoding.shape[0] > 0

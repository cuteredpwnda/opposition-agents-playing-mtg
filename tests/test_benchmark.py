import pytest

from src.training.benchmark import BenchmarkSuite
from src.agents.random_agent import RandomAgent
from src.training.deck_utils import create_mock_deck


def test_create_mock_deck_size():
    deck = create_mock_deck()
    assert len(deck) == 60
    assert all("name" in card for card in deck)


@pytest.mark.asyncio
async def test_benchmark_random_vs_random():
    agents = {
        "RandomA": lambda pid: RandomAgent(player_id=pid),
        "RandomB": lambda pid: RandomAgent(player_id=pid),
    }
    suite = BenchmarkSuite(agents, num_games_per_match=2, max_turns=20)
    results = await suite.run()

    assert "win_rates" in results
    assert "RandomA" in results["win_rates"]
    assert "RandomB" in results["win_rates"]
    assert results["win_rates"]["RandomA"]["RandomB"] >= 0.0
    assert results["win_rates"]["RandomA"]["RandomB"] <= 1.0

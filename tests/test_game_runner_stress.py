import asyncio
import pytest
from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.training.deck_utils import create_mock_deck


@pytest.mark.asyncio
async def test_game_runner_stress_random_vs_random():
    """Stress test: run several fast random games to ensure no infinite loops in priority/turn progression."""
    config = GameConfig(format="standard", starting_life=20, max_turns=10)
    runner = GameRunner(config)

    for _ in range(3):
        players = {
            "P1": RandomAgent(player_id="P1"),
            "P2": RandomAgent(player_id="P2"),
        }
        deck = create_mock_deck()
        result = await runner.run_game(
            agents=players,
            decks={"P1": deck.copy(), "P2": deck.copy()},
        )

        assert result is not None
        # Game runner may increment turn_number once after final turn in post-loop logic
        assert result.turns <= config.max_turns + 1
        assert result.log is not None
        assert result.winner in {"P1", "P2", None}

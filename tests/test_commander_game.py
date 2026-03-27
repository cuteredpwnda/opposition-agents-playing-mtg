import pytest

from src.orchestrator.game_runner import GameRunner, GameConfig
from src.agents.random_agent import RandomAgent
from src.training.deck_utils import create_mock_deck


@pytest.mark.asyncio
async def test_commander_game_initial_state():
    # Four-player commander game with random agents
    players = ["P1", "P2", "P3", "P4"]
    agents = {pid: RandomAgent(player_id=pid) for pid in players}

    runner = GameRunner(GameConfig(format="commander", starting_life=40, max_turns=25))
    deck = create_mock_deck() * 2
    if len(deck) < 100:
        deck = (deck * 2)[:100]

    result = await runner.run_game(
        agents=agents,
        decks={pid: deck.copy() for pid in players},
    )

    assert result is not None
    assert result.turns >= 1
    assert result.winner in {None, *players}

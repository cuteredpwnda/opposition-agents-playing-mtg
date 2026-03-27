import pytest

from src.agents.neural_reasoner_agent import NeuralReasonerAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.training.deck_utils import create_mock_deck


@pytest.mark.asyncio
async def test_neural_reasoner_agent_plays_game():
    player_1 = NeuralReasonerAgent(player_id="player_1")
    player_2 = NeuralReasonerAgent(player_id="player_2")

    runner = GameRunner(GameConfig(max_turns=20))
    result = await runner.run_game(
        agents={"player_1": player_1, "player_2": player_2},
        decks={"player_1": create_mock_deck(), "player_2": create_mock_deck()},
    )

    assert result is not None
    assert result.turns >= 1
    assert result.winner in {"player_1", "player_2", None}

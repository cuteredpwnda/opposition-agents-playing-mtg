import pytest

from src.training.rl_trainer import RLConfig, RLTrainer


def test_agent_pool_get_best_agents():
    config = RLConfig(num_iterations=1, games_per_iteration=1, eval_games_per_iteration=1)
    trainer = RLTrainer(config)
    trainer.setup()

    best, second = trainer.pool.get_best_agents()

    assert best in trainer.pool.agents
    assert second in trainer.pool.agents
    assert best != second


@pytest.mark.asyncio
async def test_champion_promotion_logic():
    config = RLConfig(num_iterations=1, games_per_iteration=1, eval_games_per_iteration=1)
    trainer = RLTrainer(config)
    trainer.setup()

    trainer.pool.elo = {
        "champion_0": 1300.0,
        "challenger_0": 1200.0,
    }
    trainer.pool.agents = {
        "champion_0": {"type": "random", "factory": lambda player_id: None},
        "challenger_0": {"type": "random", "factory": lambda player_id: None},
    }
    trainer.champion_id = "champion_0"

    assert trainer._pick_challenger() == "challenger_0"
    promoted = trainer._promote_candidate("challenger_0", "champion_0", 0.60)
    assert promoted is True
    assert trainer.pool.elo["challenger_0"] > 1200.0
    assert trainer.pool.elo["champion_0"] < 1300.0

    demoted = trainer._promote_candidate("challenger_0", "champion_0", 0.50)
    assert demoted is False

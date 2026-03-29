import pytest

from src.agents.active_inference_agent import ActiveInferenceAgent
from src.engine.game_state import Action, ActionType, GameState, PlayerState, Phase
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.training.deck_utils import create_mock_deck


@pytest.mark.asyncio
async def test_active_inference_agent_plays_game():
    player_1 = ActiveInferenceAgent(player_id="player_1")
    player_2 = ActiveInferenceAgent(player_id="player_2")

    runner = GameRunner(GameConfig(max_turns=20))
    result = await runner.run_game(
        agents={"player_1": player_1, "player_2": player_2},
        decks={"player_1": create_mock_deck(), "player_2": create_mock_deck()},
    )

    assert result is not None
    assert result.turns >= 1
    assert result.winner in {"player_1", "player_2", None}


@pytest.mark.asyncio
async def test_active_inference_agent_chooses_high_threat_attacker():
    agent = ActiveInferenceAgent(player_id="player_1")

    class DummyAIModule:
        async def initialize_beliefs(self, opponent_id):
            return

        def rank_actions(self, legal_actions, game_state):
            return [(a, 0.1) for a in legal_actions]

    class DummyOpponentModel:
        async def get_threat_assessment(self):
            return None

    agent.ai_module = DummyAIModule()
    agent.opponent_model = DummyOpponentModel()

    from src.engine.game_state import GameState, PlayerState, Phase, Zone, CardInstance

    game_state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.COMBAT_ATTACKERS,
        players=[
            PlayerState(player_id="player_1", name="player_1", life_total=40),
            PlayerState(player_id="player_2", name="player_2", life_total=20, commander_damage_received={"player_1": 5}),
            PlayerState(player_id="player_3", name="player_3", life_total=10, commander_damage_received={"player_1": 1}),
        ],
        cards=[],
    )

    legal_actions = [
        Action(action_type=ActionType.DECLARE_ATTACKERS, player_id="player_1", card_instance_id="c1", targets=["player_2"]),
        Action(action_type=ActionType.DECLARE_ATTACKERS, player_id="player_1", card_instance_id="c1", targets=["player_3"]),
    ]

    chosen = await agent.decide_action(game_state, legal_actions)
    assert chosen.targets == ["player_3"], "Should prefer the lower-life/higher-threat target"


@pytest.mark.asyncio
async def test_active_inference_agent_avoids_counterspell_risk(monkeypatch):
    from src.agents.opponent_model import ThreatAssessment
    from src.engine.game_state import GameState, PlayerState, Phase

    agent = ActiveInferenceAgent(player_id="player_1")

    class FakeAIModule:
        async def initialize_beliefs(self, opponent_id):
            return

        def rank_actions(self, legal_actions, game_state):
            return [
                (legal_actions[0], 0.1),
                (legal_actions[1], 0.2),
            ]

    class FakeOpponentModel:
        async def get_threat_assessment(self):
            return ThreatAssessment(
                archetype="control",
                confidence=0.9,
                predicted_hand={},
                probability_has_counterspell=0.95,
                probability_has_removal=0.3,
                probability_has_boardwipe=0.1,
                cards_remaining_in_deck=40,
                cards_in_hand_count=5,
            )

    agent.ai_module = FakeAIModule()
    agent.opponent_model = FakeOpponentModel()

    game_state = GameState(
        format="standard",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[
            PlayerState(player_id="player_1", life_total=20),
            PlayerState(player_id="player_2", life_total=20),
        ],
        cards=[],
    )

    legal_actions = [
        Action(action_type=ActionType.CAST_SPELL, player_id="player_1", card_instance_id="c1"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="player_1"),
    ]

    chosen = await agent.decide_action(game_state=game_state, legal_actions=legal_actions)
    assert chosen.action_type == ActionType.PASS_PRIORITY


@pytest.mark.asyncio
async def test_active_inference_agent_avoids_counterspell_risk(monkeypatch):
    from src.agents.opponent_model import ThreatAssessment

    agent = ActiveInferenceAgent(player_id="player_1")

    class FakeAIModule:
        async def initialize_beliefs(self, opponent_id):
            return

        def rank_actions(self, legal_actions, game_state):
            return [
                (legal_actions[0], 0.1),
                (legal_actions[1], 0.2),
            ]

    class FakeOpponentModel:
        async def get_threat_assessment(self):
            return ThreatAssessment(
                archetype="control",
                confidence=0.9,
                predicted_hand={},
                probability_has_counterspell=0.95,
                probability_has_removal=0.3,
                probability_has_boardwipe=0.1,
                cards_remaining_in_deck=40,
                cards_in_hand_count=5,
            )

    agent.ai_module = FakeAIModule()
    agent.opponent_model = FakeOpponentModel()

    legal_actions = [
        Action(action_type=ActionType.CAST_SPELL, player_id="player_1", card_instance_id="c1"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="player_1"),
    ]

    from src.engine.game_state import GameState, PlayerState

    dummy_state = GameState(
        format="standard",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[
            PlayerState(player_id="player_1", life_total=20),
            PlayerState(player_id="player_2", life_total=20),
        ],
        cards=[],
    )

    chosen = await agent.decide_action(game_state=dummy_state, legal_actions=legal_actions)
    assert chosen.action_type == ActionType.PASS_PRIORITY

import pytest
import numpy as np
import torch

from src.engine.game_state import GameState, PlayerState, CardInstance, Zone, Phase, ActionType, Action
from src.world_model.card_embeddings import CardEmbeddingModel
from src.world_model.game_tokenizer import GameTokenizer
from src.world_model.state_encoder import StateEncoder
from src.world_model.dynamics_model import DynamicsModel
from src.world_model.controller import Controller
from src.world_model.world_model import WorldModel
from src.world_model.trajectory import TrajectoryStore, Trajectory, Transition
from src.world_model.training.dream_trainer import DreamTrainer, DreamTrainerConfig


def _build_minimal_game_state():
    p1 = PlayerState(player_id="p1", name="Player 1")
    p2 = PlayerState(player_id="p2", name="Player 2")

    card = CardInstance(
        card_data={
            "name": "Lightning Bolt",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "type_line": "Instant",
            "mana_cost": "{R}",
            "cmc": 1,
        },
        zone=Zone.HAND,
        owner_id="p1",
        controller_id="p1",
        tapped=False,
        summoning_sick=False,
    )

    state = GameState(
        players=[p1, p2],
        cards=[card],
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        turn_number=1,
    )

    return state


def test_card_embedding_from_scryfall_and_trajectories():
    model = CardEmbeddingModel()
    model.build_from_scryfall([
        {
            "name": "Lightning Bolt",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "type_line": "Instant",
            "mana_cost": "{R}",
        },
        {
            "name": "Counterspell",
            "oracle_text": "Counter target spell.",
            "type_line": "Instant",
            "mana_cost": "{UU}",
        },
    ])

    assert "Lightning Bolt" in model.get_all_embeddings()
    assert model.get_embedding("Counterspell").shape == (model.embed_dim,)

    # Trajectory-based smoothing path should not raise
    traj = Trajectory(game_id="t1")
    tx = Transition(
        state_features={},
        action_encoding=np.zeros(model.embed_dim, dtype=np.float32),
        reward=1.0,
        done=True,
        action_type="CAST_SPELL",
        card_name="Lightning Bolt",
    )
    traj.add(tx)
    model.build_from_trajectories([traj])
    assert model.get_embedding("Lightning Bolt").shape == (model.embed_dim,)


def test_game_tokenizer_state_and_action_encoding():
    state = _build_minimal_game_state()
    model = CardEmbeddingModel()
    model.build_from_scryfall([
        {"name": "Lightning Bolt", "oracle_text": "Lightning Bolt deals 3 damage to any target.", "type_line": "Instant", "mana_cost": "{R}"}
    ])
    tokenizer = GameTokenizer(card_embeddings=model.get_all_embeddings())

    features = tokenizer.encode_state(state, player_id="p1")
    assert "player_features" in features
    assert features["hand_cards"].shape == (tokenizer.config.max_hand_size, tokenizer.config.card_embed_dim)

    action = Action(action_type=ActionType.CAST_SPELL, player_id="p1", card_instance_id=state.cards[0].instance_id, metadata={"card_name": "Lightning Bolt"})
    action_enc = tokenizer.encode_action(action)
    assert action_enc.shape == (tokenizer.compute_action_dim(),)


def test_state_encoder_end_to_end():
    state = _build_minimal_game_state()
    model = CardEmbeddingModel()
    model.build_from_scryfall([
        {"name": "Lightning Bolt", "oracle_text": "Lightning Bolt deals 3 damage to any target.", "type_line": "Instant", "mana_cost": "{R}"}
    ])
    tokenizer = GameTokenizer(card_embeddings=model.get_all_embeddings())

    features_np = tokenizer.encode_state(state, player_id="p1")
    features = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in features_np.items()}

    encoder = StateEncoder()
    z, mu, logvar = encoder.encode(features)
    assert z.shape == (1, encoder.config.latent_dim)

    reconstructed = encoder.decode(z)
    loss = encoder.loss(features, reconstructed, mu, logvar)
    assert loss["total"].item() >= 0.0


def test_dynamics_and_controller_and_world_dream_cycle():
    w = WorldModel()
    z = torch.randn(1, w.encoder.config.latent_dim)
    hidden = w.dynamics.initial_hidden(batch_size=1)
    action_enc = torch.zeros(1, w.controller.config.action_dim)

    z2, hidden2, done_prob = w.predict(z, action_enc, hidden, temperature=1.0)
    assert z2.shape == (1, w.encoder.config.latent_dim)
    assert 0 <= done_prob.item() <= 1

    # controller with randomized hidden vector
    h_vec = w.dynamics.get_hidden_state_vector(hidden2)
    legal = torch.zeros(1, 2, w.controller.config.action_dim)
    legal[:, 0, :] = action_enc
    legal[:, 1, :] = torch.randn_like(action_enc)
    mask = torch.tensor([[1.0, 1.0]])

    idx, log_prob = w.act(z2, h_vec, legal, mask, deterministic=True)
    assert idx in (0, 1)
    assert log_prob.shape == (1,)

    # dream search should run and return index
    best_idx = w.dream_search(z2, hidden2, legal, mask, num_rollouts=1, rollout_depth=3)[0]
    assert isinstance(best_idx, int)


def test_dream_trainer_runs_without_failure(tmp_path):
    store = TrajectoryStore(storage_dir=str(tmp_path / "trajectories"))
    controller_config = DreamTrainerConfig().controller_config
    controller_config.method = "policy_gradient"
    controller_config.pg_epochs = 1
    controller_config.pg_batch_size = 2

    trainer = DreamTrainer(
        DreamTrainerConfig(
            num_iterations=1,
            min_trajectories=0,
            trajectories_per_iteration=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            controller_config=controller_config,
        )
    )
    world_model = WorldModel()

    # Should not raise even with empty store
    w = trainer.train(store, world_model=world_model)
    assert isinstance(w, WorldModel)


def test_trajectory_store_exports_hdf5(tmp_path):
    from src.world_model.trajectory import TrajectoryStore, Trajectory, Transition

    store = TrajectoryStore(storage_dir=str(tmp_path / "traj"))
    t = Trajectory(game_id="game1", source="test")
    t.add(Transition(
        state_features={"player_features": np.zeros((10,), dtype=np.float32)},
        action_encoding=np.zeros((136,), dtype=np.float32),
        reward=0.0,
        done=False,
        action_type="PASS_PRIORITY",
    ))
    store.add(t)
    out_h5 = tmp_path / "world_model_train.h5"
    store.save_hdf5(str(out_h5))

    assert out_h5.exists()
    import h5py
    with h5py.File(str(out_h5), "r") as f:
        assert f.attrs["num_trajectories"] == 1


def test_stable_worldmodel_adapter_importable():
    try:
        import stable_pretraining  # noqa: F401
        import stable_worldmodel  # noqa: F401
    except ImportError:
        pytest.skip("stable training dependencies are not installed")

    import scripts.train_stable_worldmodel as stable_train

    assert callable(stable_train.main)


def test_stable_training_entrypoint_runs(tmp_path):
    try:
        import stable_pretraining  # noqa: F401
        import stable_worldmodel  # noqa: F401
    except ImportError:
        pytest.skip("stable training dependencies are not installed")

    from src.world_model.card_embeddings import CardEmbeddingModel
    from src.world_model.game_tokenizer import GameTokenizer
    from scripts.train_stable_worldmodel import main
    import sys

    state = _build_minimal_game_state()
    card_model = CardEmbeddingModel()
    card_model.build_from_scryfall([
        {
            "name": "Lightning Bolt",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "type_line": "Instant",
            "mana_cost": "{R}",
        }
    ])
    tokenizer = GameTokenizer(card_embeddings=card_model.get_all_embeddings())
    features = tokenizer.encode_state(state, player_id="p1")

    store = TrajectoryStore(storage_dir=str(tmp_path / "traj"))
    for game_id in ("t1", "t2"):
        traj = Trajectory(game_id=game_id, source="test")
        traj.add(Transition(
            state_features=features,
            action_encoding=np.zeros((136,), dtype=np.float32),
            reward=0.0,
            done=False,
            action_type="PASS_PRIORITY",
        ))
        traj.add(Transition(
            state_features=features,
            action_encoding=np.zeros((136,), dtype=np.float32),
            reward=0.0,
            done=True,
            action_type="PASS_PRIORITY",
        ))
        store.add(traj)
    store.save()

    checkpoint = tmp_path / "stable.pt"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_stable_worldmodel.py",
            "--trajectories",
            str(tmp_path / "traj"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--no-kg",
            "--checkpoint",
            str(checkpoint),
        ]
        main()
    finally:
        sys.argv = old_argv

    assert checkpoint.exists()


def test_schmidhuber_worldmodel_adapter_trainable(tmp_path):
    from src.world_model.training.train_schmidhuber import SchmidhuberTrainingConfig, train_schmidhuber
    from src.world_model.trajectory import TrajectoryStore, Trajectory, Transition

    store = TrajectoryStore(storage_dir=str(tmp_path / "traj"))
    trx = Transition(
        state_features={"latent": np.zeros((256,), dtype=np.float32)},
        action_encoding=np.zeros((136,), dtype=np.float32),
        reward=0.0,
        done=False,
        action_type="PASS_PRIORITY",
    )
    t = Trajectory(game_id="t1", source="test")
    t.add(trx)
    t2 = Trajectory(game_id="t2", source="test")
    t2.add(trx)
    store.add(t)
    store.add(t2)

    config = SchmidhuberTrainingConfig(epochs=1, batch_size=1, device="cpu")
    model = train_schmidhuber(store, config)

    assert model is not None


def test_world_model_agent_kg_context_encode():
    from src.agents.world_model_agent import WorldModelAgent
    from src.world_model.kg_encoder import KGContextEncoder, KGContextEncoderConfig
    from src.world_model.card_embeddings import CardEmbeddingModel
    from src.world_model.world_model import WorldModel, WorldModelConfig
    from src.world_model.game_tokenizer import GameTokenizer
    from src.engine.game_state import GameState, PlayerState, CardInstance, Zone, Phase

    card_model = CardEmbeddingModel()
    kg_enc = KGContextEncoder(card_model, KGContextEncoderConfig())
    wm_cfg = WorldModelConfig(use_jepa=True)
    wm = WorldModel(wm_cfg)
    tokenizer = GameTokenizer(card_embeddings=card_model._embeddings)

    agent = WorldModelAgent(
        player_id="p1",
        world_model=wm,
        tokenizer=tokenizer,
        kg_encoder=kg_enc,
    )

    # dummy state
    gs = GameState(
        players=[PlayerState(player_id="p1", name="p1"), PlayerState(player_id="p2", name="p2")],
        cards=[
            CardInstance(card_data={"name":"Mountain","type_line":"Land","oracle_text":"{T}: Add {R}.","cmc":0}, zone=Zone.BATTLEFIELD, owner_id="p1", controller_id="p1"),
        ],
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        turn_number=1,
    )

    z, h = agent._encode_state(gs)
    assert z.shape[1] == wm.encoder.config.latent_dim
    assert h.shape[1] == wm.dynamics.config.hidden_dim


@pytest.mark.asyncio
async def test_game_runner_collects_selfplay_trajectory(tmp_path):
    from src.orchestrator.game_runner import GameRunner, GameConfig
    from src.world_model.data_sources.self_play_collector import SelfPlayCollector
    from src.agents.random_agent import RandomAgent

    from main import build_simple_deck

    collector = SelfPlayCollector()
    runner = GameRunner(GameConfig(format="standard", starting_life=10, max_turns=4), self_play_collector=collector)

    agents = {
        "Alice": RandomAgent(player_id="Alice"),
        "Bob": RandomAgent(player_id="Bob"),
    }

    deck = build_simple_deck()
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}

    result = await runner.run_game(agents, decks)
    assert result is not None
    assert len(collector.collected_trajectories) == 1
    traj = collector.collected_trajectories[0]
    assert traj.source == "self_play"
    assert traj.num_turns > 0


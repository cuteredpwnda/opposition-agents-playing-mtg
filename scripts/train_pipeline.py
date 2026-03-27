#!/usr/bin/env python
"""
End-to-end training pipeline — Knowledge Graph + JEPA World Model.

Orchestrates the full flow from raw data to a trained dual-input JEPA
world model agent:

  Stage 1: Infrastructure check (Neo4j, GPU, dependencies)
  Stage 2: Knowledge Graph setup (Scryfall import → combos → ontology)
  Stage 3: Graph embedding training (GNN on card graph → 128-dim vectors)
  Stage 4: Self-play trajectory collection (random/simple agents)
  Stage 5: JEPA world model training (encoder + predictor + KG fusion)
  Stage 6: Full dream training (V → JEPA → M → C pipeline)
  Stage 7: Evaluation game with trained WorldModelAgent

Usage:
  python scripts/train_pipeline.py                  # Full pipeline
  python scripts/train_pipeline.py --stage 3        # Run from stage 3 onwards
  python scripts/train_pipeline.py --stage 5 --no-kg  # JEPA only, no KG fusion
  python scripts/train_pipeline.py --eval-only      # Just play a demo game

Requires:
  - Neo4j running (docker compose up neo4j)
  - pip install -r requirements.txt -r requirements-ml.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_pipeline")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 1 — Infrastructure Check                                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def stage_1_check_infra() -> dict[str, bool]:
    """Verify Python, PyTorch, Neo4j, GPU availability."""
    import platform
    status: dict[str, bool] = {}

    # Python
    py = platform.python_version()
    status["python_3.11+"] = sys.version_info >= (3, 11)
    logger.info("Python %s — %s", py, "OK" if status["python_3.11+"] else "NEED 3.11+")

    # PyTorch + GPU
    try:
        import torch
        status["pytorch"] = True
        status["cuda"] = torch.cuda.is_available()
        if status["cuda"]:
            logger.info("PyTorch %s — CUDA %s (%s)", torch.__version__,
                        torch.version.cuda, torch.cuda.get_device_name(0))
        else:
            logger.info("PyTorch %s — CPU only", torch.__version__)
    except ImportError:
        status["pytorch"] = False
        status["cuda"] = False
        logger.warning("PyTorch not installed")

    # Neo4j
    try:
        from neo4j import AsyncGraphDatabase
        from src.config import settings
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            _ = await result.single()
        await driver.close()
        status["neo4j"] = True
        logger.info("Neo4j at %s — OK", settings.neo4j_uri)
    except Exception as e:
        status["neo4j"] = False
        logger.warning("Neo4j not available: %s", e)

    # PyG
    try:
        from torch_geometric.nn import SAGEConv  # noqa: F401
        status["pyg"] = True
        logger.info("PyTorch Geometric — OK")
    except ImportError:
        status["pyg"] = False
        logger.warning("PyTorch Geometric not installed (graph embeddings will be skipped)")

    return status


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 2 — Knowledge Graph Setup                                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def stage_2_build_kg() -> bool:
    """Import Scryfall cards + combos into Neo4j, set up ontology."""
    logger.info("=== Stage 2: Building Knowledge Graph ===")

    try:
        from src.knowledge.n10s_setup import N10sSetup

        logger.info("Initialising n10s + ontology…")
        setup = N10sSetup()
        await setup.full_setup()
        logger.info("n10s setup complete")
    except Exception as e:
        logger.warning("n10s setup skipped: %s", e)

    # Scryfall import
    try:
        from scripts.import_scryfall import main as import_scryfall_main
        logger.info("Importing Scryfall card data…")
        await import_scryfall_main()
    except Exception as e:
        logger.error("Scryfall import failed: %s", e)
        return False

    # Combo import
    try:
        from scripts.import_combos import main as import_combos_main
        logger.info("Importing combos…")
        await import_combos_main()
    except Exception as e:
        logger.warning("Combo import skipped: %s", e)

    logger.info("Knowledge Graph ready")
    return True


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 3 — Graph Embedding Training                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def stage_3_train_graph_embeddings(
    embed_dim: int = 128,
    epochs: int = 200,
) -> bool:
    """Train GNN card embeddings on the Neo4j graph and cache locally."""
    logger.info("=== Stage 3: Training Graph Embeddings ===")

    try:
        from scripts.train_graph_embeddings import (
            export_graph_from_neo4j,
            save_embedding_cache,
            train_graph_embedder,
            write_embeddings_to_neo4j,
        )
    except ImportError as e:
        logger.warning("Graph embedding training not available: %s", e)
        return False

    data, name_to_idx = await export_graph_from_neo4j()
    _model, embeddings = train_graph_embedder(data, out_channels=embed_dim, epochs=epochs)
    save_embedding_cache(embeddings, name_to_idx)
    await write_embeddings_to_neo4j(embeddings, name_to_idx)
    logger.info("Graph embeddings trained and cached (%d cards, %d dims)",
                embeddings.size(0), embeddings.size(1))
    return True


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 4 — Self-Play Trajectory Collection                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def stage_4_collect_trajectories(
    num_games: int = 200,
    max_turns: int = 50,
) -> "TrajectoryStore":
    """Run self-play games to collect training trajectories.

    Uses the existing RLTrainer with random agents initially.
    Returns a TrajectoryStore with game trajectories.
    """
    logger.info("=== Stage 4: Collecting Self-Play Trajectories (%d games) ===", num_games)

    from src.world_model.trajectory import TrajectoryStore

    store = TrajectoryStore(base_dir="data/trajectories")

    # Try loading existing trajectories
    try:
        store.load()
        logger.info("Loaded %d existing trajectories", len(store))
    except Exception:
        pass

    if len(store) >= num_games:
        logger.info("Already have %d trajectories (target: %d), skipping collection",
                     len(store), num_games)
        return store

    needed = num_games - len(store)
    logger.info("Need %d more trajectories", needed)

    try:
        from src.training.rl_trainer import RLConfig, RLTrainer

        rl_config = RLConfig(
            num_iterations=max(needed // 20, 1),
            games_per_iteration=min(needed, 20),
            max_turns_per_game=max_turns,
            collect_trajectories=True,
            dream_training_interval=9999,  # disable dream training during collection
        )
        trainer = RLTrainer(rl_config)
        trainer.setup()
        await trainer.train()

        # Merge trainer's trajectories into our store
        if hasattr(trainer, "trajectory_store"):
            for traj in trainer.trajectory_store.trajectories:
                store.add(traj)
    except Exception as e:
        logger.error("Self-play collection failed: %s", e)
        logger.info("Continuing with %d trajectories", len(store))

    # Persist
    try:
        store.save()
    except Exception as e:
        logger.warning("Could not save trajectories: %s", e)

    logger.info("Trajectory collection done: %d total", len(store))
    return store


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 5 — JEPA World Model Training                              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def stage_5_train_jepa(
    store: "TrajectoryStore",
    use_kg: bool = True,
    kg_embed_dim: int = 128,
    jepa_beta: float = 1.0,
    num_epochs: int = 80,
) -> "WorldModel":
    """Train the dual-input JEPA world model on collected trajectories.

    This is the core new training step: encoder + JEPA predictor with
    optional KG context fusion.
    """
    logger.info("=== Stage 5: JEPA World Model Training ===")

    import torch
    from src.world_model.world_model import WorldModel, WorldModelConfig
    from src.world_model.state_encoder import StateEncoderConfig
    from src.world_model.jepa_predictor import JEPAPredictorConfig
    from src.world_model.training.train_jepa import JEPATrainingConfig, train_jepa

    # Build config with JEPA enabled
    encoder_cfg = StateEncoderConfig(
        kg_embed_dim=kg_embed_dim if use_kg else 0,
    )
    jepa_cfg = JEPAPredictorConfig(
        latent_dim=encoder_cfg.latent_dim,
        action_dim=136,  # 8 action types + 128 card embed
        jepa_beta=jepa_beta,
    )
    wm_cfg = WorldModelConfig(
        encoder=encoder_cfg,
        jepa=jepa_cfg,
        use_jepa=True,
    )
    world_model = WorldModel(wm_cfg)

    # Build KG encoder if enabled
    kg_encoder = None
    if use_kg:
        try:
            from src.world_model.card_embeddings import CardEmbeddingModel
            from src.world_model.kg_encoder import KGContextEncoder, KGContextEncoderConfig

            card_model = CardEmbeddingModel()

            # Try loading graph embeddings into card model
            try:
                from scripts.train_graph_embeddings import load_embedding_cache
                embeddings, name_to_idx = load_embedding_cache()
                for name, idx in name_to_idx.items():
                    card_model._embeddings[name] = embeddings[idx].numpy()
                logger.info("Loaded %d graph embeddings into card model", len(name_to_idx))
            except Exception as e:
                logger.info("No graph embeddings cached, using text embeddings: %s", e)

            kg_cfg = KGContextEncoderConfig(kg_embed_dim=kg_embed_dim)
            kg_encoder = KGContextEncoder(card_model, kg_cfg)
            logger.info("KG encoder enabled (dim=%d)", kg_embed_dim)
        except Exception as e:
            logger.warning("Could not build KG encoder: %s — training without KG", e)

    train_cfg = JEPATrainingConfig(
        num_epochs=num_epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    world_model = train_jepa(world_model, store, train_cfg, kg_encoder=kg_encoder)
    logger.info("JEPA training complete")
    return world_model


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 6 — Full Dream Training (V → JEPA → M → C)                ║
# ╚═══════════════════════════════════════════════════════════════════╝

def stage_6_dream_training(
    store: "TrajectoryStore",
    world_model: "WorldModel | None" = None,
    num_iterations: int = 3,
) -> "WorldModel":
    """Run the full V → JEPA → M → C dream training pipeline.

    If a world_model is provided (e.g. from stage 5), it continues
    training from that checkpoint.  Otherwise a fresh model is created.
    """
    logger.info("=== Stage 6: Dream Training (%d iterations) ===", num_iterations)

    from src.world_model.training.dream_trainer import DreamTrainer, DreamTrainerConfig

    dream_cfg = DreamTrainerConfig(num_iterations=num_iterations)
    trainer = DreamTrainer(dream_cfg)
    world_model = trainer.train(store, world_model=world_model)
    logger.info("Dream training complete")
    return world_model


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Stage 7 — Evaluation Game                                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def stage_7_eval_game(world_model: "WorldModel | None" = None) -> None:
    """Play a demo game with the trained agent (or random agents if no model)."""
    logger.info("=== Stage 7: Evaluation Game ===")

    try:
        from src.engine.game_runner import GameRunner
        from src.agents.random_agent import RandomAgent

        agents = {}
        if world_model is not None:
            try:
                from src.agents.world_model_agent import WorldModelAgent
                from src.world_model.card_embeddings import CardEmbeddingModel
                from src.world_model.game_tokenizer import GameTokenizer

                tokenizer = GameTokenizer(card_embeddings=CardEmbeddingModel()._embeddings)
                agents["player_0"] = WorldModelAgent(
                    "player_0", world_model, tokenizer, mode="direct",
                )
                logger.info("Player 0: WorldModelAgent (JEPA)")
            except Exception as e:
                logger.warning("Could not create WorldModelAgent: %s", e)
                agents["player_0"] = RandomAgent("player_0")
        else:
            agents["player_0"] = RandomAgent("player_0")

        agents["player_1"] = RandomAgent("player_1")

        # Build simple decks
        from main import build_simple_deck
        decks = {pid: build_simple_deck() for pid in agents}

        runner = GameRunner()
        result = await runner.run_game(agents, decks)
        logger.info("Game result: winner=%s, turns=%d", result.winner, result.turns)
    except Exception as e:
        logger.error("Evaluation game failed: %s", e)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Main Orchestrator                                                ║
# ╚═══════════════════════════════════════════════════════════════════╝

async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the training pipeline from the specified stage."""
    start = args.stage
    t0 = time.time()

    # --- Stage 1 ---
    if start <= 1:
        infra = await stage_1_check_infra()
        if not infra.get("pytorch"):
            logger.error("PyTorch is required. Install with: pip install -r requirements-ml.txt")
            return

    # --- Stage 2 ---
    if start <= 2 and not args.no_kg:
        try:
            await stage_2_build_kg()
        except Exception as e:
            logger.warning("KG setup failed: %s — continuing without KG", e)

    # --- Stage 3 ---
    if start <= 3 and not args.no_kg:
        try:
            await stage_3_train_graph_embeddings(embed_dim=args.kg_embed_dim)
        except Exception as e:
            logger.warning("Graph embedding training failed: %s — continuing", e)

    # --- Stage 4 ---
    if start <= 4:
        store = await stage_4_collect_trajectories(
            num_games=args.num_games, max_turns=args.max_turns,
        )
    else:
        from src.world_model.trajectory import TrajectoryStore
        store = TrajectoryStore(base_dir="data/trajectories")
        try:
            store.load()
        except Exception:
            pass
        if len(store) == 0:
            logger.error("No trajectories found. Run stage 4 first.")
            return

    # --- Stage 5 ---
    world_model = None
    if start <= 5:
        if args.wm_engine == "stable":
            from scripts.train_stable_worldmodel import main as stable_train_main
            import sys

            # simple entrypoint with command-line style args
            sys.argv = [sys.argv[0],
                        "--trajectories", str(store.storage_dir),
                        "--hdf5", "data/trajectories/world_model_train.h5",
                        "--epochs", str(args.jepa_epochs),
                        "--batch-size", str(args.jepa_batch_size),
                        "--device", "cuda" if hasattr(args, "cuda") and args.cuda else "cpu"]
            stable_train_main()
            # load an optional wrapper from stable output, if exists
            try:
                from src.world_model.stable_worldmodel_adapter import StableWorldModelAdapter
                world_model = StableWorldModelAdapter.load("checkpoints/stable_worldmodel.pt")
            except Exception as e:
                logger.warning("Could not load stable-worldmodel adapter artifact: %s", e)
                world_model = None
        else:
            world_model = stage_5_train_jepa(
                store,
                use_kg=not args.no_kg,
                kg_embed_dim=args.kg_embed_dim,
                jepa_beta=args.jepa_beta,
                num_epochs=args.jepa_epochs,
            )

    # --- Stage 6 ---
    if start <= 6 and not args.skip_dream:
        world_model = stage_6_dream_training(
            store,
            world_model=world_model,
            num_iterations=args.dream_iters,
        )

    # --- Stage 7 ---
    if not args.skip_eval:
        await stage_7_eval_game(world_model)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end KG + JEPA world model training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Stage control
    parser.add_argument("--stage", type=int, default=1,
                        help="Start from this stage (1-7)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run the evaluation game")

    # KG options
    parser.add_argument("--no-kg", action="store_true",
                        help="Disable knowledge graph (single-input JEPA)")
    parser.add_argument("--kg-embed-dim", type=int, default=128,
                        help="KG embedding dimension")
    parser.add_argument("--wm-engine", type=str, choices=["built_in", "stable"],
                        default="built_in",
                        help="World model training engine to use")

                        help="KG context embedding dimension")

    # Trajectory collection
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of self-play games to collect")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max turns per game")

    # JEPA training
    parser.add_argument("--jepa-beta", type=float, default=1.0,
                        help="JEPA KL regularizer weight (the ONE hyperparameter)")
    parser.add_argument("--jepa-epochs", type=int, default=80,
                        help="JEPA training epochs")
    parser.add_argument("--jepa-batch-size", type=int, default=64,
                        help="JEPA training batch size")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA for stable-worldmodel training if available")

    # Dream training
    parser.add_argument("--dream-iters", type=int, default=3,
                        help="Full V→M→C dream training iterations")
    parser.add_argument("--skip-dream", action="store_true",
                        help="Skip dream training (stages 1-5 only)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation game")

    args = parser.parse_args()

    if args.eval_only:
        args.stage = 7

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

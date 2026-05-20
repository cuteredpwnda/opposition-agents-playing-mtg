#!/usr/bin/env python
"""
End-to-end training pipeline â€” Knowledge Graph + JEPA World Model.

Orchestrates the full flow from raw data to a trained dual-input JEPA
world model agent:

  Stage 1: Infrastructure check (Neo4j, GPU, dependencies)
  Stage 2: Knowledge Graph setup (Scryfall import â†’ combos â†’ ontology)
  Stage 3: Graph embedding training (GNN on card graph â†’ 128-dim vectors)
  Stage 4: Self-play trajectory collection (random/simple agents)
  Stage 5: JEPA world model training (encoder + predictor + KG fusion)
  Stage 6: Full dream training (V â†’ JEPA â†’ M â†’ C pipeline)
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

# Force unbuffered stdout so progress shows live when piped/teed.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

print("[boot] train_pipeline.py starting...", flush=True)

# Always also write to runs/train_pipeline.log so progress survives any
# parent-shell stdout buffering (Windows Start-Process / PowerShell pipes).
_LOG_PATH = PROJECT_ROOT / "runs" / "train_pipeline.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_fmt)
_file_handler = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
_file_handler.setFormatter(_fmt)
_handler = _stream_handler  # kept for backward compat below
logging.basicConfig(
    level=logging.INFO, handlers=[_stream_handler, _file_handler], force=True
)
logger = logging.getLogger("train_pipeline")
logger.info("Logging to %s", _LOG_PATH)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 1 â€” Infrastructure Check                                  â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def stage_1_check_infra() -> dict[str, bool]:
    """Verify Python, PyTorch, Neo4j, GPU availability."""
    import platform
    status: dict[str, bool] = {}

    # Python
    py = platform.python_version()
    status["python_3.11+"] = sys.version_info >= (3, 11)
    logger.info("Python %s â€” %s", py, "OK" if status["python_3.11+"] else "NEED 3.11+")

    # PyTorch + GPU
    try:
        import torch
        status["pytorch"] = True
        status["cuda"] = torch.cuda.is_available()
        if status["cuda"]:
            logger.info("PyTorch %s â€” CUDA %s (%s)", torch.__version__,
                        torch.version.cuda, torch.cuda.get_device_name(0))
        else:
            logger.info("PyTorch %s â€” CPU only", torch.__version__)
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
        logger.info("Neo4j at %s â€” OK", settings.neo4j_uri)
    except Exception as e:
        status["neo4j"] = False
        logger.warning("Neo4j not available: %s", e)

    # PyG
    try:
        from torch_geometric.nn import SAGEConv  # noqa: F401
        status["pyg"] = True
        logger.info("PyTorch Geometric â€” OK")
    except ImportError:
        status["pyg"] = False
        logger.warning("PyTorch Geometric not installed (graph embeddings will be skipped)")

    return status


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 2 â€” Knowledge Graph Setup                                  â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def stage_2_build_kg() -> bool:
    """Import Scryfall cards + combos into Neo4j, set up ontology."""
    logger.info("=== Stage 2: Building Knowledge Graph ===")

    try:
        from src.knowledge.n10s_setup import N10sSetup

        logger.info("Initialising n10s + ontologyâ€¦")
        setup = N10sSetup()
        await setup.full_setup()
        logger.info("n10s setup complete")
    except Exception as e:
        logger.warning("n10s setup skipped: %s", e)

    # Scryfall import
    try:
        from scripts.import_scryfall import main as import_scryfall_main
        logger.info("Importing Scryfall card dataâ€¦")
        await import_scryfall_main()
    except Exception as e:
        logger.error("Scryfall import failed: %s", e)
        return False

    # Combo import
    try:
        from scripts.import_combos import main as import_combos_main
        logger.info("Importing combosâ€¦")
        await import_combos_main()
    except Exception as e:
        logger.warning("Combo import skipped: %s", e)

    logger.info("Knowledge Graph ready")
    return True


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 3 â€” Graph Embedding Training                               â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

    # Create vector index now that the embedding property is populated.
    try:
        from src.knowledge.n10s_setup import N10sSetup
        setup = N10sSetup()
        await setup.create_vector_index(dimensions=embed_dim)
        await setup.driver.close()
    except Exception as e:
        logger.warning("Vector index creation skipped: %s", e)

    logger.info("Graph embeddings trained and cached (%d cards, %d dims)",
                embeddings.size(0), embeddings.size(1))
    return True


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 4 â€” Self-Play Trajectory Collection                        â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def stage_4_collect_trajectories(
    num_games: int = 200,
    max_turns: int = 50,
) -> "TrajectoryStore":
    """Run self-play games to collect training trajectories.

    Plays games directly with HeuristicAgent vs RandomAgent using
    SelfPlayCollector to record encoded transitions.  Returns a
    TrajectoryStore with the collected games.
    """
    logger.info("=== Stage 4: Collecting Self-Play Trajectories (%d games) ===", num_games)

    from src.agents.heuristic_agent import HeuristicAgent
    from src.agents.random_agent import RandomAgent
    from src.orchestrator.game_runner import GameRunner, GameConfig
    from src.world_model.card_embeddings import CardEmbeddingModel
    from src.world_model.data_sources.self_play_collector import SelfPlayCollector
    from src.world_model.game_tokenizer import GameTokenizer
    from src.world_model.trajectory import TrajectoryStore

    store = TrajectoryStore(storage_dir="data/trajectories")

    try:
        store.load()
        logger.info("Loaded %d existing trajectories", len(store))
    except Exception:
        pass

    if len(store) >= num_games:
        logger.info("Already have %d trajectories (target: %d), skipping",
                    len(store), num_games)
        return store

    needed = num_games - len(store)
    logger.info("Need %d more trajectories", needed)

    # Set up shared tokenizer/embeddings once (expensive to build)
    card_model = CardEmbeddingModel()
    tokenizer = GameTokenizer(card_embeddings=card_model.get_all_embeddings())

    # Build deck from main.build_simple_deck for now
    from main import build_simple_deck

    for game_num in range(needed):
        try:
            collector = SelfPlayCollector(tokenizer=tokenizer, card_embeddings=card_model)
            agents = {
                "player_0": HeuristicAgent("player_0"),
                "player_1": RandomAgent("player_1"),
            }
            decks = {pid: build_simple_deck() for pid in agents}
            gc = GameConfig(format="standard", starting_life=20, max_turns=max_turns)
            runner = GameRunner(gc, self_play_collector=collector)
            result = await runner.run_game(agents, decks)
            for traj in collector.collected_trajectories:
                store.add(traj)
            logger.info(
                "Game %d/%d: winner=%s turns=%d transitions=%d",
                game_num + 1, needed, result.winner, result.turns,
                sum(len(t) for t in collector.collected_trajectories),
            )
        except Exception as e:
            logger.warning("Game %d failed: %s", game_num + 1, e)

    try:
        logger.info("Saving %d trajectories to disk...", len(store))
        t0 = time.time()
        store.save()
        logger.info("Saved trajectories in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("Could not save trajectories: %s", e)

    logger.info("Trajectory collection done: %d total", len(store))
    return store

async def stage_4_iterative_self_play(
    num_iterations: int = 8,
    games_per_iteration: int = 20,
    eval_games: int = 10,
    max_turns: int = 50,
    checkpoint_dir: str = "checkpoints/rl",
) -> "TrajectoryStore":
    """Run an iterative self-play promotion loop to collect trajectories."""
    logger.info(
        "=== Stage 4.5: Iterative Self-Play Promotion (%d iters, %d games/iter) ===",
        num_iterations,
        games_per_iteration,
    )

    from src.training.rl_trainer import RLConfig, RLTrainer
    from src.world_model.trajectory import TrajectoryStore

    rl_config = RLConfig(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        eval_games_per_iteration=eval_games,
        max_turns_per_game=max_turns,
        collect_trajectories=True,
        dream_training_interval=9999,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = RLTrainer(rl_config)
    await trainer.train()

    if trainer.trajectory_store is not None:
        return trainer.trajectory_store

    return TrajectoryStore(storage_dir="data/trajectories")

# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 5 â€” JEPA World Model Training                              â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
    logger.info(
        "GPU check: torch=%s cuda_available=%s device_count=%d cuda_version=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        torch.version.cuda,
    )
    if torch.cuda.is_available():
        logger.info("GPU 0: %s", torch.cuda.get_device_name(0))
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
    logger.info("Building WorldModel (encoder + JEPA predictor)...")
    world_model = WorldModel(wm_cfg)
    logger.info("WorldModel built: %d params",
                sum(p.numel() for p in world_model.parameters()))

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
            logger.warning("Could not build KG encoder: %s â€” training without KG", e)

    train_cfg = JEPATrainingConfig(
        num_epochs=num_epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_interval=10,
    )
    logger.info("JEPA training: device=%s epochs=%d batch=%d",
                train_cfg.device, train_cfg.num_epochs, train_cfg.batch_size)

    world_model = train_jepa(world_model, store, train_cfg, kg_encoder=kg_encoder)
    logger.info("JEPA training complete")
    return world_model


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 6 â€” Full Dream Training (V â†’ JEPA â†’ M â†’ C)                â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def stage_6_dream_training(
    store: "TrajectoryStore",
    world_model: "WorldModel | None" = None,
    num_iterations: int = 3,
) -> "WorldModel":
    """Run the full V â†’ JEPA â†’ M â†’ C dream training pipeline.

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


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Stage 7 â€” Evaluation Game                                        â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def stage_7_eval_game(world_model: "WorldModel | None" = None) -> None:
    """Play a demo game with the trained agent (or random agents if no model)."""
    logger.info("=== Stage 7: Evaluation Game ===")

    try:
        from src.orchestrator.game_runner import GameRunner
        from src.agents.random_agent import RandomAgent

        agents = {}
        if world_model is not None:
            try:
                from src.agents.world_model_agent import WorldModelAgent
                from src.world_model.card_embeddings import CardEmbeddingModel
                from src.world_model.game_tokenizer import GameTokenizer

                tokenizer = GameTokenizer(card_embeddings=CardEmbeddingModel()._embeddings)

                kg_encoder = None
                try:
                    from src.world_model.kg_encoder import KGContextEncoder, KGContextEncoderConfig

                    kg_dim = world_model.encoder.config.kg_embed_dim
                    if kg_dim and kg_dim > 0:
                        kg_cfg = KGContextEncoderConfig(kg_embed_dim=kg_dim)
                        kg_encoder = KGContextEncoder(CardEmbeddingModel(), kg_cfg)
                    else:
                        logger.info("Skipping KG encoder for evaluation: model trained with --no-kg")
                except Exception as e:
                    logger.warning("KG encoder unavailable for evaluation: %s", e)

                agents["player_0"] = WorldModelAgent(
                    "player_0",
                    world_model,
                    tokenizer,
                    kg_encoder=kg_encoder,
                    mode="direct",
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


async def stage_4_1_phase_rs_traces(
    num_games: int = 64,
    picker_name: str = "heuristic",
    ai_difficulty: str = "Medium",
) -> "TrajectoryStore":
    """Collect training traces from phase-rs engine + native MTGAgent picker.
    
    Phase-rs-first approach: Rust engine is authoritative runtime,
    collect structured JSONL traces for offline policy learning.
    """
    logger.info(
        "=== Stage 4.1: Phase-RS Trace Collection ===\n"
        "  games=%d, picker=%s, ai_difficulty=%s",
        num_games, picker_name, ai_difficulty,
    )

    import json
    import subprocess
    from pathlib import Path

    from src.world_model.trajectory import TrajectoryStore

    # Call collect_phase_rs_traces.py
    traces_dir = Path("runs/train_pipeline_traces") / f"{picker_name}_{ai_difficulty}"
    traces_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/collect_phase_rs_traces.py",
        "--games", str(num_games),
        "--picker", f"agent:{picker_name}" if not picker_name.startswith("agent:") else picker_name,
        "--ai-difficulty", ai_difficulty,
        "--autostart",
        "--output-dir", str(traces_dir),
    ]
    logger.info("Running phase-rs collector: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("Phase-RS collector exited with code %d:\n%s",
                          result.returncode, result.stderr)
        else:
            logger.info("Phase-RS trace collection succeeded")
    except subprocess.TimeoutExpired:
        logger.error("Phase-RS collection timed out after 600s")
        return TrajectoryStore(storage_dir="data/trajectories")
    except Exception as e:
        logger.error("Phase-RS collection failed: %s", e)
        return TrajectoryStore(storage_dir="data/trajectories")

    # Post-process: convert JSONL traces into TrajectoryStore format
    # TODO: Map decision events into (s,a,r,s') tuples for JEPA training
    logger.info("Post-processing phase-rs traces into TrajectoryStore format...")

    store = TrajectoryStore(storage_dir="data/trajectories")
    trace_files = list(traces_dir.glob("traces/*.jsonl"))
    logger.info("Found %d trace files", len(trace_files))

    for trace_file in trace_files[:min(num_games, 999)]:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            # TODO: Reconstruct trajectories from events
            logger.debug("Parsed %d events from %s", len(events), trace_file.name)
        except Exception as e:
            logger.warning("Could not parse %s: %s", trace_file, e)

    logger.info("Phase-RS trace collection complete")
    return store



# â•‘  Main Orchestrator                                                â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the training pipeline from the specified stage."""
    start = args.stage
    end = getattr(args, "end_stage", 7)
    t0 = time.time()

    def _in_range(n: int) -> bool:
        return start <= n <= end

    # --- Stage 1 ---
    if _in_range(1):
        infra = await stage_1_check_infra()
        if not infra.get("pytorch"):
            logger.error("PyTorch is required. Install with: pip install -r requirements-ml.txt")
            return

    # --- Stage 2 ---
    if _in_range(2) and not args.no_kg:
        try:
            await stage_2_build_kg()
        except Exception as e:
            logger.warning("KG setup failed: %s â€” continuing without KG", e)

    # --- Stage 3 ---
    if _in_range(3) and not args.no_kg:
        try:
            await stage_3_train_graph_embeddings(embed_dim=args.kg_embed_dim)
        except Exception as e:
            logger.warning("Graph embedding training failed: %s â€” continuing", e)

    # --- Stage 4 ---
    if _in_range(4):
        # Phase-RS traces (stage 4.1 NEW)
        if args.phase_rs_traces:
            logger.info("Running phase-rs-first trace collection (stage 4.1)...")
            store = await stage_4_1_phase_rs_traces(
                num_games=args.num_games,
                picker_name=args.phase_rs_picker,
                ai_difficulty=args.phase_rs_difficulty,
            )
        elif args.iterative_training:
            store = await stage_4_iterative_self_play(
                num_iterations=args.iterative_iters,
                games_per_iteration=args.iterative_games_per_iter,
                eval_games=args.iterative_eval_games,
                max_turns=args.max_turns,
                checkpoint_dir=args.checkpoint_dir,
            )
        else:
            store = await stage_4_collect_trajectories(
                num_games=args.num_games, max_turns=args.max_turns,
            )

        # Stage 4.5: KG enrichment from collected trajectories
        if not args.no_kg:
            try:
                from src.knowledge.kg_enrichment import KGEnrichment
                kg = None
                try:
                    from src.knowledge.knowledge_graph import MTGKnowledgeGraph
                    kg = MTGKnowledgeGraph()
                except Exception:
                    pass
                enrichment = KGEnrichment(kg=kg)
                report = await enrichment.enrich_from_trajectories(store)
                logger.info(
                    "KG enrichment: %d synergies, %d combos discovered",
                    report.synergies_proposed,
                    report.combos_proposed,
                )
            except Exception as e:
                logger.warning("KG enrichment skipped: %s", e)
    else:
        # Only need to load trajectories from disk if a later stage will use them
        if end >= 5:
            from src.world_model.trajectory import TrajectoryStore
            store = TrajectoryStore(storage_dir="data/trajectories")
            try:
                store.load()
            except Exception:
                pass
            if len(store) == 0:
                logger.error("No trajectories found. Run stage 4 first.")
                return
        else:
            store = None  # type: ignore[assignment]

    # --- Stage 5 ---
    world_model = None
    if _in_range(5):
        if args.wm_engine == "stable":
            from scripts.train_stable_worldmodel import main as stable_train_main
            import sys

            sys.argv = [
                sys.argv[0],
                "--trajectories",
                str(store.storage_dir),
                "--epochs",
                str(args.jepa_epochs),
                "--batch-size",
                str(args.jepa_batch_size),
                "--device",
                "cuda" if args.cuda else "cpu",
                "--jepa-beta",
                str(args.jepa_beta),
                "--kg-embed-dim",
                str(args.kg_embed_dim),
            ]
            if args.no_kg:
                sys.argv.append("--no-kg")
            stable_train_main()
            try:
                from src.world_model.world_model import WorldModel

                world_model = WorldModel.load("checkpoints/stable_worldmodel.pt")
            except Exception as e:
                logger.warning("Could not load stable JEPA artifact: %s", e)
                world_model = None

        elif args.wm_engine == "schmidhuber":
            from src.world_model.schmidhuber_worldmodel_adapter import SchmidhuberWorldModelAdapter
            from src.world_model.training.train_schmidhuber import SchmidhuberTrainingConfig, train_schmidhuber

            schm_cfg = SchmidhuberTrainingConfig(
                epochs=args.jepa_epochs,
                batch_size=args.jepa_batch_size,
                device="cuda" if args.cuda else "cpu",
            )
            world_model = train_schmidhuber(store, schm_cfg)

        else:
            world_model = stage_5_train_jepa(
                store,
                use_kg=not args.no_kg,
                kg_embed_dim=args.kg_embed_dim,
                jepa_beta=args.jepa_beta,
                num_epochs=args.jepa_epochs,
            )

    # --- Stage 6 ---
    if _in_range(6) and not args.skip_dream:
        world_model = stage_6_dream_training(
            store,
            world_model=world_model,
            num_iterations=args.dream_iters,
        )

    # --- Stage 6.5: Surprise detection + KG gap analysis ---
    if world_model is not None and not args.no_kg:
        try:
            from src.world_model.surprise_detector import SurpriseDetector
            kg = None
            try:
                from src.knowledge.knowledge_graph import MTGKnowledgeGraph
                kg = MTGKnowledgeGraph()
            except Exception:
                pass

            detector = SurpriseDetector(world_model=world_model, kg=kg)
            total_surprises = 0
            for traj in store.sample_batch(min(20, len(store))):
                surprises = await detector.analyze_trajectory(traj)
                total_surprises += len(surprises)

            summary = detector.get_surprise_summary()
            logger.info(
                "Surprise detection: %d surprises across sampled trajectories. "
                "Most surprising cards: %s",
                summary.get("total_surprises", 0),
                summary.get("most_surprising_cards", [])[:5],
            )
        except Exception as e:
            logger.warning("Surprise detection skipped: %s", e)

    # --- Stage 7 ---
    if _in_range(7) and not args.skip_eval:
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
    parser.add_argument("--end-stage", type=int, default=7,
                        help="Stop after this stage (inclusive)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run the evaluation game")

    # KG options
    parser.add_argument("--no-kg", action="store_true",
                        help="Disable knowledge graph (single-input JEPA)")
    parser.add_argument("--kg-embed-dim", type=int, default=128,
                        help="KG embedding dimension")
    parser.add_argument("--wm-engine", type=str, choices=["built_in", "stable", "schmidhuber"],
                        default="stable",
                        help="World model training engine to use")

    # Trajectory collection
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of self-play games to collect")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max turns per game")
    parser.add_argument("--iterative-training", action="store_true",
                        help="Use the iterative self-play promotion loop during stage 4")
    parser.add_argument("--iterative-iters", type=int, default=8,
                        help="Number of iterations for iterative self-play")
    parser.add_argument("--iterative-games-per-iter", type=int, default=20,
                        help="Games per iteration for iterative self-play")
    parser.add_argument("--iterative-eval-games", type=int, default=10,
                        help="Evaluation games per iteration for iterative self-play")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/rl",
                        help="Checkpoint directory for RLTrainer during iterative training")

    # Phase-RS trace collection (NEW: stage 4.1)
    parser.add_argument("--phase-rs-traces", action="store_true",
                        help="Enable phase-rs-first trace collection (stage 4.1) before JEPA training")
    parser.add_argument("--phase-rs-picker", type=str, default="heuristic",
                        help="Picker for phase-rs runs: random | heuristic | agent:<name>")
    parser.add_argument("--phase-rs-difficulty", type=str, default="Medium",
                        choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"],
                        help="AI difficulty for phase-rs training games")

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
                        help="Full Vâ†’Mâ†’C dream training iterations")
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


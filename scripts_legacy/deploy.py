#!/usr/bin/env python
"""
Deploy script — full pipeline orchestrator.

Stages:
  1. Verify infrastructure (Neo4j, Ollama, Python deps)
  2. Build Knowledge Graph (Scryfall → Neo4j, Combos, Ontology, Embeddings)
  3. Run RL self-play training (agents play & improve)
  4. Optional: Run dream training (world model)
  5. Launch a demo game

Usage:
  python scripts/deploy.py --all              # Full pipeline
  python scripts/deploy.py --check            # Infrastructure check only
  python scripts/deploy.py --kg               # Build KG only
  python scripts/deploy.py --train            # RL training only
  python scripts/deploy.py --transfer         # Standard→Commander transfer learning
  python scripts/deploy.py --play             # Play a demo game
  python scripts/deploy.py --commander        # 4-player Commander game
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("deploy")


# ── Stage 1: Infrastructure Check ─────────────────────────────────────

async def check_infrastructure() -> dict[str, bool]:
    """Verify all external dependencies are available."""
    status: dict[str, bool] = {}

    # Python version
    import platform
    py = platform.python_version()
    status["python_3.11+"] = sys.version_info >= (3, 11)
    logger.info("Python: %s — %s", py, "OK" if status["python_3.11+"] else "NEED 3.11+")

    # PyTorch
    try:
        import torch
        status["pytorch"] = True
        logger.info("PyTorch: %s — OK", torch.__version__)
    except ImportError:
        status["pytorch"] = False
        logger.warning("PyTorch: NOT INSTALLED — pip install torch")

    # Neo4j
    try:
        from neo4j import AsyncGraphDatabase
        from src.config import settings
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            record = await result.single()
            status["neo4j"] = record is not None and record["n"] == 1
        await driver.close()
        logger.info("Neo4j: Connected at %s — OK", settings.neo4j_uri)
    except Exception as e:
        status["neo4j"] = False
        logger.warning("Neo4j: NOT AVAILABLE — %s", e)
        logger.info("  Run: docker compose up -d")

    # Ollama
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        status["ollama"] = resp.status_code == 200
        logger.info("Ollama: %d models available — OK", len(models))
    except Exception:
        status["ollama"] = False
        logger.info("Ollama: not running (optional — LLM agents will use RandomAgent fallback)")

    # n10s plugin
    if status["neo4j"]:
        try:
            from src.config import settings
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
            )
            async with driver.session() as session:
                result = await session.run("RETURN n10s.version() AS v")
                record = await result.single()
                status["n10s"] = record is not None
                logger.info("n10s: %s — OK", record["v"] if record else "unknown")
            await driver.close()
        except Exception:
            status["n10s"] = False
            logger.warning("n10s plugin: NOT AVAILABLE — needed for ontology import")

    return status


# ── Stage 2: Build Knowledge Graph ───────────────────────────────────

async def build_knowledge_graph():
    """Full KG construction pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Building Knowledge Graph")
    logger.info("=" * 60)

    from src.knowledge.n10s_setup import N10sSetup

    # 2a. n10s setup + ontology import
    logger.info("--- 2a: Ontology setup ---")
    setup = N10sSetup()
    try:
        await setup.full_setup()
        logger.info("Ontology + indexes + SHACL shapes loaded")
    except Exception as e:
        logger.warning("Ontology setup issue: %s (continuing)", e)
    finally:
        await setup.close()

    # 2b. Scryfall card import
    logger.info("--- 2b: Scryfall card import ---")
    try:
        from scripts.import_scryfall import main as import_scryfall_main
        await import_scryfall_main()
    except Exception as e:
        logger.warning("Scryfall import issue: %s", e)

    # 2c. Commander Spellbook combo import
    logger.info("--- 2c: Combo import ---")
    try:
        from scripts.import_combos import main as import_combos_main
        await import_combos_main()
    except Exception as e:
        logger.warning("Combo import issue: %s", e)

    # 2d. Graph embeddings
    logger.info("--- 2d: Graph embeddings ---")
    try:
        from scripts.build_embeddings import main as build_embeddings_main
        os.makedirs("data/card_embeddings", exist_ok=True)
        await build_embeddings_main()
    except Exception as e:
        logger.warning("Embedding build issue: %s", e)

    # 2e. Validate
    logger.info("--- 2e: SHACL validation ---")
    try:
        from scripts.validate_kg import main as validate_main
        await validate_main()
    except Exception as e:
        logger.warning("Validation issue: %s", e)

    logger.info("KG build complete!")


# ── Stage 3: RL Self-Play Training ──────────────────────────────────

async def run_rl_training(
    num_iterations: int = 50,
    games_per_iter: int = 10,
    game_format: str = "standard",
):
    """Run the RL self-play training loop."""
    logger.info("=" * 60)
    logger.info("STAGE 3: RL Self-Play Training")
    logger.info("=" * 60)

    from src.training.rl_trainer import RLTrainer, RLConfig

    config = RLConfig(
        game_format=game_format,
        starting_life=40 if game_format == "commander" else 20,
        num_players=4 if game_format == "commander" else 2,
        num_iterations=num_iterations,
        games_per_iteration=games_per_iter,
        eval_games_per_iteration=max(games_per_iter // 2, 2),
        collect_trajectories=True,
        dream_training_interval=10,
    )

    trainer = RLTrainer(config)
    await trainer.train()
    logger.info("RL training complete!")


# ── Stage 4: Dream Training ─────────────────────────────────────────

def run_dream_training():
    """Run world model dream training on collected trajectories."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Dream Training (World Model)")
    logger.info("=" * 60)

    from src.world_model.training.dream_trainer import DreamTrainer, DreamTrainerConfig
    from src.world_model.trajectory import TrajectoryStore

    store = TrajectoryStore("checkpoints/rl/trajectories")
    if len(store) < 10:
        logger.warning("Only %d trajectories — need at least 10 for dream training", len(store))
        return

    config = DreamTrainerConfig(num_iterations=3, min_trajectories=5)
    trainer = DreamTrainer(config)
    trainer.train(store)
    logger.info("Dream training complete!")


# ── Stage 5: Demo Game ──────────────────────────────────────────────

async def play_demo_game(
    game_format: str = "standard",
    agent1_type: str = "random",
    agent2_type: str = "random",
    num_players: int = 2,
):
    """Run a demonstration game."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Demo Game (%s, %d players)", game_format, num_players)
    logger.info("=" * 60)

    from src.agents.random_agent import RandomAgent
    from src.orchestrator.game_runner import GameRunner, GameConfig
    from src.world_model.data_sources.self_play_collector import SelfPlayCollector

    def make_agent(agent_type: str, player_id: str):
        if agent_type == "ollama":
            try:
                from src.agents.llm_agent import OllamaAgent
                return OllamaAgent(player_id=player_id, name=f"Ollama_{player_id}")
            except Exception:
                pass
        if agent_type == "fusion":
            try:
                from src.agents.llm_fusion_agent import LLMFusionAgent
                return LLMFusionAgent(player_id=player_id, name=f"Fusion_{player_id}")
            except Exception:
                pass
        return RandomAgent(player_id=player_id, name=f"Random_{player_id}")

    # Build player list
    if num_players == 4:
        player_ids = ["Alice", "Bob", "Charlie", "Diana"]
    else:
        player_ids = ["Alice", "Bob"]

    agents = {}
    for i, pid in enumerate(player_ids):
        atype = agent1_type if i == 0 else agent2_type
        agents[pid] = make_agent(atype, pid)

    # Build decks
    from main import build_simple_deck
    if game_format == "commander":
        # Commander needs 100-card deck with command zone
        base = build_simple_deck()
        deck = base * 3  # Triplicate for 100-card
        deck = deck[:100]
    else:
        deck = build_simple_deck()

    decks = {pid: deck.copy() for pid in player_ids}

    starting_life = 40 if game_format == "commander" else 20
    config = GameConfig(format=game_format, starting_life=starting_life, max_turns=30)
    collector = SelfPlayCollector()
    runner = GameRunner(config, self_play_collector=collector)

    result = await runner.run_game(agents, decks)

    logger.info("=" * 60)
    logger.info("GAME OVER")
    logger.info("  Winner: %s", result.winner)
    logger.info("  Turns:  %d", result.turns)
    logger.info("  Events: %d", len(result.log))
    logger.info("=" * 60)

    # Print first 20 events
    for event in result.log[:20]:
        logger.info("  %s", event)
    if len(result.log) > 20:
        logger.info("  ... (%d more events)", len(result.log) - 20)

    return result


# ── Main ─────────────────────────────────────────────────────────────

async def run_all():
    """Run the complete pipeline."""
    # Stage 1: Check
    status = await check_infrastructure()
    if not status.get("python_3.11+"):
        logger.error("Python 3.11+ required")
        return

    # Stage 2: KG (requires Neo4j)
    if status.get("neo4j"):
        await build_knowledge_graph()
    else:
        logger.warning("Skipping KG build — Neo4j not available")

    # Stage 3: RL Training
    await run_rl_training(num_iterations=20, games_per_iter=5)

    # Stage 4: Dream Training
    try:
        run_dream_training()
    except Exception as e:
        logger.warning("Dream training skipped: %s", e)

    # Stage 5: Demo
    await play_demo_game()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy pipeline for MTG AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deploy.py --all              Full pipeline
  python scripts/deploy.py --check            Check infrastructure
  python scripts/deploy.py --kg               Build Knowledge Graph
  python scripts/deploy.py --train --iters 50 RL training (50 iterations)
  python scripts/deploy.py --play             Demo: random vs random
  python scripts/deploy.py --play --ollama    Demo: Ollama vs random
  python scripts/deploy.py --commander        4-player Commander game
        """,
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    parser.add_argument("--kg", action="store_true", help="Build Knowledge Graph")
    parser.add_argument("--train", action="store_true", help="RL self-play training")
    parser.add_argument("--dream", action="store_true", help="Dream training (world model)")
    parser.add_argument("--play", action="store_true", help="Play demo game")
    parser.add_argument("--commander", action="store_true", help="4-player Commander mode")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama agent")
    parser.add_argument("--fusion", action="store_true", help="Use LLM-Fusion agent")
    parser.add_argument("--iters", type=int, default=20, help="Training iterations")
    parser.add_argument("--games", type=int, default=5, help="Games per iteration")
    parser.add_argument("--transfer", action="store_true", help="Standard→Commander transfer learning")

    args = parser.parse_args()

    game_format = "commander" if args.commander else "standard"
    num_players = 4 if args.commander else 2
    a1_type = "fusion" if args.fusion else ("ollama" if args.ollama else "random")

    if args.all:
        asyncio.run(run_all())
    elif args.check:
        asyncio.run(check_infrastructure())
    elif args.kg:
        asyncio.run(build_knowledge_graph())
    elif args.train:
        asyncio.run(run_rl_training(args.iters, args.games, game_format))
    elif args.transfer:
        from src.training.transfer_learning import TransferTrainer, TransferConfig
        tc = TransferConfig(
            standard_iterations=max(args.iters // 2, 10),
            commander_iterations=max(args.iters // 3, 10),
            joint_iterations=max(args.iters // 5, 5),
        )
        asyncio.run(TransferTrainer(tc).run())
    elif args.dream:
        run_dream_training()
    elif args.play or args.commander:
        asyncio.run(play_demo_game(game_format, a1_type, "random", num_players))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

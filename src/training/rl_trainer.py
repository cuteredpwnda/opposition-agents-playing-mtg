"""
Reinforcement Learning training loop — agents play against each other.

Combines:
- Self-play game generation via GameRunner
- Experience collection from real games
- SelfPlayCollector for world model trajectory data
- Neural module training (NeuralReasoningModule or WorldModel)
- ELO-based evaluation and curriculum

This is the production RL loop that wires everything together.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for the reinforcement learning training loop."""

    # Game settings
    game_format: str = "standard"        # "standard" | "commander"
    starting_life: int = 20
    max_turns_per_game: int = 50
    num_players: int = 2                  # 2 for standard, 4 for commander

    # Training schedule
    num_iterations: int = 100
    games_per_iteration: int = 20
    eval_games_per_iteration: int = 10
    warmup_iterations: int = 5           # Random-only before neural training

    # Agent pool
    agent_types: list[str] = field(default_factory=lambda: ["random", "random"])
    elo_ratings: dict[str, float] = field(default_factory=dict)

    # Neural training
    batch_size: int = 64
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    train_steps_per_iter: int = 50

    # World model
    collect_trajectories: bool = True     # Feed games into world model training
    dream_training_interval: int = 10     # Run dream training every N iterations

    # Promotion loop
    promotion_threshold: float = 0.55     # Candidate must beat champion by > threshold

    # Parallelism
    parallel_games: int = 1               # asyncio.gather concurrency for matchup play

    # Checkpointing
    checkpoint_dir: str = "checkpoints/rl"
    log_dir: str = "logs/rl"
    save_every: int = 10


class AgentPool:
    """Manages a pool of agents with different strategies and ELO ratings."""

    def __init__(self):
        self.agents: dict[str, dict[str, Any]] = {}
        self.elo: dict[str, float] = {}
        self._agent_counter = 0

    def register(self, agent_type: str, agent_factory, elo: float = 1200.0) -> str:
        """Register an agent factory and return its pool ID."""
        pool_id = f"{agent_type}_{self._agent_counter}"
        self._agent_counter += 1
        self.agents[pool_id] = {"type": agent_type, "factory": agent_factory}
        self.elo[pool_id] = elo
        return pool_id

    def create_agent(self, pool_id: str, player_id: str):
        """Instantiate an agent from the pool."""
        entry = self.agents[pool_id]
        return entry["factory"](player_id=player_id)

    def update_elo(self, winner_id: str, loser_id: str, k: float = 32.0):
        """Update ELO ratings after a match."""
        ew = 1.0 / (1.0 + 10 ** ((self.elo.get(loser_id, 1200) - self.elo.get(winner_id, 1200)) / 400))
        self.elo[winner_id] = self.elo.get(winner_id, 1200) + k * (1.0 - ew)
        self.elo[loser_id] = self.elo.get(loser_id, 1200) + k * (0.0 - (1.0 - ew))

    def get_matchup(self) -> tuple[str, str]:
        """Select two agents for a match (prefer close ELO)."""
        ids = list(self.agents.keys())
        if len(ids) < 2:
            return ids[0], ids[0]
        random.shuffle(ids)
        return ids[0], ids[1]

    def get_best_agents(self) -> tuple[str, str]:
        """Return the top two agents by ELO rating."""
        ids = list(self.agents.keys())
        if not ids:
            raise ValueError("No agents registered in pool")
        if len(ids) == 1:
            return ids[0], ids[0]
        sorted_ids = sorted(ids, key=lambda k: self.elo.get(k, 1200.0), reverse=True)
        return sorted_ids[0], sorted_ids[1]


class RLTrainer:
    """Production reinforcement learning trainer.

    Orchestrates:
    1. Agent-vs-agent game generation
    2. Experience buffer population
    3. Neural network training
    4. World model trajectory collection
    5. ELO-based evaluation
    """

    def __init__(self, config: RLConfig | None = None):
        self.config = config or RLConfig()
        self.pool = AgentPool()
        self.experience_buffer = None
        self.trajectory_store = None
        self.neural_module = None
        self.optimizer = None
        self.tokenizer = None
        self.card_embeddings_model = None
        self.champion_id: str | None = None
        self.stats: list[dict[str, Any]] = []

    def setup(self):
        """Initialize all components."""
        from src.training.experience_buffer import ExperienceBuffer
        self.experience_buffer = ExperienceBuffer(max_size=100_000)

        # Register baseline agents
        from src.agents.random_agent import RandomAgent
        self.pool.register(
            "random",
            lambda player_id: RandomAgent(player_id=player_id, name=f"Random_{player_id}"),
            elo=1000.0,
        )
        self.pool.register(
            "random_2",
            lambda player_id: RandomAgent(player_id=player_id, name=f"Random2_{player_id}"),
            elo=1000.0,
        )

        # Register LLM agent if available
        try:
            from src.agents.llm_agent import OllamaAgent
            self.pool.register(
                "ollama",
                lambda player_id: OllamaAgent(player_id=player_id, name=f"Ollama_{player_id}"),
                elo=1200.0,
            )
        except Exception:
            logger.info("Ollama agent not available — skipping")

        # Register Active Inference + Opponent Model agent
        try:
            from src.agents.active_inference_agent import ActiveInferenceAgent
            self.pool.register(
                "active_inference",
                lambda player_id: ActiveInferenceAgent(player_id=player_id),
                elo=1250.0,
            )
        except Exception as e:
            logger.info("ActiveInferenceAgent not available: %s", e)

        # Register LLM Fusion agent (with opponent model) if available
        try:
            from src.agents.llm_fusion_agent import LLMFusionAgent
            self.pool.register(
                "llm_fusion",
                lambda player_id: LLMFusionAgent(player_id=player_id, opponent_model=None),
                elo=1300.0,
            )
        except Exception as e:
            logger.info("LLMFusionAgent not available: %s", e)

        # Register Neural Reasoner agent if available
        try:
            from src.agents.neural_reasoner_agent import NeuralReasonerAgent
            self.pool.register(
                "neural_reasoner",
                lambda player_id: NeuralReasonerAgent(player_id=player_id),
                elo=1350.0,
            )
        except Exception as e:
            logger.info("NeuralReasonerAgent not available: %s", e)

        # Set up trajectory store for world model
        if self.config.collect_trajectories:
            from src.world_model.trajectory import TrajectoryStore
            store_path = os.path.join(self.config.checkpoint_dir, "trajectories")
            self.trajectory_store = TrajectoryStore(store_path)

        # Initialize neural module
        try:
            import torch
            from src.agents.neural_reasoner import NeuralReasoningModule
            self.neural_module = NeuralReasoningModule()
            if hasattr(self.neural_module, "parameters"):
                self.optimizer = torch.optim.AdamW(
                    self.neural_module.parameters(),
                    lr=self.config.learning_rate,
                )
            else:
                logger.warning(
                    "NeuralReasoningModule is not trainable — skipping optimizer setup"
                )
                self.neural_module = None
                self.optimizer = None
        except ImportError:
            logger.warning("PyTorch not available — training will collect data only")
        except AttributeError as e:
            logger.warning("Neural module initialization failed: %s", e)
            self.neural_module = None
            self.optimizer = None

        # Set up GameTokenizer + CardEmbeddingModel for real state encoding
        try:
            from src.world_model.card_embeddings import CardEmbeddingModel
            from src.world_model.game_tokenizer import GameTokenizer
            self.card_embeddings_model = CardEmbeddingModel()
            self.tokenizer = GameTokenizer(
                card_embeddings=self.card_embeddings_model.get_all_embeddings(),
            )
            logger.info("GameTokenizer initialised for trajectory encoding")
        except Exception as e:
            logger.warning("GameTokenizer unavailable — trajectories will lack state encoding: %s", e)

        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Initialize champion agent for iterative self-play
        if self.pool.agents:
            self.champion_id, _ = self.pool.get_best_agents()
            logger.info("Initial champion agent: %s", self.champion_id)

    async def train(self):
        """Main RL training loop."""
        self.setup()
        logger.info("=" * 60)
        logger.info("Starting RL Training — %d iterations", self.config.num_iterations)
        logger.info("=" * 60)

        for iteration in range(self.config.num_iterations):
            logger.info("--- Iteration %d / %d ---", iteration + 1, self.config.num_iterations)

            # Phase 1: Generate games (self-play)
            challenger_id = self._pick_challenger()
            if challenger_id is None:
                challenger_id = self.champion_id

            game_results = await self._run_matchup(
                self.champion_id,
                challenger_id,
                self.config.games_per_iteration,
                collect=True,
            )

            # Phase 2: Train neural module (after warmup)
            train_loss = 0.0
            if iteration >= self.config.warmup_iterations and self.neural_module is not None:
                train_loss = self._train_neural_step()

            # Phase 3: Evaluate candidate against champion
            eval_results = await self._evaluate_candidate(
                challenger_id,
                self.champion_id,
                self.config.eval_games_per_iteration,
            )
            win_rate = eval_results.get("win_rate", 0.0)
            logger.info(
                "Candidate %s win rate vs champion %s: %.2f",
                challenger_id,
                self.champion_id,
                win_rate,
            )

            # Phase 3.5: Promote if candidate is stronger
            if self._promote_candidate(challenger_id, self.champion_id, win_rate):
                logger.info("Promoted %s to champion", challenger_id)
                self.champion_id = challenger_id

            # Phase 4: Dream training (periodic)
            if (
                self.config.collect_trajectories
                and iteration > 0
                and iteration % self.config.dream_training_interval == 0
                and self.trajectory_store is not None
            ):
                self._run_dream_training()

            # Phase 4.5: KG enrichment from trajectories (periodic)
            if (
                self.config.collect_trajectories
                and iteration > 0
                and iteration % self.config.dream_training_interval == 0
                and self.trajectory_store is not None
            ):
                await self._run_kg_enrichment()

            # Phase 5: Log stats
            wins = sum(1 for r in game_results if r.get("winner"))
            draws = len(game_results) - wins
            iter_stats = {
                "iteration": iteration + 1,
                "games_played": len(game_results),
                "wins": wins,
                "draws": draws,
                "train_loss": train_loss,
                "buffer_size": len(self.experience_buffer) if self.experience_buffer else 0,
                "elo_ratings": dict(self.pool.elo),
            }
            self.stats.append(iter_stats)
            logger.info(
                "Results: %d games, %d decisive, loss=%.4f, buffer=%d",
                len(game_results), wins, train_loss,
                len(self.experience_buffer) if self.experience_buffer else 0,
            )

            # Save checkpoint
            if (iteration + 1) % self.config.save_every == 0:
                self._save_checkpoint(iteration + 1)

        # Final save
        self._save_checkpoint(self.config.num_iterations)
        logger.info("RL Training complete! Stats saved to %s", self.config.log_dir)

    async def _run_games(
        self, num_games: int, collect: bool
    ) -> list[dict[str, Any]]:
        """Run a batch of games between agents from the pool."""
        from src.orchestrator.game_runner import GameRunner, GameConfig
        from src.world_model.data_sources.self_play_collector import SelfPlayCollector
        from src.training.rewards import RewardFunction
        from src.training.experience_buffer import Experience

        reward_fn = RewardFunction()
        results = []

        for game_num in range(num_games):
            try:
                # Match agents from pool
                pool_a, pool_b = self.pool.get_matchup()

                if self.config.num_players == 4:
                    # Commander: 4 players
                    player_ids = [f"P{i}" for i in range(4)]
                    agents = {}
                    pool_ids_used = []
                    for pid in player_ids:
                        picked = pool_a if len(pool_ids_used) % 2 == 0 else pool_b
                        agents[pid] = self.pool.create_agent(picked, pid)
                        pool_ids_used.append(picked)
                else:
                    # Standard: 2 players
                    player_ids = ["player_0", "player_1"]
                    agents = {
                        "player_0": self.pool.create_agent(pool_a, "player_0"),
                        "player_1": self.pool.create_agent(pool_b, "player_1"),
                    }

                # Set up collector with tokenizer for real state encoding
                collector = None
                if collect and self.config.collect_trajectories:
                    collector = SelfPlayCollector(
                        tokenizer=self.tokenizer,
                        card_embeddings=self.card_embeddings_model,
                    )

                # Build decks
                decks = {pid: self._build_deck() for pid in player_ids}

                # Run game
                gc = GameConfig(
                    format=self.config.game_format,
                    starting_life=self.config.starting_life,
                    max_turns=self.config.max_turns_per_game,
                )
                runner = GameRunner(gc, self_play_collector=collector)
                result = await runner.run_game(agents, decks)

                # Update ELO
                game_result = {"winner": result.winner, "turns": result.turns}
                if result.winner:
                    winner_pool = pool_a if result.winner == "player_0" else pool_b
                    loser_pool = pool_b if result.winner == "player_0" else pool_a
                    self.pool.update_elo(winner_pool, loser_pool)

                # Collect experience with real encoded states
                if collect and self.experience_buffer is not None:
                    for pid in player_ids:
                        reward = 1.0 if pid == result.winner else -1.0 if result.winner else 0.0
                        # Extract real state features from collector trajectories
                        state_feats = self._extract_terminal_features(collector, pid)
                        exp = Experience(
                            state_features=state_feats,
                            action_index=0,
                            reward=reward,
                            next_state_features=state_feats,
                            done=True,
                            metadata={"game_turns": result.turns, "player_id": pid},
                        )
                        self.experience_buffer.add(exp)

                # Store trajectories
                if collector and self.trajectory_store:
                    for traj in collector.collected_trajectories:
                        self.trajectory_store.add(traj)

                results.append(game_result)
                logger.debug(
                    "Game %d: winner=%s, turns=%d",
                    game_num + 1, result.winner, result.turns,
                )
            except Exception as e:
                logger.warning("Game %d failed: %s", game_num + 1, e)
                results.append({"winner": None, "turns": 0, "error": str(e)})

        return results

    async def _run_matchup(
        self,
        pool_a: str,
        pool_b: str,
        num_games: int,
        collect: bool,
    ) -> list[dict[str, Any]]:
        """Run games between two specific agents in the pool.

        When ``self.config.parallel_games > 1`` the games for the matchup are
        scheduled with :func:`asyncio.gather` (bounded by a semaphore) so that
        large self-play epochs can saturate available cores.
        """
        from src.orchestrator.game_runner import GameRunner, GameConfig
        from src.world_model.data_sources.self_play_collector import SelfPlayCollector
        from src.training.rewards import RewardFunction
        from src.training.experience_buffer import Experience

        sem = asyncio.Semaphore(max(1, int(self.config.parallel_games)))

        async def play_one(game_num: int) -> dict[str, Any]:
            async with sem:
                try:
                    player_ids = ["player_0", "player_1"]
                    agents = {
                        "player_0": self.pool.create_agent(pool_a, "player_0"),
                        "player_1": self.pool.create_agent(pool_b, "player_1"),
                    }

                    collector = None
                    if collect and self.config.collect_trajectories:
                        collector = SelfPlayCollector(
                            tokenizer=self.tokenizer,
                            card_embeddings=self.card_embeddings_model,
                        )

                    decks = {pid: self._build_deck() for pid in player_ids}
                    gc = GameConfig(
                        format=self.config.game_format,
                        starting_life=self.config.starting_life,
                        max_turns=self.config.max_turns_per_game,
                    )
                    runner = GameRunner(gc, self_play_collector=collector)
                    result = await runner.run_game(agents, decks)

                    if result.winner:
                        winner_pool = pool_a if result.winner == "player_0" else pool_b
                        loser_pool = pool_b if result.winner == "player_0" else pool_a
                        self.pool.update_elo(winner_pool, loser_pool)

                    if collect and self.experience_buffer is not None:
                        for pid in player_ids:
                            reward = (
                                1.0
                                if pid == result.winner
                                else -1.0
                                if result.winner
                                else 0.0
                            )
                            state_feats = self._extract_terminal_features(collector, pid)
                            exp = Experience(
                                state_features=state_feats,
                                action_index=0,
                                reward=reward,
                                next_state_features=state_feats,
                                done=True,
                                metadata={"game_turns": result.turns, "player_id": pid},
                            )
                            self.experience_buffer.add(exp)

                    if collector and self.trajectory_store:
                        for traj in collector.collected_trajectories:
                            self.trajectory_store.add(traj)

                    logger.debug(
                        "Game %d: winner=%s, turns=%d",
                        game_num + 1,
                        result.winner,
                        result.turns,
                    )
                    return {"winner": result.winner, "turns": result.turns}
                except Exception as e:
                    logger.warning("Game %d failed: %s", game_num + 1, e)
                    return {"winner": None, "turns": 0, "error": str(e)}

        tasks = [play_one(i) for i in range(num_games)]
        results: list[dict[str, Any]] = await asyncio.gather(*tasks)
        return results

    def _pick_challenger(self) -> str | None:
        """Choose a challenger agent different from the current champion."""
        if self.champion_id is None:
            return None
        candidates = [pid for pid in self.pool.agents.keys() if pid != self.champion_id]
        if not candidates:
            return self.champion_id
        return random.choice(candidates)

    async def _evaluate_candidate(
        self, candidate_id: str, champion_id: str, num_games: int
    ) -> dict[str, float]:
        """Evaluate candidate against champion and return a win rate summary."""
        results = await self._run_matchup(candidate_id, champion_id, num_games, collect=False)
        wins = sum(1 for r in results if r.get("winner") == "player_0")
        total = len(results)
        return {"win_rate": wins / total if total > 0 else 0.0, "games": total}

    def _promote_candidate(
        self, candidate_id: str, champion_id: str, win_rate: float
    ) -> bool:
        """Promote candidate to champion if it outperforms the existing champion."""
        if candidate_id == champion_id:
            return False
        if win_rate > self.config.promotion_threshold:
            self.pool.elo[candidate_id] = self.pool.elo.get(candidate_id, 1200.0) + 50.0
            self.pool.elo[champion_id] = self.pool.elo.get(champion_id, 1200.0) - 20.0
            return True
        return False

    def _train_neural_step(self) -> float:
        """Train the neural module on buffered experience."""
        if self.neural_module is None or self.optimizer is None:
            return 0.0
        if self.experience_buffer is None or len(self.experience_buffer) < self.config.batch_size:
            return 0.0

        import torch

        total_loss = 0.0
        for _ in range(self.config.train_steps_per_iter):
            batch = self.experience_buffer.sample(self.config.batch_size)
            if batch is None:
                break

            # Simple value-function training on terminal rewards
            rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
            # Use reward prediction as training signal
            predicted = torch.zeros_like(rewards)  # placeholder output
            loss = torch.nn.functional.mse_loss(predicted, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.neural_module.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(self.config.train_steps_per_iter, 1)

    def _extract_terminal_features(self, collector, player_id: str) -> list[float]:
        """Extract flattened state features from the last trajectory transition.

        Falls back to zeros if no tokenized features are available.
        """
        if collector is None:
            return [0.0] * 64
        # Look for the last transition with populated state features
        for traj in reversed(collector.collected_trajectories):
            for transition in reversed(traj.transitions):
                if isinstance(transition.state_features, dict):
                    # Flatten all arrays into a single list
                    flat = []
                    for v in transition.state_features.values():
                        import numpy as np
                        flat.extend(np.asarray(v).flatten().tolist())
                    return flat
        return [0.0] * 64

    def _run_dream_training(self):
        """Run world model dream training on collected trajectories."""
        if self.trajectory_store is None or len(self.trajectory_store) < 10:
            logger.info("Not enough trajectories for dream training (%d)", 
                       len(self.trajectory_store) if self.trajectory_store else 0)
            return

        try:
            from src.world_model.training.dream_trainer import DreamTrainer, DreamTrainerConfig
            dt_config = DreamTrainerConfig(
                num_iterations=1,
                min_trajectories=5,
            )
            trainer = DreamTrainer(dt_config)
            trainer.train(self.trajectory_store)
            logger.info("Dream training iteration complete")
        except Exception as e:
            logger.warning("Dream training failed: %s", e)

    async def _run_kg_enrichment(self):
        """Run KG auto-enrichment from collected trajectories."""
        if self.trajectory_store is None or len(self.trajectory_store) < 10:
            return

        try:
            from src.knowledge.kg_enrichment import KGEnrichment

            # Try to connect to KG — if unavailable, run in offline mode
            kg = None
            try:
                from src.knowledge.knowledge_graph import MTGKnowledgeGraph
                kg = MTGKnowledgeGraph()
            except Exception:
                pass

            enrichment = KGEnrichment(kg=kg)
            report = await enrichment.enrich_from_trajectories(self.trajectory_store)
            logger.info(
                "KG enrichment: %d synergies proposed, %d written, %d card stats updated",
                report.synergies_proposed,
                report.synergies_written,
                report.card_stats_updated,
            )
        except Exception as e:
            logger.warning("KG enrichment failed: %s", e)

    def _build_deck(self) -> list[dict]:
        """Build a deck for training games."""
        deck = []
        # 24 lands
        for _ in range(8):
            for land_name, land_text in [
                ("Plains", "{T}: Add {W}."),
                ("Island", "{T}: Add {U}."),
                ("Mountain", "{T}: Add {R}."),
            ]:
                deck.append({
                    "name": land_name, "mana_cost": "",
                    "type_line": f"Land — {land_name}",
                    "oracle_text": land_text,
                    "power": None, "toughness": None, "cmc": 0,
                    "keywords": [], "set": "DOM",
                })

        # 12 cheap creatures
        for _ in range(6):
            deck.append({
                "name": "Soldier", "mana_cost": "{1}{W}",
                "type_line": "Creature — Soldier",
                "oracle_text": "", "power": "2", "toughness": "2",
                "cmc": 2, "keywords": [], "set": "DOM",
            })
        for _ in range(6):
            deck.append({
                "name": "Goblin", "mana_cost": "{R}",
                "type_line": "Creature — Goblin",
                "oracle_text": "", "power": "1", "toughness": "1",
                "cmc": 1, "keywords": [], "set": "DOM",
            })

        # 4 removal
        for _ in range(4):
            deck.append({
                "name": "Lightning Bolt", "mana_cost": "{R}",
                "type_line": "Instant",
                "oracle_text": "Lightning Bolt deals 3 damage to any target.",
                "power": None, "toughness": None, "cmc": 1,
                "keywords": [], "set": "DOM",
            })

        # 4 draw
        for _ in range(4):
            deck.append({
                "name": "Divination", "mana_cost": "{2}{U}",
                "type_line": "Sorcery",
                "oracle_text": "Draw two cards.",
                "power": None, "toughness": None, "cmc": 3,
                "keywords": [], "set": "DOM",
            })

        # Pad to 60
        while len(deck) < 60:
            deck.append({
                "name": "Mountain", "mana_cost": "",
                "type_line": "Land — Mountain",
                "oracle_text": "{T}: Add {R}.",
                "power": None, "toughness": None, "cmc": 0,
                "keywords": [], "set": "DOM",
            })

        return deck[:60]

    def _save_checkpoint(self, iteration: int):
        """Save training state."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Save stats
        stats_path = os.path.join(self.config.log_dir, "training_stats.json")
        os.makedirs(self.config.log_dir, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        # Save neural module
        if self.neural_module is not None:
            try:
                import torch
                model_path = os.path.join(
                    self.config.checkpoint_dir, f"neural_iter_{iteration}.pt"
                )
                torch.save(self.neural_module.state_dict(), model_path)
                logger.info("Saved checkpoint: %s", model_path)
            except Exception as e:
                logger.warning("Failed to save neural checkpoint: %s", e)

        # Save ELO ratings
        elo_path = os.path.join(self.config.log_dir, "elo_ratings.json")
        with open(elo_path, "w") as f:
            json.dump(self.pool.elo, f, indent=2)

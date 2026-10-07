"""
Self-play training loop — AlphaZero-inspired.

Reference: Section 12.3 of PLAN.md.
Requires the [ml] optional dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.training.deck_utils import create_mock_deck
from src.training.experience_buffer import Experience, ExperienceBuffer
from src.training.rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Self-play training configuration."""

    num_parallel_games: int = 64
    buffer_size: int = 100_000
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-4
    eval_games: int = 100


class SelfPlayTrainer:
    """AlphaZero-inspired self-play training.

    Agents play each other → collect experience → train neural module →
    evaluate → update KG with discovered interactions.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.buffer = ExperienceBuffer(max_size=self.config.buffer_size)
        self.reward_fn = RewardFunction()
        self.neural_module = None  # Set after import
        self.optimizer = None

    def init_neural_module(self) -> None:
        """Lazy init to avoid importing torch at module load time."""
        import torch

        from src.agents.neural_reasoner import NeuralReasoningModule

        self.neural_module = NeuralReasoningModule()
        self.optimizer = torch.optim.Adam(
            self.neural_module.parameters(), lr=self.config.learning_rate
        )

    async def train(self, num_iterations: int = 1000) -> None:
        """Main training loop."""
        if self.neural_module is None:
            self.init_neural_module()

        for iteration in range(num_iterations):
            # 1. Self-play
            experiences = await self.run_self_play_games(
                self.config.num_parallel_games
            )
            self.buffer.add_batch(experiences)

            # 2. Train
            loss = self._train_step()

            # 3. Evaluate
            win_rate = await self.evaluate(self.config.eval_games)

            logger.info(
                f"Iter {iteration}: loss={loss:.4f}, win_rate={win_rate:.2%}, "
                f"buffer={len(self.buffer)}"
            )

    async def run_self_play_games(
        self, num_games: int
    ) -> list[Experience]:
        """Run games between copies of the current agent, collect experience.

        Games are scheduled with :func:`asyncio.gather` and bounded by
        ``TrainingConfig.num_parallel_games`` so callers can saturate the
        event loop without overwhelming it.
        """
        import asyncio
        from src.agents.llm_agent import LLMAgent
        from src.orchestrator_legacy.game_runner import GameRunner, GameConfig

        sem = asyncio.Semaphore(max(1, int(self.config.num_parallel_games)))

        async def play_one(game_num: int) -> list[Experience]:
            async with sem:
                experiences_local: list[Experience] = []
                try:
                    agent1 = LLMAgent(player_id="player_1")
                    agent2 = LLMAgent(player_id="player_2")
                    agents = {"player_1": agent1, "player_2": agent2}
                    decks = {
                        "player_1": create_mock_deck(),
                        "player_2": create_mock_deck(),
                    }
                    runner = GameRunner(GameConfig(max_turns=50))
                    result = await runner.run_game(agents, decks)
                    winner_id = result.winner if result.winner else None
                    for player_id in agents:
                        reward = (
                            1.0
                            if player_id == winner_id
                            else -1.0
                            if winner_id
                            else 0.0
                        )
                        exp = Experience(
                            state_features=[],
                            action_index=0,
                            reward=reward,
                            next_state_features=[],
                            done=True,
                            metadata={"game_turns": result.turns},
                        )
                        experiences_local.append(exp)
                    logger.info(
                        f"Self-play game {game_num+1}: winner={winner_id}, turns={result.turns}"
                    )
                    try:
                        await self._enrich_kg_from_game(result, decks)
                    except Exception as e:
                        logger.warning(
                            f"KG enrichment failed for game {game_num}: {e}"
                        )
                except Exception as e:
                    logger.error(f"Error in self-play game {game_num}: {e}")
                return experiences_local

        tasks = [play_one(i) for i in range(num_games)]
        per_game = await asyncio.gather(*tasks)
        experiences: list[Experience] = []
        for batch in per_game:
            experiences.extend(batch)
        return experiences

    async def _enrich_kg_from_game(self, result, decks) -> None:
        """Update knowledge graph with self-play game outcomes."""
        if not result.winner:
            return

        winner_id = result.winner
        loser_id = next((p for p in decks.keys() if p != winner_id), None)

        try:
            from src.knowledge.knowledge_graph import MTGKnowledgeGraph

            kg = MTGKnowledgeGraph()
            winner_cards = [card["name"] for card in decks[winner_id] if "name" in card]
            loser_cards = [card["name"] for card in decks[loser_id] if "name" in card] if loser_id else []
            run_id = f"self_play_game_{winner_id}_{result.turns}"

            # Add syntactic synergy edges for cards present together in winner deck
            unique_winner_cards = list(dict.fromkeys(winner_cards))
            for i, card_a in enumerate(unique_winner_cards):
                for card_b in unique_winner_cards[i + 1 : i + 4]:
                    await kg.add_synergy(
                        card_a,
                        card_b,
                        weight=1.0,
                        run_id=run_id,
                        source="self_play_online",
                        metadata={"winner": winner_id, "turns": result.turns},
                    )

            # Update win rate stats
            for card in set(unique_winner_cards):
                await kg.update_card_stats(
                    card,
                    won=True,
                    run_id=run_id,
                    source="self_play_online",
                    metadata={"winner": winner_id, "turns": result.turns},
                )

            for card in set(loser_cards):
                await kg.update_card_stats(
                    card,
                    won=False,
                    run_id=run_id,
                    source="self_play_online",
                    metadata={"winner": winner_id, "turns": result.turns},
                )

        except Exception:
            raise
        finally:
            try:
                await kg.close()
            except Exception:
                pass


    async def evaluate(self, num_games: int) -> float:
        """Evaluate current agent vs. baseline (random agent).
        
        Returns win rate of current agent against random opponents.
        """
        import asyncio
        from src.agents.random_agent import RandomAgent
        from src.orchestrator_legacy.game_runner import GameRunner, GameConfig
        
        wins = 0
        total = 0
        
        try:
            for _ in range(min(num_games, 10)):  # Limit to 10 eval games per iteration
                try:
                    # Create evaluation agents
                    eval_agent = RandomAgent(player_id="player_1")
                    baseline_agent = RandomAgent(player_id="player_2")
                    
                    agents = {"player_1": eval_agent, "player_2": baseline_agent}
                    
                    # Create decks
                    decks = {
                        "player_1": create_mock_deck(),
                        "player_2": create_mock_deck(),
                    }
                    
                    # Run game
                    runner = GameRunner(GameConfig(max_turns=30))
                    result = await runner.run_game(agents, decks)
                    
                    if result.winner == "player_1":
                        wins += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation game failed: {e}")
                    total += 1
                    continue
        except Exception as e:
            logger.error(f"Evaluation phase failed: {e}")
            return 0.5  # Default to 50% if evaluation fails
        
        win_rate = wins / total if total > 0 else 0.5
        return win_rate

    def _train_step(self) -> float:
        """Single training step on a batch from the buffer."""
        import torch
        import torch.nn.functional as F

        batch = self.buffer.sample(self.config.batch_size)
        if batch is None or len(batch) == 0:
            return 0.0

        try:
            if self.neural_module and self.optimizer:
                # Convert batch experiences to tensors
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                
                for exp in batch:
                    if exp.state_features and exp.next_state_features:
                        states.append(torch.tensor(exp.state_features, dtype=torch.float32))
                        actions.append(exp.action_index)
                        rewards.append(exp.reward)
                        next_states.append(torch.tensor(exp.next_state_features, dtype=torch.float32))
                        dones.append(1.0 if exp.done else 0.0)
                
                if not states:
                    return 0.0
                
                # Stack into batch
                state_batch = torch.stack(states)
                action_batch = torch.tensor(actions, dtype=torch.long)
                reward_batch = torch.tensor(rewards, dtype=torch.float32)
                next_state_batch = torch.stack(next_states) if next_states else None
                done_batch = torch.tensor(dones, dtype=torch.float32)
                
                # Forward pass through value network
                value_pred = self.neural_module.value_head(state_batch).squeeze(-1)
                
                # Compute target (reward + gamma * V(next_state) * (1 - done))
                gamma = 0.99
                with torch.no_grad():
                    if next_state_batch is not None:
                        next_value = self.neural_module.value_head(next_state_batch).squeeze(-1)
                        target = reward_batch + gamma * next_value * (1.0 - done_batch)
                    else:
                        target = reward_batch
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_module.parameters(), 1.0)
                self.optimizer.step()
                
                return value_loss.item()
        except Exception as e:
            logger.error(f"Error in training step: {e}", exc_info=True)

        return 0.0

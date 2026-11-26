"""Benchmark comparator for agent strategy evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Any

from src.orchestrator_legacy.game_runner import GameRunner, GameConfig
from src.training.deck_utils import create_mock_deck
from src.agents.random_agent import RandomAgent

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Compare multiple agent classes in a round-robin set of matches."""

    def __init__(
        self,
        agents: Dict[str, Callable[[str], Any]],
        num_games_per_match: int = 10,
        max_turns: int = 50,
    ):
        self.agents = agents
        self.num_games_per_match = num_games_per_match
        self.max_turns = max_turns
        self.results: Dict[str, Dict[str, Dict[str, float]]] = {}

    async def run(self) -> Dict[str, Any]:
        """Run the benchmark and return a results summary."""
        names = list(self.agents.keys())

        # Initialize empty results grid
        for name in names:
            self.results[name] = {
                opp: {"wins": 0, "losses": 0, "draws": 0, "games": 0}
                for opp in names
                if opp != name
            }

        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                logger.info("Benchmark match: %s vs %s", name_a, name_b)
                stats = await self._run_matchup(name_a, name_b)
                self.results[name_a][name_b] = stats[name_a]
                self.results[name_b][name_a] = stats[name_b]

        summary = self._build_summary()
        return summary

    async def _run_matchup(self, name_a: str, name_b: str) -> Dict[str, Dict[str, float]]:
        stats = {
            name_a: {"wins": 0, "losses": 0, "draws": 0, "games": 0},
            name_b: {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        }

        for game_idx in range(self.num_games_per_match):
            agent_a = self._make_agent(name_a, "player_1")
            agent_b = self._make_agent(name_b, "player_2")
            runner = GameRunner(GameConfig(max_turns=self.max_turns))
            decks = {"player_1": create_mock_deck(), "player_2": create_mock_deck()}

            try:
                result = await runner.run_game({"player_1": agent_a, "player_2": agent_b}, decks)
            except Exception as e:
                logger.warning("Game exception %s vs %s: %s", name_a, name_b, e)
                continue

            if result.winner == "player_1":
                stats[name_a]["wins"] += 1
                stats[name_a]["games"] += 1
                stats[name_b]["losses"] += 1
                stats[name_b]["games"] += 1
            elif result.winner == "player_2":
                stats[name_b]["wins"] += 1
                stats[name_b]["games"] += 1
                stats[name_a]["losses"] += 1
                stats[name_a]["games"] += 1
            else:
                stats[name_a]["draws"] += 1
                stats[name_a]["games"] += 1
                stats[name_b]["draws"] += 1
                stats[name_b]["games"] += 1

        return stats

    def _make_agent(self, name: str, player_id: str) -> Any:
        factory = self.agents.get(name)
        if not factory:
            raise ValueError(f"Unknown agent name {name}")
        return factory(player_id)

    def _build_summary(self) -> Dict[str, Any]:
        return {
            "matrix": self.results,
            "win_rates": {
                name: {
                    opp: stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0
                    for opp, stats in opp_stats.items()
                }
                for name, opp_stats in self.results.items()
            },
        }


async def run_default_benchmark() -> None:
    agent_factories = {
        "Random": lambda pid: RandomAgent(player_id=pid),
        # Add additional agents as available:
        # "LLM": lambda pid: OllamaAgent(player_id=pid),
        # "WorldModel": lambda pid: WorldModelAgent(...)
    }

    suite = BenchmarkSuite(agent_factories, num_games_per_match=20, max_turns=50)
    results = await suite.run()
    print("Benchmark complete")
    for name, row in results["win_rates"].items():
        print(f"{name}:")
        for opp, wr in row.items():
            print(f"  vs {opp}: {wr:.2%}")


if __name__ == "__main__":
    asyncio.run(run_default_benchmark())

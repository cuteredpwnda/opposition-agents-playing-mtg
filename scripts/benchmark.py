#!/usr/bin/env python
"""Benchmark comparator entrypoint.

Usage:
  python scripts/benchmark.py --games 10 --max-turns 50
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src is importable when run from repository root
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.training.benchmark import BenchmarkSuite
from src.agents.random_agent import RandomAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark comparator for agents")
    parser.add_argument("--games", type=int, default=10, help="Games per pairing")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns per game")
    args = parser.parse_args()

    agent_factories = {
        "Random": lambda pid: RandomAgent(player_id=pid),
    }

    try:
        from src.agents.active_inference_agent import ActiveInferenceAgent
        agent_factories["ActiveInference"] = lambda pid: ActiveInferenceAgent(player_id=pid)
    except Exception as e:
        print("ActiveInferenceAgent not available:", e)

    try:
        from src.agents.neural_reasoner_agent import NeuralReasonerAgent
        agent_factories["NeuralReasoner"] = lambda pid: NeuralReasonerAgent(player_id=pid)
    except Exception as e:
        print("NeuralReasonerAgent not available:", e)

    # Add your own agents by un-commenting below and adjusting parameters:
    # "LLM": lambda pid: OllamaAgent(player_id=pid),
    # "WorldModel": lambda pid: WorldModelAgent(player_id=pid, ...),
    # "LLMFusion": lambda pid: LLMFusionAgent(player_id=pid, ...),

    suite = BenchmarkSuite(agent_factories, num_games_per_match=args.games, max_turns=args.max_turns)

    results = asyncio.run(suite.run())

    print("\n=== Benchmark Results ===")
    for name, row in results["win_rates"].items():
        print(f"{name}:")
        for opp, wr in row.items():
            print(f"  vs {opp}: {wr:.2%}")


if __name__ == "__main__":
    main()

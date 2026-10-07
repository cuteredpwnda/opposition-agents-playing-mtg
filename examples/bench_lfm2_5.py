#!/usr/bin/env python
"""Quick benchmark: LFM2.5:8b vs Gemma4:e2b (or other baseline).

Test LFM2.5:8b response quality and speed in the MTG decision context.
This is a lightweight check to see if the model is available and responds reasonably.

Usage:
    python examples/bench_lfm2_5.py
    python examples/bench_lfm2_5.py --model1 lfm2.5:8b --model2 qwen2.5:7b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_model_availability(model_tag: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        import httpx
        
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return any(m.get("name", "").startswith(model_tag.split(":")[0]) for m in models)
    except Exception as e:
        logger.warning(f"Could not check model availability: {e}")
        return False


async def benchmark_response_time(model_tag: str, num_samples: int = 3) -> dict:
    """Benchmark the response time for a model."""
    from src.agents.llm_agent import OllamaAgent
    from src.engine_legacy.game_state import GameState, PlayerState, ActionType, Action
    
    agent = OllamaAgent("test", name=f"Test ({model_tag})", model=model_tag)
    
    if not agent._ollama_available:
        logger.warning(f"Ollama not available for model {model_tag}")
        return {"model": model_tag, "available": False, "error": "Ollama unavailable"}
    
    times = []
    logger.info(f"Testing {model_tag} ({num_samples} samples)...")
    
    # Create minimal players and game state
    players = [
        PlayerState(player_id="p0", name="Player 0", life_total=20),
        PlayerState(player_id="p1", name="Player 1", life_total=20),
    ]
    state = GameState(game_id="bench", format="duel", players=players)
    
    # Create some dummy legal actions
    legal_actions = [
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p0"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p0"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p0"),
    ]
    
    for i in range(num_samples):
        start = time.time()
        try:
            action = await agent.decide_action(state, legal_actions)
            elapsed = time.time() - start
            times.append(elapsed)
            logger.info(f"  Sample {i+1}: {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"  Sample {i+1}: ERROR: {e}")
            return {"model": model_tag, "available": True, "error": str(e)}
    
    if not times:
        return {"model": model_tag, "available": True, "error": "no samples completed"}
    
    avg_time = sum(times) / len(times)
    return {
        "model": model_tag,
        "available": True,
        "num_samples": len(times),
        "avg_time_s": avg_time,
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


async def main_async(args) -> int:
    logger.info("LFM2.5:8b Benchmark")
    logger.info("=" * 60)
    
    # Check Ollama availability
    try:
        import httpx
        httpx.get("http://localhost:11434/api/tags", timeout=2.0)
    except Exception:
        logger.error("Ollama not running on localhost:11434")
        logger.error("Start Ollama with: ollama serve")
        return 1
    
    logger.info(f"Testing {args.model1}...")
    result1 = await benchmark_response_time(args.model1, args.samples)
    
    logger.info(f"\nTesting {args.model2}...")
    result2 = await benchmark_response_time(args.model2, args.samples)
    
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    for result in [result1, result2]:
        model = result.get("model", "?")
        if result.get("available"):
            avg = result.get("avg_time_s", 0)
            logger.info(f"{model:20s}  {avg:6.2f}s avg (n={result.get('num_samples')})")
        else:
            logger.info(f"{model:20s}  UNAVAILABLE or ERROR")
            if "error" in result:
                logger.info(f"                       {result['error']}")
    
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model1",
        default="lfm2.5:8b",
        help="First model to test (default: lfm2.5:8b)",
    )
    ap.add_argument(
        "--model2",
        default="gemma4:e2b",
        help="Second model to test (default: gemma4:e2b)",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of response samples per model (default: 2)",
    )

    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

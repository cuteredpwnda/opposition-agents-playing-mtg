"""End-to-end demo: MTG agents playing a real game.

This script demonstrates agentic play with a full pipeline:
1. Create agents (RandomAgent, HeuristicAgent, OllamaAgent, WorldModelAgent, etc.)
2. Build or load decklists
3. Run a complete game through the simulator or async runner
4. Display results including mulligan choices, moves, and final state

Usage:
    python examples/demo_agents_play.py                    # Random vs Random
    python examples/demo_agents_play.py --agent1 ollama --agent2 heuristic
    python examples/demo_agents_play.py --agent1 ollama --agent2 ollama --turns 40
    python examples/demo_agents_play.py --async               # Use async GameRunner
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import MTGAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.agents.llm_agent import OllamaAgent
from src.engine.agent_strategies import Strategy
from src.engine.game_simulator import GameSimulator
from src.engine.game_execution import AgentGamePlayer
from src.utils.seeding import set_global_seed


# Pre-built basic decks for demo
def _build_basic_red_deck():
    """60-card mono-red deck: lands + creatures + spells."""
    return [
        {"name": "Mountain", "type_line": "Basic Land - Mountain", "cmc": 0}
        for _ in range(24)
    ] + [
        {"name": "Goblin Recruit", "type_line": "Creature - Goblin", "cmc": 1}
        for _ in range(12)
    ] + [
        {"name": "Goblin Guide", "type_line": "Creature - Goblin Scout", "cmc": 1}
        for _ in range(12)
    ] + [
        {"name": "Lightning Bolt", "type_line": "Instant", "cmc": 1}
        for _ in range(12)
    ]


def _build_basic_blue_deck():
    """60-card mono-blue deck: lands + creatures + spells."""
    return [
        {"name": "Island", "type_line": "Basic Land - Island", "cmc": 0}
        for _ in range(24)
    ] + [
        {"name": "Merfolk Looter", "type_line": "Creature - Merfolk", "cmc": 2}
        for _ in range(12)
    ] + [
        {"name": "Wind Drake", "type_line": "Creature - Drake", "cmc": 3}
        for _ in range(12)
    ] + [
        {"name": "Counterspell", "type_line": "Instant", "cmc": 2}
        for _ in range(12)
    ]


AGENT_PRESETS = {
    "random": {
        "name": "Random Agent",
        "factory": lambda pid: RandomAgent(pid),
        "strategy": Strategy.AGGRESSIVE,
    },
    "heuristic": {
        "name": "Heuristic Agent",
        "factory": lambda pid: HeuristicAgent(pid),
        "strategy": Strategy.AGGRESSIVE,
    },
    "ollama": {
        "name": "Ollama Agent (gemma4:e2b)",
        "factory": lambda pid: OllamaAgent(pid, model="gemma4:e2b"),
        "strategy": Strategy.REACTIVE,
    },
}

DECK_PRESETS = {
    "red": {"name": "Mono-Red Aggro", "builder": _build_basic_red_deck},
    "blue": {"name": "Mono-Blue Control", "builder": _build_basic_blue_deck},
}


def build_agent(agent_type: str, player_id: str) -> MTGAgent:
    """Instantiate an agent by type."""
    if agent_type not in AGENT_PRESETS:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                         f"Available: {sorted(AGENT_PRESETS)}")
    return AGENT_PRESETS[agent_type]["factory"](player_id)


def build_deck(deck_type: str) -> list[dict]:
    """Build a decklist by type."""
    if deck_type not in DECK_PRESETS:
        raise ValueError(f"Unknown deck type: {deck_type}. "
                         f"Available: {sorted(DECK_PRESETS)}")
    return DECK_PRESETS[deck_type]["builder"]()


async def play_async_game(agent1_type: str, agent2_type: str,
                          deck1_type: str, deck2_type: str,
                          max_turns: int, seed: Optional[int]) -> None:
    """Play a game using the async GameRunner."""
    from src.orchestrator.game_runner import GameConfig, GameRunner

    set_global_seed(seed)

    agent1 = build_agent(agent1_type, "player1")
    agent2 = build_agent(agent2_type, "player2")

    deck1 = build_deck(deck1_type)
    deck2 = build_deck(deck2_type)

    config = GameConfig(
        max_turns=max_turns,
        mulligan_enabled=True,
        max_mulligans=3,
    )
    runner = GameRunner(config)

    print(f"\n{'='*70}")
    print(f"MTG AGENTS PLAY — ASYNC MODE")
    print(f"{'='*70}")
    print(f"Player 1: {AGENT_PRESETS[agent1_type]['name']}")
    print(f"Player 2: {AGENT_PRESETS[agent2_type]['name']}")
    print(f"Decks: {deck1_type} vs {deck2_type}")
    print(f"Max turns: {max_turns}, Mulligan: enabled (cap {config.max_mulligans})")
    print(f"{'='*70}\n")

    try:
        result = await runner.run_game(
            agents={agent1.player_id: agent1, agent2.player_id: agent2},
            decks={agent1.player_id: deck1, agent2.player_id: deck2},
        )
        print(f"\nGame finished: {result.winner} in {result.turns} turns")
        if result.log:
            print(f"Game log: {result.log[:500]}...")
    except Exception as e:
        print(f"Game error: {e}")
        import traceback
        traceback.print_exc()


def play_sync_game(agent1_type: str, agent2_type: str,
                   deck1_type: str, deck2_type: str,
                   max_turns: int, seed: Optional[int]) -> None:
    """Play a game using the sync GameSimulator."""
    set_global_seed(seed)

    # Create agents wrapped as AgentGamePlayer
    a1 = AgentGamePlayer("player1", strategy=AGENT_PRESETS[agent1_type]["strategy"])
    a2 = AgentGamePlayer("player2", strategy=AGENT_PRESETS[agent2_type]["strategy"])

    # For now, inject the MTGAgent as a stub; full integration with
    # AgentGamePlayer will come in a follow-up.
    a1.llm_agent = build_agent(agent1_type, "player1")
    a2.llm_agent = build_agent(agent2_type, "player2")

    sim = GameSimulator(a1, a2, max_turns=max_turns)

    deck1 = build_deck(deck1_type)
    deck2 = build_deck(deck2_type)

    print(f"\n{'='*70}")
    print(f"MTG AGENTS PLAY — SYNC MODE")
    print(f"{'='*70}")
    print(f"Player 1: {AGENT_PRESETS[agent1_type]['name']}")
    print(f"Player 2: {AGENT_PRESETS[agent2_type]['name']}")
    print(f"Decks: {deck1_type} vs {deck2_type}")
    print(f"Max turns: {max_turns}, Mulligan: enabled (cap 3)")
    print(f"{'='*70}\n")

    try:
        game_state = sim.setup_game(
            deck1=deck1,
            deck2=deck2,
            shuffle=True,
            mulligan_enabled=True,
            max_mulligans=3,
        )

        print("Game initialized.")
        print(f"P1 mulligans: {game_state.players[0].mulligans_taken}")
        print(f"P2 mulligans: {game_state.players[1].mulligans_taken}")
        print(f"P1 hand size: {len([c for c in game_state.cards if c.zone.value == 'HAND' and c.owner_id == 'player1'])}")
        print(f"P2 hand size: {len([c for c in game_state.cards if c.zone.value == 'HAND' and c.owner_id == 'player2'])}")
        print()

        # NOTE: Full game execution loop would go here once the sync simulator
        # is fully wired to call agent.decide_action() and handle phases.
        # For now, this is a smoke test showing setup works.

    except Exception as e:
        print(f"Game error: {e}")
        import traceback
        traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent1", choices=list(AGENT_PRESETS), default="random",
                   help="Type of agent for player 1.")
    p.add_argument("--agent2", choices=list(AGENT_PRESETS), default="random",
                   help="Type of agent for player 2.")
    p.add_argument("--deck1", choices=list(DECK_PRESETS), default="red",
                   help="Deck for player 1.")
    p.add_argument("--deck2", choices=list(DECK_PRESETS), default="blue",
                   help="Deck for player 2.")
    p.add_argument("--turns", type=int, default=40,
                   help="Maximum game length in turns.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility.")
    p.add_argument("--async", action="store_true", dest="use_async",
                   help="Use async GameRunner instead of sync GameSimulator.")
    return p


async def async_main(args) -> int:
    await play_async_game(
        args.agent1, args.agent2,
        args.deck1, args.deck2,
        args.turns, args.seed
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.use_async:
        return asyncio.run(async_main(args))
    else:
        play_sync_game(
            args.agent1, args.agent2,
            args.deck1, args.deck2,
            args.turns, args.seed
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

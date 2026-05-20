"""
LangGraph state machine for MTG game flow.

Encodes the full MTG turn structure as a LangGraph StateGraph with
nodes for each phase/step and conditional edges for priority passing,
stack resolution, and game-over checks.

Reference: Section 5.2 of PLAN.md.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.engine.game_state import GameState, Phase, Zone
from src.engine.phases import advance_phase, is_combat_phase, is_main_phase
from src.engine.stack import is_empty as stack_is_empty


def build_game_graph(num_players: int = 2) -> StateGraph:
    """Build the LangGraph state machine for an MTG game.

    Nodes:
        untap_step, upkeep_step, draw_step, main_phase,
        combat_phase, end_step, cleanup_step,
        priority_loop, resolve_stack,
        agent_decision, check_game_over

    Returns a compiled StateGraph.
    """
    graph = StateGraph(dict)

    # Phase nodes
    graph.add_node("untap_step", untap_step)
    graph.add_node("upkeep_step", upkeep_step)
    graph.add_node("draw_step", draw_step)
    graph.add_node("main_phase", main_phase)
    graph.add_node("combat_phase", combat_phase)
    graph.add_node("end_step", end_step)
    graph.add_node("cleanup_step", cleanup_step)

    # Decision nodes
    graph.add_node("priority_loop", priority_loop)
    graph.add_node("resolve_stack", resolve_stack)
    graph.add_node("agent_decision", agent_decision)
    graph.add_node("check_game_over", check_game_over)

    # Entry
    graph.set_entry_point("untap_step")

    # Phase transitions: each phase → priority_loop
    graph.add_edge("untap_step", "upkeep_step")
    graph.add_edge("upkeep_step", "priority_loop")
    graph.add_edge("draw_step", "priority_loop")

    # Priority loop routing
    graph.add_conditional_edges(
        "priority_loop",
        route_priority,
        {
            "agent_decision": "agent_decision",
            "resolve_stack": "resolve_stack",
            "advance_phase": "check_game_over",
        },
    )

    # Agent decision → back to priority loop
    graph.add_edge("agent_decision", "priority_loop")

    # Stack resolution → priority loop (players get priority again)
    graph.add_edge("resolve_stack", "priority_loop")

    # Game-over check routing
    graph.add_conditional_edges(
        "check_game_over",
        route_game_over,
        {
            "continue": "next_phase_router",
            "game_over": END,
        },
    )

    # Phase router
    graph.add_node("next_phase_router", next_phase_router)
    graph.add_conditional_edges(
        "next_phase_router",
        route_next_phase,
        {
            "main_phase": "main_phase",
            "combat_phase": "combat_phase",
            "end_step": "end_step",
            "cleanup_step": "cleanup_step",
            "untap_step": "untap_step",  # new turn
            "draw_step": "draw_step",
            "upkeep_step": "upkeep_step",
            "priority_loop": "priority_loop",
        },
    )

    graph.add_edge("main_phase", "priority_loop")
    graph.add_edge("combat_phase", "priority_loop")
    graph.add_edge("end_step", "priority_loop")
    graph.add_edge("cleanup_step", "untap_step")

    return graph


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def untap_step(state: dict) -> dict:
    """Untap all permanents controlled by the active player."""
    gs: GameState = state["game_state"]
    active = gs.active_player
    bf = gs.cards_in_zone.get((active, "battlefield"), [])
    for card in bf:
        card.tapped = False
        card.summoning_sick = False
    gs.phase = Phase.UNTAP
    return state


def upkeep_step(state: dict) -> dict:
    """Upkeep step — trigger upkeep abilities and check SBAs."""
    gs: GameState = state["game_state"]
    gs.phase = Phase.UPKEEP
    
    # Trigger upkeep abilities from active player's permanents
    active_player_id = gs.players[gs.active_player_index].player_id
    upkeep_triggers = []
    
    for card in gs.cards:
        if card.controller_id == active_player_id and card.zone == Zone.BATTLEFIELD:
            text = card.oracle_text.lower()
            # Check for "at the beginning of your upkeep"
            if "upkeep" in text:
                upkeep_triggers.append(card)
    
    # Queue upkeep triggers (in real implementation, would use proper stack/trigger queue)
    # For now, just log them
    if upkeep_triggers:
        trigger_names = [c.card_name for c in upkeep_triggers]
        gs.log.append(f"Upkeep triggers: {', '.join(trigger_names)}")
    
    return state


def draw_step(state: dict) -> dict:
    """Active player draws a card (skip on turn 1 in 2-player)."""
    gs: GameState = state["game_state"]
    gs.phase = Phase.DRAW
    active = gs.active_player
    library = gs.cards_in_zone.get((active, "library"), [])
    hand = gs.cards_in_zone.setdefault((active, "hand"), [])
    if library and not (gs.turn_number == 1 and len(gs.players) == 2):
        hand.append(library.pop(0))
    return state


def main_phase(state: dict) -> dict:
    gs: GameState = state["game_state"]
    # Set to main 1 or main 2 depending on whether combat happened
    if gs.phase.value < Phase.COMBAT_BEGIN.value:
        gs.phase = Phase.MAIN_1
    else:
        gs.phase = Phase.MAIN_2
    return state


def combat_phase(state: dict) -> dict:
    gs: GameState = state["game_state"]
    gs.phase = Phase.COMBAT_BEGIN
    return state


def end_step(state: dict) -> dict:
    gs: GameState = state["game_state"]
    gs.phase = Phase.END
    return state


def cleanup_step(state: dict) -> dict:
    """Discard to hand size, remove damage, advance turn."""
    gs: GameState = state["game_state"]
    gs.phase = Phase.CLEANUP
    gs = advance_phase(gs)  # Advances turn
    state["game_state"] = gs
    return state


def priority_loop(state: dict) -> dict:
    """Set up priority passing — APNAP order."""
    return state


def resolve_stack(state: dict) -> dict:
    """Resolve the top item on the stack."""
    from src.engine.stack import resolve_top

    gs: GameState = state["game_state"]
    gs = resolve_top(gs)
    state["game_state"] = gs
    return state


def agent_decision(state: dict) -> dict:
    """Placeholder — the actual decision is made by the game_runner
    which calls the agent's decide_action method."""
    return state


def check_game_over(state: dict) -> dict:
    """Check state-based actions and whether the game has ended."""
    from src.engine.rules_engine import RulesEngine

    gs: GameState = state["game_state"]
    engine = RulesEngine()
    gs = engine.check_state_based_actions(gs)
    state["game_state"] = gs
    return state


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_priority(state: dict) -> str:
    gs: GameState = state["game_state"]
    if not stack_is_empty(gs):
        # If there are items on the stack and all players passed, resolve
        if state.get("all_passed"):
            state["all_passed"] = False
            return "resolve_stack"
        return "agent_decision"
    # Stack empty and all passed → advance phase
    if state.get("all_passed"):
        state["all_passed"] = False
        return "advance_phase"
    return "agent_decision"


def route_game_over(state: dict) -> str:
    gs: GameState = state["game_state"]
    if gs.game_over:
        return "game_over"
    return "continue"


def next_phase_router(state: dict) -> dict:
    gs: GameState = state["game_state"]
    gs = advance_phase(gs)
    state["game_state"] = gs
    return state


def route_next_phase(state: dict) -> str:
    gs: GameState = state["game_state"]
    phase = gs.phase
    mapping = {
        Phase.UNTAP: "untap_step",
        Phase.UPKEEP: "upkeep_step",
        Phase.DRAW: "draw_step",
        Phase.MAIN_1: "main_phase",
        Phase.COMBAT_BEGIN: "combat_phase",
        Phase.MAIN_2: "main_phase",
        Phase.END: "end_step",
        Phase.CLEANUP: "cleanup_step",
    }
    return mapping.get(phase, "priority_loop")

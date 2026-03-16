"""
Priority passing logic — APNAP order for multiplayer.

In MTG, after any game event players receive priority in Active Player,
Non-Active Player (APNAP) order. An item on the stack resolves only when
all players pass priority in succession without adding anything.

Reference: Section 5.3 of PLAN.md, CR 117.
"""

from __future__ import annotations

from src.engine.game_state import GameState, ActionType
from src.engine.stack import is_empty as stack_is_empty, resolve_top


def get_priority_order(game_state: GameState) -> list[str]:
    """Return player IDs in APNAP order starting from active player."""
    player_ids = [p.player_id for p in game_state.players]
    active_player_id = game_state.active_player.player_id
    
    if active_player_id not in player_ids:
        return player_ids
    
    idx = player_ids.index(active_player_id)
    return player_ids[idx:] + player_ids[:idx]


def advance_priority(game_state: GameState) -> GameState:
    """Move priority to the next player in APNAP order."""
    order = get_priority_order(game_state)
    current_priority_id = game_state.priority_player.player_id
    
    if current_priority_id not in order:
        game_state.priority_player_index = 0
        return game_state
    
    idx = order.index(current_priority_id)
    next_idx = (idx + 1) % len(order)
    next_player_id = order[next_idx]
    
    # Find index of next player in players list
    game_state.priority_player_index = next(
        i for i, p in enumerate(game_state.players) if p.player_id == next_player_id
    )
    return game_state


def all_players_passed(
    game_state: GameState, passed: set[str]
) -> bool:
    """Check if all players have passed priority in order."""
    all_player_ids = {p.player_id for p in game_state.players}
    return all_player_ids == passed


def priority_action_result(
    game_state: GameState,
    passed_players: set[str],
) -> str:
    """Determine what happens after a priority pass.

    Returns:
        'resolve'       — all passed, stack non-empty → resolve top
        'advance_phase' — all passed, stack empty → move to next phase
        'continue'      — not all passed, keep going
    """
    if all_players_passed(game_state, passed_players):
        if stack_is_empty(game_state):
            return "advance_phase"
        return "resolve"
    return "continue"


async def run_priority_loop(
    game_state: GameState,
    agents: dict[str, object],
    rules_engine: object,
) -> GameState:
    """Execute a full priority loop.
    
    This is the core MTG action resolution mechanic:
    1. Current priority player acts (cast spell, activate ability, pass)
    2. If they pass, priority moves to next player in APNAP order
    3. When all players pass in succession, resolve top of stack (if any)
    4. Repeat until stack is empty and all pass
    
    Args:
        game_state: Current game state with priority tracking
        agents: Mapping of player_id → agent with decide_action() method
        rules_engine: RulesEngine instance for get_legal_actions() and execute_action()
    
    Returns:
        Updated game state with priority loop completed
    """
    passed_players: set[str] = set()
    
    while True:
        if game_state.game_over:
            return game_state
        
        # Get current priority player
        priority_player_id = game_state.priority_player.player_id
        agent = agents.get(priority_player_id)
        
        # Get legal actions for this player given current game state
        legal_actions = rules_engine.get_legal_actions(game_state, priority_player_id)
        
        if not legal_actions:
            # This shouldn't happen; PASS_PRIORITY should always be legal
            game_state.log(f"ERROR: No legal actions for {priority_player_id}")
            return game_state
        
        # Agent decides what to do
        action = await agent.decide_action(game_state, legal_actions)
        
        # Notify all agents of action
        for agent_obj in agents.values():
            await agent_obj.observe(game_state, action)
        
        if action.action_type == ActionType.PASS_PRIORITY:
            # Player passed priority
            passed_players.add(priority_player_id)
            game_state.log(f"{priority_player_id} passes priority")
            
            # Check if all players passed
            if all_players_passed(game_state, passed_players):
                # All players passed in sequence
                if stack_is_empty(game_state):
                    # Stack is empty, priority loop ends
                    game_state.log("All players passed, stack empty → end of priority loop")
                    return game_state
                else:
                    # Stack has items, resolve top
                    game_state.log("All players passed, stack non-empty → resolving top of stack")
                    
                    # Resolve top of stack using the new resolve_stack_item method
                    game_state = rules_engine.resolve_stack_item(game_state)
                    
                    # After resolution, reset passed set and give AP priority again
                    passed_players = set()
                    game_state.priority_player_index = game_state.active_player_index
                    continue
            else:
                # Not all passed yet, move priority to next player
                game_state = advance_priority(game_state)
                continue
        else:
            # Player took an action (cast spell, activate ability, etc.)
            game_state = rules_engine.execute_action(game_state, action)
            
            # After action, reset passed set (new action added to stack or board changed)
            passed_players = set()
            
            # Priority goes to next player in APNAP order
            game_state = advance_priority(game_state)
            
            # Check SBAs immediately after action
            rules_engine.check_state_based_actions(game_state)
            
            continue

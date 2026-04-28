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


def _summarize_actions(actions, game_state: GameState, max_items: int = 10) -> str:
    """Render a compact one-line summary of legal actions for the log.

    Groups by ``(kind, card name)`` so that e.g. three identical
    ``Activate(Mountain)`` actions render as ``Activate(Mountain) ×3``.
    Truncates to ``max_items`` distinct entries.
    """
    cards_by_id = {c.instance_id: c for c in game_state.cards}
    KIND_SHORT = {
        "PLAY_LAND": "Play",
        "CAST_SPELL": "Cast",
        "ACTIVATE_ABILITY": "Activate",
        "DECLARE_ATTACKER": "Attack",
        "DECLARE_BLOCKER": "Block",
    }
    # Preserve insertion order while counting duplicates.
    counts: dict[str, int] = {}
    for a in actions:
        kind = a.action_type.name if hasattr(a.action_type, "name") else str(a.action_type)
        kind_short = KIND_SHORT.get(kind, kind.title())
        cid = getattr(a, "card_instance_id", None)
        if cid and cid in cards_by_id:
            label = f"{kind_short}({cards_by_id[cid].name})"
        else:
            label = kind_short
        counts[label] = counts.get(label, 0) + 1

    parts: list[str] = []
    for i, (label, n) in enumerate(counts.items()):
        if i >= max_items:
            parts.append("…")
            break
        parts.append(label if n == 1 else f"{label} ×{n}")
    return ", ".join(parts)


def _is_noise_only(actions) -> bool:
    """True if the only non-pass action is CONCEDE — not worth logging."""
    if not actions:
        return True
    for a in actions:
        kind = a.action_type.name if hasattr(a.action_type, "name") else str(a.action_type)
        if kind != "CONCEDE":
            return False
    return True


def _sync_priority_player(game_state: GameState) -> None:
    """Ensure priority player is valid when players are removed mid-game."""
    if not game_state.players:
        game_state.priority_player_index = 0
        return

    if game_state.priority_player_index < 0 or game_state.priority_player_index >= len(game_state.players):
        game_state.priority_player_index = 0
        return

    current = game_state.players[game_state.priority_player_index].player_id
    if current not in [p.player_id for p in game_state.players]:
        game_state.priority_player_index = 0


def advance_priority(game_state: GameState) -> GameState:
    """Move priority to the next player in APNAP order."""
    _sync_priority_player(game_state)

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
    collector: object | None = None,
) -> GameState:
    """Execute a full priority loop.
    
    Optional `collector` is a SelfPlayCollector (or equivalent) used to record
    states and actions during self-play.
    
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
    
    loop_iterations = 0
    # Pure safety net. In a correctly-implemented agent the loop terminates
    # in O(#players × actions-this-turn) iterations because every player
    # eventually passes. The cap protects against pathological agents that
    # never pass; it should never fire in normal play.
    max_iterations = 500

    while True:
        loop_iterations += 1
        if loop_iterations > max_iterations:
            game_state.log(
                "Priority loop aborted: exceeded max iterations (forcing phase end)"
            )
            return game_state

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

        # Log legal actions (other than PASS) so it's visible in the game
        # log what the player could have done from the current board state.
        # Skip pure-pass turns to keep noise down.
        non_pass = [a for a in legal_actions if a.action_type != ActionType.PASS_PRIORITY]
        if non_pass and not _is_noise_only(non_pass):
            pp = next(
                (p for p in game_state.players if p.player_id == priority_player_id),
                None,
            )
            label = (pp.name or pp.player_id) if pp else priority_player_id
            game_state.log(
                f"      ? {label} legal actions ({len(non_pass)}): "
                + _summarize_actions(non_pass, game_state)
            )

# Collector sees current state before action
        if collector is not None:
            collector.on_state(game_state, game_state.priority_player_index)

        # Agent decides what to do
        action = await agent.decide_action(game_state, legal_actions)

        # Notify all agents of action
        for agent_obj in agents.values():
            await agent_obj.observe(game_state, action)

        # Collector records action
        if collector is not None:
            collector.on_action(action, reward=0.0, done=game_state.game_over)
        
        if action.action_type == ActionType.PASS_PRIORITY:
            # Player passed priority
            passed_players.add(priority_player_id)
            pp = next(
                (p for p in game_state.players if p.player_id == priority_player_id),
                None,
            )
            label = (pp.name or pp.player_id) if pp else priority_player_id
            game_state.log(f"      · {label} passes")
            
            # Check if all players passed
            if all_players_passed(game_state, passed_players):
                # All players passed in sequence
                if stack_is_empty(game_state):
                    # Stack is empty, priority loop ends
                    return game_state
                else:
                    # Stack has items, resolve top
                    game_state.log("      · all pass → resolving stack")
                    
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

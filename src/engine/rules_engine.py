"""
Rules engine — legal action generation and game action execution.

References:
- open-mtg (MIT): get_moves() → make_move() pattern
- mtg-python-engine (MIT): check_state_based_actions() pattern
- mtg-player (MIT): tool-based validation (get_legal_actions, execute_action)
- Forge (Java, GPL-3.0): behavioral reference for correctness
"""

from __future__ import annotations

from .game_state import Action, ActionType, GameState, Phase, Zone, StackItem
from .phases import is_main_phase
from .stack import is_empty as stack_is_empty
from .stack import push_to_stack
from .mana import can_pay, pay_cost, parse_mana_cost


class RulesEngine:
    """Validates and executes game actions according to MTG Comprehensive Rules."""

    def get_legal_actions(self, state: GameState, player_id: str) -> list[Action]:
        """Return all legal actions for a player given current state and priority.

        This is the core interface that agents use — adapted from open-mtg's
        ``game.get_moves()`` pattern, extended with priority and instant-speed.
        
        Legal actions are restricted by timing:
        - Sorcery-speed (land play, creature/sorcery cast): only during own main + empty stack
        - Instant-speed (instants, flash creatures): anytime with priority
        - Activated mana abilities: anytime (simplified to simple lands for now)
        """
        actions: list[Action] = []
        player = next((p for p in state.players if p.player_id == player_id), None)
        if not player:
            return []

        # Always can pass priority
        actions.append(Action(action_type=ActionType.PASS_PRIORITY, player_id=player_id))

        # NOTE: CONCEDE is not offered as a legal action for agents (they should stay in game)

        hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == player_id]
        battlefield = [c for c in state.cards if c.zone == Zone.BATTLEFIELD and c.owner_id == player_id]

        # Sorcery-speed actions: only during own main phase with empty stack
        if (
            state.players[state.active_player_index].player_id == player_id
            and is_main_phase(state.phase)
            and stack_is_empty(state)
        ):
            # Play a land (once per turn)
            if player.land_plays_remaining > 0:
                for card in hand:
                    if card.is_land():
                        actions.append(
                            Action(
                                action_type=ActionType.PLAY_LAND,
                                player_id=player_id,
                                card_instance_id=card.instance_id,
                            )
                        )

            # Cast sorceries and creatures (non-instant, non-flash)
            for card in hand:
                if card.is_land():
                    continue  # Already handled
                if card.is_instant():
                    continue  # Instant-speed only
                if "flash" in card.oracle_text.lower():
                    continue  # Flash is instant-speed
                    
                cost = parse_mana_cost(card.mana_cost or "")
                if can_pay(player, cost):
                    actions.append(
                        Action(
                            action_type=ActionType.CAST_SPELL,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                        )
                    )

        # Instant-speed actions: any time you have priority
        # This includes instants, flash creatures, activated abilities, etc.
        for card in hand:
            if card.is_instant() or "flash" in card.oracle_text.lower():
                cost = parse_mana_cost(card.mana_cost or "")
                if can_pay(player, cost):
                    actions.append(
                        Action(
                            action_type=ActionType.CAST_SPELL,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                        )
                    )

        # Activated abilities of permanents (basic impl: tap for mana)
        # Mana abilities can be used anytime, but for simplicity we restrict to available mana
        for card in battlefield:
            if card.is_land() and not card.tapped:
                actions.append(
                    Action(
                        action_type=ActionType.ACTIVATE_ABILITY,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                    )
                )

        # Combat actions: declare attackers/blockers during combat phases
        from .game_state import Phase
        if state.phase == Phase.COMBAT_ATTACKERS and state.players[state.active_player_index].player_id == player_id:
            # During declare attackers, generate options for attacking creatures
            # For now, simple approach: each untapped creature can attack
            for card in battlefield:
                if card.is_creature() and not card.tapped and not card.summoning_sick:
                    # For simplicity, creatures attack the defending player (next player in turn order)
                    defender_idx = (state.active_player_index + 1) % len(state.players)
                    defender_id = state.players[defender_idx].player_id
                    actions.append(
                        Action(
                            action_type=ActionType.DECLARE_ATTACKERS,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                            targets=[defender_id],
                        )
                    )

        return actions

    def execute_action(self, state: GameState, action: Action) -> GameState:
        """Apply an action, update state, check SBAs, queue triggers."""
        from .zones import move_card
        
        if action.action_type == ActionType.PASS_PRIORITY:
            # No state change on pass
            return state
        
        if action.action_type == ActionType.CONCEDE:
            state.game_over = True
            # Winner is next player
            players_list = state.players
            idx = next((i for i, p in enumerate(players_list) if p.player_id == action.player_id), 0)
            state.winner = players_list[(idx + 1) % len(players_list)]
            return state
        
        if action.action_type == ActionType.PLAY_LAND:
            if action.card_instance_id:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    # Move card from hand to battlefield
                    state = move_card(
                        state, action.card_instance_id, Zone.HAND, Zone.BATTLEFIELD,
                        action.player_id
                    )
                    card.tapped = False
                    player = next((p for p in state.players if p.player_id == action.player_id), None)
                    if player:
                        player.land_plays_remaining -= 1
            return state
        
        if action.action_type == ActionType.CAST_SPELL:
            if action.card_instance_id:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    # Pay mana cost
                    cost = parse_mana_cost(card.mana_cost or "")
                    player = next((p for p in state.players if p.player_id == action.player_id), None)
                    if player:
                        pay_cost(player, cost)
                    
                    # Move card from hand to stack
                    state = move_card(
                        state, action.card_instance_id, Zone.HAND, Zone.STACK,
                        action.player_id
                    )
                    
                    # Create stack item and push to stack
                    stack_item = StackItem(
                        source_card_id=action.card_instance_id,
                        controller_id=action.player_id,
                        is_spell=True,
                        card_data=card.card_data.copy(),
                    )
                    state = push_to_stack(state, stack_item)
                    state.log(f"{card.name} is cast")
            return state
        
        if action.action_type == ActionType.ACTIVATE_ABILITY:
            if action.card_instance_id:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if card and card.is_land():
                    # Tap for mana
                    card.tapped = True
                    player = next((p for p in state.players if p.player_id == action.player_id), None)
                    if player:
                        # Basic land produces one mana of appropriate color
                        if "Plains" in card.name:
                            player.mana_pool["W"] += 1
                        elif "Island" in card.name:
                            player.mana_pool["U"] += 1
                        elif "Swamp" in card.name:
                            player.mana_pool["B"] += 1
                        elif "Mountain" in card.name:
                            player.mana_pool["R"] += 1
                        elif "Forest" in card.name:
                            player.mana_pool["G"] += 1
                        else:
                            player.mana_pool["C"] += 1
            return state
        
        if action.action_type == ActionType.DECLARE_ATTACKERS:
            if action.card_instance_id and action.targets:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                defender_id = action.targets[0]
                if card:
                    from .combat import declare_attackers
                    declare_attackers(state, {action.card_instance_id: defender_id})
                    card.tapped = True
            return state
        
        # Default: no change
        return state

    def resolve_spell(self, state: GameState, stack_item) -> GameState:
        """Resolve a spell from the stack.
        
        For creatures, this moves the spell to the battlefield.
        For other spells, effects are applied (not implemented in Phase 1).
        """
        from .zones import move_card
        from .game_state import Zone, StackItem
        from .triggers import check_enters_battlefield_triggers, resolve_trigger
        
        source_card_id = stack_item.source_card_id
        card = next((c for c in state.cards if c.instance_id == source_card_id), None)
        
        if not card:
            return state
        
        # Check if this is still in the stack (wasn't fizzled)
        if card.zone != Zone.STACK:
            return state
        
        # If it's a creature, move to battlefield with summoning sickness
        if card.is_creature():
            state = move_card(state, source_card_id, Zone.STACK, Zone.BATTLEFIELD, card.owner_id)
            card.summoning_sick = True
            card.turn_entered = state.turn_number
            state.log(f"{card.name} enters the battlefield")
            
            # CHECK TRIGGERED ABILITIES
            # Check what ETB triggers should fire from other permanents and this card
            etb_triggers = check_enters_battlefield_triggers(state, card)
            
            for trigger in etb_triggers:
                # Add trigger to the stack (triggered abilities go on stack)
                # Create a stack item for this triggered ability
                trigger_stack_item = StackItem(
                    source_card_id=trigger.source_card_id,
                    controller_id=trigger.controller_id,
                    is_spell=False,  # It's an ability, not a spell
                    card_data={
                        "name": f"[Trigger] {trigger.description}",
                        "type_line": "Ability",
                    }
                )
                state.stack.append(trigger_stack_item)
                state.log(f"[TRIGGER] {trigger.description} added to stack")
                
                # Store the trigger for resolution
                state.triggered_abilities.append(trigger)
        
        else:
            # For non-creatures, move to graveyard (they've resolved)
            # TODO: Implement actual spell resolution effects
            state = move_card(state, source_card_id, Zone.STACK, Zone.GRAVEYARD, card.owner_id)
            state.log(f"{card.name} resolves")
        
        return state

    def resolve_stack_item(self, state: GameState) -> GameState:
        """Resolve the top of the stack (spell or ability).
        
        Handles both spells (creatures) and triggered abilities.
        """
        from .triggers import resolve_trigger
        
        if len(state.stack) == 0:
            return state
        
        stack_item = state.stack.pop()
        
        # Check if this is a triggered ability
        if not stack_item.is_spell:
            # This is a triggered ability on the stack
            state.log(f"{stack_item.card_data.get('name', 'Ability')} resolves")
            
            # Find the corresponding trigger
            trigger = next(
                (t for t in state.triggered_abilities if t.source_card_id == stack_item.source_card_id),
                None
            )
            
            if trigger:
                # Apply the trigger's effect
                state = resolve_trigger(state, trigger)
                state.triggered_abilities.remove(trigger)
            
            return state
        
        else:
            # This is a spell, resolve it normally (put it back and call resolve_spell)
            state.stack.append(stack_item)
            return self.resolve_spell(state, stack_item)

    def check_state_based_actions(self, state: GameState) -> list[str]:
        """CR 704 — check and apply state-based actions.

        Returns list of events that occurred.
        """
        from .zones import move_card
        
        events: list[str] = []
        
        # Track which players survive
        died_this_check = []

        # Players at 0 or less life lose
        for player in list(state.players):
            if player.life_total <= 0:
                events.append(f"{player.name} loses the game (life <= 0)")
                died_this_check.append(player.player_id)

        # Creatures with lethal damage or 0 toughness die
        for card in list(state.cards):
            if card.zone != Zone.BATTLEFIELD or not card.is_creature():
                continue
            
            toughness = _parse_int(card.toughness)
            if toughness is not None and card.damage_marked >= toughness:
                events.append(f"{card.name} dies (lethal damage)")
                state = move_card(state, card.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, card.owner_id)
                card.damage_marked = 0
            elif toughness is not None and toughness <= 0:
                events.append(f"{card.name} dies (0 toughness)")
                state = move_card(state, card.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, card.owner_id)

        # Commander damage check (21+)
        if state.format == "commander":
            for player in list(state.players):
                if player.player_id in died_this_check:
                    continue
                for cmd_id, dmg in list(player.commander_damage_received.items()):
                    if dmg >= 21:
                        events.append(f"{player.name} loses (21+ commander damage)")
                        died_this_check.append(player.player_id)
                        break

        # Tokens cease to exist when they leave battlefield
        for card in list(state.cards):
            if card.is_token and card.zone != Zone.BATTLEFIELD:
                if card in state.cards:
                    state.cards.remove(card)
                    events.append(f"{card.name} token ceases to exist")

        # Remove dead players from the game
        orig_count = len(state.players)
        state.players = [p for p in state.players if p.player_id not in died_this_check]

        # Game ends if only one player remains or fewer
        if len(state.players) == 1:
            state.game_over = True
            state.winner = state.players[0]
            events.append(f"{state.players[0].name} wins the game!")
        elif len(state.players) == 0:
            state.game_over = True
            events.append("Draw!")

        return events


def _parse_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None

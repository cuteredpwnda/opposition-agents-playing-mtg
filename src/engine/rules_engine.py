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
from .mana import auto_tap_for_cost, can_pay, can_pay_with_lands, pay_cost, parse_mana_cost
from .abilities import (
    parse_abilities,
    get_legal_activated_abilities,
    is_mana_ability,
    resolve_ability
)
from .spell_effects import apply_spell_effect, auto_pick_targets


def _get_color_identity(card):
    if not card:
        return set()
    ci = card.card_data.get("color_identity") or card.card_data.get("colors") or []
    if isinstance(ci, str):
        return set(ci.upper())
    return set(ci)


def _commander_color_identity(state: GameState, player_id: str) -> set:
    if state.format != "commander":
        return set()
    commander_id = getattr(state, "commanders", {}).get(player_id, None)
    if commander_id is None:
        return set()
    commander_card = next((c for c in state.cards if c.instance_id == commander_id), None)
    return _get_color_identity(commander_card)


def _check_color_identity(state: GameState, player_id: str, card) -> bool:
    if state.format != "commander":
        return True
    commander_ci = _commander_color_identity(state, player_id)
    if not commander_ci:
        return True
    card_ci = _get_color_identity(card)
    return card_ci.issubset(commander_ci)


def _get_effective_cost(state: GameState, player: "PlayerState", card) -> dict[str, int]:
    cost = parse_mana_cost(card.mana_cost or "")
    # Commander tax: +2 generic for each previous time commander was cast from command zone
    if state.format == "commander" and card.zone == Zone.COMMAND_ZONE:
        cost["generic"] = cost.get("generic", 0) + (player.commander_tax * 2)
    return cost


def _parse_mana_cost_safe(text: str) -> dict[str, int]:
    try:
        return parse_mana_cost(text)
    except Exception:
        return {}


def _can_afford(state: GameState, player, cost: dict[str, int]) -> bool:
    if can_pay(player, cost):
        return True
    snapshot = dict(player.mana_pool)
    auto_tap_for_cost(state, player, cost)
    ok = can_pay(player, cost)
    player.mana_pool = snapshot
    return ok


def _safe_power(card) -> int:
    try:
        return int(card.power) if card.power is not None else 0
    except (TypeError, ValueError):
        return 0



class RulesEngine:
    """Validates and executes game actions according to MTG Comprehensive Rules."""

    def _card_color_identity(self, card: CardInstance) -> set[str]:
        identity = set(card.card_data.get("color_identity", []))
        if not identity:
            for c in card.card_data.get("colors", []):
                identity.add(c)
        return identity

    def _is_legal_for_commander(self, state: GameState, player_id: str, card: CardInstance) -> bool:
        if state.format != "commander":
            return True

        commanders = getattr(state, "commanders", {}) or {}
        commander_id = commanders.get(player_id)
        if not commander_id:
            return True

        commander_card = next((c for c in state.cards if c.instance_id == commander_id), None)
        if not commander_card:
            return True

        cmd_identity = self._card_color_identity(commander_card)
        card_identity = self._card_color_identity(card)
        if not card_identity:
            return True

        # Commander color identity is inclusive: card must be subset
        return card_identity.issubset(cmd_identity)

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

        # Offer concede as a legal action for early termination / debug modes
        actions.append(Action(action_type=ActionType.CONCEDE, player_id=player_id))

        hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == player_id]
        battlefield = [c for c in state.cards if c.zone == Zone.BATTLEFIELD and c.owner_id == player_id]
        command_zone = [c for c in state.cards if c.zone == Zone.COMMAND_ZONE and c.owner_id == player_id]

        def commander_color_identity() -> set[str]:
            if state.format != "commander":
                return set()
            commander_id = getattr(state, "commanders", {}).get(player_id, "")
            commander_card = next((c for c in state.cards if c.instance_id == commander_id), None)
            if commander_card is None:
                return set()
            return set(commander_card.card_data.get("color_identity", []))

        def legal_for_commander(card):
            if state.format != "commander":
                return True
            card_id = set(card.card_data.get("color_identity", []))
            if not card_id:
                return True
            cmd_id = commander_color_identity()
            return card_id.issubset(cmd_id)

        def cost_with_commander_tax(card_cost):
            if state.format != "commander":
                return card_cost
            # commander tax building may be in player state
            if player.commander_tax > 0:
                card_cost = dict(card_cost)
                card_cost["generic"] = card_cost.get("generic", 0) + player.commander_tax
            return card_cost

        # Sorcery-speed actions: only during own main phase with empty stack
        if (
            state.players[state.active_player_index].player_id == player_id
            and is_main_phase(state.phase)
            and stack_is_empty(state)
        ):
            # Play a land (once per turn)
            if player.land_plays_remaining > 0:
                for card in hand:
                    if card.is_land() and _check_color_identity(state, player_id, card) and self._is_legal_for_commander(state, player_id, card):
                        actions.append(
                            Action(
                                action_type=ActionType.PLAY_LAND,
                                player_id=player_id,
                                card_instance_id=card.instance_id,
                            )
                        )

            # Cast sorceries and creatures (non-instant, non-flash)
            castable_cards = [
                c for c in state.cards
                if c.owner_id == player_id and c.zone in (Zone.HAND, Zone.COMMAND_ZONE)
            ]
            for card in castable_cards:
                if card.is_land():
                    continue  # Land is handled separately
                if card.is_instant():
                    continue  # Instant-speed only
                if "flash" in card.oracle_text.lower():
                    continue  # Flash is instant-speed
                if not _check_color_identity(state, player_id, card):
                    continue
                if not self._is_legal_for_commander(state, player_id, card):
                    continue

                cost = _get_effective_cost(state, player, card)
                if can_pay_with_lands(state, player, cost):
                    actions.append(
                        Action(
                            action_type=ActionType.CAST_SPELL,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                        )
                    )

        # Instant-speed actions: any time you have priority
        # This includes instants, flash creatures, activated abilities, etc.
        castable_cards = [
            c for c in state.cards
            if c.owner_id == player_id and c.zone in (Zone.HAND, Zone.COMMAND_ZONE)
        ]
        for card in castable_cards:
            if card.is_instant() or "flash" in card.oracle_text.lower():
                if not _check_color_identity(state, player_id, card):
                    continue
                if not self._is_legal_for_commander(state, player_id, card):
                    continue

                cost = _get_effective_cost(state, player, card)
                if can_pay_with_lands(state, player, cost):
                    actions.append(
                        Action(
                            action_type=ActionType.CAST_SPELL,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                        )
                    )

        # Activated abilities of permanents (skip mana abilities — those are
        # implicit during cast resolution; surfacing them as discrete actions
        # would let naive agents livelock the priority loop tapping lands).
        for card in battlefield:
            legal_abilities = get_legal_activated_abilities(state, card, player_id)
            for ability in legal_abilities:
                if ability.can_use_any_time:  # mana abilities are flagged here
                    continue
                actions.append(
                    Action(
                        action_type=ActionType.ACTIVATE_ABILITY,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                        metadata={"ability_id": ability.ability_id},
                    )
                )

        # Cycling: any card in hand with "cycling {cost}" can be discarded
        # to draw a card (CR 702.32). Available at instant speed any time
        # the player has priority.
        from .cycling import parse_cycling_cost, can_pay_cycling
        for card in hand:
            cyc_cost = parse_cycling_cost(card)
            if cyc_cost is not None and can_pay_cycling(state, player, cyc_cost):
                actions.append(
                    Action(
                        action_type=ActionType.SPECIAL_ACTION,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                        metadata={"special": "cycle"},
                    )
                )

        # Flashback (CR 702.34): cast eligible spells from graveyard for the
        # flashback cost. Sorcery-speed for sorceries, instant-speed for
        # instants.
        from .equip import (
            parse_flashback_cost, can_flashback, parse_equip_cost,
            parse_crew_cost, parse_kicker_cost, kicker_cost_dict,
        )
        graveyard = [c for c in state.cards if c.zone == Zone.GRAVEYARD and c.owner_id == player_id]
        for card in graveyard:
            fb_cost = parse_flashback_cost(card)
            if fb_cost is None:
                continue
            sorcery_speed = not card.is_instant() and "flash" not in (card.oracle_text or "").lower()
            ok_timing = (
                (state.players[state.active_player_index].player_id == player_id
                 and is_main_phase(state.phase)
                 and stack_is_empty(state))
                if sorcery_speed
                else True
            )
            if not ok_timing:
                continue
            if can_flashback(card, player, state):
                actions.append(
                    Action(
                        action_type=ActionType.SPECIAL_ACTION,
                        player_id=player_id,
                        card_instance_id=card.instance_id,
                        metadata={"special": "flashback"},
                    )
                )

        # Equip {N} (CR 702.6): activated ability of an Equipment.
        # Sorcery-speed only.
        if (
            state.players[state.active_player_index].player_id == player_id
            and is_main_phase(state.phase)
            and stack_is_empty(state)
        ):
            for equipment in battlefield:
                eq_cost = parse_equip_cost(equipment)
                if eq_cost is None:
                    continue
                cost = _parse_mana_cost_safe(eq_cost)
                if not _can_afford(state, player, cost):
                    continue
                # Surface one action per legal target creature.
                for target in battlefield:
                    if not target.is_creature():
                        continue
                    if target.instance_id == equipment.instance_id:
                        continue
                    if equipment.attached_to == target.instance_id:
                        continue
                    actions.append(
                        Action(
                            action_type=ActionType.SPECIAL_ACTION,
                            player_id=player_id,
                            card_instance_id=equipment.instance_id,
                            targets=[target.instance_id],
                            metadata={"special": "equip"},
                        )
                    )

            # Crew {N} (CR 702.121): tap creatures totalling N power.
            for vehicle in battlefield:
                crew_n = parse_crew_cost(vehicle)
                if crew_n is None:
                    continue
                # Greedy crew set: smallest creatures first that hit N power.
                creatures = sorted(
                    [c for c in battlefield if c.is_creature() and not c.tapped
                     and c.instance_id != vehicle.instance_id],
                    key=lambda c: _safe_power(c),
                )
                picked: list[str] = []
                total = 0
                for c in creatures:
                    if total >= crew_n:
                        break
                    picked.append(c.instance_id)
                    total += _safe_power(c)
                if total >= crew_n and picked:
                    actions.append(
                        Action(
                            action_type=ActionType.SPECIAL_ACTION,
                            player_id=player_id,
                            card_instance_id=vehicle.instance_id,
                            targets=picked,
                            metadata={"special": "crew"},
                        )
                    )

        # Kicker {cost} (CR 702.33): surface a *kicked* alternative for any
        # castable spell with a kicker cost we can afford.
        if (
            state.players[state.active_player_index].player_id == player_id
            and is_main_phase(state.phase)
            and stack_is_empty(state)
        ):
            for card in hand:
                if card.is_land() or card.is_instant():
                    continue
                kc = kicker_cost_dict(card)
                if kc is None:
                    continue
                base_cost = _get_effective_cost(state, player, card)
                merged = dict(base_cost)
                for k, v in kc.items():
                    merged[k] = merged.get(k, 0) + v
                if can_pay_with_lands(state, player, merged):
                    actions.append(
                        Action(
                            action_type=ActionType.CAST_SPELL,
                            player_id=player_id,
                            card_instance_id=card.instance_id,
                            metadata={"kicked": True},
                        )
                    )

        # Treasure: tap+sacrifice for {C} (or any color). Surface once per
        # untapped Treasure on the battlefield as a mana ability.
        for treasure in battlefield:
            if "Treasure" not in treasure.type_line or treasure.tapped:
                continue
            actions.append(
                Action(
                    action_type=ActionType.SPECIAL_ACTION,
                    player_id=player_id,
                    card_instance_id=treasure.instance_id,
                    metadata={"special": "sacrifice_treasure"},
                )
            )

        # Combat actions: declare attackers/blockers during combat phases
        from .game_state import Phase
        from .combat import can_attack, can_block
        if state.phase == Phase.COMBAT_ATTACKERS and state.players[state.active_player_index].player_id == player_id:
            opponents = [p.player_id for p in state.players if p.player_id != player_id]
            for card in battlefield:
                if can_attack(card, state.turn_number):
                    for defender_id in opponents:
                        actions.append(
                            Action(
                                action_type=ActionType.DECLARE_ATTACKERS,
                                player_id=player_id,
                                card_instance_id=card.instance_id,
                                targets=[defender_id],
                            )
                        )

        if state.phase == Phase.COMBAT_BLOCKERS and state.players[state.active_player_index].player_id != player_id and state.combat is not None:
            # Defender enumerates legal (attacker, blocker) pairs.
            for attacker_id in state.combat.attackers.keys():
                attacker = next((c for c in state.cards if c.instance_id == attacker_id), None)
                if attacker is None:
                    continue
                for card in battlefield:
                    if can_block(attacker, card):
                        actions.append(
                            Action(
                                action_type=ActionType.DECLARE_BLOCKERS,
                                player_id=player_id,
                                card_instance_id=card.instance_id,
                                targets=[attacker_id],
                            )
                        )

        return actions

    def execute_action(self, state: GameState, action: Action) -> GameState:
        """Apply an action, update state, check SBAs, queue triggers."""
        from .zones import move_card
        
        if action.action_type == ActionType.PASS_PRIORITY:
            # No state change on pass
            if action.metadata:
                state.log(f"{action.player_id} pass metadata: {action.metadata}")
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
                    # Replacement effect: "enters the battlefield tapped"
                    if "enters the battlefield tapped" in card.oracle_text.lower() \
                            or "enters tapped" in card.oracle_text.lower():
                        card.tapped = True
                        state.log(f"{card.name} enters tapped")
                    else:
                        card.tapped = False
                    player = next((p for p in state.players if p.player_id == action.player_id), None)
                    if player:
                        player.land_plays_remaining -= 1

                    # Landfall: trigger on every permanent the player controls
                    # whose oracle text contains "landfall" (CR 702.124).
                    from .triggers import check_landfall_triggers
                    landfall = check_landfall_triggers(state, action.player_id, card)
                    for trig in landfall:
                        stack_item = StackItem(
                            source_card_id=trig.source_card_id,
                            controller_id=trig.controller_id,
                            is_spell=False,
                            card_data={
                                "name": f"[Landfall] {trig.description}",
                                "type_line": "Ability",
                            },
                        )
                        state.stack.append(stack_item)
                        state.triggered_abilities.append(trig)
                        state.log(f"[TRIGGER (Landfall)] {trig.description}")
            return state
        
        if action.action_type == ActionType.CAST_SPELL:
            if action.metadata:
                state.log(f"{action.player_id} cast metadata: {action.metadata}")
            if action.card_instance_id:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    player = next((p for p in state.players if p.player_id == action.player_id), None)
                    if player is None:
                        return state

                    if not _check_color_identity(state, action.player_id, card):
                        return state

                    cost = _get_effective_cost(state, player, card)
                    # Kicker: if action metadata says kicked, fold the kicker
                    # cost into the effective cost (CR 702.33).
                    if (action.metadata or {}).get("kicked"):
                        from .equip import kicker_cost_dict
                        kc = kicker_cost_dict(card) or {}
                        for k, v in kc.items():
                            cost[k] = cost.get(k, 0) + v
                    # Auto-tap untapped lands so naive agents don't have to
                    # explicitly activate mana abilities before each cast.
                    if not can_pay(player, cost):
                        auto_tap_for_cost(state, player, cost)
                    if not can_pay(player, cost):
                        return state

                    # Additional costs (CR 601.2f). We currently support the
                    # most common pattern: "As an additional cost to cast
                    # this spell, sacrifice <a/an/N> <type>." The agent
                    # consents to the cost by casting the spell at all; we
                    # pick the cheapest valid permanent automatically.
                    extra_sacs = _parse_additional_sac_cost(card.oracle_text or "")
                    sac_targets: list[str] = []
                    for required_type, count in extra_sacs:
                        candidates = [
                            c for c in state.cards
                            if c.zone == Zone.BATTLEFIELD
                            and c.controller_id == player.player_id
                            and c.instance_id != card.instance_id
                            and required_type.lower() in c.type_line.lower()
                        ]
                        # Prefer cheapest, prefer non-creatures (keep board).
                        candidates.sort(
                            key=lambda c: (c.is_creature(), int(c.card_data.get("cmc", 0) or 0))
                        )
                        if len(candidates) < count:
                            state.log(
                                f"    {card.name} cannot be cast: need to sacrifice "
                                f"{count} {required_type}, only {len(candidates)} available"
                            )
                            return state
                        sac_targets.extend(c.instance_id for c in candidates[:count])

                    pay_cost(player, cost)

                    # Pay additional sacrifice cost(s) now.
                    for sid in sac_targets:
                        scard = next((c for c in state.cards if c.instance_id == sid), None)
                        if scard is None:
                            continue
                        state.log(
                            f"    \u2716 {player.name} sacrifices {scard.name} "
                            f"(additional cost of {card.name})"
                        )
                        state = move_card(
                            state, sid, Zone.BATTLEFIELD, Zone.GRAVEYARD, player.player_id
                        )

                    # Commander tax is applied when casting from command zone
                    if state.format == "commander" and card.zone == Zone.COMMAND_ZONE:
                        player.commander_tax += 1

                    # Move card from current zone to stack
                    state = move_card(
                        state, action.card_instance_id, card.zone, Zone.STACK,
                        action.player_id
                    )

                    # Pick targets if the agent didn't supply any.
                    targets = list(action.targets) if action.targets else auto_pick_targets(
                        state, card, action.player_id
                    )

                    # Validate targets against hexproof/shroud/protection;
                    # if illegal targets remain, refuse to cast (the spell
                    # has no legal targets, CR 601.2c).
                    from .keywords import (
                        can_be_targeted,
                        ward_cost as _ward_cost,
                        apply_prowess_on_cast,
                    )
                    valid_targets: list[str] = []
                    extra_ward_cost = 0
                    for tid in targets:
                        tcard = next((c for c in state.cards if c.instance_id == tid), None)
                        if tcard is None:
                            # Player or stack-item id — accept as-is.
                            valid_targets.append(tid)
                            continue
                        if not can_be_targeted(tcard, card, action.player_id):
                            state.log(f"{card.name} cannot target {tcard.name} (hexproof/shroud/protection)")
                            continue
                        # Ward (CR 702.21): caster pays extra generic mana or
                        # the spell is countered. We charge it as additional
                        # generic cost from the pool/lands; fail to cast if
                        # the controller can't pay.
                        if tcard.controller_id != action.player_id:
                            wc = _ward_cost(tcard)
                            extra_ward_cost += wc
                        valid_targets.append(tid)

                    if targets and not valid_targets:
                        state.log(f"{card.name} fizzles (no legal targets)")
                        return state

                    if extra_ward_cost > 0:
                        ward_pay_cost = {"generic": extra_ward_cost}
                        if not can_pay(player, ward_pay_cost):
                            auto_tap_for_cost(state, player, ward_pay_cost)
                        if not can_pay(player, ward_pay_cost):
                            state.log(f"{card.name} cannot pay ward {{{extra_ward_cost}}} — countered")
                            return state
                        pay_cost(player, ward_pay_cost)
                        state.log(f"{card.name} pays ward {{{extra_ward_cost}}}")

                    targets = valid_targets

                    # Create stack item and push to stack
                    stack_item = StackItem(
                        source_card_id=action.card_instance_id,
                        controller_id=action.player_id,
                        is_spell=True,
                        targets=targets,
                        card_data=card.card_data.copy(),
                    )
                    state = push_to_stack(state, stack_item)
                    # Prowess (CR 702.108): non-creature spells boost
                    # prowess creatures controller controls.
                    apply_prowess_on_cast(state, action.player_id, card)
                    caster = next(
                        (p for p in state.players if p.player_id == action.player_id),
                        None,
                    )
                    cname = (caster.name or caster.player_id) if caster else action.player_id
                    if targets:
                        readable: list[str] = []
                        for tid in targets:
                            tcard = next((c for c in state.cards if c.instance_id == tid), None)
                            if tcard is not None:
                                tplayer = next(
                                    (p for p in state.players
                                     if p.player_id == tcard.controller_id),
                                    None,
                                )
                                tlabel = (tplayer.name or tplayer.player_id) if tplayer else tcard.controller_id
                                readable.append(f"{tcard.name} ({tlabel})")
                                continue
                            tplayer = next(
                                (p for p in state.players if p.player_id == tid),
                                None,
                            )
                            if tplayer is not None:
                                readable.append(
                                    f"{tplayer.name or tplayer.player_id} "
                                    f"(life={tplayer.life_total})"
                                )
                                continue
                            readable.append(tid)
                        target_phrase = " targeting " + ", ".join(readable)
                    else:
                        target_phrase = ""
                    state.log(f"    \u25b6 {cname} casts {card.name}{target_phrase}")
            return state
        
        if action.action_type == ActionType.ACTIVATE_ABILITY:
            if action.card_instance_id:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    # Get abilities on this card
                    abilities = parse_abilities(card)

                    # Find the specific ability to activate (if provided in metadata)
                    ability_to_activate = None
                    ability_id = action.metadata.get("ability_id") if action.metadata else None
                    if ability_id:
                        ability_to_activate = next(
                            (a for a in abilities if a.ability_id == ability_id),
                            None
                        )
                        if ability_to_activate is None:
                            legal = get_legal_activated_abilities(state, card, action.player_id)
                            ability_to_activate = next(
                                (a for a in legal if a.ability_id == ability_id),
                                None
                            )
                    else:
                        # If no specific ability, use first legal one (for legacy compatibility)
                        legal = get_legal_activated_abilities(state, card, action.player_id)
                        ability_to_activate = legal[0] if legal else None
                    
                    if ability_to_activate:
                        # Resolve the ability
                        state = resolve_ability(state, ability_to_activate, action.player_id)
                        
                        # Non-mana abilities go on the stack
                        if not is_mana_ability(ability_to_activate.effect):
                            stack_item = StackItem(
                                source_card_id=card.instance_id,
                                controller_id=action.player_id,
                                is_spell=False,  # It's an ability
                                card_data={
                                    "name": f"[Ability] {card.name}",
                                    "type_line": "Ability",
                                }
                            )
                            state = push_to_stack(state, stack_item)
                            state.log(f"{card.name} ability activated")
                        else:
                            # Mana abilities resolve immediately
                            state.log(f"{card.name} mana ability activated")
            return state

        if action.action_type == ActionType.SPECIAL_ACTION:
            special = (action.metadata or {}).get("special")
            if special == "cycle" and action.card_instance_id:
                from .cycling import execute_cycle
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                player = next((p for p in state.players if p.player_id == action.player_id), None)
                if card and player:
                    state = execute_cycle(state, card, player)
            elif special == "equip" and action.card_instance_id and action.targets:
                from .equip import execute_equip
                eq = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                tgt = next((c for c in state.cards if c.instance_id == action.targets[0]), None)
                if eq and tgt:
                    execute_equip(state, eq, tgt)
            elif special == "crew" and action.card_instance_id and action.targets:
                from .equip import execute_crew
                vehicle = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                crew = [c for c in state.cards if c.instance_id in action.targets]
                if vehicle and crew:
                    execute_crew(state, vehicle, crew)
            elif special == "flashback" and action.card_instance_id:
                state = self._execute_flashback(state, action)
            elif special == "sacrifice_treasure" and action.card_instance_id:
                from .tokens import sacrifice_for_mana
                tok = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                if tok:
                    color = (action.metadata or {}).get("color", "C")
                    sacrifice_for_mana(state, tok, color)
            return state
        
        if action.action_type == ActionType.DECLARE_ATTACKERS:
            if action.card_instance_id and action.targets:
                card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
                defender_id = action.targets[0]
                if card:
                    from .combat import declare_attackers, has_kw
                    from .triggers import check_attack_triggers

                    # Ensure CombatState exists (priority loop normally creates
                    # this in COMBAT_BEGIN, but tests/legal-action paths may
                    # call execute_action without that hook).
                    if state.combat is None:
                        from .combat import begin_combat
                        begin_combat(state)

                    declare_attackers(state, {action.card_instance_id: defender_id})
                    # `declare_attackers` already taps unless vigilance — don't
                    # double-tap here.

                    # ATTACKS TRIGGERS
                    attack_triggers = check_attack_triggers(state, card)
                    for trigger in attack_triggers:
                        trigger_stack_item = StackItem(
                            source_card_id=trigger.source_card_id,
                            controller_id=trigger.controller_id,
                            is_spell=False,
                            card_data={
                                "name": f"[Trigger] {trigger.description}",
                                "type_line": "Ability",
                            }
                        )
                        state.stack.append(trigger_stack_item)
                        state.log(f"[TRIGGER (ATTACK)] {trigger.description} added to stack")
                        state.triggered_abilities.append(trigger)
            return state

        if action.action_type == ActionType.DECLARE_BLOCKERS:
            if action.card_instance_id and action.targets:
                from .combat import declare_blockers
                attacker_id = action.targets[0]
                if state.combat is None:
                    return state
                # Append to existing blocker list for this attacker.
                existing = list(state.combat.blockers.get(attacker_id, []))
                if action.card_instance_id not in existing:
                    existing.append(action.card_instance_id)
                declare_blockers(state, {attacker_id: existing})
            return state

        # Default: no change
        return state

    def resolve_spell(self, state: GameState, stack_item) -> GameState:
        """Resolve a spell from the stack.
        
        For creatures, this moves the spell to the battlefield.
        For other spells, effects are applied (not implemented in Phase 1).
        Triggers cast and death abilities as appropriate.
        """
        from .zones import move_card
        from .triggers import (
            check_enters_battlefield_triggers,
            check_cast_triggers,
            resolve_trigger
        )
        
        source_card_id = stack_item.source_card_id
        card = next((c for c in state.cards if c.instance_id == source_card_id), None)
        
        if not card:
            return state
        
        # Check if this is still in the stack (wasn't fizzled)
        if card.zone != Zone.STACK:
            return state
        
        # CAST TRIGGERS: Fire "when you cast" triggers from permanents
        # This happens before the spell resolves
        controller_id = card.controller_id
        cast_triggers = check_cast_triggers(state, controller_id, card)
        
        for trigger in cast_triggers:
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
            state.log(f"[TRIGGER (CAST)] {trigger.description} added to stack")
            state.triggered_abilities.append(trigger)
        
        # If it's a creature, move to battlefield with summoning sickness
        if card.is_creature():
            state = move_card(state, source_card_id, Zone.STACK, Zone.BATTLEFIELD, card.owner_id)
            card.summoning_sick = True
            card.turn_entered = state.turn_number
            state.log(f"{card.name} enters the battlefield")
            
            # ETB TRIGGERS: Check what ETB triggers should fire
            etb_triggers = check_enters_battlefield_triggers(state, card)
            
            for trigger in etb_triggers:
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
                state.log(f"[TRIGGER (ETB)] {trigger.description} added to stack")
                
                # Store the trigger for resolution
                state.triggered_abilities.append(trigger)
        
        else:
            type_line = (card.card_data.get("type_line") or "").lower()
            is_permanent = any(t in type_line for t in
                               ("artifact", "enchantment", "planeswalker", "battle", "land"))
            if is_permanent:
                # Permanent spell: enters the battlefield (CR 608.3).
                state = move_card(
                    state, source_card_id, Zone.STACK, Zone.BATTLEFIELD,
                    card.owner_id,
                )
                card.summoning_sick = True
                card.turn_entered = state.turn_number
                # Auras attach to their target if one was chosen.
                if "aura" in type_line and stack_item.targets:
                    card.attached_to = stack_item.targets[0]
                # ETB triggers fire for any permanent.
                etb_triggers = check_enters_battlefield_triggers(state, card)
                for trigger in etb_triggers:
                    trigger_stack_item = StackItem(
                        source_card_id=trigger.source_card_id,
                        controller_id=trigger.controller_id,
                        is_spell=False,
                        card_data={
                            "name": f"[Trigger] {trigger.description}",
                            "type_line": "Ability",
                        },
                    )
                    state.stack.append(trigger_stack_item)
                    state.log(f"[TRIGGER (ETB)] {trigger.description} added to stack")
                    state.triggered_abilities.append(trigger)
                state.log(f"    ◀ {card.name} resolves")
            else:
                # Instant / Sorcery: apply effect, then to graveyard.
                state = apply_spell_effect(state, stack_item)
                # If the spell wasn't already moved (e.g., counter spells
                # don't move themselves), send it to the graveyard now.
                if card.zone == Zone.STACK:
                    # Flashback: exile instead of graveyard (CR 702.34).
                    dest = Zone.EXILE if card.card_data.get("flashback_exile") else Zone.GRAVEYARD
                    state = move_card(state, source_card_id, Zone.STACK, dest, card.owner_id)
                    if dest == Zone.EXILE:
                        card.card_data.pop("flashback_exile", None)
                state.log(f"    ◀ {card.name} resolves")

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
            # This is a spell. resolve_spell expects the StackItem to no longer
            # be on state.stack (it operates by looking up the card by id and
            # moving it from Zone.STACK to its destination). We already popped
            # it above, so just resolve directly.
            return self.resolve_spell(state, stack_item)

    def _execute_flashback(self, state: GameState, action: Action) -> GameState:
        """Cast a spell from graveyard via flashback (CR 702.34).

        Flashback exiles the card on resolution instead of going to
        graveyard. We mark it via card_data['flashback_exile'] so
        ``resolve_spell`` does the right thing.
        """
        from .equip import parse_flashback_cost
        from .zones import move_card

        card = next((c for c in state.cards if c.instance_id == action.card_instance_id), None)
        player = next((p for p in state.players if p.player_id == action.player_id), None)
        if card is None or player is None:
            return state
        cost_text = parse_flashback_cost(card)
        if cost_text is None:
            return state
        cost = parse_mana_cost(cost_text)
        if not can_pay(player, cost):
            auto_tap_for_cost(state, player, cost)
        if not can_pay(player, cost):
            state.log(f"{player.name} cannot pay flashback {cost_text} for {card.name}")
            return state
        pay_cost(player, cost)
        card.card_data["flashback_exile"] = True
        # Move from graveyard to stack.
        move_card(state, card.instance_id, Zone.GRAVEYARD, Zone.STACK, action.player_id)
        stack_item = StackItem(
            source_card_id=card.instance_id,
            controller_id=action.player_id,
            is_spell=True,
            targets=list(action.targets) if action.targets else auto_pick_targets(state, card, action.player_id),
            card_data=card.card_data.copy(),
        )
        push_to_stack(state, stack_item)
        state.log(f"{card.name} is flashed back")
        return state

    def check_state_based_actions(self, state: GameState) -> list[str]:
        """CR 704 — check and apply state-based actions.

        Returns list of events that occurred.
        Also fires death triggers when creatures die.
        """
        from .zones import move_card
        from .triggers import check_death_triggers
        from .counters import apply_counter_sbas

        events: list[str] = []

        # Counter SBAs: cancel +1/+1 vs -1/-1, then poison-loss check.
        events.extend(apply_counter_sbas(state))

        # Track which players survive
        died_this_check = []

        # Players at 0 or less life lose
        for player in list(state.players):
            if player.life_total <= 0:
                events.append(f"{player.name} loses the game (life <= 0)")
                died_this_check.append(player.player_id)

        # Creatures with lethal damage or 0 toughness die.
        # Indestructible (CR 702.12) creatures ignore lethal damage and "destroy"
        # but still die from 0 (or less) toughness.
        for card in list(state.cards):
            if card.zone != Zone.BATTLEFIELD or not card.is_creature():
                continue

            from .counters import effective_toughness
            base_toughness = _parse_int(card.toughness)
            eff_toughness = effective_toughness(card)
            indestructible = "indestructible" in (card.oracle_text or "").lower()
            creature_dies = False

            # Lethal damage uses *effective* toughness (CR 704.5g).
            if base_toughness is not None and card.damage_marked >= eff_toughness and not indestructible:
                events.append(f"{card.name} dies (lethal damage)")
                creature_dies = True

            elif base_toughness is not None and eff_toughness <= 0:
                events.append(f"{card.name} dies (0 toughness)")
                creature_dies = True
            
            # If creature dies, fire death triggers and move to graveyard
            if creature_dies:
                # DEATH TRIGGERS: Fire "when creature dies" triggers
                death_triggers = check_death_triggers(state, card)
                
                for trigger in death_triggers:
                    trigger_stack_item = StackItem(
                        source_card_id=trigger.source_card_id,
                        controller_id=trigger.controller_id,
                        is_spell=False,  # It's an ability
                        card_data={
                            "name": f"[Trigger] {trigger.description}",
                            "type_line": "Ability",
                        }
                    )
                    state.stack.append(trigger_stack_item)
                    state.log(f"[TRIGGER (DEATH)] {trigger.description} added to stack")
                    state.triggered_abilities.append(trigger)
                
                # Commander goes to command zone; others go to graveyard
                if state.format == "commander" and card.instance_id == getattr(state, "commanders", {}).get(card.owner_id):
                    state = move_card(state, card.instance_id, Zone.BATTLEFIELD, Zone.COMMAND_ZONE, card.owner_id)
                    state.log(f"{card.name} returns to the command zone")
                    events.append(f"{card.name} returns to the command zone")
                else:
                    state = move_card(state, card.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, card.owner_id)

                card.damage_marked = 0

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

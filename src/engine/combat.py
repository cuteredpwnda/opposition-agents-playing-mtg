"""
Combat system — declare attackers, declare blockers, assign damage.

Supports the following CR 702 evergreen keywords:
- Flying / Reach (block legality)
- Shadow (CR 702.28) — only shadow blocks shadow
- Fear (CR 702.35) — only artifact or black can block
- Intimidate (CR 702.74) — only artifact or same color can block
- Horsemanship (CR 702.42) — only horsemanship blocks
- Skulk (CR 702.116) — can't be blocked by power > attacker's power
- Menace (block legality: requires 2+ blockers)
- Vigilance (attacker doesn't tap)
- Haste (ignores summoning sickness)
- First strike / Double strike (two-step damage)
- Trample (excess damage to defending player)
- Lifelink (controller gains life equal to damage dealt)
- Deathtouch (any damage from this source is lethal to creatures)
- Indestructible (survives lethal damage and "destroy" effects)
- Infect (CR 702.90) — damage as -1/-1 to creatures, poison to players
- Wither (CR 702.80) — damage as -1/-1 to creatures
- Flanking (CR 702.25) — blockers without flanking get -1/-1 until EOT
- Annihilator N (CR 702.86) — defender sacrifices N permanents on attack
- Exalted (CR 702.90) — +1/+1 if only creature attacking
- Battle cry (CR 702.91) — other attackers get +1/+0
- Toxic N (modern phyrexia) — damage also gives N poison counters
- Myriad (CR 702.116) — creates token copies attacking other opponents

References:
- open-mtg (MIT): CR 509/510-compliant damage assignment ordering
- mtg-python-engine (MIT): handle_combat_phase() pattern
"""

from __future__ import annotations

import random

from .game_state import CardInstance, CombatState, GameState, Zone
from .keywords import (
    has as has_kw,
    can_be_targeted,
    effective_power,
    effective_toughness,
    can_be_blocked_by,
    annihilator_count,
    toxic_count,
)


_KEYWORDS = (
    "flying", "reach", "menace", "vigilance", "haste",
    "first strike", "double strike", "trample", "lifelink",
    "deathtouch", "indestructible", "defender",
    "shadow", "fear", "intimidate", "horsemanship", "skulk",
    "infect", "wither", "flanking", "exalted", "battle cry", "myriad",
)


def _fire_damage_triggers(state: GameState, source: CardInstance, target_kind: str) -> None:
    """Push 'whenever ~ deals damage to a player/creature' triggers and
    resolve them immediately so combat math reflects their effects."""
    from .triggers import check_damage_triggers, resolve_trigger
    from .game_state import StackItem
    triggers = check_damage_triggers(state, source, target_kind)
    for trig in triggers:
        item = StackItem(
            source_card_id=trig.source_card_id,
            controller_id=trig.controller_id,
            is_spell=False,
            card_data={"name": f"[Trigger] {source.name}: {trig.description}",
                       "type_line": "Ability"},
        )
        state.log(f"[TRIGGER (DAMAGE)] {source.name}: {trig.description} added to stack")
        # Resolve directly (no priority window in this minimal engine).
        state.triggered_abilities.append(trig)
        resolve_trigger(state, trig)
        try:
            state.triggered_abilities.remove(trig)
        except ValueError:
            pass
        state.log(f"[Trigger] {trig.description} resolves")


def can_attack(card: CardInstance, turn_number: int) -> bool:
    if card is None or not card.is_creature():
        return False
    if card.zone != Zone.BATTLEFIELD or card.tapped:
        return False
    if has_kw(card, "defender"):
        return False
    # Summoning sickness: cleared by haste.
    if card.summoning_sick and not has_kw(card, "haste"):
        return False
    return True


def can_block(attacker: CardInstance, blocker: CardInstance) -> bool:
    """CR 509: legality of a single blocker for ``attacker``.

    Uses extended keyword logic from ``can_be_blocked_by`` which handles
    shadow, fear, intimidate, skulk, horsemanship, etc.
    """
    return can_be_blocked_by(attacker, blocker)


def menace_satisfied(attacker: CardInstance, blocker_count: int) -> bool:
    if has_kw(attacker, "menace") and blocker_count == 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase entry points
# ---------------------------------------------------------------------------


def begin_combat(state: GameState) -> CombatState:
    combat = CombatState()
    state.combat = combat
    state.log("Combat begins")
    return combat


def _fire_exalted(state: GameState, sole_attacker: CardInstance) -> None:
    """Apply exalted bonuses to the sole attacker (CR 702.90)."""
    exalted_count = 0
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.controller_id == sole_attacker.controller_id and has_kw(card, "exalted"):
            exalted_count += 1
    if exalted_count > 0:
        sole_attacker.eot_power_bonus = getattr(sole_attacker, "eot_power_bonus", 0) + exalted_count
        sole_attacker.eot_toughness_bonus = getattr(sole_attacker, "eot_toughness_bonus", 0) + exalted_count
        state.log(f"Exalted x{exalted_count}: {sole_attacker.name} gets +{exalted_count}/+{exalted_count} until EOT")


def _fire_battle_cry(state: GameState, attacker_ids: list[str]) -> None:
    """Apply battle cry: each attacking creature with battle cry gives
    all *other* attacking creatures +1/+0 until EOT (CR 702.91)."""
    battle_cry_count = 0
    for aid in attacker_ids:
        c = _get_card(state, aid)
        if c and has_kw(c, "battle cry"):
            battle_cry_count += 1
    if battle_cry_count > 0:
        for aid in attacker_ids:
            c = _get_card(state, aid)
            if c:
                c.eot_power_bonus = getattr(c, "eot_power_bonus", 0) + battle_cry_count
        state.log(f"Battle cry x{battle_cry_count}: attacking creatures each get +{battle_cry_count}/+0 until EOT")


def _fire_annihilator(state: GameState, attacker: CardInstance, defender_id: str) -> None:
    """Force defending player to sacrifice N permanents (CR 702.86)."""
    n = annihilator_count(attacker)
    if n <= 0:
        return
    defender = next((p for p in state.players if p.player_id == defender_id), None)
    if defender is None:
        return
    from .zones import move_card
    sacrificed = 0
    # Sacrifice N permanents controlled by the defending player (heuristic: pick lands last)
    candidates = [
        c for c in state.cards
        if c.zone == Zone.BATTLEFIELD and c.controller_id == defender_id and not c.is_token
    ]
    # Sort: sacrifice non-creatures first, then lowest-power creatures
    candidates.sort(key=lambda c: (c.is_creature(), effective_power(c) if c.is_creature() else 0))
    for c in candidates:
        if sacrificed >= n:
            break
        state.log(f"Annihilator {n}: {defender.name} sacrifices {c.name}")
        move_card(state, c.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, c.owner_id)
        sacrificed += 1


def _fire_myriad(state: GameState, attacker: CardInstance, primary_defender_id: str) -> None:
    """Create a copy of the attacker attacking each other opponent (CR 702.116)."""
    if not has_kw(attacker, "myriad"):
        return
    from .game_state import CardInstance as CI
    other_defenders = [
        p.player_id for p in state.players
        if p.player_id != attacker.controller_id and p.player_id != primary_defender_id
    ]
    token_ids: list[str] = []
    for def_id in other_defenders:
        # Create a token copy of the attacker
        token = CI(
            instance_id=f"myriad_{attacker.instance_id}_{def_id}",
            card_data=dict(attacker.card_data),
            zone=Zone.BATTLEFIELD,
            owner_id=attacker.controller_id,
            controller_id=attacker.controller_id,
        )
        token.is_token = True
        token.summoning_sick = False
        token.counters = dict(attacker.counters)
        state.cards.append(token)
        if state.combat:
            state.combat.attackers[token.instance_id] = def_id
        state.log(f"Myriad: {attacker.name} token attacks {def_id}")
        token_ids.append(token.instance_id)
    # Schedule EOT exile of myriad tokens (CR 702.116c)
    if token_ids:
        try:
            from .delayed_triggers import schedule, TriggerPoint
            from .zones import move_card as _mc

            def _exile_tokens(s: GameState, _ids: list[str] = token_ids) -> None:
                for tid in _ids:
                    from .zones import move_card as _mc2
                    try:
                        _mc2(s, tid, Zone.BATTLEFIELD, Zone.EXILE, attacker.controller_id)
                        s.log(f"Myriad: token exiled at end of combat")
                    except Exception:
                        pass

            schedule(
                state, TriggerPoint.END_OF_COMBAT,
                _exile_tokens, attacker.controller_id,
                description=f"Myriad exile tokens for {attacker.name}",
            )
        except Exception:
            pass


def declare_attackers(
    state: GameState, attacker_assignments: dict[str, str]
) -> None:
    """Declare attackers. attacker_assignments: {card_instance_id: defending_player_id}"""
    if state.combat is None:
        return

    declared_ids: list[str] = []
    for attacker_id, defender_id in attacker_assignments.items():
        card = _get_card(state, attacker_id)
        if card is None or not can_attack(card, state.turn_number):
            continue
        # Vigilance: don't tap when attacking.
        if not has_kw(card, "vigilance"):
            card.tapped = True
        state.combat.attackers[attacker_id] = defender_id
        declared_ids.append(attacker_id)
        ep = effective_power(card)
        et = effective_toughness(card)
        state.log(f"{card.name} ({ep}/{et}) attacks {defender_id}")

    # Annihilator triggers — fire for each attacker with annihilator
    for attacker_id, defender_id in list(state.combat.attackers.items()):
        attacker = _get_card(state, attacker_id)
        if attacker:
            _fire_annihilator(state, attacker, defender_id)

    # Myriad — create copies for other opponents
    for attacker_id, defender_id in list(state.combat.attackers.items()):
        attacker = _get_card(state, attacker_id)
        if attacker:
            _fire_myriad(state, attacker, defender_id)

    # Exalted — if exactly one creature attacks, apply exalted bonus
    if len(state.combat.attackers) == 1:
        sole_id = next(iter(state.combat.attackers))
        sole = _get_card(state, sole_id)
        if sole:
            _fire_exalted(state, sole)

    # Battle cry — each attacking battle-cry creature gives others +1/+0
    _fire_battle_cry(state, list(state.combat.attackers.keys()))

    # Training (CR 702.153): if another attacking creature has greater power, +1/+1
    from .triggers import check_training_triggers
    check_training_triggers(state, list(state.combat.attackers.keys()))
    # Watcher notifications (H3) — one event per attacking creature
    try:
        from .watchers import on_creature_attacked
        attacking_player = state.active_player.player_id
        for aid in list(state.combat.attackers.keys()):
            on_creature_attacked(state, attacking_player, source_id=aid)
    except Exception:
        pass


def declare_blockers(
    state: GameState, blocker_assignments: dict[str, list[str]]
) -> None:
    """Declare blockers. blocker_assignments: {attacker_id: [blocker_card_ids]}"""
    if state.combat is None:
        return

    for attacker_id, blocker_ids in blocker_assignments.items():
        if attacker_id not in state.combat.attackers:
            continue
        attacker = _get_card(state, attacker_id)
        if attacker is None:
            continue
        legal: list[str] = []
        for bid in blocker_ids:
            blocker = _get_card(state, bid)
            if blocker and can_block(attacker, blocker):
                legal.append(bid)
                state.log(
                    f"{blocker.name} ({effective_power(blocker)}/{effective_toughness(blocker)})"
                    f" blocks {attacker.name} ({effective_power(attacker)}/{effective_toughness(attacker)})"
                )
        # Enforce menace.
        if not menace_satisfied(attacker, len(legal)):
            state.log(f"{attacker.name} has menace and is blocked by only {len(legal)} — block illegal, ignored")
            legal = []
        state.combat.blockers[attacker_id] = legal

        # Flanking (CR 702.25): each blocker without flanking gets -1/-1 until EOT
        if has_kw(attacker, "flanking"):
            from .counters import add_counter
            for bid in legal:
                blocker = _get_card(state, bid)
                if blocker and not has_kw(blocker, "flanking"):
                    blocker.eot_toughness_bonus = getattr(blocker, "eot_toughness_bonus", 0) - 1
                    blocker.eot_power_bonus = getattr(blocker, "eot_power_bonus", 0) - 1
                    state.log(f"Flanking: {blocker.name} gets -1/-1 until end of turn")


# ---------------------------------------------------------------------------
# Damage resolution (with first/double strike)
# ---------------------------------------------------------------------------


def resolve_combat_damage(state: GameState) -> None:
    """CR 510 damage step. Handles first strike, double strike, trample,
    lifelink, deathtouch, infect, wither, and player damage."""
    if state.combat is None:
        return

    attackers = list(state.combat.attackers.items())

    # Determine if a first-strike step is needed.
    has_first_strike = False
    for attacker_id, _ in attackers:
        if _has_first_or_double(_get_card(state, attacker_id)):
            has_first_strike = True
            break
        for bid in state.combat.blockers.get(attacker_id, []):
            if _has_first_or_double(_get_card(state, bid)):
                has_first_strike = True
                break
        if has_first_strike:
            break

    if has_first_strike:
        _deal_step(state, attackers, first_strike_step=True)
        # SBA between steps would normally apply; defer to caller's SBA loop
        # but remove dead creatures for the regular step.
        _remove_dead(state)
        # Re-fetch attackers (some may have died).
        attackers = [
            (aid, did) for aid, did in attackers
            if _get_card(state, aid) and _get_card(state, aid).zone == Zone.BATTLEFIELD
        ]

    _deal_step(state, attackers, first_strike_step=False)
    state.combat = None


def _deal_infect_or_wither_damage(
    state: GameState, source: CardInstance, target: CardInstance | None,
    player_target, amount: int
) -> None:
    """Deal damage using infect/wither rules.

    * Target is a creature → mark as -1/-1 counters.
    * Target is a player AND source has infect → add poison counters.
    * Target is a player AND source has only wither → deal normal damage.
    """
    from .counters import add_counter, add_poison
    if target is not None:
        # Creature — always -1/-1 counters for infect OR wither
        add_counter(target, "-1/-1", amount)
        state.log(f"{source.name} deals {amount} damage to {target.name} as -1/-1 counters (infect/wither)")
    elif player_target is not None:
        if has_kw(source, "infect"):
            add_poison(player_target, amount)
            state.log(f"{source.name} deals {amount} poison to {player_target.name} (infect)")
        else:
            # Wither doesn't give poison, just normal player damage
            player_target.life_total -= amount
            state.log(f"{source.name} deals {amount} damage to {player_target.name} (wither)")


def _deal_step(
    state: GameState,
    attackers: list[tuple[str, str]],
    first_strike_step: bool,
) -> None:
    for attacker_id, defender_id in attackers:
        attacker = _get_card(state, attacker_id)
        if attacker is None or attacker.zone != Zone.BATTLEFIELD:
            continue
        # Filter participation in this step.
        if first_strike_step and not _has_first_or_double(attacker):
            atk_deals = False
        else:
            atk_deals = True
            # In second step, first-strike-only creatures don't deal again.
            if not first_strike_step and has_kw(attacker, "first strike") and not has_kw(attacker, "double strike"):
                atk_deals = False

        uses_infect = has_kw(attacker, "infect")
        uses_wither = has_kw(attacker, "wither")
        uses_alt_damage = uses_infect or uses_wither

        blockers = [
            _get_card(state, bid)
            for bid in state.combat.blockers.get(attacker_id, [])
        ]
        blockers = [b for b in blockers if b is not None and b.zone == Zone.BATTLEFIELD]

        atk_power = effective_power(attacker)

        if not blockers:
            if atk_deals and atk_power > 0:
                defender = next((p for p in state.players if p.player_id == defender_id), None)
                if defender:
                    if uses_alt_damage:
                        _deal_infect_or_wither_damage(state, attacker, None, defender, atk_power)
                    else:
                        defender.life_total -= atk_power
                        state.log(f"{attacker.name} deals {atk_power} damage to {defender.name}")
                    _apply_lifelink(state, attacker, atk_power)
                    _track_commander_damage(state, attacker, defender, atk_power)
                    _fire_damage_triggers(state, attacker, "player")
                    # Toxic — give poison counters even when dealing normal damage
                    tx = toxic_count(attacker)
                    if tx > 0 and defender:
                        from .counters import add_poison
                        add_poison(defender, tx)
                        state.log(f"{attacker.name} toxic {tx}: {defender.name} gets {tx} poison")
            continue

        # Blocked — assign attacker damage across blockers in declared order.
        if atk_deals:
            remaining = atk_power
            damage_dealt_by_attacker = 0
            for blocker in blockers:
                if remaining <= 0:
                    break
                if uses_alt_damage:
                    # Infect/wither: deal -1/-1 counters; lethal = 1 if deathtouch else toughness
                    blk_toughness = effective_toughness(blocker) - blocker.damage_marked
                    assign = min(remaining, max(1, blk_toughness))
                    _deal_infect_or_wither_damage(state, attacker, blocker, None, assign)
                    damage_dealt_by_attacker += assign
                    remaining -= assign
                else:
                    # Standard damage
                    blk_toughness = effective_toughness(blocker) - blocker.damage_marked
                    lethal = 1 if has_kw(attacker, "deathtouch") else max(1, blk_toughness)
                    assign = min(remaining, lethal) if has_kw(attacker, "deathtouch") else min(remaining, max(1, blk_toughness))
                    blocker.damage_marked += assign
                    damage_dealt_by_attacker += assign
                    remaining -= assign
            # Trample: leftover to defending player.
            if has_kw(attacker, "trample") and remaining > 0:
                defender = next((p for p in state.players if p.player_id == defender_id), None)
                if defender:
                    if uses_alt_damage:
                        _deal_infect_or_wither_damage(state, attacker, None, defender, remaining)
                    else:
                        defender.life_total -= remaining
                        state.log(f"{attacker.name} tramples over for {remaining} to {defender.name}")
                    damage_dealt_by_attacker += remaining
                    _fire_damage_triggers(state, attacker, "player")
                    # Toxic also applies to combat damage from trample.
                    tx = toxic_count(attacker)
                    if tx > 0:
                        from .counters import add_poison
                        add_poison(defender, tx)
                        state.log(f"{attacker.name} toxic {tx}: {defender.name} gets {tx} poison")
            _apply_lifelink(state, attacker, damage_dealt_by_attacker)
            if damage_dealt_by_attacker > 0 and blockers:
                _fire_damage_triggers(state, attacker, "creature")

        # Blockers strike back (subject to step filter).
        for blocker in blockers:
            if first_strike_step and not _has_first_or_double(blocker):
                continue
            if not first_strike_step and has_kw(blocker, "first strike") and not has_kw(blocker, "double strike"):
                continue
            blk_power = effective_power(blocker)
            if blk_power <= 0:
                continue
            blk_infect = has_kw(blocker, "infect") or has_kw(blocker, "wither")
            if blk_infect:
                _deal_infect_or_wither_damage(state, blocker, attacker, None, blk_power)
            else:
                attacker.damage_marked += blk_power
            _apply_lifelink(state, blocker, blk_power)


def _has_first_or_double(card: CardInstance | None) -> bool:
    return has_kw(card, "first strike") or has_kw(card, "double strike")


def _apply_lifelink(state: GameState, source: CardInstance, damage: int) -> None:
    if damage <= 0 or not has_kw(source, "lifelink"):
        return
    controller = next((p for p in state.players if p.player_id == source.controller_id), None)
    if controller:
        controller.life_total += damage
        state.log(f"{source.name} lifelink: {controller.name} gains {damage} life")
        _fire_lifegain_triggers(state, controller.player_id)


def _fire_lifegain_triggers(state: GameState, gaining_player_id: str) -> None:
    from .triggers import check_lifegain_triggers, resolve_trigger
    for trig in check_lifegain_triggers(state, gaining_player_id):
        state.log(f"[TRIGGER (LIFE_GAIN)] {trig.description}")
        state.triggered_abilities.append(trig)
        resolve_trigger(state, trig)
        try:
            state.triggered_abilities.remove(trig)
        except ValueError:
            pass


def _track_commander_damage(
    state: GameState, attacker: CardInstance, defender, amount: int
) -> None:
    if state.format != "commander":
        return
    if not hasattr(defender, "commander_damage_received"):
        defender.commander_damage_received = {}
    is_commander = (
        attacker.instance_id in (
            state.commander_ids(attacker.controller_id)
            if hasattr(state, "commander_ids")
            else [state.commanders.get(attacker.controller_id)]
        )
        or attacker.card_data.get("is_commander", False)
    )
    if is_commander:
        defender.commander_damage_received.setdefault(attacker.controller_id, 0)
        defender.commander_damage_received[attacker.controller_id] += amount


def _remove_dead(state: GameState) -> None:
    """Inter-step SBA helper: move creatures with lethal damage (and not
    indestructible) to the graveyard so they don't deal damage in the
    second damage step."""
    from .zones import move_card
    for c in list(state.cards):
        if c.zone != Zone.BATTLEFIELD or not c.is_creature():
            continue
        if has_kw(c, "indestructible"):
            continue
        toughness = effective_toughness(c)
        if toughness > 0 and c.damage_marked >= toughness:
            move_card(state, c.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, c.owner_id)
            state.log(f"{c.name} dies (first-strike damage)")


def end_combat(state: GameState) -> None:
    state.combat = None
    state.log("Combat ends")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_card(state: GameState, cid: str) -> CardInstance | None:
    return next((c for c in state.cards if c.instance_id == cid), None)


def _card_name(state: GameState, cid: str) -> str:
    card = _get_card(state, cid)
    return card.name if card else "Unknown"


def _parse_pt(value: str | None) -> int:
    """Parse power/toughness — returns 0 for * or None."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0
    """Push 'whenever ~ deals damage to a player/creature' triggers and
    resolve them immediately so combat math reflects their effects."""
    from .triggers import check_damage_triggers, resolve_trigger
    from .game_state import StackItem
    triggers = check_damage_triggers(state, source, target_kind)
    for trig in triggers:
        item = StackItem(
            source_card_id=trig.source_card_id,
            controller_id=trig.controller_id,
            is_spell=False,
            card_data={"name": f"[Trigger] {source.name}: {trig.description}",
                       "type_line": "Ability"},
        )
        state.log(f"[TRIGGER (DAMAGE)] {source.name}: {trig.description} added to stack")
        # Resolve directly (no priority window in this minimal engine).
        state.triggered_abilities.append(trig)
        resolve_trigger(state, trig)
        try:
            state.triggered_abilities.remove(trig)
        except ValueError:
            pass
        state.log(f"[Trigger] {trig.description} resolves")


def can_attack(card: CardInstance, turn_number: int) -> bool:
    if card is None or not card.is_creature():
        return False
    if card.zone != Zone.BATTLEFIELD or card.tapped:
        return False
    if has_kw(card, "defender"):
        return False
    # Summoning sickness: cleared by haste.
    if card.summoning_sick and not has_kw(card, "haste"):
        return False
    return True


def can_block(attacker: CardInstance, blocker: CardInstance) -> bool:
    """CR 509: legality of a single blocker for ``attacker``."""
    if blocker is None or not blocker.is_creature():
        return False
    if blocker.zone != Zone.BATTLEFIELD or blocker.tapped:
        return False
    if has_kw(attacker, "flying") and not (has_kw(blocker, "flying") or has_kw(blocker, "reach")):
        return False
    return True


def menace_satisfied(attacker: CardInstance, blocker_count: int) -> bool:
    if has_kw(attacker, "menace") and blocker_count == 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase entry points
# ---------------------------------------------------------------------------


def begin_combat(state: GameState) -> CombatState:
    combat = CombatState()
    state.combat = combat
    state.log("Combat begins")
    return combat


def declare_attackers(
    state: GameState, attacker_assignments: dict[str, str]
) -> None:
    """Declare attackers. attacker_assignments: {card_instance_id: defending_player_id}"""
    if state.combat is None:
        return

    for attacker_id, defender_id in attacker_assignments.items():
        card = _get_card(state, attacker_id)
        if card is None or not can_attack(card, state.turn_number):
            continue
        # Vigilance: don't tap when attacking.
        if not has_kw(card, "vigilance"):
            card.tapped = True
        state.combat.attackers[attacker_id] = defender_id
        ep = effective_power(card)
        et = effective_toughness(card)
        state.log(f"{card.name} ({ep}/{et}) attacks {defender_id}")


def declare_blockers(
    state: GameState, blocker_assignments: dict[str, list[str]]
) -> None:
    """Declare blockers. blocker_assignments: {attacker_id: [blocker_card_ids]}"""
    if state.combat is None:
        return

    for attacker_id, blocker_ids in blocker_assignments.items():
        if attacker_id not in state.combat.attackers:
            continue
        attacker = _get_card(state, attacker_id)
        if attacker is None:
            continue
        legal: list[str] = []
        for bid in blocker_ids:
            blocker = _get_card(state, bid)
            if blocker and can_block(attacker, blocker):
                legal.append(bid)
                state.log(
                    f"{blocker.name} ({effective_power(blocker)}/{effective_toughness(blocker)})"
                    f" blocks {attacker.name} ({effective_power(attacker)}/{effective_toughness(attacker)})"
                )
        # Enforce menace.
        if not menace_satisfied(attacker, len(legal)):
            state.log(f"{attacker.name} has menace and is blocked by only {len(legal)} — block illegal, ignored")
            legal = []
        state.combat.blockers[attacker_id] = legal


# ---------------------------------------------------------------------------
# Damage resolution (with first/double strike)
# ---------------------------------------------------------------------------


def resolve_combat_damage(state: GameState) -> None:
    """CR 510 damage step. Handles first strike, double strike, trample,
    lifelink, deathtouch, and player damage."""
    if state.combat is None:
        return

    attackers = list(state.combat.attackers.items())

    # Determine if a first-strike step is needed.
    has_first_strike = False
    for attacker_id, _ in attackers:
        if _has_first_or_double(_get_card(state, attacker_id)):
            has_first_strike = True
            break
        for bid in state.combat.blockers.get(attacker_id, []):
            if _has_first_or_double(_get_card(state, bid)):
                has_first_strike = True
                break
        if has_first_strike:
            break

    if has_first_strike:
        _deal_step(state, attackers, first_strike_step=True)
        # SBA between steps would normally apply; defer to caller's SBA loop
        # but remove dead creatures for the regular step.
        _remove_dead(state)
        # Re-fetch attackers (some may have died).
        attackers = [
            (aid, did) for aid, did in attackers
            if _get_card(state, aid) and _get_card(state, aid).zone == Zone.BATTLEFIELD
        ]

    _deal_step(state, attackers, first_strike_step=False)
    state.combat = None


def _deal_step(
    state: GameState,
    attackers: list[tuple[str, str]],
    first_strike_step: bool,
) -> None:
    for attacker_id, defender_id in attackers:
        attacker = _get_card(state, attacker_id)
        if attacker is None or attacker.zone != Zone.BATTLEFIELD:
            continue
        # Filter participation in this step.
        if first_strike_step and not _has_first_or_double(attacker):
            atk_deals = False
        else:
            atk_deals = True
            # In second step, first-strike-only creatures don't deal again.
            if not first_strike_step and has_kw(attacker, "first strike") and not has_kw(attacker, "double strike"):
                atk_deals = False

        blockers = [
            _get_card(state, bid)
            for bid in state.combat.blockers.get(attacker_id, [])
        ]
        blockers = [b for b in blockers if b is not None and b.zone == Zone.BATTLEFIELD]

        atk_power = effective_power(attacker)

        if not blockers:
            if atk_deals and atk_power > 0:
                defender = next((p for p in state.players if p.player_id == defender_id), None)
                if defender:
                    defender.life_total -= atk_power
                    _apply_lifelink(state, attacker, atk_power)
                    _track_commander_damage(state, attacker, defender, atk_power)
                    state.log(f"{attacker.name} deals {atk_power} damage to {defender.name}")
                    _fire_damage_triggers(state, attacker, "player")
            continue

        # Blocked — assign attacker damage across blockers in declared order.
        if atk_deals:
            remaining = atk_power
            damage_dealt_by_attacker = 0
            for blocker in blockers:
                if remaining <= 0:
                    break
                # Deathtouch lets us assign just 1 to satisfy "lethal".
                blk_toughness = effective_toughness(blocker) - blocker.damage_marked
                lethal = 1 if has_kw(attacker, "deathtouch") else max(1, blk_toughness)
                assign = min(remaining, lethal) if has_kw(attacker, "deathtouch") else min(remaining, max(1, blk_toughness))
                blocker.damage_marked += assign
                damage_dealt_by_attacker += assign
                remaining -= assign
            # Trample: leftover to defending player.
            if has_kw(attacker, "trample") and remaining > 0:
                defender = next((p for p in state.players if p.player_id == defender_id), None)
                if defender:
                    defender.life_total -= remaining
                    damage_dealt_by_attacker += remaining
                    state.log(f"{attacker.name} tramples over for {remaining} to {defender.name}")
                    _fire_damage_triggers(state, attacker, "player")
            _apply_lifelink(state, attacker, damage_dealt_by_attacker)
            if damage_dealt_by_attacker > 0 and blockers:
                _fire_damage_triggers(state, attacker, "creature")

        # Blockers strike back (subject to step filter).
        for blocker in blockers:
            if first_strike_step and not _has_first_or_double(blocker):
                continue
            if not first_strike_step and has_kw(blocker, "first strike") and not has_kw(blocker, "double strike"):
                continue
            blk_power = effective_power(blocker)
            if blk_power <= 0:
                continue
            attacker.damage_marked += blk_power
            _apply_lifelink(state, blocker, blk_power)


def _has_first_or_double(card: CardInstance | None) -> bool:
    return has_kw(card, "first strike") or has_kw(card, "double strike")


def _apply_lifelink(state: GameState, source: CardInstance, damage: int) -> None:
    if damage <= 0 or not has_kw(source, "lifelink"):
        return
    controller = next((p for p in state.players if p.player_id == source.controller_id), None)
    if controller:
        controller.life_total += damage
        state.log(f"{source.name} lifelink: {controller.name} gains {damage} life")
        _fire_lifegain_triggers(state, controller.player_id)


def _fire_lifegain_triggers(state: GameState, gaining_player_id: str) -> None:
    from .triggers import check_lifegain_triggers, resolve_trigger
    for trig in check_lifegain_triggers(state, gaining_player_id):
        state.log(f"[TRIGGER (LIFE_GAIN)] {trig.description}")
        state.triggered_abilities.append(trig)
        resolve_trigger(state, trig)
        try:
            state.triggered_abilities.remove(trig)
        except ValueError:
            pass


def _track_commander_damage(
    state: GameState, attacker: CardInstance, defender, amount: int
) -> None:
    if state.format != "commander":
        return
    if not hasattr(defender, "commander_damage_received"):
        defender.commander_damage_received = {}
    is_commander = (
        attacker.instance_id in (
            state.commander_ids(attacker.controller_id)
            if hasattr(state, "commander_ids")
            else [state.commanders.get(attacker.controller_id)]
        )
        or attacker.card_data.get("is_commander", False)
    )
    if is_commander:
        defender.commander_damage_received.setdefault(attacker.controller_id, 0)
        defender.commander_damage_received[attacker.controller_id] += amount


def _remove_dead(state: GameState) -> None:
    """Inter-step SBA helper: move creatures with lethal damage (and not
    indestructible) to the graveyard so they don't deal damage in the
    second damage step."""
    from .zones import move_card
    for c in list(state.cards):
        if c.zone != Zone.BATTLEFIELD or not c.is_creature():
            continue
        if has_kw(c, "indestructible"):
            continue
        toughness = effective_toughness(c)
        if toughness > 0 and c.damage_marked >= toughness:
            move_card(state, c.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, c.owner_id)
            state.log(f"{c.name} dies (first-strike damage)")


def end_combat(state: GameState) -> None:
    state.combat = None
    state.log("Combat ends")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_card(state: GameState, cid: str) -> CardInstance | None:
    return next((c for c in state.cards if c.instance_id == cid), None)


def _card_name(state: GameState, cid: str) -> str:
    card = _get_card(state, cid)
    return card.name if card else "Unknown"


def _parse_pt(value: str | None) -> int:
    """Parse power/toughness — returns 0 for * or None."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0

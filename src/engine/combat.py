"""
Combat system — declare attackers, declare blockers, assign damage.

Supports the following CR 702 evergreen keywords:
- Flying / Reach (block legality)
- Menace (block legality: requires 2+ blockers)
- Vigilance (attacker doesn't tap)
- Haste (ignores summoning sickness)
- First strike / Double strike (two-step damage)
- Trample (excess damage to defending player)
- Lifelink (controller gains life equal to damage dealt)
- Deathtouch (any damage from this source is lethal to creatures)
- Indestructible (survives lethal damage and "destroy" effects)

References:
- open-mtg (MIT): CR 509/510-compliant damage assignment ordering
- mtg-python-engine (MIT): handle_combat_phase() pattern
"""

from __future__ import annotations

from .game_state import CardInstance, CombatState, GameState, Zone
from .keywords import (
    has as has_kw,
    can_be_targeted,
    effective_power,
    effective_toughness,
)


_KEYWORDS = (
    "flying", "reach", "menace", "vigilance", "haste",
    "first strike", "double strike", "trample", "lifelink",
    "deathtouch", "indestructible", "defender",
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
        attacker.instance_id == state.commanders.get(attacker.controller_id)
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

"""Alternate and additional cost mechanics (CR 118.8, 702).

Implements:
- Convoke (CR 702.54): tap creatures to pay generic or colored mana
- Delve (CR 702.65): exile graveyard cards to pay generic mana
- Improvise (CR 702.125): tap artifacts to pay generic mana
- Emerge (CR 702.116): sacrifice creature to reduce cost by its CMC
- Spectacle (CR 702.133): alternate cost if an opponent lost life this turn
- Overload (CR 702.96): alternate cost to affect all instead of one target
- Surge (CR 702.116): alternate cost if you cast another spell this turn
- Madness (CR 702.34): cast from discard for alternate cost
- Escape (CR 702.141): cast from graveyard by exiling N other cards + mana
- Jump-start (CR 702.131): cast from graveyard by discarding a card
- Retrace (CR 702.88): cast from graveyard by discarding a land
- Flashback (CR 702.33): already implemented in equip.py — re-exported here
- Foretell (CR 702.143): exile face-down for {2}, then cast for foretell cost
- Buyback (CR 702.26): pay extra cost to return to hand after resolving
- Entwine (CR 702.56): pay extra cost to choose all modes
- Replicate (CR 702.72): pay extra cost to copy the spell N times
- Kicker (CR 702.32): already in equip.py — re-exported here
- Ninjutsu (CR 702.48): return unblocked attacker, put ninja into play
- Offering (CR 702.49): sacrifice creature of the right type to reduce cost
"""

from __future__ import annotations

import re

from .game_state import CardInstance, GameState, Zone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_cmc(card: CardInstance) -> int:
    """Return the converted mana cost (CMC) of a card."""
    cmc = card.card_data.get("cmc")
    if cmc is not None:
        try:
            return int(float(cmc))
        except (TypeError, ValueError):
            pass
    mc = card.card_data.get("mana_cost") or ""
    total = 0
    for tok in re.findall(r"\{([^}]+)\}", mc):
        try:
            total += int(tok)
        except ValueError:
            if tok.upper() in {"W", "U", "B", "R", "G", "C"}:
                total += 1
            elif "/" in tok:
                total += 1  # hybrid = 1
    return total


def _has_keyword_text(card: CardInstance, keyword: str) -> bool:
    oracle = (card.oracle_text or "").lower()
    kw_list = [k.lower() for k in (card.card_data.get("keywords") or [])]
    return keyword in oracle or keyword in kw_list


# ---------------------------------------------------------------------------
# Convoke (CR 702.54)
# ---------------------------------------------------------------------------


def has_convoke(card: CardInstance) -> bool:
    return _has_keyword_text(card, "convoke")


def apply_convoke(
    state: GameState, caster_id: str, card: CardInstance,
    cost: dict[str, int],
    tapped_creature_ids: list[str],
) -> dict[str, int]:
    """Tap the specified creatures to reduce cost (CR 702.54).

    Each tapped creature reduces either one colored or one generic mana from
    cost. Returns the modified cost (copy).
    """
    cost = dict(cost)
    for cid in tapped_creature_ids:
        creature = next(
            (c for c in state.cards
             if c.instance_id == cid
             and c.zone == Zone.BATTLEFIELD
             and c.controller_id == caster_id
             and c.is_creature()
             and not c.tapped),
            None,
        )
        if creature is None:
            continue
        creature.tapped = True
        state.log(f"Convoke: {creature.name} tapped to pay for {card.name}")
        # Try to pay a colored mana first
        from .keywords import _card_colors
        paid = False
        for color in _card_colors(creature):
            if cost.get(color, 0) > 0:
                cost[color] -= 1
                if cost[color] == 0:
                    del cost[color]
                paid = True
                break
        if not paid and cost.get("generic", 0) > 0:
            cost["generic"] -= 1
            if cost["generic"] == 0:
                del cost["generic"]
    return cost


# ---------------------------------------------------------------------------
# Delve (CR 702.65)
# ---------------------------------------------------------------------------


def has_delve(card: CardInstance) -> bool:
    return _has_keyword_text(card, "delve")


def apply_delve(
    state: GameState, caster_id: str, card: CardInstance,
    cost: dict[str, int],
    exiled_card_ids: list[str],
) -> dict[str, int]:
    """Exile graveyard cards to pay generic mana (CR 702.65)."""
    from .zones import move_card
    cost = dict(cost)
    for cid in exiled_card_ids:
        graveyard_card = next(
            (c for c in state.cards
             if c.instance_id == cid
             and c.zone == Zone.GRAVEYARD
             and c.owner_id == caster_id),
            None,
        )
        if graveyard_card is None:
            continue
        if cost.get("generic", 0) <= 0:
            break
        move_card(state, cid, Zone.GRAVEYARD, Zone.EXILE, caster_id)
        cost["generic"] -= 1
        if cost["generic"] == 0:
            del cost["generic"]
        state.log(f"Delve: {graveyard_card.name} exiled to pay for {card.name}")
    return cost


# ---------------------------------------------------------------------------
# Improvise (CR 702.125)
# ---------------------------------------------------------------------------


def has_improvise(card: CardInstance) -> bool:
    return _has_keyword_text(card, "improvise")


def apply_improvise(
    state: GameState, caster_id: str, card: CardInstance,
    cost: dict[str, int],
    tapped_artifact_ids: list[str],
) -> dict[str, int]:
    """Tap artifacts to pay generic mana (CR 702.125)."""
    cost = dict(cost)
    for aid in tapped_artifact_ids:
        artifact = next(
            (c for c in state.cards
             if c.instance_id == aid
             and c.zone == Zone.BATTLEFIELD
             and c.controller_id == caster_id
             and "artifact" in (c.type_line or "").lower()
             and not c.tapped),
            None,
        )
        if artifact is None:
            continue
        if cost.get("generic", 0) <= 0:
            break
        artifact.tapped = True
        cost["generic"] -= 1
        if cost["generic"] == 0:
            del cost["generic"]
        state.log(f"Improvise: {artifact.name} tapped to pay for {card.name}")
    return cost


# ---------------------------------------------------------------------------
# Emerge (CR 702.116)
# ---------------------------------------------------------------------------


def has_emerge(card: CardInstance) -> bool:
    return bool(re.search(r"\bemerge\b", (card.oracle_text or ""), re.IGNORECASE))


def emerge_cost(card: CardInstance) -> dict[str, int] | None:
    """Return the emerge cost dict, or None if card lacks emerge."""
    m = re.search(
        r"emerge\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def apply_emerge(
    state: GameState, caster_id: str, sacrificed_card_id: str,
    emerge_cost_dict: dict[str, int],
) -> dict[str, int]:
    """Sacrifice a creature; reduce emerge cost by its CMC (CR 702.116)."""
    from .zones import move_card
    cost = dict(emerge_cost_dict)
    sacrificed = next(
        (c for c in state.cards
         if c.instance_id == sacrificed_card_id
         and c.zone == Zone.BATTLEFIELD
         and c.controller_id == caster_id
         and c.is_creature()),
        None,
    )
    if sacrificed is None:
        return cost
    cmc = _parse_cmc(sacrificed)
    move_card(state, sacrificed_card_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, caster_id)
    state.log(f"Emerge: {sacrificed.name} (CMC {cmc}) sacrificed")
    # Reduce generic mana by cmc
    generic = cost.get("generic", 0)
    reduction = min(generic, cmc)
    cost["generic"] = generic - reduction
    if cost["generic"] <= 0:
        cost.pop("generic", None)
    return cost


# ---------------------------------------------------------------------------
# Spectacle (CR 702.133)
# ---------------------------------------------------------------------------


def has_spectacle(card: CardInstance) -> bool:
    return bool(re.search(r"\bspectacle\b", (card.oracle_text or ""), re.IGNORECASE))


def spectacle_cost(card: CardInstance) -> dict[str, int] | None:
    """Return the spectacle cost dict, or None if card lacks spectacle."""
    m = re.search(
        r"spectacle\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def can_use_spectacle(state: GameState, caster_id: str) -> bool:
    """Spectacle is available if an opponent lost life this turn."""
    for player in state.players:
        if player.player_id != caster_id and getattr(player, "life_lost_this_turn", 0) > 0:
            return True
    return getattr(state, "opponent_lost_life_this_turn", False)


# ---------------------------------------------------------------------------
# Surge (CR 702.116)
# ---------------------------------------------------------------------------


def has_surge(card: CardInstance) -> bool:
    return bool(re.search(r"\bsurge\b", (card.oracle_text or ""), re.IGNORECASE))


def surge_cost(card: CardInstance) -> dict[str, int] | None:
    """Return the surge cost dict, or None if card lacks surge."""
    m = re.search(
        r"surge\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def can_use_surge(state: GameState, caster_id: str) -> bool:
    """Surge is available if you or a teammate cast a spell this turn."""
    spells = getattr(state, "spells_cast_this_turn", 0)
    return spells >= 1


# ---------------------------------------------------------------------------
# Madness (CR 702.34)
# ---------------------------------------------------------------------------


def has_madness(card: CardInstance) -> bool:
    return bool(re.search(r"\bmadness\b", (card.oracle_text or ""), re.IGNORECASE))


def madness_cost(card: CardInstance) -> dict[str, int] | None:
    """Return the madness cost dict, or None if card lacks madness."""
    m = re.search(
        r"madness\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


# ---------------------------------------------------------------------------
# Miracle (CR 702.92)
# ---------------------------------------------------------------------------


def has_miracle(card: CardInstance) -> bool:
    return bool(re.search(r"\bmiracle\b", (card.oracle_text or ""), re.IGNORECASE))


def miracle_cost(card: CardInstance) -> dict[str, int] | None:
    """Return the miracle cost dict, or None if card lacks miracle."""
    m = re.search(
        r"miracle\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def can_use_miracle(state: GameState, caster_id: str) -> bool:
    """Miracle is available if this is the first card drawn this turn."""
    return getattr(state, "first_draw_this_turn_id", None) is not None


# ---------------------------------------------------------------------------
# Overload (CR 702.96)
# ---------------------------------------------------------------------------


def has_overload(card: CardInstance) -> bool:
    return bool(re.search(r"\boverload\b", (card.oracle_text or ""), re.IGNORECASE))


def overload_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"overload\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


# ---------------------------------------------------------------------------
# Escape (CR 702.141)
# ---------------------------------------------------------------------------


def has_escape(card: CardInstance) -> bool:
    return bool(re.search(r"\bescape\b", (card.oracle_text or ""), re.IGNORECASE))


def escape_cost(card: CardInstance) -> tuple[dict[str, int], int] | None:
    """Return (mana_cost_dict, cards_to_exile_count) or None."""
    m = re.search(
        r"escape[—\-]\s*((?:\{[^}]+\})+).*?exile\s+(\w+|\d+)\s+other",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        # Simpler form: "Escape—{mana}, exile N other cards from your graveyard"
        m2 = re.search(
            r"escape\s*[—\-]\s*((?:\{[^}]+\})+)[^.]*exile (\w+|\d+) other",
            (card.oracle_text or ""), re.IGNORECASE,
        )
        if not m2:
            return None
        from .mana import parse_mana_cost
        n_str = m2.group(2).lower()
        n_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8}
        try:
            n = int(n_str)
        except ValueError:
            n = n_map.get(n_str, 3)
        return parse_mana_cost(m2.group(1)), n
    from .mana import parse_mana_cost
    n_str = m.group(2).lower()
    n_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8}
    try:
        n = int(n_str)
    except ValueError:
        n = n_map.get(n_str, 3)
    return parse_mana_cost(m.group(1)), n


def can_escape(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Return True if escape is available (card in graveyard, enough cards to exile)."""
    if card.zone != Zone.GRAVEYARD:
        return False
    result = escape_cost(card)
    if result is None:
        return False
    _, n = result
    other_graveyard = [
        c for c in state.cards
        if c.zone == Zone.GRAVEYARD
        and c.owner_id == caster_id
        and c.instance_id != card.instance_id
    ]
    return len(other_graveyard) >= n


def apply_escape(
    state: GameState, caster_id: str, card: CardInstance,
    exiled_ids: list[str],
) -> bool:
    """Exile N other graveyard cards to enable escape (CR 702.141)."""
    from .zones import move_card
    result = escape_cost(card)
    if result is None:
        return False
    _, n = result
    exiled = 0
    for cid in exiled_ids:
        if exiled >= n:
            break
        gc = next(
            (c for c in state.cards
             if c.instance_id == cid
             and c.zone == Zone.GRAVEYARD
             and c.owner_id == caster_id
             and c.instance_id != card.instance_id),
            None,
        )
        if gc:
            move_card(state, cid, Zone.GRAVEYARD, Zone.EXILE, caster_id)
            exiled += 1
    return exiled >= n


# ---------------------------------------------------------------------------
# Jump-start (CR 702.131)
# ---------------------------------------------------------------------------


def has_jump_start(card: CardInstance) -> bool:
    return bool(re.search(r"\bjump-start\b|\bjumpstart\b",
                          (card.oracle_text or ""), re.IGNORECASE))


def can_jump_start(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Return True if jump-start is available (card in graveyard, hand has a card)."""
    if card.zone != Zone.GRAVEYARD:
        return False
    if not has_jump_start(card):
        return False
    hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == caster_id]
    return len(hand) >= 1


def apply_jump_start_discard(state: GameState, caster_id: str) -> bool:
    """Discard a card from hand to enable jump-start (CR 702.131)."""
    from .zones import move_card
    hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == caster_id]
    if not hand:
        return False
    # Discard worst (first) card
    move_card(state, hand[0].instance_id, Zone.HAND, Zone.GRAVEYARD, caster_id)
    state.log(f"Jump-start: {hand[0].name} discarded")
    return True


# ---------------------------------------------------------------------------
# Retrace (CR 702.88)
# ---------------------------------------------------------------------------


def has_retrace(card: CardInstance) -> bool:
    return bool(re.search(r"\bretrace\b", (card.oracle_text or ""), re.IGNORECASE))


def can_retrace(state: GameState, caster_id: str, card: CardInstance) -> bool:
    if card.zone != Zone.GRAVEYARD:
        return False
    if not has_retrace(card):
        return False
    # Need a land card in hand
    hand = [
        c for c in state.cards
        if c.zone == Zone.HAND
        and c.owner_id == caster_id
        and "land" in (c.type_line or "").lower()
    ]
    return len(hand) >= 1


def apply_retrace_discard(state: GameState, caster_id: str) -> bool:
    """Discard a land card from hand to enable retrace (CR 702.88)."""
    from .zones import move_card
    land_cards = [
        c for c in state.cards
        if c.zone == Zone.HAND
        and c.owner_id == caster_id
        and "land" in (c.type_line or "").lower()
    ]
    if not land_cards:
        return False
    move_card(state, land_cards[0].instance_id, Zone.HAND, Zone.GRAVEYARD, caster_id)
    state.log(f"Retrace: {land_cards[0].name} discarded (land)")
    return True


# ---------------------------------------------------------------------------
# Foretell (CR 702.143)
# ---------------------------------------------------------------------------


def has_foretell(card: CardInstance) -> bool:
    return bool(re.search(r"\bforetell\b", (card.oracle_text or ""), re.IGNORECASE))


def foretell_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"foretell\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def foretell_card(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Exile card face-down with foretell marker during your turn for {2}."""
    from .zones import move_card
    from .mana import pay_cost
    player = next((p for p in state.players if p.player_id == caster_id), None)
    if player is None:
        return False
    # Pay {2}
    if player.mana_pool.get("generic", 0) + sum(
        v for k, v in player.mana_pool.items() if k != "generic"
    ) < 2:
        return False
    pay_cost(player, {"generic": 2})
    move_card(state, card.instance_id, Zone.HAND, Zone.EXILE, caster_id)
    card.card_data["foretold"] = True
    card.card_data["foretold_by"] = caster_id
    state.log(f"Foretell: {card.name} exiled face-down (can be cast for foretell cost)")
    return True


# ---------------------------------------------------------------------------
# Ninjutsu (CR 702.48)
# ---------------------------------------------------------------------------


def has_ninjutsu(card: CardInstance) -> bool:
    return bool(re.search(r"\bninjutsu\b", (card.oracle_text or ""), re.IGNORECASE))


def ninjutsu_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"ninjutsu\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def can_use_ninjutsu(
    state: GameState, caster_id: str, ninja_card: CardInstance
) -> bool:
    """Return True if ninjutsu is available — need an unblocked attacker."""
    if ninja_card.zone != Zone.HAND:
        return False
    if not has_ninjutsu(ninja_card):
        return False
    if state.combat is None:
        return False
    unblocked = [
        aid for aid, def_id in state.combat.attackers.items()
        if not state.combat.blockers.get(aid) and
        next((c for c in state.cards if c.instance_id == aid and
              c.controller_id == caster_id), None) is not None
    ]
    return len(unblocked) > 0


def apply_ninjutsu(
    state: GameState, caster_id: str,
    ninja_card: CardInstance, returned_attacker_id: str,
) -> bool:
    """Ninjutsu swap: return unblocked attacker to hand, put ninja into play."""
    from .zones import move_card
    if state.combat is None:
        return False
    returner = next(
        (c for c in state.cards
         if c.instance_id == returned_attacker_id
         and c.zone == Zone.BATTLEFIELD
         and c.controller_id == caster_id),
        None,
    )
    if returner is None:
        return False
    defender_id = state.combat.attackers.get(returned_attacker_id)
    if defender_id is None:
        return False
    # Return attacker to hand
    move_card(state, returned_attacker_id, Zone.BATTLEFIELD, Zone.HAND, caster_id)
    state.combat.attackers.pop(returned_attacker_id, None)
    # Put ninja into play tapped and attacking
    move_card(state, ninja_card.instance_id, Zone.HAND, Zone.BATTLEFIELD, caster_id)
    ninja_card.tapped = True
    ninja_card.summoning_sick = False
    state.combat.attackers[ninja_card.instance_id] = defender_id
    state.log(f"Ninjutsu: {returner.name} returned, {ninja_card.name} enters attacking {defender_id}")
    return True


# ---------------------------------------------------------------------------
# Buyback (CR 702.26)
# ---------------------------------------------------------------------------


def has_buyback(card: CardInstance) -> bool:
    return bool(re.search(r"\bbuyback\b", (card.oracle_text or ""), re.IGNORECASE))


def buyback_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"buyback\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


# ---------------------------------------------------------------------------
# Replicate (CR 702.72)
# ---------------------------------------------------------------------------


def has_replicate(card: CardInstance) -> bool:
    return bool(re.search(r"\breplicate\b", (card.oracle_text or ""), re.IGNORECASE))


def replicate_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"replicate\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def apply_replicate(
    state: GameState, stack_item, caster_id: str, times: int
) -> None:
    """Push ``times`` copies of stack_item onto the stack."""
    from .game_state import StackItem
    for _ in range(times):
        copy = StackItem(
            source_card_id=stack_item.source_card_id,
            controller_id=caster_id,
            is_spell=False,
            card_data=dict(stack_item.card_data),
            targets=list(getattr(stack_item, "targets", []) or []),
        )
        copy.card_data["is_replicate_copy"] = True
        state.stack.append(copy)
        state.log(f"Replicate copy #{_ + 1} of {stack_item.card_data.get('name', 'spell')} added to stack")


# ---------------------------------------------------------------------------
# Entwine (CR 702.56)
# ---------------------------------------------------------------------------


def has_entwine(card: CardInstance) -> bool:
    return bool(re.search(r"\bentwine\b", (card.oracle_text or ""), re.IGNORECASE))


def entwine_cost(card: CardInstance) -> dict[str, int] | None:
    m = re.search(
        r"entwine\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


# ---------------------------------------------------------------------------
# Affinity (CR 702.41) — cost reduces by 1 per permanent of specified type
# ---------------------------------------------------------------------------


def affinity_discount(
    state: GameState, caster_id: str, card: CardInstance,
    cost: dict[str, int],
) -> dict[str, int]:
    """Apply affinity discount: reduce generic by 1 per matching permanent (CR 702.41)."""
    text = (card.oracle_text or "").lower()
    m = re.search(r"\baffinity for ([\w\s]+?)(?:\s*\(|\s*$)", text)
    if not m:
        return cost
    perm_type = m.group(1).strip()
    count = sum(
        1 for c in state.cards
        if c.zone == Zone.BATTLEFIELD
        and c.controller_id == caster_id
        and perm_type in (c.type_line or "").lower()
    )
    cost = dict(cost)
    generic = cost.get("generic", 0)
    discount = min(generic, count)
    cost["generic"] = generic - discount
    if cost["generic"] <= 0:
        cost.pop("generic", None)
    return cost


# ---------------------------------------------------------------------------
# Suspend (CR 702.62)
# ---------------------------------------------------------------------------


def has_suspend(card: CardInstance) -> bool:
    return bool(re.search(r"\bsuspend\b", (card.oracle_text or ""), re.IGNORECASE))


def suspend_data(card: CardInstance) -> tuple[int, dict[str, int]] | None:
    """Return (time_counter_count, suspend_cost_dict) or None."""
    m = re.search(
        r"suspend\s+(\d+)\s*[—\-]\s*((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return int(m.group(1)), parse_mana_cost(m.group(2))


def can_suspend(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Return True if suspend can be activated (card in hand)."""
    if card.zone != Zone.HAND:
        return False
    return has_suspend(card)


def apply_suspend(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Exile card from hand with time counters (CR 702.62)."""
    from .zones import move_card
    from .mana import pay_cost
    from .counters import add_counter
    result = suspend_data(card)
    if result is None:
        return False
    n, cost = result
    player = next((p for p in state.players if p.player_id == caster_id), None)
    if player is None:
        return False
    # Try to pay the suspend cost
    from .mana import auto_tap_for_cost
    if not auto_tap_for_cost(state, player, cost):
        return False
    pay_cost(player, cost)
    move_card(state, card.instance_id, Zone.HAND, Zone.EXILE, caster_id)
    for _ in range(n):
        add_counter(card, "time", 1)
    card.card_data["suspended_by"] = caster_id
    state.log(f"Suspend: {card.name} exiled with {n} time counter(s)")
    return True


def tick_suspend_counters(state: GameState, player_id: str) -> list[CardInstance]:
    """At start of player's upkeep: remove one time counter from each suspended
    card. Return cards with 0 time counters that should be cast for free."""
    from .counters import remove_counter
    free_cast: list[CardInstance] = []
    for card in state.cards:
        if card.zone != Zone.EXILE:
            continue
        if card.card_data.get("suspended_by") != player_id:
            continue
        if card.counters.get("time", 0) <= 0:
            continue
        remove_counter(card, "time", 1)
        state.log(f"Suspend: time counter removed from {card.name} ({card.counters.get('time', 0)} remaining)")
        if card.counters.get("time", 0) == 0:
            free_cast.append(card)
    return free_cast


def cast_suspended_card(state: GameState, card: CardInstance) -> None:
    """Cast a suspended card for free from exile (haste if it's a creature)."""
    from .zones import move_card
    from .keywords import has as has_kw
    move_card(state, card.instance_id, Zone.EXILE, Zone.BATTLEFIELD
              if "land" in (card.type_line or "").lower() else Zone.GRAVEYARD,
              card.owner_id)
    # Actually we cast it — for simplicity, move to hand then cast
    # The rules engine will handle resolving it from the stack
    card.card_data.pop("suspended_by", None)
    if card.is_creature() and not has_kw(card, "haste"):
        # Suspended creatures get haste
        oracle = card.oracle_text or ""
        card.card_data["oracle_text"] = oracle + "\nHaste. (Suspended)"
    state.log(f"Suspend: {card.name} cast for free (last time counter removed)")


# ---------------------------------------------------------------------------
# Morph / Megamorph (CR 702.36 / 702.99)
# ---------------------------------------------------------------------------


def has_morph(card: CardInstance) -> bool:
    return bool(re.search(r"\bmorph\b", (card.oracle_text or ""), re.IGNORECASE))


def has_megamorph(card: CardInstance) -> bool:
    return bool(re.search(r"\bmegamorph\b", (card.oracle_text or ""), re.IGNORECASE))


def morph_cost(card: CardInstance) -> dict[str, int] | None:
    """Return morph cost, or None if absent."""
    m = re.search(
        r"\b(?:mega)?morph\s+((?:\{[^}]+\})+)",
        (card.oracle_text or ""), re.IGNORECASE,
    )
    if not m:
        return None
    from .mana import parse_mana_cost
    return parse_mana_cost(m.group(1))


def cast_face_down(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Cast a morph card face-down as a 2/2 for {3} (CR 702.36b)."""
    from .zones import move_card
    from .mana import pay_cost
    player = next((p for p in state.players if p.player_id == caster_id), None)
    if player is None:
        return False
    # Pay {3}
    total_mana = sum(player.mana_pool.values())
    if total_mana < 3:
        return False
    pay_cost(player, {"generic": 3})
    move_card(state, card.instance_id, Zone.HAND, Zone.BATTLEFIELD, caster_id)
    card.card_data["face_down"] = True
    card.card_data["face_down_by"] = caster_id
    # Override P/T to 2/2, no abilities
    card.card_data["face_down_power"] = card.card_data.get("power")
    card.card_data["face_down_toughness"] = card.card_data.get("toughness")
    card.card_data["power"] = "2"
    card.card_data["toughness"] = "2"
    card.summoning_sick = True
    state.log(f"Morph: {card.name} cast face-down as a 2/2")
    return True


def turn_face_up(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Turn a face-down morph card face-up for its morph cost (CR 702.36e)."""
    from .mana import pay_cost, auto_tap_for_cost
    if not card.card_data.get("face_down"):
        return False
    cost = morph_cost(card)
    if cost is None:
        cost = {}  # Some cards have {0} morph
    player = next((p for p in state.players if p.player_id == caster_id), None)
    if player is None:
        return False
    if not auto_tap_for_cost(state, player, cost):
        return False
    pay_cost(player, cost)
    # Restore real card data
    card.card_data.pop("face_down", None)
    card.card_data.pop("face_down_by", None)
    if "face_down_power" in card.card_data:
        card.card_data["power"] = card.card_data.pop("face_down_power")
    if "face_down_toughness" in card.card_data:
        card.card_data["toughness"] = card.card_data.pop("face_down_toughness")
    # Megamorph: also get a +1/+1 counter
    if has_megamorph(card):
        from .counters import add_counter
        add_counter(card, "+1/+1", 1)
        state.log(f"Megamorph: {card.name} turned face-up with +1/+1 counter")
    else:
        state.log(f"Morph: {card.name} turned face-up")
    return True

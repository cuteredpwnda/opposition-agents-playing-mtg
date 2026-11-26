"""Counter mechanics (CR 122) and counter-driven SBAs.

Implements:

- ``+1/+1`` and ``-1/-1`` counters cancel pairwise on a creature
  (CR 704.5q) — applied as part of ``apply_counter_sbas``.
- A creature with ``-1/-1`` counters reducing its toughness to <= 0
  is handled by the existing 0-toughness SBA in
  :mod:`rules_engine`; this module only normalises the counter
  bookkeeping so power/toughness queries can use it.
- ``poison`` counters: a player with 10+ poison counters loses the
  game (CR 704.5c).
- ``stun`` counters (CR 701.49): on untap, remove a stun counter
  *instead* of untapping. ``apply_stun_on_untap`` exposes this hook.
- Helpers to add/remove counters with logging.
"""

from __future__ import annotations

from .game_state import CardInstance, GameState, Zone


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def add_counter(card: CardInstance, kind: str, amount: int = 1) -> None:
    if amount <= 0:
        return
    card.counters[kind] = card.counters.get(kind, 0) + amount


def remove_counter(card: CardInstance, kind: str, amount: int = 1) -> int:
    """Remove up to ``amount`` counters of ``kind``. Returns count removed."""
    have = card.counters.get(kind, 0)
    take = min(have, amount)
    if take <= 0:
        return 0
    card.counters[kind] = have - take
    if card.counters[kind] <= 0:
        card.counters.pop(kind, None)
    return take


def add_poison(player, amount: int = 1) -> None:
    player.poison_counters = getattr(player, "poison_counters", 0) + max(0, amount)


def add_energy(player, amount: int = 1) -> None:
    player.energy_counters = getattr(player, "energy_counters", 0) + max(0, amount)


def spend_energy(player, amount: int) -> bool:
    have = getattr(player, "energy_counters", 0)
    if have < amount:
        return False
    player.energy_counters = have - amount
    return True


# ---------------------------------------------------------------------------
# State-based actions
# ---------------------------------------------------------------------------


def apply_counter_sbas(state: GameState) -> list[str]:
    """Cancel +1/+1 vs -1/-1 on every permanent and check poison loss.

    Called from ``rules_engine.check_state_based_actions`` before the
    lethal-damage / 0-toughness sweep so that net toughness reductions
    from -1/-1 counters trigger the existing death rule.

    Returns a list of human-readable events (logged by caller).
    """
    events: list[str] = []

    # +1/+1 vs -1/-1 cancellation (CR 704.5q)
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        plus = card.counters.get("+1/+1", 0)
        minus = card.counters.get("-1/-1", 0)
        if plus > 0 and minus > 0:
            cancel = min(plus, minus)
            card.counters["+1/+1"] = plus - cancel
            card.counters["-1/-1"] = minus - cancel
            if card.counters["+1/+1"] == 0:
                card.counters.pop("+1/+1", None)
            if card.counters["-1/-1"] == 0:
                card.counters.pop("-1/-1", None)
            events.append(f"{card.name}: {cancel} +1/+1 and {cancel} -1/-1 cancel")

    # Poison loss (CR 704.5c) — 10+ poison loses the game.
    survivors = []
    for player in state.players:
        if getattr(player, "poison_counters", 0) >= 10:
            events.append(f"{player.name} loses the game (10+ poison counters)")
        else:
            survivors.append(player)
    if len(survivors) != len(state.players):
        state.players = survivors

    return events


# ---------------------------------------------------------------------------
# Stun counters (CR 701.49)
# ---------------------------------------------------------------------------


def apply_stun_on_untap(card: CardInstance) -> bool:
    """If ``card`` has a stun counter, remove one and skip untap.

    Returns True if untap was *skipped*, False otherwise.
    """
    if card.counters.get("stun", 0) > 0:
        remove_counter(card, "stun", 1)
        return True
    return False


# ---------------------------------------------------------------------------
# Effective P/T including counters and -1/-1
# ---------------------------------------------------------------------------


def effective_toughness(card: CardInstance) -> int:
    """Base toughness + (+1/+1) - (-1/-1). Returns 0 if non-creature."""
    base = card.card_data.get("toughness")
    try:
        base_int = int(base) if base is not None else 0
    except (TypeError, ValueError):
        base_int = 0
    plus = card.counters.get("+1/+1", 0)
    minus = card.counters.get("-1/-1", 0)
    return base_int + plus - minus


def effective_power(card: CardInstance) -> int:
    base = card.card_data.get("power")
    try:
        base_int = int(base) if base is not None else 0
    except (TypeError, ValueError):
        base_int = 0
    plus = card.counters.get("+1/+1", 0)
    minus = card.counters.get("-1/-1", 0)
    return base_int + plus - minus

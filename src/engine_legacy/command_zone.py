"""Command-zone operational helpers.

Covers the non-card command-zone state that lives on ``GameState``:

- :class:`Emblem` — created by ultimate planeswalker abilities and other
  effects (CR 114).  Persistent, never leaves the command zone.
- :class:`Dungeon` — chosen via "venture into the dungeon" (CR 309).
  Players advance one room at a time; completing the bottom room removes
  the dungeon from the command zone.
- Day / Night designation (CR 726).  The game tracks a global designation
  that flips based on spells cast per turn; ``DayNight.NEITHER`` is the
  pre-game default.
- The Monarch (CR 716) and The Initiative (Commander Legends: BG)
  designations — held by at most one player at a time and transferred on
  combat damage to a player.
- Generic :class:`CommandZoneObject` for vanguards, conspiracies, planes,
  schemes, attractions, and phenomena.

This module is **operational** only — pure state mutators with logging.
The card-effect parser (``triggers.py`` / ``spell_effects.py``) is
responsible for *invoking* these helpers when an oracle text says so.

References:
    CR 114 (Emblems), 309 (Dungeons), 716 (The Monarch),
    726 (Day and Night), 113.6.1 (command-zone objects).
"""

from __future__ import annotations

from typing import Optional

from src.engine.game_state import (
    CommandZoneObject,
    DayNight,
    Dungeon,
    Emblem,
    GameState,
)


# ---------------------------------------------------------------------------
# Emblems (CR 114)
# ---------------------------------------------------------------------------


def create_emblem(state: GameState, controller_id: str, source: str, text: str) -> Emblem:
    """Put an emblem into ``controller_id``'s command zone.

    Returns the created :class:`Emblem` so callers can reference it later
    (emblem abilities are static / triggered and need a stable id).
    """
    emblem = Emblem(controller_id=controller_id, source=source, text=text)
    state.emblems.append(emblem)
    state.log(f"  ⊕ Emblem ({source}) enters the command zone for {controller_id}")
    return emblem


def emblems_for(state: GameState, player_id: str) -> list[Emblem]:
    """Every emblem currently controlled by ``player_id``."""
    return [e for e in state.emblems if e.controller_id == player_id]


# ---------------------------------------------------------------------------
# Dungeons (CR 309)
# ---------------------------------------------------------------------------


# Standard dungeons from AFR + CLB.  Each entry maps name -> ordered list
# of room names (top entry is the first room you enter).  The room *effect*
# text isn't modelled here yet; callers that want to fire effects on entry
# should look up the room by ``(dungeon.name, dungeon.rooms[dungeon.current_room])``.
DUNGEONS: dict[str, list[str]] = {
    "Tomb of Annihilation": [
        "Trapped Entry",
        "Veiled Path",
        "Oubliette",
        "Sandfall Cell",
        "Cradle of the Death God",
    ],
    "Lost Mine of Phandelver": [
        "Goblin Hideout",
        "Storeroom",
        "Dark Pool",
    ],
    "Dungeon of the Mad Mage": [
        "Yawning Portal",
        "Secret Entrance",
        "Goblin Bazaar",
        "Twisted Caverns",
        "Skullport",
        "Trollskull Alley",
        "Muiral's Graveyard",
        "Lord Robe's Manor",
        "Runic Chamber",
        "Mad Wizard's Lair",
    ],
    "Undercity": [
        "Forum of the Stone Giant Lord",
        "Stash",
        "Goblin Bazaar",
        "Hall of Records",
        "Stash of the Spider Cult",
        "Throne of the Dead Three",
    ],
}


def venture_into_dungeon(
    state: GameState, controller_id: str, dungeon_name: str = ""
) -> Optional[Dungeon]:
    """Advance ``controller_id`` one room into a dungeon (CR 309.4).

    If the player isn't currently in a dungeon, they choose a new one
    (``dungeon_name`` is required in that case; defaults to the smallest
    dungeon, "Lost Mine of Phandelver", if omitted) and enter its top
    room.  If they already have an active dungeon, they advance to the
    next room of that dungeon, ignoring ``dungeon_name``.
    """
    active = next(
        (d for d in state.dungeons if d.controller_id == controller_id and not d.completed),
        None,
    )
    if active is None:
        chosen = dungeon_name or "Lost Mine of Phandelver"
        rooms = DUNGEONS.get(chosen)
        if rooms is None:
            state.log(f"[venture] unknown dungeon '{chosen}'")
            return None
        active = Dungeon(
            controller_id=controller_id,
            name=chosen,
            rooms=list(rooms),
            current_room=0,
        )
        state.dungeons.append(active)
        state.log(f"  ⛬ {controller_id} ventures into {chosen} → '{rooms[0]}'")
        return active

    # Already in a dungeon: advance one room.
    next_room = active.current_room + 1
    if next_room >= len(active.rooms):
        active.completed = True
        state.dungeons.remove(active)
        state.log(f"  ⛬ {controller_id} completes {active.name}")
        return active
    active.current_room = next_room
    state.log(
        f"  ⛬ {controller_id} ventures further into {active.name} "
        f"→ '{active.rooms[next_room]}'"
    )
    return active


def active_dungeon(state: GameState, controller_id: str) -> Optional[Dungeon]:
    """The dungeon ``controller_id`` is currently in, or ``None``."""
    return next(
        (d for d in state.dungeons if d.controller_id == controller_id and not d.completed),
        None,
    )


# ---------------------------------------------------------------------------
# Day / Night (CR 726)
# ---------------------------------------------------------------------------


def set_day_night(state: GameState, value: DayNight) -> None:
    """Set the day/night designation.  Logs the transition (transform-y!)."""
    if state.day_night == value:
        return
    prev = state.day_night
    state.day_night = value
    state.log(f"  ☀☾  Day/Night: {prev.value} → {value.value}")


def update_day_night_for_upkeep(state: GameState, spells_cast_last_turn: int) -> None:
    """Apply the upkeep day/night flip (CR 726.3).

    During the active player's upkeep:
      * If it's neither day nor night, do nothing.
      * If it became day and the previous turn's active player cast no
        spells → it becomes night.
      * If it became night and the previous turn's active player cast two
        or more spells → it becomes day.
    """
    if state.day_night == DayNight.NEITHER:
        return
    if state.day_night == DayNight.DAY and spells_cast_last_turn == 0:
        set_day_night(state, DayNight.NIGHT)
    elif state.day_night == DayNight.NIGHT and spells_cast_last_turn >= 2:
        set_day_night(state, DayNight.DAY)


# ---------------------------------------------------------------------------
# The Monarch / The Initiative
# ---------------------------------------------------------------------------


def become_monarch(state: GameState, player_id: str) -> None:
    """``player_id`` becomes the monarch (CR 716)."""
    if state.monarch == player_id:
        return
    prev = state.monarch
    state.monarch = player_id
    if prev:
        state.log(f"  ♛ The monarch passes from {prev} to {player_id}")
    else:
        state.log(f"  ♛ {player_id} becomes the monarch")


def take_initiative(state: GameState, player_id: str) -> None:
    """``player_id`` takes the Initiative (CLB).  Triggers a venture into Undercity."""
    if state.the_initiative == player_id:
        return
    prev = state.the_initiative
    state.the_initiative = player_id
    if prev:
        state.log(f"  ⚔ The Initiative passes from {prev} to {player_id}")
    else:
        state.log(f"  ⚔ {player_id} takes the Initiative")
    venture_into_dungeon(state, player_id, "Undercity")


# ---------------------------------------------------------------------------
# Generic command-zone objects (vanguard / conspiracy / plane / scheme / ...)
# ---------------------------------------------------------------------------


def add_command_zone_object(
    state: GameState,
    controller_id: str,
    kind: str,
    name: str,
    card_data: Optional[dict] = None,
) -> CommandZoneObject:
    """Place a non-card command-zone object (vanguard, conspiracy, etc.)."""
    obj = CommandZoneObject(
        controller_id=controller_id,
        kind=kind,
        name=name,
        card_data=card_data or {},
    )
    state.command_zone_objects.append(obj)
    state.log(f"  ⌬ {kind.title()}: '{name}' enters {controller_id}'s command zone")
    return obj


def command_zone_objects_for(
    state: GameState, player_id: str, kind: Optional[str] = None
) -> list[CommandZoneObject]:
    """Return ``player_id``'s command-zone objects, optionally filtered by ``kind``."""
    out = [o for o in state.command_zone_objects if o.controller_id == player_id]
    if kind is not None:
        out = [o for o in out if o.kind == kind]
    return out

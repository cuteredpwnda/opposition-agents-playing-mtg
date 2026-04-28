"""Tests for command-zone designations: emblems, dungeons, day/night, initiative,
the monarch, and partner / background multi-commander setup."""

from __future__ import annotations

from src.engine.command_zone import (
    DUNGEONS,
    active_dungeon,
    add_command_zone_object,
    become_monarch,
    command_zone_objects_for,
    create_emblem,
    emblems_for,
    set_day_night,
    take_initiative,
    update_day_night_for_upkeep,
    venture_into_dungeon,
)
from src.engine.game_state import DayNight, GameState, PlayerState, Phase


def _state(*pids: str) -> GameState:
    return GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id=p) for p in pids],
    )


def test_emblem_lifetime():
    s = _state("A", "B")
    e = create_emblem(s, "A", "Sorin, Solemn Visitor", "Creatures get +1/+0 ...")
    assert e in s.emblems
    assert emblems_for(s, "A") == [e]
    assert emblems_for(s, "B") == []


def test_venture_creates_then_advances():
    s = _state("A")
    d1 = venture_into_dungeon(s, "A", "Lost Mine of Phandelver")
    assert d1.current_room == 0 and d1.name == "Lost Mine of Phandelver"
    venture_into_dungeon(s, "A")  # no name -> advance current
    assert active_dungeon(s, "A").current_room == 1
    venture_into_dungeon(s, "A")
    assert active_dungeon(s, "A").current_room == 2
    venture_into_dungeon(s, "A")  # last room -> completes & removes
    assert active_dungeon(s, "A") is None


def test_venture_default_dungeon_is_phandelver():
    s = _state("A")
    d = venture_into_dungeon(s, "A")
    assert d.name == "Lost Mine of Phandelver"


def test_initiative_triggers_undercity_venture():
    s = _state("A", "B")
    take_initiative(s, "A")
    assert s.the_initiative == "A"
    d = active_dungeon(s, "A")
    assert d is not None and d.name == "Undercity"


def test_day_night_flip_rules():
    s = _state("A")
    set_day_night(s, DayNight.DAY)
    update_day_night_for_upkeep(s, spells_cast_last_turn=0)
    assert s.day_night == DayNight.NIGHT
    update_day_night_for_upkeep(s, spells_cast_last_turn=2)
    assert s.day_night == DayNight.DAY
    update_day_night_for_upkeep(s, spells_cast_last_turn=1)
    assert s.day_night == DayNight.DAY  # 1 spell on day -> stays day


def test_monarch_passes_between_players():
    s = _state("A", "B")
    become_monarch(s, "A")
    assert s.monarch == "A"
    become_monarch(s, "B")
    assert s.monarch == "B"


def test_command_zone_object_filter_by_kind():
    s = _state("A", "B")
    add_command_zone_object(s, "A", "vanguard", "Urza Avatar")
    add_command_zone_object(s, "A", "conspiracy", "Backup Plan")
    add_command_zone_object(s, "B", "vanguard", "Mishra Avatar")
    vanguards_a = command_zone_objects_for(s, "A", kind="vanguard")
    assert len(vanguards_a) == 1 and vanguards_a[0].name == "Urza Avatar"
    assert len(command_zone_objects_for(s, "A")) == 2
    assert len(command_zone_objects_for(s, "B", kind="vanguard")) == 1


def test_commanders_setter_accepts_strings_and_lists():
    s = _state("A", "B")
    s.commanders = {"A": "id-a1", "B": ["id-b1", "id-b2"]}
    # Backward-compat: ``commanders`` view returns first id per player.
    assert s.commanders == {"A": "id-a1", "B": "id-b1"}
    assert s.commander_ids("A") == ["id-a1"]
    assert s.commander_ids("B") == ["id-b1", "id-b2"]


def test_add_commander_dedupes():
    s = _state("A")
    s.add_commander("A", "id1")
    s.add_commander("A", "id2")
    s.add_commander("A", "id1")  # dup
    assert s.commander_ids("A") == ["id1", "id2"]


def test_dungeons_registry_has_known_entries():
    assert "Lost Mine of Phandelver" in DUNGEONS
    assert "Undercity" in DUNGEONS
    assert len(DUNGEONS["Lost Mine of Phandelver"]) == 3

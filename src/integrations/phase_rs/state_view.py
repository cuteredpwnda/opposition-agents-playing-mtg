"""Typed read-only view over phase-rs's ``state`` snapshot.

phase-server delivers ``StateUpdate`` payloads as a single deep JSON object
mirroring its internal :rust-struct:`GameState`. Picker / agent code wants
a small, stable surface — life totals, hand size, lands available, what
phase we're in, what kind of legal actions are on offer — without having
to memorise the snapshot schema.

:class:`PhaseRsStateView` provides that surface with the fields most
heuristics care about. Anything more exotic should reach into
:attr:`PhaseRsStateView.raw`, which exposes the full dict so we never
fence agents off from advanced data.

This view is **derived data only**. The authoritative state lives
server-side; our local view is rebuilt from each ``StateUpdate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class PhaseRsPlayerView:
    """Per-seat slice of the snapshot, oriented around ``our_seat``."""

    seat: int
    life: int
    hand_size: int
    library_size: int
    graveyard_size: int
    lands_played_this_turn: int
    poison_counters: int
    is_eliminated: bool

    @classmethod
    def from_state(cls, player: dict[str, Any]) -> "PhaseRsPlayerView":
        return cls(
            seat=int(player.get("id", -1)),
            life=int(player.get("life", 0)),
            hand_size=len(player.get("hand", []) or []),
            library_size=len(player.get("library", []) or []),
            graveyard_size=len(player.get("graveyard", []) or []),
            lands_played_this_turn=int(player.get("lands_played_this_turn", 0)),
            poison_counters=int(player.get("poison_counters", 0)),
            is_eliminated=bool(player.get("is_eliminated", False)),
        )


@dataclass(frozen=True)
class PhaseRsStateView:
    """Read-only projection of a phase-rs ``state`` snapshot.

    Most heuristics only need a handful of fields; everything else stays
    accessible via :attr:`raw`.
    """

    our_seat: int
    turn_number: int
    phase: str
    active_player: int
    priority_player: int
    waiting_for_type: str
    me: PhaseRsPlayerView
    opponents: tuple[PhaseRsPlayerView, ...]
    battlefield_ids: tuple[int, ...]
    stack_size: int
    legal_action_types: tuple[str, ...]
    raw: dict[str, Any] = field(repr=False)

    # ---- convenience predicates that heuristics keep re-deriving --------

    @property
    def is_my_turn(self) -> bool:
        return self.active_player == self.our_seat

    @property
    def i_have_priority(self) -> bool:
        return self.priority_player == self.our_seat

    @property
    def is_main_phase(self) -> bool:
        return self.phase in {"PreCombatMain", "PostCombatMain", "Main1", "Main2"}

    @property
    def is_combat_phase(self) -> bool:
        return self.phase in {
            "BeginningOfCombat",
            "DeclareAttackers",
            "DeclareBlockers",
            "CombatDamage",
            "EndOfCombat",
        }

    @property
    def stack_is_empty(self) -> bool:
        return self.stack_size == 0

    @property
    def can_play_land(self) -> bool:
        """A ``PlayLand`` action is in the offered legal actions."""
        return "PlayLand" in self.legal_action_types

    def battlefield_objects(self) -> list[dict[str, Any]]:
        """Resolve battlefield object-IDs against ``raw['objects']``.

        Returns a list of object dicts (mana cost, name, types, controller).
        Falls back to skipping IDs that aren't in ``objects`` (which
        shouldn't happen in a well-formed snapshot, but we never want a
        rendering helper to blow up an agent decision).
        """
        objects = self.raw.get("objects") or {}
        out: list[dict[str, Any]] = []
        for oid in self.battlefield_ids:
            obj = objects.get(str(oid)) or objects.get(oid)
            if isinstance(obj, dict):
                out.append(obj)
        return out

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_snapshot(
        cls,
        state: dict[str, Any],
        legal_actions: Sequence[dict[str, Any]],
        our_seat: int,
    ) -> "PhaseRsStateView":
        """Build a view from a phase-rs ``state`` dict + ``legal_actions``.

        ``state`` is exactly the dict in ``StateUpdate.state`` /
        ``GameStarted.state``. ``legal_actions`` is the most recent
        ``legal_actions`` field for ``our_seat`` (passed alongside because
        the server sometimes splits the two across messages).
        """
        players = state.get("players") or []
        me_dict = next(
            (p for p in players if int(p.get("id", -1)) == our_seat),
            None,
        )
        if me_dict is None:
            # Degenerate snapshot — synthesise an empty player so downstream
            # code doesn't have to None-check.
            me_dict = {"id": our_seat}
        me = PhaseRsPlayerView.from_state(me_dict)
        opps = tuple(
            PhaseRsPlayerView.from_state(p)
            for p in players
            if int(p.get("id", -1)) != our_seat
        )

        waiting_for = state.get("waiting_for") or {}
        legal_types = tuple(
            str(a.get("type", "")) for a in legal_actions if isinstance(a, dict)
        )

        return cls(
            our_seat=our_seat,
            turn_number=int(state.get("turn_number", 0)),
            phase=str(state.get("phase", "Unknown")),
            active_player=int(state.get("active_player", -1)),
            priority_player=int(state.get("priority_player", -1)),
            waiting_for_type=str(waiting_for.get("type", "")),
            me=me,
            opponents=opps,
            battlefield_ids=tuple(int(i) for i in (state.get("battlefield") or [])),
            stack_size=len(state.get("stack") or []),
            legal_action_types=legal_types,
            raw=state,
        )

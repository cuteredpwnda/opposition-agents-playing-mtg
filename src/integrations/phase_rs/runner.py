"""Drive a single phase-rs game from a Python ``ActionPicker``.

This is the smallest useful loop: connect to a running ``phase-server``,
create a 1v1 game with one Python-controlled seat and one phase-ai opponent,
then drive the human seat by repeatedly receiving ``StateUpdate`` and
sending the picker's chosen action back as an ``Action`` envelope.

The server picks the seat order and runs all rules/SBAs/triggers — our side
just chooses one ``legal_actions`` index per turn.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any


def _is_our_turn_to_act(
    waiting_for: Any, our_seat: int, legal_actions: list[dict[str, Any]]
) -> bool:
    """Return True if our seat is genuinely the actor for the current state.

    The server sends ``legal_actions`` as the *union* across every pending
    player for simultaneous-decision states (MulliganDecision,
    MulliganBottomCards). A non-empty list does not on its own mean we are
    expected to act — the AI seat may still be the only pending player.

    Check ``waiting_for.pending`` for our seat. For non-simultaneous states
    (the common case) we trust that a non-empty ``legal_actions`` implies it
    is our turn (the server only routes legal actions to the active player).
    """
    if not legal_actions:
        return False
    if not isinstance(waiting_for, dict):
        return True
    wf_type = waiting_for.get("type", "")
    if wf_type not in {"MulliganDecision", "MulliganBottomCards"}:
        return True
    data = waiting_for.get("data") or {}
    pending = data.get("pending") or []
    for entry in pending:
        # `pending` entries are MulliganDecisionEntry { player: PlayerId(u8), ... }
        # PlayerId serializes transparently as a u8.
        if isinstance(entry, dict):
            p = entry.get("player")
        else:
            p = entry
        if isinstance(p, dict):
            p = p.get("0") if "0" in p else None
        if p == our_seat:
            return True
    return False


def _our_simultaneous_round_key(
    waiting_for: Any, our_seat: int
) -> tuple[str, int, str] | None:
    """Return a (waiting_for_type, our_mulligan_count, phase_type) key identifying the
    *current* simultaneous-decision round for our seat.

    The server keeps a player in ``pending`` after they submit (until both
    decide). Within one round the ``waiting_for`` JSON can mutate (e.g. the
    other player's ``chosen`` field flips) without the round actually
    advancing. Protocol v106 also moves from choosing Keep to BottomCards
    inside the same MulliganDecision variant and mulligan count. Include
    that per-seat phase so bottoming is not suppressed as a duplicate Keep.

    Returns ``None`` if waiting_for is not a simultaneous type or our seat
    is not in pending.
    """
    if not isinstance(waiting_for, dict):
        return None
    wf_type = waiting_for.get("type", "")
    if wf_type not in {"MulliganDecision", "MulliganBottomCards"}:
        return None
    data = waiting_for.get("data") or {}
    for entry in data.get("pending") or []:
        if not isinstance(entry, dict):
            continue
        p = entry.get("player")
        if isinstance(p, dict):
            p = p.get("0")
        if p != our_seat:
            continue
        mc = entry.get("mulligan_count", 0)
        try:
            mc = int(mc)
        except (TypeError, ValueError):
            mc = 0
        phase = entry.get("phase")
        phase_type = str(phase.get("type", "")) if isinstance(phase, dict) else ""
        return (wf_type, mc, phase_type)
    return None


def _dedup_legal_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate action payloads.

    For simultaneous-decision waiting_for variants (MulliganDecision,
    MulliganBottomCards), the engine's ``legal_actions_full`` emits one set
    of candidates per pending player, with the per-player metadata stripped
    when serialized to JSON. The resulting list looks like
    ``[Keep, Mulligan, Keep, Mulligan]`` with no way for the client to know
    which entry belongs to which player. Since the server resolves the actor
    from the WebSocket token (not the action payload), the duplicates are
    semantically identical from our seat's perspective — picking the second
    ``Keep`` would actually apply Keep to *us*, but if we've already kept,
    the server rejects the action.

    Dedup by stable JSON repr so the picker can only choose among distinct
    payloads.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for a in actions:
        key = json.dumps(a, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _trace_state_snippet(state: dict[str, Any], our_seat: int) -> dict[str, Any]:
    """Return a compact, explainability-friendly state snapshot for traces.

    Keep this intentionally small and stable so downstream analytics can
    reason about why actions were taken without serializing full game state.
    """

    def _to_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return None

    snippet: dict[str, Any] = {
        "turn": _to_int(state.get("turn_number")),
        "phase": str(state.get("phase", "Unknown")),
    }

    waiting_for = state.get("waiting_for")
    if isinstance(waiting_for, dict):
        snippet["waiting_for_type"] = str(waiting_for.get("type", ""))

    active_player = _to_int(state.get("active_player"))
    if active_player is not None:
        snippet["active_player"] = active_player

    priority_player = _to_int(state.get("priority_player"))
    if priority_player is not None:
        snippet["priority_player"] = priority_player

    stack = state.get("stack")
    if isinstance(stack, list):
        snippet["stack_depth"] = len(stack)

    players = state.get("players")
    player_entries: list[tuple[int, dict[str, Any]]] = []
    if isinstance(players, list):
        for seat, pdata in enumerate(players):
            if isinstance(pdata, dict):
                player_entries.append((seat, pdata))
    elif isinstance(players, dict):
        for seat_key, pdata in players.items():
            if not isinstance(pdata, dict):
                continue
            try:
                seat = int(seat_key)
            except (TypeError, ValueError):
                continue
            player_entries.append((seat, pdata))

    if player_entries:
        summaries: list[dict[str, Any]] = []
        for seat, pdata in sorted(player_entries, key=lambda x: x[0]):
            life = pdata.get("life")
            if life is None:
                life = pdata.get("life_total")

            hand_size = pdata.get("hand_size")
            if hand_size is None and isinstance(pdata.get("hand"), list):
                hand_size = len(pdata["hand"])

            battlefield_size = pdata.get("battlefield_size")
            if battlefield_size is None and isinstance(pdata.get("battlefield"), list):
                battlefield_size = len(pdata["battlefield"])

            graveyard_size = pdata.get("graveyard_size")
            if graveyard_size is None and isinstance(pdata.get("graveyard"), list):
                graveyard_size = len(pdata["graveyard"])

            library_size = pdata.get("library_size")
            if library_size is None and isinstance(pdata.get("library"), list):
                library_size = len(pdata["library"])

            p_summary: dict[str, Any] = {"seat": seat}
            for field_name, value in (
                ("life", life),
                ("hand_size", hand_size),
                ("battlefield_size", battlefield_size),
                ("graveyard_size", graveyard_size),
                ("library_size", library_size),
            ):
                v = _to_int(value)
                if v is not None:
                    p_summary[field_name] = v
            summaries.append(p_summary)

        snippet["players"] = summaries
        for p in summaries:
            if p.get("seat") == our_seat:
                snippet["our_player"] = p
                break

    return snippet

from src.integrations.phase_rs.agent_bridge import ActionPicker, PickerError, RandomActionPicker
from src.integrations.phase_rs.client import (
    ActionFailed,
    ActionRejected,
    GameOver,
    GameStarted,
    PhaseServerClient,
    PhaseServerConfig,
    PhaseServerError,
    StateUpdate,
)
from src.integrations.phase_rs.server_lock import acquire_server_lock, release_server_lock
from src.integrations.phase_rs.server_process import PhaseServerProcess

# Import websockets exceptions if available (for better error handling)
try:
    from websockets.exceptions import ConnectionClosedError
except ImportError:
    ConnectionClosedError = ConnectionError  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class GameRunResult:
    """Outcome of a single ``run_game`` call."""

    winner_seat: int | None
    reason: str
    our_seat: int
    turns_observed: int
    actions_sent: int
    final_state: dict[str, Any] | None
    log: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


async def run_game(
    *,
    deck: dict[str, Any],
    picker: ActionPicker | None = None,
    config: PhaseServerConfig | None = None,
    ai_difficulty: str = "Medium",
    ai_deck_name: str = "Red Deck Wins",
    display_name: str = "OppositionAgent",
    max_actions: int = 2000,
    autostart_server: bool = False,
    server_process: PhaseServerProcess | None = None,
    reconnect_attempts: int = 2,
    format_name: str | None = None,
    ai_decks: list[dict[str, Any]] | None = None,
    max_game_seconds: float | None = None,
    max_turns: int | None = None,
) -> GameRunResult:
    """Play one game against native phase-ai opponents and return the result.

    ``deck`` is the JSON ``DeckData`` our seat will use (see
    ``src.integrations.phase_rs.decks``). ``ai_deck_name`` selects one of
    phase-rs's built-in starter decks for the AI seat -- the server resolves
    the card list internally, so we don't need to know its contents.
    ``ai_decks`` instead supplies explicit decks for seats 1 onward, allowing
    a four-player Commander pod with three native AI opponents.

    ``max_actions`` is a hard cap so a misbehaving picker can't spin
    forever; the loop exits with ``reason="action_cap"`` if hit.

    ``format_name`` selects the game format (Standard, Pioneer, Modern,
    Commander, Brawl, etc.). Defaults to Standard if not specified.

    When ``autostart_server=True`` (or ``server_process`` is supplied) we
    bring up a local ``phase-server`` subprocess for the lifetime of this
    call. If the port is already open we adopt the existing instance and
    don't kill anything on exit. See :class:`PhaseServerProcess`.
    """
    picker = picker or RandomActionPicker()
    if max_game_seconds is not None and (
        not math.isfinite(max_game_seconds) or max_game_seconds <= 0
    ):
        raise ValueError("max_game_seconds must be finite and positive")
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turns must be positive")
    cfg = config or PhaseServerConfig()
    log: list[str] = []

    owned_server: PhaseServerProcess | None = None
    server_lock_fd: int | None = None
    
    if server_process is not None:
        owned_server = server_process
        if owned_server._proc is None and not owned_server._adopted:  # noqa: SLF001
            owned_server.start()
        cfg = PhaseServerConfig(
            uri=owned_server.uri,
            headers=cfg.headers,
            request_timeout_s=cfg.request_timeout_s,
            stream_timeout_s=cfg.stream_timeout_s,
            client_version=cfg.client_version,
            build_commit=cfg.build_commit,
            max_message_bytes=cfg.max_message_bytes,
        )
    elif autostart_server:
        # Acquire lock to prevent multiple processes from starting their own servers
        server_lock_fd = acquire_server_lock(timeout=5.0)
        if server_lock_fd is not None:
            owned_server = PhaseServerProcess().start()
            cfg = PhaseServerConfig(
                uri=owned_server.uri,
                headers=cfg.headers,
                request_timeout_s=cfg.request_timeout_s,
                stream_timeout_s=cfg.stream_timeout_s,
                client_version=cfg.client_version,
                build_commit=cfg.build_commit,
                max_message_bytes=cfg.max_message_bytes,
            )
        else:
            log.append("could not acquire server lock; attempting to connect to existing server")
            # Fall through: cfg already has default URI

    try:
        return await _play(
            cfg=cfg,
            deck=deck,
            picker=picker,
            ai_difficulty=ai_difficulty,
            ai_deck_name=ai_deck_name,
            display_name=display_name,
            max_actions=max_actions,
            log=log,
            reconnect_attempts=reconnect_attempts,
            format_name=format_name,
            ai_decks=ai_decks,
            max_game_seconds=max_game_seconds,
            max_turns=max_turns,
        )
    finally:
        # Clean up server lock
        if server_lock_fd is not None:
            release_server_lock(server_lock_fd)
        # Only stop a server we spun up implicitly (autostart_server=True).
        # When the caller passed an explicit ``server_process`` we leave
        # lifecycle to them.
        if autostart_server and owned_server is not None and server_process is None:
            owned_server.stop()


async def _play(
    *,
    cfg: PhaseServerConfig,
    deck: dict[str, Any],
    picker: ActionPicker,
    ai_difficulty: str,
    ai_deck_name: str,
    display_name: str,
    max_actions: int,
    log: list[str],
    reconnect_attempts: int = 2,
    format_name: str | None = None,
    ai_decks: list[dict[str, Any]] | None = None,
    max_game_seconds: float | None = None,
    max_turns: int | None = None,
) -> GameRunResult:
    # Never block forever in the streaming loop. If the server goes quiet
    # unexpectedly (dropped GameOver, stale socket, etc.), attempt to
    # reconnect and resume; if all reconnection attempts fail, end cleanly.
    # Default 180s allows phase-ai time to chain many decisions per turn.
    stream_timeout_s = cfg.stream_timeout_s if cfg.stream_timeout_s is not None else 180.0
    deadline = time.perf_counter() + max_game_seconds if max_game_seconds is not None else None

    async with PhaseServerClient(cfg) as client:
        hello = await client.handshake()
        log.append(f"connected: server={hello.server_version} build={hello.build_commit}")

        created = await client.create_game_with_ai(
            deck=deck,
            display_name=display_name,
            ai_difficulty=ai_difficulty,
            ai_deck_name=ai_deck_name,
            format_name=format_name,
            ai_decks=ai_decks,
        )
        log.append(f"game created: code={created.game_code}")

        started = await client.expect("GameStarted")
        assert isinstance(started, GameStarted)
        full_key = started.full_key or created.full_key
        if created.full_key is not None and full_key != created.full_key:
            raise PhaseServerError("GameStarted session identity differs from GameCreated")
        our_seat = started.your_player
        log.append(f"game started: our_seat={our_seat} opp={started.opponent_name!r}")

        latest_state: dict[str, Any] = started.state
        state_revision = started.state_revision
        legal_actions: list[dict[str, Any]] = _dedup_legal_actions(list(started.legal_actions))
        trace: list[dict[str, Any]] = [
            {
                "event": "game_started",
                "server_version": hello.server_version,
                "server_build_commit": hello.build_commit,
                "protocol_version": hello.protocol_version,
                "our_seat": our_seat,
                "player_count": 2 if ai_decks is None else len(ai_decks) + 1,
                "max_message_bytes": cfg.max_message_bytes,
                "turn": int(started.state.get("turn_number", 0)),
                "phase": str(started.state.get("phase", "Unknown")),
                "legal_action_types": [str(a.get("type", "")) for a in legal_actions],
                "state": _trace_state_snippet(started.state, our_seat),
            }
        ]

        actions_sent = 0
        turns_observed = 1
        # For simultaneous-decision rounds (mulligan), record the per-seat
        # round key at submit-time so we don't re-submit before the round
        # advances. See _our_simultaneous_round_key.
        submitted_round_key: tuple[str, int, str] | None = None

        while True:
            waiting = latest_state.get("waiting_for") or {}
            if isinstance(waiting, dict) and waiting.get("type") == "GameOver":
                winner = (waiting.get("data") or {}).get("winner")
                log.append(f"game over (restored state): winner={winner}")
                trace.append({
                    "event": "game_over", "winner": winner, "reason": "game_rules",
                    "turn": int(latest_state.get("turn_number", 0)),
                    "state": _trace_state_snippet(latest_state, our_seat),
                })
                return GameRunResult(
                    winner_seat=winner, reason="game_rules", our_seat=our_seat,
                    turns_observed=turns_observed, actions_sent=actions_sent,
                    final_state=latest_state, log=log, trace=trace,
                )
            if (deadline is not None and time.perf_counter() >= deadline) or (
                max_turns is not None and int(latest_state.get("turn_number", 0)) > max_turns
            ):
                reason = (
                    "game_timeout" if deadline is not None and time.perf_counter() >= deadline
                    else "turn_cap"
                )
                log.append(f"{reason}: stopped at the configured experiment budget")
                trace.append({"event": reason, "turn": latest_state.get("turn_number", 0)})
                return GameRunResult(
                    winner_seat=None, reason=reason, our_seat=our_seat,
                    turns_observed=turns_observed, actions_sent=actions_sent,
                    final_state=latest_state, log=log, trace=trace,
                )
            # If it's our turn to act (the server only sends legal_actions
            # when we have priority / a decision), send one.
            # For simultaneous mulligan states the server broadcasts the
            # *union* of all pending players' actions, so check pending too.
            # Skip if we already submitted in this simultaneous round and
            # the round hasn't advanced yet (our pending entry persists
            # until both players decide; intra-round waiting_for mutations
            # — e.g. the other seat's chosen flipping — must NOT trigger a
            # re-submit).
            current_round_key = _our_simultaneous_round_key(
                latest_state.get("waiting_for"), our_seat
            )
            already_submitted = (
                submitted_round_key is not None
                and current_round_key == submitted_round_key
            )
            if already_submitted:
                legal_actions = []
            if legal_actions and _is_our_turn_to_act(
                latest_state.get("waiting_for"), our_seat, legal_actions
            ):
                # Drain any queued StateUpdates from the server before acting.
                # phase-server schedules AI-follow-up broadcasts with 100ms
                # delays after GameStarted and after each player action. If we
                # act on the first StateUpdate we see (e.g., GameStarted with
                # AI still pending), the server may apply our action against a
                # newer state and reject downstream actions because the broadcast
                # we acted on was stale. Drain with a short grace window so
                # the picker always sees the latest state.
                drain_deadline = asyncio.get_event_loop().time() + 0.15
                early_exit_result: GameRunResult | None = None
                while asyncio.get_event_loop().time() < drain_deadline:
                    try:
                        extra = await client.recv(timeout=0.02)
                    except (asyncio.TimeoutError, ConnectionResetError, ConnectionError, ConnectionClosedError, OSError):
                        break
                    if isinstance(extra, StateUpdate):
                        if extra.state.get("turn_number", 0) != latest_state.get("turn_number", 0):
                            turns_observed += 1
                        latest_state = extra.state
                        state_revision = extra.state_revision
                        legal_actions = _dedup_legal_actions(list(extra.legal_actions))
                        wf = extra.state.get("waiting_for") or {}
                        if isinstance(wf, dict) and wf.get("type") == "GameOver":
                            winner_data = wf.get("data") or {}
                            winner = winner_data.get("winner") if isinstance(winner_data, dict) else None
                            log.append(f"game over (from state, drain): winner={winner}")
                            trace.append({
                                "event": "game_over",
                                "winner": winner,
                                "reason": "game_rules",
                                "turn": int(latest_state.get("turn_number", 0)),
                                "state": _trace_state_snippet(latest_state, our_seat),
                            })
                            early_exit_result = GameRunResult(
                                winner_seat=winner,
                                reason="game_rules",
                                our_seat=our_seat,
                                turns_observed=turns_observed,
                                actions_sent=actions_sent,
                                final_state=latest_state,
                                log=log,
                                trace=trace,
                            )
                            break
                    elif isinstance(extra, GameOver):
                        log.append(f"game over (drain): winner={extra.winner}")
                        trace.append({
                            "event": "game_over",
                            "winner": extra.winner,
                            "reason": extra.reason,
                            "turn": int(latest_state.get("turn_number", 0)),
                            "state": _trace_state_snippet(latest_state, our_seat),
                        })
                        early_exit_result = GameRunResult(
                            winner_seat=extra.winner,
                            reason=extra.reason,
                            our_seat=our_seat,
                            turns_observed=turns_observed,
                            actions_sent=actions_sent,
                            final_state=latest_state,
                            log=log,
                            trace=trace,
                        )
                        break
                    elif isinstance(extra, ActionFailed):
                        log.append(f"{extra.reason}: {extra.message}")
                        trace.append({"event": extra.reason, "message": extra.message})
                        early_exit_result = GameRunResult(
                            winner_seat=None,
                            reason=extra.reason,
                            our_seat=our_seat,
                            turns_observed=turns_observed,
                            actions_sent=actions_sent,
                            final_state=latest_state,
                            log=log,
                            trace=trace,
                        )
                        break
                    elif isinstance(extra, ActionRejected):
                        log.append(f"action rejected during drain: {extra.reason}")
                        trace.append(
                            {
                                "event": "action_rejected",
                                "reason": extra.reason,
                                "turn": int(latest_state.get("turn_number", 0)),
                                "state": _trace_state_snippet(latest_state, our_seat),
                            }
                        )
                        early_exit_result = GameRunResult(
                            winner_seat=None,
                            reason="action_rejected",
                            our_seat=our_seat,
                            turns_observed=turns_observed,
                            actions_sent=actions_sent,
                            final_state=latest_state,
                            log=log,
                            trace=trace,
                        )
                        break
                if early_exit_result is not None:
                    return early_exit_result
                if (deadline is not None and time.perf_counter() >= deadline) or (
                    max_turns is not None and int(latest_state.get("turn_number", 0)) > max_turns
                ):
                    continue
                # After drain, re-check that it's still our turn.
                if not (legal_actions and _is_our_turn_to_act(
                    latest_state.get("waiting_for"), our_seat, legal_actions
                )):
                    # AI is now acting (or game ended) — fall through to recv().
                    pass
                else:
                    if actions_sent >= max_actions:
                        log.append(f"hit action cap ({max_actions}); conceding")
                        await client.concede()
                        return GameRunResult(
                            winner_seat=None,
                            reason="action_cap",
                            our_seat=our_seat,
                            turns_observed=turns_observed,
                            actions_sent=actions_sent,
                            final_state=latest_state,
                            log=log,
                            trace=trace,
                        )
                    decision_started = time.perf_counter()
                    try:
                        if hasattr(picker, "pick_async"):
                            idx = await picker.pick_async(legal_actions, latest_state, our_seat)  # type: ignore[attr-defined]
                        else:
                            idx = picker.pick(legal_actions, latest_state, our_seat)
                    except PickerError as exc:
                        log.append(f"{exc.reason}: {exc}")
                        trace.append({
                            "event": exc.reason,
                            "message": str(exc),
                            "decision_time_sec": time.perf_counter() - decision_started,
                            "picker_reasoning": copy.deepcopy(
                                getattr(picker, "last_reasoning", None)
                            ),
                        })
                        return GameRunResult(
                            winner_seat=None,
                            reason=exc.reason,
                            our_seat=our_seat,
                            turns_observed=turns_observed,
                            actions_sent=actions_sent,
                            final_state=latest_state,
                            log=log,
                            trace=trace,
                        )
                    decision_time = time.perf_counter() - decision_started
                    if not 0 <= idx < len(legal_actions):
                        raise ValueError(
                            f"picker {picker.name} returned index {idx} "
                            f"out of range [0, {len(legal_actions)})"
                        )
                    chosen = legal_actions[idx]
                    logger.debug("phase-rs: seat %d -> %s", our_seat, chosen.get("type"))
                    trace.append(
                        {
                            "event": "decision",
                            "turn": int(latest_state.get("turn_number", 0)),
                            "phase": str(latest_state.get("phase", "Unknown")),
                            "seat": our_seat,
                            "chosen_index": idx,
                            "chosen_type": str(chosen.get("type", "")),
                            "chosen_action": chosen,
                            "decision_time_sec": decision_time,
                            "picker_reasoning": copy.deepcopy(
                                getattr(picker, "last_reasoning", None)
                            ),
                            "legal_action_types": [str(a.get("type", "")) for a in legal_actions],
                            "state": _trace_state_snippet(latest_state, our_seat),
                        }
                    )
                    await client.send_action(chosen)
                    actions_sent += 1
                    # Record submit-time round key so we don't re-submit
                    # before the simultaneous round advances.
                    submitted_round_key = _our_simultaneous_round_key(
                        latest_state.get("waiting_for"), our_seat
                    )
                    legal_actions = []  # wait for next StateUpdate

            # Wait for the next state. Could be StateUpdate or GameOver, plus
            # incidental traffic (events, timer ticks). Use the streaming
            # timeout so AI thinking pauses don't trigger spurious TimeoutErrors.
            try:
                remaining = stream_timeout_s
                if deadline is not None:
                    remaining = max(0.001, min(remaining, deadline - time.perf_counter()))
                msg = await client.recv(timeout=remaining)
            except (
                asyncio.TimeoutError,
                ConnectionResetError,
                ConnectionError,
                ConnectionClosedError,
                OSError,
            ) as e:
                if deadline is not None and time.perf_counter() >= deadline:
                    # Return through the budget guard, never through reconnect.
                    continue
                exc_name = type(e).__name__
                stream_error = str(e)
                log.append(
                    f"stream error ({exc_name}: {stream_error}) "
                    f"with {stream_timeout_s:.1f}s timeout; "
                    f"attempting reconnect"
                )
                attempts_allowed = reconnect_attempts if full_key is not None else 0
                if full_key is None:
                    log.append("cannot reconnect: server did not issue a Full session key")
                for attempt in range(1, attempts_allowed + 1):
                    log.append(
                        f"reconnect attempt {attempt}/{reconnect_attempts}"
                    )
                    try:
                        await client.close()

                        async def restore_session() -> GameStarted:
                            await client.__aenter__()
                            await client.handshake()
                            assert full_key is not None
                            return await client.reconnect(
                                created.game_code, created.player_token, full_key,
                            )

                        timeout = cfg.request_timeout_s * 3
                        if deadline is not None:
                            timeout = max(0.001, min(timeout, deadline - time.perf_counter()))
                        resumed = await asyncio.wait_for(restore_session(), timeout=timeout)
                        if resumed.your_player != our_seat:
                            raise PhaseServerError("Reconnect restored a different controlled seat")
                        if resumed.state_revision < state_revision:
                            raise PhaseServerError("Reconnect restored a stale state revision")
                        if resumed.state.get("turn_number") != latest_state.get("turn_number"):
                            turns_observed += 1
                        latest_state = resumed.state
                        state_revision = resumed.state_revision
                        legal_actions = _dedup_legal_actions(list(resumed.legal_actions))
                        log.append(f"session restored (attempt {attempt})")
                        trace.append(
                            {
                                "event": "reconnect_success",
                                "attempt": attempt,
                                "turn": int(latest_state.get("turn_number", 0)),
                                "state_revision": state_revision,
                                "state": _trace_state_snippet(latest_state, our_seat),
                            }
                        )
                        break
                    except (
                        asyncio.TimeoutError, ConnectionError, ConnectionClosedError,
                        OSError, PhaseServerError,
                    ) as reconnect_error:
                        log.append(f"reconnect attempt {attempt} failed: {reconnect_error}")
                        trace.append({
                            "event": "reconnect_failure", "attempt": attempt,
                            "error": str(reconnect_error),
                        })
                        if deadline is not None and time.perf_counter() >= deadline:
                            break
                else:
                    # All reconnection attempts failed.
                    log.append(
                        f"stream timeout; {attempts_allowed} session restore attempts failed"
                    )
                    trace.append(
                        {
                            "event": "stream_timeout_final",
                            "error": stream_error,
                            "turn": int(latest_state.get("turn_number", 0)),
                            "phase": str(latest_state.get("phase", "Unknown")),
                            "timeout_s": stream_timeout_s,
                            "reconnect_attempts": attempts_allowed,
                            "state": _trace_state_snippet(latest_state, our_seat),
                        }
                    )
                    return GameRunResult(
                        winner_seat=None,
                        reason="stream_timeout",
                        our_seat=our_seat,
                        turns_observed=turns_observed,
                        actions_sent=actions_sent,
                        final_state=latest_state,
                        log=log,
                        trace=trace,
                    )
                continue
            if isinstance(msg, StateUpdate):
                if msg.state.get("turn_number", 0) != latest_state.get("turn_number", 0):
                    turns_observed += 1
                latest_state = msg.state
                state_revision = msg.state_revision
                # legal_actions arrives whenever it is *our* seat's turn to
                # choose. An empty list means the AI seat has priority — keep
                # looping until phase-server sends us another decision.
                legal_actions = _dedup_legal_actions(list(msg.legal_actions))
                # phase-server does NOT emit ServerMessage::GameOver for normal
                # rule-based game ends — it only sets state.waiting_for to
                # GameOver and broadcasts a final StateUpdate. Detect that here
                # so games actually terminate instead of hanging on recv().
                waiting_for = msg.state.get("waiting_for") or {}
                if isinstance(waiting_for, dict) and waiting_for.get("type") == "GameOver":
                    winner_data = waiting_for.get("data") or {}
                    winner = winner_data.get("winner") if isinstance(winner_data, dict) else None
                    log.append(f"game over (from state): winner={winner}")
                    trace.append(
                        {
                            "event": "game_over",
                            "winner": winner,
                            "reason": "game_rules",
                            "turn": int(latest_state.get("turn_number", 0)),
                            "state": _trace_state_snippet(latest_state, our_seat),
                        }
                    )
                    return GameRunResult(
                        winner_seat=winner,
                        reason="game_rules",
                        our_seat=our_seat,
                        turns_observed=turns_observed,
                        actions_sent=actions_sent,
                        final_state=latest_state,
                        log=log,
                        trace=trace,
                    )
            elif isinstance(msg, GameOver):
                log.append(f"game over: winner={msg.winner} reason={msg.reason}")
                trace.append(
                    {
                        "event": "game_over",
                        "winner": msg.winner,
                        "reason": msg.reason,
                        "turn": int(latest_state.get("turn_number", 0)),
                        "state": _trace_state_snippet(latest_state, our_seat),
                    }
                )
                return GameRunResult(
                    winner_seat=msg.winner,
                    reason=msg.reason,
                    our_seat=our_seat,
                    turns_observed=turns_observed,
                    actions_sent=actions_sent,
                    final_state=latest_state,
                    log=log,
                    trace=trace,
                )
            elif isinstance(msg, ActionFailed):
                log.append(f"{msg.reason}: {msg.message}")
                trace.append({"event": msg.reason, "message": msg.message})
                return GameRunResult(
                    winner_seat=None,
                    reason=msg.reason,
                    our_seat=our_seat,
                    turns_observed=turns_observed,
                    actions_sent=actions_sent,
                    final_state=latest_state,
                    log=log,
                    trace=trace,
                )
            elif isinstance(msg, ActionRejected):
                log.append(f"action rejected: {msg.reason}")
                trace.append(
                    {
                        "event": "action_rejected",
                        "reason": msg.reason,
                        "turn": int(latest_state.get("turn_number", 0)),
                        "state": _trace_state_snippet(latest_state, our_seat),
                    }
                )
                return GameRunResult(
                    winner_seat=None,
                    reason="action_rejected",
                    our_seat=our_seat,
                    turns_observed=turns_observed,
                    actions_sent=actions_sent,
                    final_state=latest_state,
                    log=log,
                    trace=trace,
                )
            else:
                # Lobby / timer / emote / opponent (re)connect. Logged at debug
                # so we can spot oddities without polluting the game log.
                msg_type = msg[0] if isinstance(msg, tuple) else type(msg).__name__
                if msg_type == "Error":
                    payload = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else {}
                    message = str(payload.get("message", "server error"))
                    raise PhaseServerError(message)
                logger.debug("phase-rs: incidental %s", msg_type)


def run_game_sync(**kwargs: Any) -> GameRunResult:
    """Synchronous wrapper for callers who don't want asyncio in their loop."""
    return asyncio.run(run_game(**kwargs))

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
import logging
from dataclasses import dataclass, field
from typing import Any

from src.integrations.phase_rs.agent_bridge import ActionPicker, RandomActionPicker
from src.integrations.phase_rs.client import (
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
) -> GameRunResult:
    """Play one game against a phase-ai opponent and return the result.

    ``deck`` is the JSON ``DeckData`` our seat will use (see
    ``src.integrations.phase_rs.decks``). ``ai_deck_name`` selects one of
    phase-rs's built-in starter decks for the AI seat -- the server resolves
    the card list internally, so we don't need to know its contents.

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
) -> GameRunResult:
    # Never block forever in the streaming loop. If the server goes quiet
    # unexpectedly (dropped GameOver, stale socket, etc.), attempt to
    # reconnect and resume; if all reconnection attempts fail, end cleanly.
    # Default 180s allows phase-ai time to chain many decisions per turn.
    stream_timeout_s = cfg.stream_timeout_s if cfg.stream_timeout_s is not None else 180.0

    async with PhaseServerClient(cfg) as client:
        hello = await client.handshake()
        log.append(f"connected: server={hello.server_version} build={hello.build_commit}")

        created = await client.create_game_with_ai(
            deck=deck,
            display_name=display_name,
            ai_difficulty=ai_difficulty,
            ai_deck_name=ai_deck_name,
            format_name=format_name,
        )
        log.append(f"game created: code={created.game_code}")

        started = await client.expect("GameStarted")
        assert isinstance(started, GameStarted)
        our_seat = started.your_player
        log.append(f"game started: our_seat={our_seat} opp={started.opponent_name!r}")

        latest_state: dict[str, Any] = started.state
        legal_actions: list[dict[str, Any]] = list(started.legal_actions)
        trace: list[dict[str, Any]] = [
            {
                "event": "game_started",
                "our_seat": our_seat,
                "turn": int(started.state.get("turn_number", 0)),
                "phase": str(started.state.get("phase", "Unknown")),
                "legal_action_types": [str(a.get("type", "")) for a in legal_actions],
            }
        ]

        actions_sent = 0
        turns_observed = 1

        while True:
            # If it's our turn to act (the server only sends legal_actions
            # when we have priority / a decision), send one.
            if legal_actions:
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
                if hasattr(picker, "pick_async"):
                    idx = await picker.pick_async(legal_actions, latest_state, our_seat)  # type: ignore[attr-defined]
                else:
                    idx = picker.pick(legal_actions, latest_state, our_seat)
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
                        "legal_action_types": [str(a.get("type", "")) for a in legal_actions],
                    }
                )
                await client.send_action(chosen)
                actions_sent += 1
                legal_actions = []  # wait for next StateUpdate

            # Wait for the next state. Could be StateUpdate or GameOver, plus
            # incidental traffic (events, timer ticks). Use the streaming
            # timeout so AI thinking pauses don't trigger spurious TimeoutErrors.
            try:
                msg = await client.recv(timeout=stream_timeout_s)
            except (
                asyncio.TimeoutError,
                ConnectionResetError,
                ConnectionError,
                ConnectionClosedError,
                OSError,
            ) as e:
                # Stream timeout or connection errors: phase-ai's turn may have chained many
                # decisions, or the network connection dropped. Attempt to reconnect and resume.
                exc_name = type(e).__name__
                log.append(
                    f"stream error ({exc_name}) after {stream_timeout_s:.1f}s; "
                    f"attempting reconnect"
                )
                for attempt in range(1, reconnect_attempts + 1):
                    log.append(
                        f"reconnect attempt {attempt}/{reconnect_attempts}"
                    )
                    try:
                        # Close old connection and open a new one, then attempt
                        # to resume the existing game using the player token.
                        player_token = client.config.headers.get("X-Player-Token")
                        game_code = None  # Would need to track this from GameCreated
                        
                        await client.close()
                        await asyncio.sleep(0.5)
                        new_client = PhaseServerClient(cfg)
                        await new_client.__aenter__()
                        # Handshake to verify connection is live.
                        await new_client.handshake()
                        logger.debug(
                            "phase-rs: reconnect attempt %d established new connection",
                            attempt,
                        )
                        # Patch the client in the outer context so we can keep looping.
                        client = new_client  # noqa: F841 - reassign for next iteration
                        log.append(f"reconnected successfully (attempt {attempt})")
                        trace.append(
                            {
                                "event": "reconnect_success",
                                "attempt": attempt,
                                "turn": int(latest_state.get("turn_number", 0)),
                            }
                        )
                        # Note: Full game state recovery is not yet implemented.
                        # The server may continue from where it left off if the
                        # websocket transport itself handles it. Otherwise the
                        # game is lost and we should implement a proper rejoin.
                        break  # Success; continue main loop
                    except Exception as e:
                        logger.debug(
                            "phase-rs: reconnect attempt %d failed: %s", attempt, e
                        )
                        await asyncio.sleep(0.5)
                else:
                    # All reconnection attempts failed.
                    log.append(
                        f"stream timeout and all {reconnect_attempts} reconnect attempts failed"
                    )
                    trace.append(
                        {
                            "event": "stream_timeout_final",
                            "turn": int(latest_state.get("turn_number", 0)),
                            "phase": str(latest_state.get("phase", "Unknown")),
                            "timeout_s": stream_timeout_s,
                            "reconnect_attempts": reconnect_attempts,
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
                continue  # Reconnect succeeded; try to recv() again
            if isinstance(msg, StateUpdate):
                if msg.state.get("turn_number", 0) != latest_state.get("turn_number", 0):
                    turns_observed += 1
                latest_state = msg.state
                # legal_actions arrives whenever it is *our* seat's turn to
                # choose. An empty list means the AI seat has priority — keep
                # looping until phase-server sends us another decision.
                legal_actions = list(msg.legal_actions)
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
            elif isinstance(msg, ActionRejected):
                log.append(f"action rejected: {msg.reason}")
                trace.append(
                    {
                        "event": "action_rejected",
                        "reason": msg.reason,
                        "turn": int(latest_state.get("turn_number", 0)),
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

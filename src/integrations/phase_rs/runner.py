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
    GameOver,
    GameStarted,
    PhaseServerClient,
    PhaseServerConfig,
    StateUpdate,
)

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


async def run_game(
    *,
    deck: dict[str, Any],
    picker: ActionPicker | None = None,
    config: PhaseServerConfig | None = None,
    ai_difficulty: str = "Medium",
    ai_deck_name: str = "Red Deck Wins",
    display_name: str = "OppositionAgent",
    max_actions: int = 2000,
) -> GameRunResult:
    """Play one game against a phase-ai opponent and return the result.

    ``deck`` is the JSON ``DeckData`` our seat will use (see
    ``src.integrations.phase_rs.decks``). ``ai_deck_name`` selects one of
    phase-rs's built-in starter decks for the AI seat — the server resolves
    the card list internally, so we don't need to know its contents.

    ``max_actions`` is a hard cap so a misbehaving picker can't spin
    forever; the loop exits with ``reason="action_cap"`` if hit.
    """
    picker = picker or RandomActionPicker()
    cfg = config or PhaseServerConfig()
    log: list[str] = []

    async with PhaseServerClient(cfg) as client:
        hello = await client.handshake()
        log.append(f"connected: server={hello.server_version} build={hello.build_commit}")

        created = await client.create_game_with_ai(
            deck=deck,
            display_name=display_name,
            ai_difficulty=ai_difficulty,
            ai_deck_name=ai_deck_name,
        )
        log.append(f"game created: code={created.game_code}")

        started = await client.expect("GameStarted")
        assert isinstance(started, GameStarted)
        our_seat = started.your_player
        log.append(f"game started: our_seat={our_seat} opp={started.opponent_name!r}")

        latest_state: dict[str, Any] = started.state
        legal_actions: list[dict[str, Any]] = list(started.legal_actions)

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
                    )
                idx = picker.pick(legal_actions, latest_state, our_seat)
                if not 0 <= idx < len(legal_actions):
                    raise ValueError(
                        f"picker {picker.name} returned index {idx} "
                        f"out of range [0, {len(legal_actions)})"
                    )
                chosen = legal_actions[idx]
                logger.debug("phase-rs: seat %d -> %s", our_seat, chosen.get("type"))
                await client.send_action(chosen)
                actions_sent += 1
                legal_actions = []  # wait for next StateUpdate

            # Wait for the next state. Could be StateUpdate or GameOver, plus
            # incidental traffic (events, timer ticks). Use the streaming
            # timeout (``None`` by default) so AI thinking pauses don't
            # trigger spurious TimeoutErrors.
            msg = await client.recv(timeout=client.config.stream_timeout_s)
            if isinstance(msg, StateUpdate):
                if msg.state.get("turn_number", 0) != latest_state.get("turn_number", 0):
                    turns_observed += 1
                latest_state = msg.state
                # legal_actions arrives whenever it is *our* seat's turn to
                # choose. An empty list means the AI seat has priority — keep
                # looping until phase-server sends us another decision.
                legal_actions = list(msg.legal_actions)
            elif isinstance(msg, GameOver):
                log.append(f"game over: winner={msg.winner} reason={msg.reason}")
                return GameRunResult(
                    winner_seat=msg.winner,
                    reason=msg.reason,
                    our_seat=our_seat,
                    turns_observed=turns_observed,
                    actions_sent=actions_sent,
                    final_state=latest_state,
                    log=log,
                )
            else:
                # Lobby / timer / emote / opponent (re)connect. Logged at debug
                # so we can spot oddities without polluting the game log.
                msg_type = msg[0] if isinstance(msg, tuple) else type(msg).__name__
                logger.debug("phase-rs: incidental %s", msg_type)


def run_game_sync(**kwargs: Any) -> GameRunResult:
    """Synchronous wrapper for callers who don't want asyncio in their loop."""
    return asyncio.run(run_game(**kwargs))

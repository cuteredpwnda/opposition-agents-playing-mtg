"""WebSocket client for phase-rs ``phase-server`` (protocol v6).

Schema source: ``external/phase-rs/crates/server-core/src/protocol.rs``.

The wire format is JSON with serde's ``#[serde(tag = "type", content = "data")]``
envelope on every message::

    {"type": "ClientHello", "data": {...}}

We model only the messages our adapter needs (handshake, create game with AI
opponent, apply action, receive state). Everything else is passed through as
``dict[str, Any]`` so we don't have to track every upstream field addition.

Heavy deps (``websockets``) are lazy-imported inside method bodies, per the
project convention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


# Mirror of PROTOCOL_VERSION in server-core/src/protocol.rs. Bump when the
# upstream constant changes; mismatched clients are rejected at handshake.
PROTOCOL_VERSION = 6

# Our identity advertised to phase-server. The version string is informational;
# only ``protocol_version`` gates compatibility.
DEFAULT_CLIENT_VERSION = "opposition-agents-mtg/0.1"
DEFAULT_BUILD_COMMIT = "dev"


@dataclass
class PhaseServerConfig:
    """Connection settings for a running ``phase-server`` instance."""

    # phase-server's default port is 9374 (see Cli in phase-server/src/main.rs).
    uri: str = "ws://127.0.0.1:9374/ws"
    headers: dict[str, str] = field(default_factory=dict)
    # Short timeout for control-plane traffic (handshake, CreateGame ack).
    request_timeout_s: float = 15.0
    # Per-message timeout while a game is streaming. phase-ai's wall-clock
    # budget per decision is 1.5 s and a single AI turn can chain dozens of
    # decisions (drawing, untap triggers, casting, attacking, blocking).
    # ``None`` disables the timeout entirely; set a finite value to bound
    # how long the client will wait for the next ServerMessage.
    stream_timeout_s: float | None = None
    client_version: str = DEFAULT_CLIENT_VERSION
    build_commit: str = DEFAULT_BUILD_COMMIT


@dataclass
class ServerHello:
    server_version: str
    build_commit: str
    protocol_version: int
    mode: str  # "Full" | "LobbyOnly"


@dataclass
class GameCreated:
    game_code: str
    player_token: str


@dataclass
class GameStarted:
    """First in-game message after CreateGame*. Mirrors ServerMessage::GameStarted."""

    state: dict[str, Any]
    your_player: int
    opponent_name: str | None
    player_names: list[str]
    legal_actions: list[dict[str, Any]]
    auto_pass_recommended: bool
    spell_costs: dict[str, Any]
    legal_actions_by_object: dict[str, list[dict[str, Any]]]
    derived: dict[str, Any]
    player_token: str | None
    raw: dict[str, Any]


@dataclass
class StateUpdate:
    state: dict[str, Any]
    events: list[dict[str, Any]]
    legal_actions: list[dict[str, Any]]
    auto_pass_recommended: bool
    eliminated_players: list[int]
    log_entries: list[dict[str, Any]]
    spell_costs: dict[str, Any]
    legal_actions_by_object: dict[str, list[dict[str, Any]]]
    derived: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class GameOver:
    winner: int | None
    reason: str


@dataclass
class ActionRejected:
    reason: str


class PhaseServerError(RuntimeError):
    """Raised when the server sends an ``Error`` frame or rejects an action."""


# ---- Parsers ---------------------------------------------------------------


def parse_envelope(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(type, data)`` from a serde tag/content envelope.

    Raises ``ValueError`` if the shape doesn't match.
    """
    if not isinstance(payload, dict) or "type" not in payload:
        raise ValueError(f"missing 'type' field in envelope: {payload!r}")
    return payload["type"], payload.get("data", {}) or {}


def parse_server_message(
    payload: dict[str, Any],
) -> ServerHello | GameCreated | GameStarted | StateUpdate | GameOver | ActionRejected | tuple[str, dict[str, Any]]:
    """Decode a ``ServerMessage`` into a typed dataclass when we model it.

    Returns the raw ``(type, data)`` tuple for messages we don't yet model so
    callers can introspect lobby / draft / spectator traffic without us having
    to chase every upstream variant.
    """
    msg_type, data = parse_envelope(payload)
    if msg_type == "ServerHello":
        return ServerHello(
            server_version=data["server_version"],
            build_commit=data["build_commit"],
            protocol_version=int(data["protocol_version"]),
            mode=data["mode"],
        )
    if msg_type == "GameCreated":
        return GameCreated(game_code=data["game_code"], player_token=data["player_token"])
    if msg_type == "GameStarted":
        return GameStarted(
            state=data["state"],
            your_player=int(data["your_player"]),
            opponent_name=data.get("opponent_name"),
            player_names=list(data.get("player_names", [])),
            legal_actions=list(data.get("legal_actions", [])),
            auto_pass_recommended=bool(data.get("auto_pass_recommended", False)),
            spell_costs=dict(data.get("spell_costs", {})),
            legal_actions_by_object=dict(data.get("legal_actions_by_object", {})),
            derived=dict(data.get("derived", {})),
            player_token=data.get("player_token"),
            raw=payload,
        )
    if msg_type == "StateUpdate":
        return StateUpdate(
            state=data["state"],
            events=list(data.get("events", [])),
            legal_actions=list(data.get("legal_actions", [])),
            auto_pass_recommended=bool(data.get("auto_pass_recommended", False)),
            eliminated_players=list(data.get("eliminated_players", [])),
            log_entries=list(data.get("log_entries", [])),
            spell_costs=dict(data.get("spell_costs", {})),
            legal_actions_by_object=dict(data.get("legal_actions_by_object", {})),
            derived=dict(data.get("derived", {})),
            raw=payload,
        )
    if msg_type == "GameOver":
        return GameOver(winner=data.get("winner"), reason=str(data.get("reason", "")))
    if msg_type == "ActionRejected":
        return ActionRejected(reason=str(data.get("reason", "")))
    return (msg_type, data)


# ---- Client ---------------------------------------------------------------


class PhaseServerClient:
    """Async context-managed WebSocket client speaking phase-server protocol v6.

    Typical use::

        async with PhaseServerClient() as client:
            hello = await client.handshake()
            created = await client.create_game_with_ai(
                deck={"main_deck": [...], "sideboard": [], "commander": []},
                display_name="Bot",
                ai_difficulty="Medium",
            )
            started = await client.expect("GameStarted")
            ...

    All ``send_*`` methods write JSON; ``recv`` returns the next typed
    ``ServerMessage`` dataclass (or a raw ``(type, data)`` tuple for unmodelled
    variants).
    """

    def __init__(self, config: PhaseServerConfig | None = None) -> None:
        self.config = config or PhaseServerConfig()
        self._ws: Any = None

    async def __aenter__(self) -> "PhaseServerClient":
        try:
            import websockets  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — surfaced at runtime
            raise RuntimeError(
                "phase-rs adapter requires the 'websockets' package. "
                "Install with: pip install 'opposition-agents-mtg[phase_rs]' "
                "or pip install websockets>=12"
            ) from exc

        self._ws = await websockets.connect(
            self.config.uri,
            additional_headers=list(self.config.headers.items()) or None,
            max_size=2 * 1024 * 1024,  # phase-rs StateUpdate can be large
        )
        logger.debug("phase-rs: connected to %s", self.config.uri)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # -- low-level I/O ----------------------------------------------------

    async def _send(self, msg_type: str, data: dict[str, Any] | None = None) -> None:
        if self._ws is None:
            raise RuntimeError("PhaseServerClient is not connected")
        envelope = {"type": msg_type, "data": data or {}}
        await self._ws.send(json.dumps(envelope))
        logger.debug("phase-rs: sent %s", msg_type)

    async def _recv_raw(self, *, timeout: float | None = -1.0) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("PhaseServerClient is not connected")
        # ``timeout=-1.0`` is the sentinel for "use the config default";
        # ``None`` means no timeout (block forever). Callers in the streaming
        # loop pass ``self.config.stream_timeout_s`` directly.
        if timeout == -1.0:
            timeout = self.config.request_timeout_s
        if timeout is None:
            raw = await self._ws.recv()
        else:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode("utf-8")
        return json.loads(raw)

    async def recv(self, *, timeout: float | None = -1.0) -> Any:
        """Receive and decode the next ``ServerMessage``.

        ``timeout`` defaults to ``config.request_timeout_s``; pass ``None``
        to block indefinitely (used by the streaming game loop where AI
        thinking can pause server traffic for several seconds).
        """
        return parse_server_message(await self._recv_raw(timeout=timeout))

    async def expect(self, msg_type: str, *, max_skip: int = 8) -> Any:
        """Wait for a message of the given type, skipping unrelated traffic
        (Ping, lobby chatter, PlayerCount, etc.) up to ``max_skip`` frames.

        Raises ``PhaseServerError`` if an ``Error`` or ``ActionRejected``
        arrives before the expected type.
        """
        for _ in range(max_skip + 1):
            msg = await self.recv()
            actual_type: str
            if isinstance(msg, tuple):
                actual_type = msg[0]
            else:
                actual_type = type(msg).__name__
            if actual_type == msg_type:
                return msg
            if actual_type == "Error":
                _, data = msg  # type: ignore[misc]
                raise PhaseServerError(str(data.get("message", "server error")))
            if actual_type == "ActionRejected":
                raise PhaseServerError(f"action rejected: {msg.reason}")  # type: ignore[union-attr]
            logger.debug("phase-rs: skipping %s while waiting for %s", actual_type, msg_type)
        raise TimeoutError(f"did not receive {msg_type} within {max_skip + 1} frames")

    # -- protocol helpers -------------------------------------------------

    async def handshake(self) -> ServerHello:
        """Receive ``ServerHello``, send ``ClientHello``, return server hello.

        Aborts with ``PhaseServerError`` on protocol mismatch.
        """
        hello = await self.expect("ServerHello")
        assert isinstance(hello, ServerHello)
        if hello.protocol_version != PROTOCOL_VERSION:
            raise PhaseServerError(
                f"protocol mismatch: server={hello.protocol_version} "
                f"client={PROTOCOL_VERSION}. Update PROTOCOL_VERSION in "
                f"src/integrations/phase_rs/client.py after reviewing upstream "
                f"protocol.rs changes."
            )
        await self._send(
            "ClientHello",
            {
                "client_version": self.config.client_version,
                "build_commit": self.config.build_commit,
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        return hello

    async def create_game_with_ai(
        self,
        *,
        deck: dict[str, Any],
        display_name: str = "OppositionAgent",
        ai_difficulty: str = "Medium",
        ai_deck_name: str | None = None,
        format_name: str | None = None,
    ) -> GameCreated:
        """Create a 2-player game with one AI opponent on seat 1.

        ``ai_difficulty`` is one of ``VeryEasy``, ``Easy``, ``Medium``,
        ``Hard``, ``VeryHard`` (see ``AiDifficulty`` in
        ``phase-ai/src/config.rs``).
        
        ``format_name`` is one of ``Standard``, ``Pioneer``, ``Modern``,
        ``Legacy``, ``Vintage``, ``Commander``, ``Brawl``, ``HistoricBrawl``,
        etc. Defaults to ``Standard`` if not specified.
        """
        # Per ``ClientMessage::CreateGameWithSettings`` in
        # ``external/phase-rs/crates/server-core/src/protocol.rs`` — fields
        # are snake_case (no ``rename_all`` attribute on the enum). Optional
        # fields with ``#[serde(default)]`` (``match_config``,
        # ``format_config``, ``ai_seats``, ``room_name``, ``host_peer_id``,
        # ``draft_metadata``) may be omitted; we omit them and let the
        # server pick defaults.
        payload: dict[str, Any] = {
            "deck": deck,
            "display_name": display_name,
            "public": False,
            "password": None,
            "timer_seconds": None,
            "player_count": 2,
            "ai_seats": [
                {
                    # ``AiSeatRequest`` itself carries
                    # ``#[serde(rename_all = "camelCase")]`` — these fields
                    # are camelCase even though their parent envelope is
                    # snake_case.
                    "seatIndex": 1,
                    "difficulty": ai_difficulty,
                    "deckName": ai_deck_name,
                }
            ],
        }
        # Add format_config if specified (e.g., for Commander games)
        if format_name:
            payload["format_config"] = {
                "format": format_name,
            }
        await self._send("CreateGameWithSettings", payload)
        created = await self.expect("GameCreated")
        assert isinstance(created, GameCreated)
        return created

    async def send_action(self, action: dict[str, Any]) -> None:
        """Send a ``GameAction`` envelope. ``action`` is the raw tagged-union
        dict received in ``legal_actions``."""
        await self._send("Action", {"action": action})

    async def concede(self) -> None:
        await self._send("Concede", {})

    async def ping(self, *, nonce: int | None = None) -> None:
        await self._send(
            "Ping",
            {"timestamp": nonce if nonce is not None else (uuid.uuid4().int >> 96)},
        )

    # -- streaming --------------------------------------------------------

    async def stream(self) -> AsyncIterator[Any]:
        """Yield successive parsed messages until the connection closes."""
        try:
            while True:
                yield await self.recv()
        except (asyncio.IncompleteReadError, ConnectionError):
            return

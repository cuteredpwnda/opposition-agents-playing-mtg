"""phase-rs integration — adapter to use the Rust/WASM MTG engine
[phase-rs/phase](https://github.com/phase-rs/phase) as an alternative
rules backend behind our Python ``Agent`` interface.

This package is **experimental**. See ``README.md`` in this directory for
status, transport options, and the protocol-discovery TODO list. Nothing
here is wired into the default runner yet — agents continue to talk to
``src/engine/`` as the authoritative engine.

Heavy deps (``websockets``) are imported lazily inside method bodies, so
``import src.integrations.phase_rs`` stays cheap.
"""

from src.integrations.phase_rs.agent_bridge import (
    ActionPicker,
    AgentActionPicker,
    HeuristicActionPicker,
    OllamaActionPicker,
    PreferNonPassPicker,
    RandomActionPicker,
)
from src.integrations.phase_rs.adapter import (
    engine_action_to_phase_action,
    legal_actions_to_engine_actions,
    phase_state_to_game_state,
)
from src.integrations.phase_rs.client import (
    PROTOCOL_VERSION,
    GameCreated,
    GameOver,
    GameStarted,
    PhaseServerClient,
    PhaseServerConfig,
    PhaseServerError,
    ServerHello,
    StateUpdate,
    parse_server_message,
)
from src.integrations.phase_rs.decks import (
    STARTER_DECK_NAMES,
    decklist_to_deck_data,
    load_deck_data,
)
from src.integrations.phase_rs.runner import GameRunResult, run_game, run_game_sync
from src.integrations.phase_rs.server_process import PhaseServerProcess
from src.integrations.phase_rs.state_view import PhaseRsPlayerView, PhaseRsStateView

__all__ = [
    "PROTOCOL_VERSION",
    "ActionPicker",
    "AgentActionPicker",
    "GameCreated",
    "GameOver",
    "GameRunResult",
    "GameStarted",
    "HeuristicActionPicker",
    "OllamaActionPicker",
    "PhaseRsPlayerView",
    "PhaseRsStateView",
    "PhaseServerClient",
    "PhaseServerConfig",
    "PhaseServerError",
    "PhaseServerProcess",
    "PreferNonPassPicker",
    "RandomActionPicker",
    "STARTER_DECK_NAMES",
    "ServerHello",
    "StateUpdate",
    "decklist_to_deck_data",
    "engine_action_to_phase_action",
    "legal_actions_to_engine_actions",
    "load_deck_data",
    "parse_server_message",
    "phase_state_to_game_state",
    "run_game",
    "run_game_sync",
]

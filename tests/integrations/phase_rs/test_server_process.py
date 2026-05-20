"""Lifecycle smoke test for :class:`PhaseServerProcess`.

These tests are gated on the vendored phase-server binary existing under
``external/phase-rs/target/{release,debug}/`` so CI without a Rust
toolchain skips cleanly. When the binary is available we verify:

* the context manager starts the server,
* the WebSocket port becomes reachable,
* an already-running instance is adopted (not double-launched),
* the server is torn down on context exit when we owned it.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

from src.integrations.phase_rs.server_process import (
    DEFAULT_PORT,
    PhaseServerProcess,
    _port_open,
    _resolve_binary,
)

SUBMODULE = (
    Path(__file__).resolve().parents[3] / "external" / "phase-rs"
)


def _binary_available() -> bool:
    return _resolve_binary(SUBMODULE) is not None


pytestmark = pytest.mark.skipif(
    not _binary_available(),
    reason=(
        "phase-server binary not built; "
        "run `cargo build --release --bin phase-server` "
        "inside external/phase-rs to enable"
    ),
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_stop_opens_and_closes_port():
    port = _free_port()
    proc = PhaseServerProcess(port=port, startup_timeout_s=30.0)
    assert not _port_open("127.0.0.1", port, timeout=0.2)
    with proc as server:
        assert _port_open("127.0.0.1", port, timeout=2.0)
        assert server.uri == f"ws://127.0.0.1:{port}/ws"
    # Give the OS a beat to release the socket; some platforms hold it briefly.
    for _ in range(20):
        if not _port_open("127.0.0.1", port, timeout=0.1):
            break
        time.sleep(0.1)
    assert not _port_open("127.0.0.1", port, timeout=0.2)


def test_adopts_running_instance_without_double_launch():
    port = _free_port()
    # Start one server, then try to start a second on the same port.
    first = PhaseServerProcess(port=port, startup_timeout_s=30.0).start()
    try:
        adopter = PhaseServerProcess(port=port, startup_timeout_s=5.0).start()
        # Adopter must not have launched its own subprocess.
        assert adopter._adopted is True  # noqa: SLF001
        assert adopter._proc is None  # noqa: SLF001
        # stop() on the adopter must be a no-op (port stays open).
        adopter.stop()
        assert _port_open("127.0.0.1", port, timeout=1.0)
    finally:
        first.stop()

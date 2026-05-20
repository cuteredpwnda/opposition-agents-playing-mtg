"""Start / stop a local ``phase-server`` process from Python.

The agent demos and benchmarks need phase-server running on
``ws://127.0.0.1:9374/ws``. Asking every user to keep a second terminal
open with ``cargo serve`` is annoying — this module spawns the server
as a managed subprocess and tears it down on exit.

Resolution order for the server binary:

1. Explicit ``binary`` argument to :class:`PhaseServerProcess`.
2. ``PHASE_RS_SERVER_BINARY`` environment variable.
3. ``external/phase-rs/target/release/phase-server[.exe]`` (vendored
   submodule, release build).
4. ``external/phase-rs/target/debug/phase-server[.exe]`` (debug build).
5. ``cargo`` fallback — runs ``cargo run --release --bin phase-server``
   from the submodule, building on demand. Requires a Rust toolchain.

Use as a context manager::

    from src.integrations.phase_rs.server_process import PhaseServerProcess

    with PhaseServerProcess() as server:
        # server.uri is now reachable
        result = run_game_sync(deck=..., config=PhaseServerConfig(uri=server.uri))

The context manager waits for the server's WebSocket port to accept
connections before yielding (default 30 s), so the next line can
immediately open a client.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBMODULE = REPO_ROOT / "external" / "phase-rs"
DEFAULT_PORT = 9374
DEFAULT_HOST = "127.0.0.1"


def _binary_name() -> str:
    return "phase-server.exe" if sys.platform.startswith("win") else "phase-server"


def _resolve_binary(submodule_root: Path) -> Optional[Path]:
    """Find an existing phase-server binary in ``target/{release,debug}``.

    Returns ``None`` if neither exists; the caller then falls back to
    ``cargo run``.
    """
    name = _binary_name()
    for profile in ("release", "debug"):
        candidate = submodule_root / "target" / profile / name
        if candidate.is_file():
            return candidate
    return None


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return ``True`` iff a TCP connection to ``(host, port)`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class PhaseServerProcess:
    """Context-manager wrapper around a local ``phase-server`` subprocess.

    Parameters
    ----------
    host, port
        Where phase-server will bind. ``port=0`` is **not** supported by
        phase-server itself — pick a real free port (default 9374).
        ``host`` is the address Python uses to probe the server and to
        construct :attr:`uri`; phase-server itself listens on all
        interfaces, so ``host`` here is for the client side only.
    binary
        Override the server-binary path. ``None`` means auto-resolve from
        the vendored submodule, then fall back to ``cargo run``.
    submodule_root
        Where to look for ``target/release/phase-server`` and where to
        invoke ``cargo`` from when falling back.
    startup_timeout_s
        How long to wait for the server's port to start accepting
        connections before considering the launch failed.
    extra_env
        Extra environment variables to pass to the subprocess (e.g.
        ``RUST_LOG=info``).
    quiet
        When ``True`` (default) the server's stdout/stderr is redirected
        to ``DEVNULL``. Set ``False`` to inherit the parent's streams —
        useful for debugging crashes.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    binary: Optional[Path] = None
    submodule_root: Path = DEFAULT_SUBMODULE
    startup_timeout_s: float = 30.0
    extra_env: dict[str, str] = field(default_factory=dict)
    quiet: bool = True

    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _adopted: bool = field(default=False, init=False, repr=False)

    @property
    def uri(self) -> str:
        """The WebSocket URI clients should connect to."""
        return f"ws://{self.host}:{self.port}/ws"

    def _build_command(self) -> tuple[list[str], Path]:
        """Return ``(argv, cwd)`` for launching phase-server.

        Tries (a) explicit binary, (b) ``$PHASE_RS_SERVER_BINARY``,
        (c) auto-located ``target/{release,debug}/phase-server``,
        (d) ``cargo run --release --bin phase-server`` as a last resort.
        """
        # phase-server's CLI flags (see crates/phase-server/src/main.rs Cli):
        #   -p, --port <u16>      (default 9374)
        #   -d, --data-dir <path> (default "data" -- relative to cwd, which
        #                          is the submodule root, so the bundled
        #                          mtgjson test fixture is picked up)
        # There is no --host arg; the server binds 0.0.0.0 unconditionally.
        args_tail = ["--port", str(self.port)]

        if self.binary is not None:
            return ([str(self.binary), *args_tail], self.submodule_root)

        env_bin = os.environ.get("PHASE_RS_SERVER_BINARY")
        if env_bin:
            return ([env_bin, *args_tail], self.submodule_root)

        resolved = _resolve_binary(self.submodule_root)
        if resolved is not None:
            return ([str(resolved), *args_tail], self.submodule_root)

        # cargo fallback. Builds on demand; slow first-run, ~instant after.
        return (
            ["cargo", "run", "--release", "--bin", "phase-server", "--", *args_tail],
            self.submodule_root,
        )

    def start(self) -> "PhaseServerProcess":
        """Spawn the server and block until its port accepts connections.

        Idempotent: if the port is already open, we assume someone else
        (e.g. a manually started ``cargo serve``) owns it and do nothing.
        In that case :meth:`stop` is a no-op too — we never kill a process
        we didn't start.
        """
        if _port_open(self.host, self.port, timeout=0.5):
            logger.info(
                "phase-server already running on %s:%d; adopting existing instance",
                self.host,
                self.port,
            )
            self._adopted = True
            return self

        if not self.submodule_root.exists():
            raise FileNotFoundError(
                f"phase-rs submodule missing at {self.submodule_root!s}; "
                "run: git submodule update --init external/phase-rs"
            )

        argv, cwd = self._build_command()
        env = {**os.environ, **self.extra_env}
        # Default RUST_LOG keeps the server quiet (warn+) unless caller
        # overrode it; phase-server is chatty on info.
        env.setdefault("RUST_LOG", "warn,phase_server=info")
        logger.info("starting phase-server: %s (cwd=%s)", " ".join(argv), cwd)

        stdio = subprocess.DEVNULL if self.quiet else None
        self._proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=stdio,
            stderr=stdio,
            # On Windows, create a new process group so Ctrl-C in the
            # parent doesn't immediately kill the server before we have
            # a chance to send our shutdown.
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform.startswith("win")
                else 0
            ),
        )

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"phase-server exited during startup with code "
                    f"{self._proc.returncode}; rerun with quiet=False to see logs"
                )
            if _port_open(self.host, self.port, timeout=0.5):
                logger.info(
                    "phase-server ready on %s:%d (pid=%d)",
                    self.host,
                    self.port,
                    self._proc.pid,
                )
                return self
            time.sleep(0.25)

        # Timeout — kill what we started and report.
        self.stop()
        raise TimeoutError(
            f"phase-server did not open port {self.host}:{self.port} within "
            f"{self.startup_timeout_s:.1f}s"
        )

    def stop(self, *, terminate_timeout_s: float = 5.0) -> None:
        """Terminate the server if we own it. No-op when we adopted an
        externally-started instance."""
        if self._adopted:
            return
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("stopping phase-server (pid=%d)", self._proc.pid)
        try:
            self._proc.terminate()
            self._proc.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("phase-server did not exit on SIGTERM; killing")
            self._proc.kill()
            self._proc.wait(timeout=terminate_timeout_s)

    def __enter__(self) -> "PhaseServerProcess":
        return self.start()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.stop()

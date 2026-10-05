from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.integrations.phase_rs import runner, server_lock


@pytest.mark.parametrize("succeeds", [True, False])
def test_unix_lock_polls_and_closes_on_timeout(tmp_path, monkeypatch, succeeds):
    path = tmp_path / "server.lock"
    monkeypatch.setattr(server_lock, "LOCK_FILE", path)
    monkeypatch.setattr(server_lock, "sys", SimpleNamespace(platform="linux"))
    clock = [0.0]
    attempts = []

    def flock(fd, flags):
        attempts.append(fd)
        if not succeeds or len(attempts) < 3:
            raise BlockingIOError

    monkeypatch.setattr(server_lock, "fcntl",
                        SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2), raising=False)
    monkeypatch.setattr(server_lock.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(server_lock.time, "sleep",
                        lambda delay: clock.__setitem__(0, clock[0] + delay))
    fd = server_lock.acquire_server_lock(timeout=0.25)
    assert len(attempts) >= 3
    if succeeds:
        assert fd is not None
        os.close(fd)
    else:
        assert fd is None
        with pytest.raises(OSError):
            os.fstat(attempts[0])


def test_unix_release_preserves_lock_inode_for_waiters(tmp_path, monkeypatch):
    path = tmp_path / "server.lock"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY)
    monkeypatch.setattr(server_lock, "LOCK_FILE", path)
    monkeypatch.setattr(server_lock, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(server_lock, "fcntl",
                        SimpleNamespace(flock=lambda *args: None, LOCK_UN=8), raising=False)
    server_lock.release_server_lock(fd)
    assert path.exists()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_windows_lock_excludes_second_owner_and_releases(tmp_path, monkeypatch):
    path = tmp_path / "server.lock"
    monkeypatch.setattr(server_lock, "LOCK_FILE", path)
    monkeypatch.setattr(server_lock, "sys", SimpleNamespace(platform="win32"))
    fd = server_lock.acquire_server_lock(timeout=0)
    assert fd is not None
    assert server_lock.acquire_server_lock(timeout=0) is None
    server_lock.release_server_lock(fd)
    assert not path.exists()
    fd = server_lock.acquire_server_lock(timeout=0)
    assert fd is not None
    server_lock.release_server_lock(fd)


@pytest.mark.parametrize("failure", [None, "play", "stop", "start"])
async def test_runner_holds_lock_until_server_stops(monkeypatch, failure):
    events = []

    class Server:
        uri = "ws://localhost:9374/ws"

        def start(self):
            events.append("start")
            if failure == "start":
                raise RuntimeError("start")
            return self

        def stop(self):
            events.append("stop")
            if failure == "stop":
                raise RuntimeError("stop")

    async def play(**kwargs):
        events.append("play")
        if failure == "play":
            raise RuntimeError("play")
        return "result"

    monkeypatch.setattr(runner, "PhaseServerProcess", Server)
    monkeypatch.setattr(runner, "acquire_server_lock", lambda **kwargs: 123)
    monkeypatch.setattr(runner, "release_server_lock", lambda fd: events.append("release"))
    monkeypatch.setattr(runner, "_play", play)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            await runner.run_game(deck={}, autostart_server=True)
    else:
        assert await runner.run_game(deck={}, autostart_server=True) == "result"
    assert events == (
        ["start", "release"] if failure == "start"
        else ["start", "play", "stop", "release"]
    )

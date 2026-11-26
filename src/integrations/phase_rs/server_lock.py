"""Process lock utilities for phase-rs server management.

Prevents multiple Python processes from simultaneously starting their own
phase-rs server instances, which would cause port contention.
"""

import os
import sys
import time
from pathlib import Path

# fcntl is Unix-only
if not sys.platform.startswith("win"):
    import fcntl

LOCK_FILE = Path(".phase-rs-server.lock")


def acquire_server_lock(timeout: float = 30.0) -> int | None:
    """
    Acquire a lock for starting the phase-rs server.

    Returns the lock file descriptor on success, None on timeout.
    Only one process can hold the lock at a time.
    """
    if sys.platform.startswith("win"):
        # Windows: use file existence as a simple lock
        # (fcntl not available on Windows)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                return fd
            except FileExistsError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(0.1, remaining))
    else:
        # Unix: use fcntl (atomic, more robust)
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_WRONLY)
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        os.close(fd)
                        return None
                    time.sleep(min(0.1, remaining))
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except OSError:
            os.close(fd)
            raise


def release_server_lock(lock_fd: int | None) -> None:
    """Release the server lock."""
    if lock_fd is None:
        return
    if sys.platform.startswith("win"):
        os.close(lock_fd)
        LOCK_FILE.unlink(missing_ok=True)
    else:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

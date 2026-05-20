"""Process lock utilities for phase-rs server management.

Prevents multiple Python processes from simultaneously starting their own
phase-rs server instances, which would cause port contention.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# fcntl is Unix-only
if not sys.platform.startswith("win"):
    import fcntl

LOCK_FILE = Path(".phase-rs-server.lock")


def acquire_server_lock(timeout: float = 30.0) -> Optional[int]:
    """
    Acquire a lock for starting the phase-rs server.

    Returns the lock file descriptor on success, None on timeout.
    Only one process can hold the lock at a time.
    """
    if sys.platform.startswith("win"):
        # Windows: use file existence as a simple lock
        # (fcntl not available on Windows)
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                return fd
            except FileExistsError:
                time.sleep(0.1)
        return None
    else:
        # Unix: use fcntl (atomic, more robust)
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_WRONLY)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except (IOError, BlockingIOError):
            return None


def release_server_lock(lock_fd: Optional[int]) -> None:
    """Release the server lock."""
    if lock_fd is None:
        return
    try:
        if sys.platform.startswith("win"):
            os.close(lock_fd)
            LOCK_FILE.unlink(missing_ok=True)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)  # type: ignore
            os.close(lock_fd)
            LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass  # Best effort cleanup

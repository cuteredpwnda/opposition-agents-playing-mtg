"""
Push and run the training pipeline on a remote machine.

Syncs the project to a remote server via SSH/rsync, then starts the
Docker-based training stack (Neo4j + Ollama + Trainer) on that machine.

Usage:
  python scripts/push_remote.py user@gpu-server.example.com
  python scripts/push_remote.py user@server --transfer         # Standard→Commander curriculum
  python scripts/push_remote.py user@server --train --iters 200
  python scripts/push_remote.py user@server --sync-only        # Just push files, don't start
  python scripts/push_remote.py user@server --pull-checkpoints # Pull trained models back

Requires: SSH key auth to remote, Docker + Docker Compose on remote.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REMOTE_DIR = "~/mtg-agents"

EXCLUDE_PATTERNS = [
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    "*.pyc",
    ".git/",
    "neo4j_data/",
    "*.log",
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream output."""
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def sync_to_remote(host: str):
    """Rsync the project to the remote machine."""
    logger.info("\n=== Syncing project to %s:%s ===", host, REMOTE_DIR)

    excludes = []
    for pat in EXCLUDE_PATTERNS:
        excludes.extend(["--exclude", pat])

    run([
        "rsync", "-avz", "--delete",
        *excludes,
        ".", f"{host}:{REMOTE_DIR}/",
    ])
    logger.info("Sync complete.")


def remote_exec(host: str, cmd: str):
    """Execute a command on the remote machine via SSH."""
    run(["ssh", host, f"cd {REMOTE_DIR} && {cmd}"])


def start_training(host: str, extra_args: list[str]):
    """Start the Docker-based training stack on the remote machine."""
    logger.info("\n=== Starting training on %s ===", host)

    # Build images
    remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml build")

    # Start Neo4j + Ollama first
    remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml up -d neo4j ollama")

    # Pull the Ollama model (if using LLM)
    if "--ollama" in extra_args or "--fusion" in extra_args:
        logger.info("Pulling Ollama model (mistral)...")
        remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml exec ollama ollama pull mistral")

    # Run trainer
    trainer_args = " ".join(extra_args) if extra_args else "--all --commander"
    remote_exec(host, f"docker compose -f docker-compose.yml -f docker-compose.remote.yml run -d trainer {trainer_args}")

    logger.info("\nTraining started! Monitor with:")
    logger.info("  ssh %s 'cd %s && docker compose -f docker-compose.yml -f docker-compose.remote.yml logs -f trainer'", host, REMOTE_DIR)


def start_transfer_training(host: str):
    """Start Standard→Commander transfer learning on the remote machine."""
    logger.info("\n=== Starting Transfer Learning on %s ===", host)

    remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml build")
    remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml up -d neo4j ollama")
    remote_exec(host, "docker compose -f docker-compose.yml -f docker-compose.remote.yml run -d trainer --transfer")

    logger.info("\nTransfer learning started! Monitor with:")
    logger.info("  ssh %s 'cd %s && docker compose -f docker-compose.yml -f docker-compose.remote.yml logs -f trainer'", host, REMOTE_DIR)


def pull_checkpoints(host: str):
    """Pull trained model checkpoints from the remote machine."""
    logger.info("\n=== Pulling checkpoints from %s ===", host)
    run([
        "rsync", "-avz",
        f"{host}:{REMOTE_DIR}/checkpoints/",
        "checkpoints/",
    ])
    logger.info("Checkpoints synced to local checkpoints/ directory.")


def main():
    parser = argparse.ArgumentParser(description="Push training to a remote machine")
    parser.add_argument("host", help="SSH host (e.g. user@gpu-server.example.com)")
    parser.add_argument("--sync-only", action="store_true", help="Only sync files, don't start training")
    parser.add_argument("--transfer", action="store_true", help="Run Standard→Commander transfer learning")
    parser.add_argument("--pull-checkpoints", action="store_true", help="Pull checkpoints from remote")
    parser.add_argument("--train", action="store_true", help="Start RL training")
    parser.add_argument("--iters", type=int, default=100, help="Training iterations")
    parser.add_argument("--commander", action="store_true", help="Commander format")
    parser.add_argument("--ollama", action="store_true", help="Include Ollama agent")
    parser.add_argument("--fusion", action="store_true", help="Include LLM-Fusion agent")
    args = parser.parse_args()

    if args.pull_checkpoints:
        pull_checkpoints(args.host)
        return

    # Sync project to remote
    sync_to_remote(args.host)

    if args.sync_only:
        logger.info("Sync-only mode — done.")
        return

    if args.transfer:
        start_transfer_training(args.host)
        return

    # Build extra args for the trainer container
    extra_args = []
    if args.train:
        extra_args.append("--train")
    if args.commander:
        extra_args.append("--commander")
    if args.ollama:
        extra_args.append("--ollama")
    if args.fusion:
        extra_args.append("--fusion")
    extra_args.extend(["--iters", str(args.iters)])

    start_training(args.host, extra_args)


if __name__ == "__main__":
    main()

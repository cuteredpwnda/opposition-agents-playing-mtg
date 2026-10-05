"""Create an isolated local Tev1 context alias; preserve the original model."""

from __future__ import annotations

import argparse
import json
import urllib.request


def configure(source: str, target: str, context: int) -> dict:
    source, target = source.strip(), target.strip()
    source_tag = source if ":" in source.rsplit("/", 1)[-1] else f"{source}:latest"
    target_tag = target if ":" in target.rsplit("/", 1)[-1] else f"{target}:latest"
    if not source or not target or source_tag == target_tag:
        raise ValueError("Source and target must be nonempty distinct model tags")
    if not 2050 <= context <= 32768:
        raise ValueError("Context must be between 2050 and 32768 tokens")
    body = {"model": target, "from": source,
            "parameters": {"num_ctx": context}, "stream": False}
    request = urllib.request.Request(
        "http://localhost:11434/api/create", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result.get("status") != "success":
        raise ValueError(f"Ollama did not create context alias: {result}")
    request = urllib.request.Request(
        "http://localhost:11434/api/show",
        data=json.dumps({"model": target}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    configured = {
        line.split()[0]: line.split()[1]
        for line in metadata.get("parameters", "").splitlines() if len(line.split()) == 2
    }
    if configured.get("num_ctx") != str(context):
        raise ValueError(f"Context alias verification failed: {configured}")
    return {"source": source, "target": target, "num_ctx": context,
            "details": metadata.get("details")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tev1:0.8b")
    parser.add_argument("--target", default="tev1-mtg-8k:0.8b")
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(configure(args.source, args.target, args.context), indent=2))


if __name__ == "__main__":
    main()

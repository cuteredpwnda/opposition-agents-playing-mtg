"""Export format defaults from already-built phase-rs Rust libraries.

Run after updating/building the engine, in the same MSVC environment on Windows.
Pass the engine and serde_json .rlib files from that build explicitly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from src.integrations.phase_rs.client import PROTOCOL_VERSION
from src.integrations.phase_rs.server_process import DEFAULT_SUBMODULE, REPO_ROOT

SOURCE = """
fn main() {
    let registry = engine::types::format::GameFormat::registry();
    println!("{}", serde_json::to_string(&registry).unwrap());
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-rlib", required=True, type=Path)
    parser.add_argument("--serde-json-rlib", required=True, type=Path)
    args = parser.parse_args()
    engine = args.engine_rlib.resolve(strict=True)
    serde = args.serde_json_rlib.resolve(strict=True)
    if engine.parent != serde.parent:
        raise ValueError("Both libraries must come from the same build dependency directory")
    with tempfile.TemporaryDirectory(prefix="phase-format-export-") as directory:
        source = Path(directory) / "export.rs"
        binary = Path(directory) / "export.exe"
        source.write_text(SOURCE, encoding="utf-8")
        subprocess.run(
            [
                "rustc", "--edition=2021", str(source), "-o", str(binary),
                "-C", "panic=abort",
                "-C", "opt-level=z", "-C", "lto=thin", "-C", "codegen-units=1",
                "--extern", f"engine={engine}", "--extern", f"serde_json={serde}",
                "-L", f"dependency={engine.parent}",
            ],
            cwd=DEFAULT_SUBMODULE, check=True,
        )
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    registry = json.loads(result.stdout)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEFAULT_SUBMODULE,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "phase_rs_commit": revision,
        "protocol_version": PROTOCOL_VERSION,
        "configs": {entry["format"]: entry["default_config"] for entry in registry},
    }
    destination = REPO_ROOT / "src" / "integrations" / "phase_rs" / "format_defaults.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(payload['configs'])} engine-authored formats to {destination}")


if __name__ == "__main__":
    main()

"""Export format defaults from the pinned phase-rs Rust engine.

Provide matching engine and serde_json libraries from a local build, or the
JavaScript bindings and WASM binary from a verified build of the same revision.
"""

from __future__ import annotations

import argparse
import hashlib
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

WASM_SOURCE = """
const fs = require("node:fs");
const { pathToFileURL } = require("node:url");
(async () => {
    const engine = await import(pathToFileURL(process.argv[1]).href);
    const wasm = await WebAssembly.compile(fs.readFileSync(process.argv[2]));
    await engine.default({ module_or_path: wasm });
    console.log(JSON.stringify(engine.getFormatRegistry()));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--engine-rlib", type=Path)
    source.add_argument("--wasm-bindings", type=Path)
    parser.add_argument("--serde-json-rlib", type=Path)
    parser.add_argument("--engine-wasm", type=Path)
    args = parser.parse_args()
    if args.wasm_bindings:
        if not args.engine_wasm or args.serde_json_rlib:
            parser.error("--wasm-bindings requires --engine-wasm and excludes --serde-json-rlib")
        bindings = args.wasm_bindings.resolve(strict=True)
        wasm = args.engine_wasm.resolve(strict=True)
        result = subprocess.run(
            ["node", "-e", WASM_SOURCE, str(bindings), str(wasm)],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        provenance = {
            "kind": "wasm",
            "wasm_sha256": hashlib.sha256(wasm.read_bytes()).hexdigest(),
            "bindings_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
        }
    else:
        if not args.serde_json_rlib or args.engine_wasm:
            parser.error("--engine-rlib requires --serde-json-rlib and excludes --engine-wasm")
        engine = args.engine_rlib.resolve(strict=True)
        serde = args.serde_json_rlib.resolve(strict=True)
        if engine.parent != serde.parent:
            raise ValueError("Both libraries must come from the same build dependency directory")
        with tempfile.TemporaryDirectory(prefix="phase-format-export-") as directory:
            source_file = Path(directory) / "export.rs"
            binary = Path(directory) / "export.exe"
            source_file.write_text(SOURCE, encoding="utf-8")
            subprocess.run(
                [
                    "rustc", "--edition=2021", str(source_file), "-o", str(binary),
                    "--extern", f"engine={engine}", "--extern", f"serde_json={serde}",
                    "-L", f"dependency={engine.parent}",
                ],
                cwd=DEFAULT_SUBMODULE, check=True,
            )
            result = subprocess.run([str(binary)], check=True, stdout=subprocess.PIPE, text=True)
        provenance = {"kind": "rust-rlib"}
    registry = json.loads(result.stdout)
    configs = {entry["format"]: entry["default_config"] for entry in registry}
    if not configs or len(configs) != len(registry):
        raise ValueError("Engine registry must contain unique format entries")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEFAULT_SUBMODULE,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "phase_rs_commit": revision,
        "protocol_version": PROTOCOL_VERSION,
        "registry_source": provenance,
        "configs": configs,
    }
    destination = REPO_ROOT / "src" / "integrations" / "phase_rs" / "format_defaults.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(payload['configs'])} engine-authored formats to {destination}")


if __name__ == "__main__":
    main()

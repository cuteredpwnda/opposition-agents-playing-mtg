# phase-rs integration (experimental)

Thin adapter to drive the [phase-rs](https://github.com/phase-rs/phase)
Rust/WASM MTG rules engine from our Python ``Agent`` interface.

**Status**: scaffold only. Protocol discovery is the first sub-task — see
``IMPLEMENTATION_PLAN.md`` queue entry *"phase-rs engine adapter"*. The
Python engine in ``src/engine/`` remains the authoritative substrate for
training, evaluation, and the trajectory pipeline.

## Why this exists

phase-rs implements layers, replacement effects, and 34k+ cards from
MTGJSON — areas where our engine is partial. Using it as an *oracle*
backend lets us:

1. Run differential tests: replay a recorded action trace through both
   engines and diff the resulting state.
2. Side-step mechanics our engine does not yet model when an experiment
   needs them, without blocking on our own implementation.

This is **not** a switch away from our engine. Agents stay in Python;
only the rules layer is swapped behind an adapter.

## Transport

WebSocket via ``phase-server`` was chosen first because it requires no
Rust toolchain in our repo — we just point at a running
``cargo serve`` instance (or the public preview) and speak JSON.

```powershell
# In a phase-rs checkout:
cargo serve   # binds 127.0.0.1:8080 by default — confirm in their docs
```

Then from Python:

```python
from src.integrations.phase_rs import PhaseServerClient, PhaseServerConfig

cfg = PhaseServerConfig(uri="ws://127.0.0.1:8080/ws")
async with PhaseServerClient(cfg) as client:
    ...  # protocol calls — see TODOs in client.py
```

## What's missing

The message schema is not yet pinned down. The phase-rs README documents
the high-level architecture (Axum + WebSocket, discriminated unions
serialised via ``serde`` + ``tsify``) but the on-the-wire envelope
shapes need to be read out of ``crates/phase-server/src/`` and the
``fixtures/adapter-contract`` directory. Until that is done, ``client.py``
exposes only connection plumbing.

See also their LLM-card-contribution flow at
<https://raw.githubusercontent.com/phase-rs/phase/main/docs/AI-CONTRIBUTOR.md>
— this is the other half of the work tracked in the plan file.

## License & attribution

phase-rs is dual-licensed MIT / Apache-2.0; both are compatible with this
repository. phase-rs is a non-commercial fan project — see their
``DMCA.md`` / ``NOTICE`` files. We never bundle their card data; the
adapter expects a running phase-rs instance to provide it.

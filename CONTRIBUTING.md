# Contributing

Thanks for wanting to contribute to opposition-agents-playing-mtg — a
non-commercial fan / research project (see [NOTICE.md](NOTICE.md)).

This file is the entry point for **humans and AI coding agents alike**. The
rules below apply equally to both.

## Read first

1. **[AGENTS.md](AGENTS.md)** — describes the agent contract, the lineup of
   playable agents, and how to add a new one.
2. **[.github/copilot-instructions.md](.github/copilot-instructions.md)** —
   project conventions: file layout, plan-file workflow (the
   `IMPLEMENTATION_PLAN.md` single-source-of-truth rule), test commands,
   typing conventions, and the engine-mutation rules.
3. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — single source of
   truth for what is done / in progress / queued. Pick a task from the
   Queue and update the plan in the same PR.
4. **[NOTICE.md](NOTICE.md)** — fan-content policy and what we do not
   redistribute.

## Setup

```powershell
# Windows / PowerShell
git clone --recurse-submodules https://github.com/maximegmd/opposition-agents-playing-mtg
cd opposition-agents-playing-mtg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/fetch_card_data.py     # downloads Scryfall bulk into data/scryfall/
python scripts/fetch_rules.py         # downloads CR into data/rules/
```

Optional extras: `pip install -e ".[ml,ontology,phase_rs]"`.

## Make a change

- Search for existing patterns before writing new ones (`grep_search` /
  semantic search). Reuse helpers in `src/engine/`, `src/agents/`,
  `src/world_model/` rather than duplicating logic.
- Don't reach into `GameState` to flip flags directly — go through the
  helpers in `src/engine/`.
- Card data is **immutable Scryfall JSON**. Per-instance state lives on
  `CardInstance`. Mutations to oracle text go on `card_data["oracle_text"]`,
  never via the read-only `oracle_text` property.
- Random / heuristic agents must remain deterministic with a `seed`.
- Keep the gameplay log readable — no `print`, no stack traces. Use
  `state.log(...)` for player-visible events; standard `logging` for
  internals (`logger.debug`).

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe examples/play_edh_pod.py --max-turns 6 --seed 7 --model none
```

Both must pass before opening a PR.

For phase-rs integration changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integrations/phase_rs -q
```

The fixture-based parser tests skip automatically when `external/phase-rs`
isn't checked out, so CI without submodules still passes.

## Open the PR

- Branch naming: `feat/...`, `fix/...`, `docs/...`, `refactor/...`.
- Title in conventional-commit form (`feat(engine): ...`).
- Body must:
  - Link the `IMPLEMENTATION_PLAN.md` queue entry the work resolves (or
    add one if the task wasn't queued yet).
  - List files touched if more than ten.
  - Note any follow-up work appended to the plan.
- **Do not** add new top-level Markdown files for one-off notes — append to
  `IMPLEMENTATION_PLAN.md` instead.
- **Do not** use `--no-verify`, `git push --force` on shared branches, or
  `git reset --hard` on anything you don't own, without explicit approval.
- **Do not** commit `print()` debug statements, `data/scryfall/` payloads,
  `data/rules/` snapshots, or anything from `runs/`.

## Contributing to the phase-rs engine itself

The [phase-rs](https://github.com/phase-rs/phase) project is vendored as a
submodule. If you find a bug in phase-rs or want to implement a card there,
file the issue / PR **upstream**, not here. See their
[AI-CONTRIBUTOR.md](https://github.com/phase-rs/phase/blob/main/docs/AI-CONTRIBUTOR.md)
for their LLM-friendly contribution flow. Their card-coverage feed is at
<https://pub-fc5b5c2c6e774356ae3e730bb0326394.r2.dev/staging/coverage-data.json>.

## Code of conduct

Be kind. Assume good faith. This is a side project and we all have day jobs.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).

## graphify

Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` if it exists.
If `graphify-out/wiki/index.md` exists, navigate it for deep questions.
Type `/graphify` in Copilot Chat to build or update the knowledge graph.

## Single source of truth: IMPLEMENTATION_PLAN.md

`IMPLEMENTATION_PLAN.md` is the **single source of truth** for what is
implemented, what is in progress, and what is queued. Before starting work:

1. Read the **Status Snapshot** and **Active Work Log** sections.
2. Pick a task from the **Queue** (High → Medium → Low priority).
3. When you finish, **update the plan in the same PR**:
   - Move the item from "Queue" / "In progress" to "Done" with a one-line
     description of what landed and the file(s) touched.
   - If new follow-up work is required, append it to the appropriate
     Queue bucket — never delete known-pending work silently.
4. If the work touches an architectural section (4.x), update the matching
   subsection's status (`✅ Implemented` / `🟡 Partial` / `❌ Missing`).

`PLAN.md`, `ARCHITECTURE.md`, and the slide decks under `presentations/`
are **design rationale / research context** only. They are *not* updated
per-task; cross-reference them, don't rewrite them.

## Project conventions

- Python 3.10+, idiomatic typing (`list[str]`, `dict[str, int]`, no `Tuple`/`Dict`).
- Engine state is mutated through dedicated modules; never reach into
  `GameState` to flip flags from agent or example code — call helpers in
  `src/engine/`.
- Card data is **immutable Scryfall JSON**. Per-instance state lives on
  `CardInstance`. Mutations to oracle text live on `card_data["oracle_text"]`,
  never via the `oracle_text` property (it is read-only).
- Random/heuristic agents must remain deterministic when a `seed` is
  supplied to the runner.
- Keep the human game log readable — no debug `print`, no stack traces.
  Use `state.log(...)`. Verbose internal events go through the standard
  `logging` module.
- Tests live under `tests/` and use plain `pytest`. New mechanics get a
  focused unit test plus, ideally, a touch from the EDH pod sim
  (`examples/play_edh_pod.py --max-turns 6 --seed 7 --model none`).

## Where things live

- **Comprehensive Rules text**: `data/rules/` — populated by
  `python scripts/fetch_rules.py` (downloads from
  https://magic.wizards.com/en/rules). The judge vectorstore reads
  `data/rules/latest.txt`.
- **Card data**: `data/scryfall/` — populated by
  `python scripts/fetch_card_data.py`.
- **Decks**: `data/decks/` — plain-text decklists with a `Commander` /
  `Mainboard` / `Sideboard` section header.
- **Game logs**: `runs/` — produced by example scripts.
- **Engine code**: `src/engine/` — rules, zones, mana, stack, triggers,
  combat, command zone, companion.
- **Orchestration**: `src/orchestrator/` — `game_runner.py` and
  `priority_loop.py` are the only places that drive turns.

## Editing rules of thumb

- When changing a public engine signature, search for and update **all**
  call sites (`grep_search` first).
- New command-zone objects (emblems, dungeons, vanguards, conspiracies,
  designations) go through `src/engine/command_zone.py`. Don't reach into
  `GameState.emblems` etc. directly.
- New mechanics that need oracle-text parsing add a small handler to
  `src/engine/spell_effects.py` or `src/engine/triggers.py`. Keep regexes
  conservative; prefer false negatives over wrong matches.
- After any engine change, run:
  ```
  .\.venv\Scripts\python.exe -m pytest tests/ -q
  .\.venv\Scripts\python.exe examples/play_edh_pod.py --max-turns 6 --seed 7 --model none
  ```
  and confirm the pod log still completes.

## Don'ts

- Don't add new top-level Markdown files for one-off notes — append to
  `IMPLEMENTATION_PLAN.md` instead.
- Don't bypass safety guards (`--no-verify`, `git push --force`,
  `git reset --hard` on shared branches) without explicit user approval.
- Don't generate URLs unless the user asked or you are linking to known
  documentation (e.g. magic.wizards.com, scryfall.com).
- Don't leave `print()` debug statements behind. Either convert to
  `state.log` (gameplay) or `logger.debug` (internals).

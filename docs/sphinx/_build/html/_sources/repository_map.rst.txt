Repository Map
==============

Top-level files
---------------

- ``README.md``: project overview and usage entry point.
- ``IMPLEMENTATION_PLAN.md``: single source of truth for progress tracking.
- ``ARCHITECTURE.md``: architecture rationale and subsystem narratives.
- ``DEVELOPMENT.md``: contributor and environment workflows.
- ``pyproject.toml`` / ``requirements*.txt``: package and dependency definitions.

Primary directories
-------------------

- ``src/``: all runtime code.

  - ``src/agents/``: policy implementations (heuristic, world-model, active inference, fusion, LLM).
  - ``src/engine/``: game-rules engine (zones, stack, phases, combat, triggers, SBAs).
  - ``src/orchestrator/``: game loop and priority orchestration.
  - ``src/knowledge/``: ontology and graph integration.
  - ``src/world_model/``: latent dynamics and controller models.
  - ``src/training/``: self-play and training loop machinery.
  - ``src/judge/``: rules QA and vectorstore integration.

- ``data/``: card/rules/deck/ontology assets.
- ``scripts/``: training, benchmarking, and ETL entrypoints.
- ``examples/``: deterministic demos and smoke scenarios.
- ``tests/``: pytest suite.
- ``paper/``: publication artifacts and figures.
- ``docs/``: design notes and setup guides.

Execution flow summary
----------------------

1. Data and ontology import populate symbolic sources.
2. The orchestrator queries the engine for legal actions.
3. Agents select a legal action from the provided set.
4. Engine mutates state and logs traces.
5. Training code consumes traces/replay to update models.

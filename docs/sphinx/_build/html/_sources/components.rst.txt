Components
==========

Engine
------

The engine under ``src/engine/`` is the authoritative state transition system.
It generates legal actions, resolves stack interactions, applies state-based
actions, and enforces turn/phase sequencing. Agents never mutate state
directly; they only select from legal actions returned by the engine.

Agent Zoo
---------

Agents under ``src/agents/`` share one contract: select exactly one action
from ``legal_actions``. Implementations include baseline stochastic policies,
rule-heuristic policies, KG-aware policies, world-model policies, active
inference, and LLM fusion.

Knowledge Graph
---------------

The symbolic layer under ``src/knowledge/`` models cards, mechanics,
archetypes, and combos in Neo4j/OWL-backed schemas. Query utilities expose
combo detection and synergy scoring to relevant agents.

World Model
-----------

The latent model under ``src/world_model/`` uses encoder-dynamics-controller
decomposition (V+M+C) plus JEPA-style prediction in latent space.

Training and Evaluation
-----------------------

The training stack under ``src/training/`` and scripts under ``scripts/``
provide self-play, replay collection, promotion loops, benchmark matchups,
and overnight experiment orchestration.

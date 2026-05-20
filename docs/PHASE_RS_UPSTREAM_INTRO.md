# Draft: upstream introduction to phase-rs

This file is a draft message we propose opening as a **GitHub Discussion**
(or, if Discussions are off, an **Issue** with label `discussion`) on
<https://github.com/phase-rs/phase>. It introduces our research project and
asks a few orientation questions. It is **not** a pull request — we are not
proposing code changes upstream yet.

Posting requires explicit maintainer-side confirmation in this repo. Until
then, this file is the canonical draft.

---

**Suggested title:**
`Research integration: opposition-agents-playing-mtg using phase-rs as an engine backend`

**Suggested labels:** `question`, `discussion` (whichever exist upstream).

**Body:**

> Hi Matt, and the phase.rs community —
>
> First: thank you. Finding [phase-rs/phase](https://github.com/phase-rs/phase)
> has been a small revelation. The combination of idiomatic Rust, layers /
> replacement effects / a real stack, 30k+ implemented cards, and a tunable
> per-difficulty AI opponent is exactly what an MTG research framework
> needs and we'd otherwise have to build from scratch (and badly).
>
> We're a research project at
> <https://github.com/maximegmd/opposition-agents-playing-mtg> studying
> **reasoning under uncertainty** in MTG as a testbed for neuro-symbolic
> AI:
>
> - From-scratch CR engine (now being treated as a differential-test
>   substrate while we migrate to phase-rs as the primary backend)
> - Neo4j knowledge graph of cards / keywords / archetypes / combos with
>   GraphSAGE embeddings
> - V+M+C+JEPA world model (Set-Transformer + VAE encoder, MDN-LSTM
>   dynamics, JEPA latent predictor)
> - Active-inference LLM agents with per-opponent belief modules that do
>   exact library/hand inference when decklists are public
> - Collective append-only graph memory across self-play runs
>
> We just landed a WebSocket adapter against phase-server protocol v6:
> `src/integrations/phase_rs/{client,decks,agent_bridge,runner}.py`.
> Our random / heuristic pickers complete games end-to-end against
> `Medium` `AiDifficulty`. The fixture tests under
> `tests/integrations/phase_rs/` validate parsing against your
> `fixtures/adapter-contract/*.json` so we'll catch protocol drift the
> moment we `git submodule update`.
>
> **A few questions, no rush on any:**
>
> 1. **Stability of `protocol.rs`.** Are breaking changes to the
>    `PROTOCOL_VERSION = 6` envelope expected in the near term? We pin
>    the version in our handshake; happy to bump as you ship, just want
>    to calibrate cadence.
> 2. **Headless / "spectator" mode.** The web/Tauri client is gorgeous
>    but we don't need it — we drive games purely over `ws://…/ws`. Is
>    there a recommended way to expose a richer per-seat snapshot than
>    `StateUpdate` carries today (e.g. for opponent-modelling research
>    we'd love structured legal-action metadata, not just opaque tagged
>    unions)? We're happy to upstream a PR if you have an opinion on
>    shape.
> 3. **Contribution etiquette.** We'd like to send card implementations
>    via your [AI-CONTRIBUTOR.md](https://github.com/phase-rs/phase/blob/main/docs/AI-CONTRIBUTOR.md)
>    flow, picking unimplemented cards from our pod decks that appear in
>    your coverage feed. Any cards / sets you'd particularly like prioritised?
> 4. **Data contributions.** We maintain curated combo data
>    (`data/combos_merged.json`, `data/combos_edhrec.json`) and commander
>    decklists. Would either of those be useful to your scraper or to a
>    sibling data feed?
>
> Either way: cheers for shipping this. We've added a fan-content notice
> and a link to phase-rs in our README and `NOTICE.md`, and we point
> anyone wanting to fix card behaviour at *your* repo, not ours.
>
> — the opposition-agents-playing-mtg team

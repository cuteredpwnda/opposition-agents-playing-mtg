# Research pipelines and decision boundaries

These diagrams describe the intended whole system, not a claim that every
research component is already connected to native gameplay. The status and
experiment gates live in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md#6-benchmark--evaluation-plan).
Commands live in [EXPERIMENTS.md](EXPERIMENTS.md).

**Legend:** green = verified current path; amber = partial integration or a
research gate; dashed grey = queued. The present Commander wrapper controls
one Python seat and three native AI seats, not four independent Python agents.

## 1. Data lineage: two databases, not one interchangeable cache

```mermaid
flowchart LR
    SF["Scryfall oracle cards and rulings"] --> DL["Validated gzip-JSONL download"]
    DL --> CACHE["Immutable JSON snapshots + name index"]
    CACHE --> HASH["Dates, counts and SHA-256 manifest"]
    CACHE --> AB["CardDesign / CardPrinting ABox importer"]
    CR["Comprehensive Rules facts"] --> ONT["DOLCE-aligned OWL vocabulary + constraints"]
    ONT --> CHECK["Vocabulary, OWL, SHACL and competency checks"]
    AB --> CHECK
    CHECK --> KG["Curated strategic graph"]
    RELEASE["Pinned phase-rs signed release data"] --> NATIVE["Compiled engine card definitions"]
    NATIVE --> RULES["Native legality and state transitions"]
    HASH --> FREEZE["Experiment data freeze"]
    NATIVE --> FREEZE
    KG --> CONTEXT["Grounded agent context"]
    RULES --> CONTEXT
    classDef ready fill:#e2f3df,stroke:#377b32
    classDef partial fill:#fff2cc,stroke:#9b7315
    class SF,DL,CACHE,HASH,RELEASE,NATIVE,RULES ready
    class AB,ONT,CHECK,KG,CONTEXT,FREEZE partial
```

Refreshing Scryfall does not implement new Rust card effects or automatically
rebuild Neo4j. The native signed dataset was refreshed separately. ABox
re-import and conformance checks must be versioned before a KG experiment.

## 2. One online decision: observation, candidates, choice, validation

```mermaid
flowchart TD
    S["Authoritative phase-rs state"] --> V["Controlled-perspective observation"]
    S --> A["Engine-authored legal actions / interaction opportunities"]
    V --> F["Context features and hidden-information belief"]
    A --> C["Candidate builder with original IDs and targets"]
    C --> ONE{"Only one eligible choice?"}
    ONE -->|yes| FORCE["Forced choice; no model call"]
    ONE -->|no| POLICY{"Configured policy stack"}
    F --> POLICY
    POLICY --> H["Random or heuristic"]
    POLICY --> KL["KL planning or legacy EFE scoring"]
    POLICY --> DM["Tev1: compact card-aware context"]
    POLICY -.-> FUTURE["Matched chat, KG, JEPA and learned rollouts"]
    DM --> LIMIT{"Fits native request and prompt budgets?"}
    LIMIT -->|no| ERR["Explicit incomplete run; never random fallback"]
    LIMIT -->|yes| ANSWER["Choice + option probabilities"]
    ANSWER --> MAP["Validate answer and map to original legal action"]
    H --> MAP
    KL --> MAP
    FORCE --> MAP
    FUTURE -.-> MAP
    MAP --> SUB["Submit action or native interaction"]
    SUB --> ACCEPT{"Engine accepts it?"}
    ACCEPT -->|yes| NEXT["New state + observable events"]
    ACCEPT -->|no| FAIL["Record rejection / operational failure"]
    NEXT --> TRACE["Decision timing, candidates, reasoning and outcome trace"]
    FAIL --> TRACE
    ERR --> TRACE
    NEXT --> S
    classDef queued fill:#eee,stroke:#666,stroke-dasharray:5 5
    classDef partial fill:#fff2cc,stroke:#9b7315
    class FUTURE queued
    class F,KL,DM,LIMIT,SUB partial
```

The native interaction planner remains queued. Current policies chiefly use
legacy legal-action lists. KL's current horizon planner and the one-step EFE
arms differ in budget; this is not an objective-only experiment.

## 3. Four-player Commander: who actually controls each seat

```mermaid
sequenceDiagram
    participant P as Python host policy - seat 0
    participant W as WebSocket bridge
    participant E as phase-rs rules engine
    participant A as Native AI seats 1, 2, 3
    P->>W: Host deck + three explicit Commander decks
    W->>E: CreateGameWithSettings, player_count=4
    E->>E: Validate format, 100-card decks and commanders
    E-->>W: GameStarted, four players and viewer state
    W-->>P: Seat 0 observation and legal choices
    loop Priority, mulligans, combat and commander decisions
        E-->>A: Native seat-specific decisions
        A->>E: Native AI actions
        E-->>W: Controlled-perspective state update
        W-->>P: Original legal actions / decision context
        P->>W: Selected original action
        W->>E: Authenticated action submission
        E->>E: Stack, triggers, SBA and elimination
    end
    alt Genuine terminal state
        E-->>W: Winner or rules-terminal draw
        W-->>P: Completed result with terminal evidence
    else Time, turn, transport or action budget exceeded
        W-->>P: Incomplete result with explicit reason
    end
```

Four Python policies, seat/deck rotation, multiplayer opponent beliefs and
politics-aware reasoning require separate work. Reaching a turn cap proves
bounded execution, not a completed pod or a draw.

## 4. Learning loop: immutable facts versus induced strategic evidence

```mermaid
flowchart LR
    FACTS["Immutable card facts + curated ontology"] --> PRIOR["Policy priors / grounded context"]
    PRIOR --> PLAY["Self-play or native-AI episodes"]
    PLAY --> EVENTS["Perspective-safe traces and terminal outcomes"]
    EVENTS --> EXTRACT["Evidence extraction with game, seat and data provenance"]
    EXTRACT --> INDUCED["Append-only induced synergy / counter-evidence"]
    INDUCED --> GATE{"Support, refutation and human-review gate"}
    GATE -->|not ready| RETAIN["Retain as induced; do not overwrite facts"]
    GATE -->|validated| PROMOTE["Versioned curated promotion"]
    INDUCED --> QUERY["Later agents query learned priors"]
    QUERY --> PRIOR
    EVENTS -.-> TRAIN["Train JEPA / transition model on training split"]
    TRAIN -.-> FREEZE["Freeze checkpoint before held-out evaluation"]
    FREEZE -.-> PRIOR
    PROMOTE -.-> FACTS
    classDef queued fill:#eee,stroke:#666,stroke-dasharray:5 5
    classDef partial fill:#fff2cc,stroke:#9b7315
    class TRAIN,FREEZE,GATE,PROMOTE queued
    class EXTRACT,INDUCED,QUERY,PRIOR partial
```

Offline learning modules exist, but an end-to-end native play-to-graph loop
and the induced-to-curated promotion gate are not verified publication results.
No evaluation traces enter training or graph induction before holdout reporting.

## 5. Publication campaign: claims must pass gates before adding more arms

```mermaid
flowchart TD
    CLAIMS["Choose narrow claims and prespecified comparisons"] --> DATA["Freeze data, decks, engine and policy revisions"]
    DATA --> G0{"G0: protocol, card support and runtime correct?"}
    G0 -->|no| FIX["Fix blocker; keep failure evidence"]
    FIX --> G0
    G0 -->|yes| PILOT["Bounded random / heuristic / objective-stack pilot"]
    PILOT --> G1{"G1: completion and policy semantics acceptable?"}
    G1 -->|no| DIAG["Audit targets, action translation, timeouts and bias"]
    DIAG --> G0
    G1 -->|yes| VALID["Match budget, context, RNG / seats or restrict claims"]
    VALID --> G2{"G2: comparator genuinely deployable?"}
    G2 -->|no| EXCLUDE["Exclude from strength analysis; report feasibility failure"]
    G2 -->|yes| FINAL["Freeze training / tuning and held-out evaluation schedule"]
    FINAL --> RUN["Run planned cells; retain all attempts and incomplete runs"]
    RUN --> STATS["Per-cell rates, intervals, effect sizes and sensitivity checks"]
    STATS --> REPRO["Audited configs, code revision, hashes and reproducibility package"]
    REPRO --> PAPER["Write only supported claims; negative results are allowed"]
    EXCLUDE --> PAPER
```

The initial pilot is a calibration step, not statistical evidence of superiority.
Tev1's local input ceiling and chat fallback accounting currently block their
inclusion as clean strength comparators.

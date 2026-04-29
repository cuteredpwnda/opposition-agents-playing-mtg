# How the World Model + Agents Play Magic: The Gathering

> **Engine status (April 2026):** Games run end-to-end. Each game now begins
> with a London mulligan (configurable cap, default 3), the cleanup phase
> discards down to `PlayerState.max_hand_size`, drawing from an empty library
> ends the game (CR 104.3c), and max-turn timeouts resolve via a
> life → battlefield → hand → library tie-breaker rather than auto-DRAW. The
> world-model and agent layers described below now sit on top of a complete
> two-player rules pipeline.

## Table of Contents

1. [Overview — What Is This?](#1-overview)
2. [The World Models Lineage (Ha & Schmidhuber → IRIS → DIAMOND → Ours)](#2-world-models-lineage)
3. [Why World Models for MTG?](#3-why-world-models-for-mtg)
4. [Architecture: How All the Pieces Fit Together](#4-architecture)
5. [Component-by-Component Walkthrough](#5-component-walkthrough)
6. [How an Agent Actually Plays a Game](#6-how-an-agent-plays)
7. [Where LLM Capabilities Fit In](#7-llm-integration)
8. [Training: From Raw Data to a Playing Agent](#8-training-pipeline)
9. [Data Sources & the Commander Question](#9-data-sources)
10. [What DIAMOND Does Differently (and What We Borrow)](#10-diamond)
11. [Module Map — Every File and What It Does](#11-module-map)
12. [Open Questions & Next Steps](#12-next-steps)
13. [The Knowledge Graph as Long-Term Semantic Memory](#13-kg-as-long-term-memory)
14. [Transfer Learning: Standard → Commander](#14-transfer-learning)
15. [Remote Deployment](#15-remote-deployment)

---

## 1. Overview

This project builds AI agents that play Magic: The Gathering. The **world model** is one of several agent capabilities — it gives agents the ability to *imagine future game states* without actually running the rules engine, enabling fast lookahead planning.

The core loop:

```
Real Game State
    → Encode into latent vector z (State Encoder / V)
    → Predict what happens next (Dynamics Model / M)  
    → Choose the best action (Controller / C)
    → Execute in real game
```

The agent can also "dream" — simulate entire games inside the world model at orders of magnitude faster than the rules engine — and use those imagined games to train itself.

---

## 2. The World Models Lineage

### Ha & Schmidhuber (2018) — "World Models"
**Paper:** worldmodels.github.io  
**Key idea:** Train a 3-part model (V + M + C):
- **V** (Vision / VAE): Compresses 64×64 pixel frames into a latent vector z ∈ ℝ³²
- **M** (Memory / MDN-RNN): Predicts the next z given current z, action a, and hidden state h. Uses a *Mixture Density Network* to model multi-modal futures (multiple possible outcomes)
- **C** (Controller): A tiny linear model that maps (z, h) → action. Deliberately small (~900 params) so it can be trained with evolution (CMA-ES)

**Groundbreaking insight:** The controller can be trained *entirely inside the world model's dream* — the agent never touches the real environment during controller optimization. This is massively cheaper.

**Applied to:** VizDoom, CarRacing-v0. The agent learned to dodge fireballs and navigate tracks by dreaming.

### IRIS (Micheli, Alonso & Fleuret, 2023)
**Paper:** "Transformers are Sample-Efficient World Models" (ICLR 2023, notable top 5%)  
**Key evolution:** Replaced the VAE + RNN with:
- A **discrete autoencoder** (VQ-VAE style) — compress frames into discrete token sequences
- An **autoregressive Transformer** — predict next tokens given action + previous tokens

This basically turns "predict next game frame" into a *language modeling problem*. The Transformer can capture longer-term dependencies than RNNs. Achieved 1.046 mean human-normalized score on Atari 100k — outperforming humans on 10/26 games.

### DIAMOND (Alonso, Jelley et al., 2024)
**Paper:** "Diffusion for World Modeling: Visual Details Matter in Atari" (NeurIPS 2024 Spotlight)  
**Key evolution:** Replaced discrete tokens with **diffusion models**:
- Uses EDM (Karras et al.) as the denoising backbone
- Predicts the *actual next frame* via iterative denoising, conditioned on agent actions + previous frames
- Critical insight: visual details lost by VQ-VAE tokenization (like small projectiles or score changes) actually matter for RL. Diffusion preserves them.
- Uses n=3 denoising steps for speed while keeping multi-modal fidelity
- Achieved 1.46 mean HNS on Atari 100k — new best for world-model-only agents
- Also trained on 87h of static CS:GO gameplay → playable interactive neural game engine at ~10 FPS (381M params, from 4.4M for Atari)

**Key design choices:**  
- EDM over DDPM (more stable under low denoising steps)
- Multiple denoising steps handle multi-modal transitions (e.g., in Boxing, the opponent's moves can't be predicted from actions alone — 1-step blurs, 3-step selects a mode)
- Two-stage pipeline for CS:GO: low-res dynamics prediction → upsampler

### Our Approach for MTG

We take the V-M-C structure from Ha & Schmidhuber but **fundamentally adapt** it for a structured, partially-observable card game:

| Aspect | Original World Models | DIAMOND | Ours (MTG) |
|--------|----------------------|---------|------------|
| **Input** | 64×64 RGB pixels | 64×64 RGB pixels | Structured game state (zones, cards, life, mana, stack) |
| **V model** | CNN-based VAE | Diffusion model | Set Transformers + MLP VAE on game features |
| **M model** | MDN-RNN | Diffusion (conditioned on actions) | MDN-LSTM or Transformer (predicts latent z) |
| **C model** | Linear (900 params) | PPO policy | Linear or small MLP (~1-5K params) |
| **Multi-modality** | Gaussian mixture | Denoising steps | Gaussian mixture (MDN) |
| **Training data** | Agent plays the game | Static gameplay recordings + RL | 17Lands, self-play, MTGA logs |
| **Observation** | Full (pixels show everything) | Full | Partial (opponent hand hidden) |
| **Action space** | 3 continuous (steering) | 18 discrete (Atari) | Variable discrete (play land, cast spell, attack, block, pass...) |

We don't need diffusion because we're not rendering pixels — our game state is already structured data. The MDN's Gaussian mixture handles the multi-modal nature of MTG (will the opponent counter? will we topdeck a land?).

---

## 3. Why World Models for MTG?

MTG is uniquely challenging for AI:

1. **Huge action space** — At any priority window, you might have 0-50+ legal actions
2. **Hidden information** — You don't know the opponent's hand or library order
3. **Multi-modal outcomes** — Casting a spell can resolve, get countered, or trigger a chain of abilities
4. **Long-horizon planning** — Holding a card for 3 turns to bait a counterspell requires deep lookahead
5. **28,000+ unique cards** — Each with complex rules text producing emergent interactions

A world model helps because:

- **Fast simulation** — Imagining "what if I cast Lightning Bolt?" in latent space is ~1000x faster than running the rules engine
- **Opponent modeling** — The M model learns *statistical patterns* of how opponents respond, including hidden information effects
- **Planning depth** — Dream search can evaluate 8 rollouts × 10 steps deep in the time it takes to run one real game action
- **Sample efficiency** — Can learn from 17Lands data (millions of games) without needing to simulate every one

---

## 4. Architecture: How All the Pieces Fit Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REAL GAME                                   │
│                                                                     │
│  GameState ──→ GameCoordinator ──→ Rules Engine ──→ Next GameState  │
│      ↑                                                   │          │
│      └──────── Agent decides action ─────────────────────┘          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   AGENTS    │
                    │             │
         ┌─────────┤  MTGAgent   ├──────────┐
         │         │  (base)     │          │
         │         └──────┬──────┘          │
         │                │                 │
    ┌────▼────┐    ┌──────▼──────┐   ┌──────▼───────┐
    │  LLM    │    │   Random    │   │ World Model  │
    │  Agent  │    │   Agent     │   │    Agent     │
    │         │    │             │   │              │
    │ GPT/    │    │ Uniform     │   │ V+M+C model  │
    │ Claude  │    │ random      │   │ Dream search │
    │ + tools │    │ selection   │   │ or direct    │
    └─────────┘    └─────────────┘   └──────┬───────┘
                                            │
                           ┌────────────────┼────────────────┐
                           │                │                │
                    ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
                    │ V: State    │  │ M: Dynamics  │  │ C: Control │
                    │ Encoder     │  │ Model        │  │            │
                    │             │  │              │  │ (z,h) → a  │
                    │ GameState   │  │ MDN-LSTM:    │  │            │
                    │ → z ∈ ℝ²⁵⁶ │  │ P(z'|z,a,h)  │  │ Linear or  │
                    │             │  │              │  │ small MLP  │
                    │ Uses:       │  │ Predicts:    │  │            │
                    │ • Tokenizer │  │ • Next z     │  │ Scores     │
                    │ • Card emb  │  │ • Done prob  │  │ legal      │
                    │ • Set pools │  │ • Reward     │  │ actions    │
                    └─────────────┘  └──────────────┘  └────────────┘
                                            │
                                     ┌──────▼──────┐
                                     │   DREAM     │
                                     │ ENVIRONMENT │
                                     │             │
                                     │ Simulate    │
                                     │ entire      │
                                     │ games in    │
                                     │ latent      │
                                     │ space       │
                                     └─────────────┘
```

### The Two Data Paths

**Real game path:** `GameState → GameTokenizer → StateEncoder → z → Controller → Action → RulesEngine → next GameState`

**Dream path:** `z → Controller → Action → DynamicsModel → z' → Controller → ... (loop until done)`

---

## 5. Component-by-Component Walkthrough

### 5.1 GameTokenizer (`src/world_model/game_tokenizer.py`)

**What:** Converts a `GameState` object into fixed-size numpy arrays that neural networks can consume.

**How:** Takes the messy, variable-length game state (hand of 3 cards? 7 cards? battlefield with 2 creatures? 15?) and produces a standardized dict of arrays:

| Output Key | Shape | Content |
|-----------|-------|---------|
| `player_features` | (12,) | Life/40, mana pool (6 colors normalized), hand/lib/battlefield counts |
| `opponent_features` | (12,) | Same for opponent |
| `hand_cards` | (10, 128) | Card embeddings for up to 10 cards in hand |
| `hand_mask` | (10,) | Which slots have real cards vs padding |
| `battlefield_cards` | (20, 128) | Card embeddings for up to 20 permanents |
| `battlefield_state` | (20, 4) | Tapped, power/15, toughness/15, summoning_sick |
| `opp_battlefield_cards` | (20, 128) | Opponent's permanents |
| `graveyard_cards` | (20, 128) | Cards in graveyard |
| `stack_features` | (5, 130) | Cards on the stack + controller info |
| `phase_encoding` | (12,) | One-hot phase vector |
| `turn_features` | (4,) | Turn number/30, active player, priority holder, stack depth |

### 5.2 CardEmbeddingModel (`src/world_model/card_embeddings.py`)

**What:** Maps any card name → 128-dimensional vector.

**How:** Two approaches, combined:
1. **Text-based bootstrapping:** Runs `"Lightning Bolt — Instant — {R} — Lightning Bolt deals 3 damage to any target."` through a sentence-transformer (`all-MiniLM-L6-v2`), then projects to 128d. This gives zero-shot embeddings for *any* card ever printed.
2. **Gameplay-based (planned):** Learn embeddings from actual game trajectories — cards that function similarly in games end up near each other, regardless of flavor text.

### 5.3 StateEncoder / V Model (`src/world_model/state_encoder.py`)

**What:** VAE that compresses the tokenized game features into a latent vector z ∈ ℝ²⁵⁶.

**How:**
1. **SetEncoders** handle variable-length card zones — applies MLP to each card, then masked mean-pooling. This is permutation-invariant (order of cards on battlefield doesn't matter).
2. **Player/phase/turn encoders** handle fixed-size features via small MLPs.
3. **Fusion layer** concatenates all zone encodings → MLP → hidden representation.
4. **VAE bottleneck:** hidden → μ, log(σ²) → sample z using reparameterization trick.
5. **Decoder** reconstructs the fused features from z (for training via reconstruction + KL loss).

**Key insight:** Unlike pixel VAEs, we don't need to reconstruct the original image. We reconstruct the *fused feature vector*, which is much lower-dimensional.

### 5.4 DynamicsModel / M Model (`src/world_model/dynamics_model.py`)

**What:** Predicts the probability distribution of the next game state given the current state and action.

**How:**
- Input: `[z_t, action_encoding]` concatenated
- Backbone: 2-layer LSTM (or Transformer variant for long games)
- Output heads:
  - **MDN (Mixture Density Network):** 5 Gaussian components, each with mean μ_k and std σ_k, plus mixing coefficients π_k. This models *multi-modal* futures.
  - **Done head:** Predicts probability the game is over
  - **Reward head:** Predicts immediate reward

**Why MDN?** In MTG, the same action can lead to very different outcomes:
- You cast a creature → it resolves (mode 1) or gets countered (mode 2)
- You attack with everything → opponent blocks favorably (mode 1) or doesn't block (mode 2)

Each Gaussian component captures one plausible outcome. The mixing coefficients π give the probability of each.

**Temperature τ:** Controls dream stochasticity. τ > 1 makes dreams harder (more divergent futures), preventing the controller from exploiting artifacts of the model. τ < 1 is more deterministic. Default: τ = 1.15.

### 5.5 Controller / C Model (`src/world_model/controller.py`)

**What:** Maps (z, h) → action selection. Deliberately tiny.

**How:**
- Default: single linear layer `a = W[z; h] + b` (~200K params for 256+512 → 136)
- Optional: one hidden layer with tanh activation
- Scores legal actions via dot-product: `score_i = preference · legal_action_i`
- Returns log-probabilities over legal actions

**Why so small?** Following Ha & Schmidhuber's insight: all the complexity should live in V and M. The controller just needs to read the "summary" (z + h) and pick well. With fewer parameters, it's:
- Faster to train (CMA-ES works with ~1K-200K params)
- Less prone to overfitting
- More interpretable

**CMA-ES interface:** `get_flat_params()` / `set_flat_params()` allow evolution strategies to optimize the controller by treating all weights as a flat vector.

### 5.6 WorldModel (`src/world_model/world_model.py`)

**What:** The orchestrator — wires V + M + C together and provides high-level APIs.

**Key methods:**
- `encode(features)` → z, μ, logvar
- `predict(z, action, hidden)` → z_next, hidden_next, done_prob
- `act(z, h, legal_actions)` → action_idx, log_prob
- `dream(z_start)` → full trajectory in latent space
- `dream_search(z, hidden, legal_actions)` → best action via MCTS in latent space
- `save(path)` / `load(path)` → checkpoint V+M+C together

### 5.7 WorldModelAgent (`src/agents/world_model_agent.py`)

**What:** An `MTGAgent` subclass that plugs the world model into the game loop.

**Two modes:**
1. **Direct policy** (fast): Encode state → run controller → pick action. Suitable for real-time play.
2. **Dream search** (strong): For each legal action, simulate multiple futures via the dynamics model, score by cumulative predicted reward, pick the action with the best average outcome.

**observe() hook:** When any player takes an action (including the opponent), the agent feeds it through the dynamics model to keep the LSTM hidden state h updated. This gives the dynamics model memory of the game history.

---

## 6. How an Agent Actually Plays a Game

Here's the full chain when `WorldModelAgent.decide_action()` is called:

```
1. GameCoordinator says: "It's your turn, here are 7 legal actions"

2. WorldModelAgent receives GameState + legal_actions

3. ENCODE STATE:
   a. GameTokenizer.encode_state(game_state, player_id=0)
      → dict of numpy arrays (hand_cards, battlefield, life, mana, etc.)
   b. Convert to batched PyTorch tensors
   c. StateEncoder.encode(features)
      → z (256d latent vector), μ, logvar

4. ENCODE LEGAL ACTIONS:
   For each legal action:
      GameTokenizer.encode_action(action)
      → 136d vector (8d action type one-hot + 128d card embedding)
   Pad to fixed size, create mask

5. DECIDE (Direct mode):
   Controller.score_actions(z, h, action_encodings, mask)
   → log-probabilities over actions
   → pick highest (deterministic) or sample (stochastic)

   OR DECIDE (Dream search mode):
   For each legal action:
      For each rollout (8 times):
         Simulate: action → DynamicsModel.step() → z' → Controller picks next
         → DynamicsModel.step() → z'' → ... (10 steps deep)
         Sum up predicted rewards with discounting
      Average reward across rollouts
   Pick action with highest average imagined reward

6. Return the chosen Action to GameCoordinator

7. Observe outcome: opponent's response feeds through DynamicsModel
   to keep hidden state h current
```

---

## 7. Where LLM Capabilities Fit In

The world model doesn't replace the LLM — it **augments** it. Here's how they compose:

### The LLM Agent (`src/agents/llm_agent.py`)
- Uses GPT-4 / Claude to read game state as text, reason about strategy, pick actions
- Has access to tools: search cards, evaluate boards, check rules
- Strong at reasoning about card text, combos, and novel interactions
- **Weakness:** Slow (API call per decision), can't efficiently search decision trees

### The World Model Agent (`src/agents/world_model_agent.py`)
- Uses the V+M+C neural network to encode states and pick actions
- Fast (milliseconds per decision), can dream-search many rollouts
- **Weakness:** Limited to patterns seen in training data, may miss novel combos

### Potential Fusion: LLM-Guided World Model

The most powerful approach combines both:

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│ LLM Agent   │     │ World Model  │     │ Final      │
│             │     │              │     │ Decision   │
│ "I should   │────▶│ Dream search │────▶│            │
│ hold up     │     │ confirms:    │     │ Hold mana, │
│ mana for    │     │ +0.3 EV if   │     │ pass turn  │
│ counterspell│     │ we hold vs   │     │            │
│ because..." │     │ -0.1 if tap  │     │            │
└─────────────┘     └──────────────┘     └────────────┘
```

Ways LLMs enhance the world model:

1. **Card Embeddings:** The text-based card embeddings use a language model (sentence-transformer) to encode card rules text. The LLM's understanding of English gives meaningful zero-shot embeddings for any card.

2. **Strategic Guidance:** An LLM can propose 2-3 candidate strategies ("play aggressively" vs "hold up removal"), then the world model evaluates each via dream rollouts.

3. **Novel Card Interactions:** When the world model encounters card interactions it hasn't seen in training (e.g., a new set release), the LLM can reason about the rules text to fill in gaps.

4. **Explanation:** After the world model picks an action, the LLM can explain *why* in natural language: "I'm casting Lightning Bolt on your Ragavan because dream search shows this leads to +15% win rate versus holding it."

5. **Belief Updates:** The Active Inference module (`src/agents/active_inference.py`) maintains probabilistic beliefs about opponent's hand. The LLM can reason about these beliefs and update the world model's hidden state interpretation.

---

## 8. Training: From Raw Data to a Playing Agent

### Phase 1: Collect Trajectories

```
Data Sources:
├── 17Lands public datasets (millions of draft games)
├── Self-play via our game engine
├── MTGA log parsing (local player's detailed game logs)
└── (Future: XMage game recordings)
         │
         ▼
    TrajectoryStore
    └── data/trajectories/
        ├── metadata.json
        ├── game_001.npz
        ├── game_002.npz
        └── ...
```

### Phase 2: Train Card Embeddings

```
Scryfall bulk data (all 28K+ cards)
    → sentence-transformer("Lightning Bolt — Instant — {R} — deals 3 damage...")
    → project to 128d
    → save to data/card_embeddings/embeddings.npz
```

### Phase 3: Train State Encoder (V)

```
For each transition in TrajectoryStore:
    tokenize state → VAE encode → z, μ, logvar
    decode z → reconstructed features
    Loss = reconstruction_loss + β * KL_divergence
    
β-VAE warmup: gradually increase β over 10 epochs
Result: V can compress any game state to 256d
```

### Phase 4: Train Dynamics Model (M)

```
Pre-encode all states: V(game_state) → z for every transition
Create sequences: [(z₀,a₀), (z₁,a₁), ..., (zₜ,aₜ)]

For each sequence:
    Feed through LSTM → hidden states
    MDN head predicts: P(z_{t+1}) as mixture of 5 Gaussians
    Done head predicts: game over probability
    
Loss = -log_likelihood(actual z_{t+1} under mixture) + BCE(done) + MSE(reward)
Gradient clipping at norm 1.0
Result: M can predict what happens next from any (z, action)
```

### Phase 5: Train Controller (C) — In Dreams!

```
CMA-ES approach (original World Models):
    Population of 64 candidate controllers
    For each candidate:
        Set controller weights
        Run 16 dream rollouts (V+M simulate the game)
        Score = average cumulative reward
    CMA-ES updates parameter distribution
    Repeat for 100 generations
    
    OR

Policy Gradient approach:
    Run dream rollouts using current controller
    Compute discounted returns
    REINFORCE: ∇θ J = E[∇θ log π(a|z,h) * R]
    Update controller with Adam
```

### Phase 6: Dream Training Loop (Full Pipeline)

```
for iteration in 1..5:
    1. Train V on collected data
    2. Train M on V-encoded sequences  
    3. Train C inside dreams (V+M frozen)
    4. Deploy C to real games, collect new trajectories
    5. Add trajectories to store
    6. Increase dream temperature τ (harder dreams)
```

This iterative process is managed by `DreamTrainer` (`src/world_model/training/dream_trainer.py`).

---

## 9. Data Sources & the Commander Question

### Available Data Sources

| Source | Format | Type | Volume | Usability |
|--------|--------|------|--------|-----------|
| **17Lands** | CSV | Limited/Draft (Arena) | Millions of games | High — structured per-game stats, card-level data |
| **Self-play** | NPZ | Any format (our engine) | Unlimited (compute-bound) | Highest — full state transitions |
| **MTGA Logs** | JSON | Standard/Limited (Arena) | Your games only | Medium — detailed but single-player perspective |
| **Scryfall** | JSON | Card database | 28K+ cards | For card embeddings, not game logs |
| **MTGJSON** | JSON | Card database | All printed cards | Alternative to Scryfall for card data |
| **XMage** | Java logs | Any format | Varies | Potential future source (open-source engine) |

### Commander / EDH Game Log Databases

**Short answer: There is no large-scale Commander game log database comparable to 17Lands.**

Here's why and what exists:

- **EDHREC** (edhrec.com) — The dominant Commander data source. Tracks *decklist popularity* (which cards appear in which commander decks, % inclusion rates) but **not game-by-game play logs**. It tells you "72% of Atraxa decks run Doubling Season" but not "on turn 5, player cast Doubling Season and won 3 turns later."

- **Moxfield / Archidekt** — Deck building platforms with huge Commander deck databases. Decklists only, no game logs.

- **TopDeck.gg** — Competitive EDH (cEDH) tournament platform. Tracks tournament results (who won, with what deck) and has an API, but **not turn-by-turn game logs**. Useful for meta-game analysis.

- **Commander Spellbook** — Combo database for Commander. Lists card combinations and their effects. Useful for the knowledge graph, not for game trajectories.

- **MTG Arena** — Does not support Commander/EDH format at all. 17Lands only covers Limited and some Constructed Arena formats.

- **MTGO (Magic Online)** — Supports Commander and logs exist, but there's no public API or bulk data export for game replays. Some players manually share game logs.

- **Spelltable** — The primary online Commander platform (webcam-based). No game state logging whatsoever.

**For Commander specifically, the best approach is self-play** — have our own engine simulate Commander games and collect trajectories. The engine already supports multi-player games conceptually (the `GameState` has a `players: list[PlayerState]`).

### What 17Lands Data Actually Contains

The 17Lands public datasets (at 17lands.com/public_datasets) include:

- **Game data:** Per-game records with: draft_id, game_number, on_play/draw, won/lost, num_turns, and for every card in the pool: was it in the deck, was it drawn, was it in the opening hand, how many turns was it in hand
- **Draft data:** Every pick in every draft — what was in the pack, what was picked
- **Replay data:** Some per-action replay data (limited availability)

This is primarily **Limited** (Draft/Sealed) format on Arena. Not Commander, not Constructed.

---

## 10. What DIAMOND Does Differently (and What We Borrow)

### DIAMOND's Architecture

```
Previous frames [f_{t-3}, f_{t-2}, f_{t-1}] + action a_t
    → Diffusion model (EDM backbone, 3 denoising steps)
    → Predicted next frame f_t
    → Agent trained via PPO inside this diffusion world model
```

**Key differences from our approach:**

1. **Pixel-based vs Structured:** DIAMOND operates on raw pixels because Atari games are visual. MTG game state is symbolic/structured — we don't need pixel-level prediction.

2. **Diffusion vs MDN:** DIAMOND uses iterative denoising to handle multi-modal transitions (e.g., opponent's unpredictable movements cause blurring with 1-step, resolved with 3-step). We use Mixture Density Networks for the same purpose — our "modes" are things like {spell_resolves, spell_countered, combat_trade_favorable, combat_trade_unfavorable}.

3. **Scale:** DIAMOND's CS:GO model is 381M params trained on 87h of video. Our MDN-LSTM is ~2-5M params trainable on structured game logs. Much more tractable.

### What We Borrow from DIAMOND

- **The concept that details matter:** DIAMOND showed that lossy compression (VQ-VAE tokenization) loses important details. Our structured encoding avoids this — we encode exact life totals, exact mana, exact card identities. No information loss in the V model.

- **The training paradigm:** Train world model on external data (DIAMOND used static CS:GO recordings; we use 17Lands), then train the agent's policy inside the world model.

- **Multi-modal handling:** DIAMOND's insight that you need multiple denoising steps for transitions with ambiguous outcomes directly maps to our MDN's multiple Gaussian components.

---

## 11. Module Map — Every File and What It Does

### World Model Core (`src/world_model/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Exports: WorldModel, StateEncoder, DynamicsModel, Controller, GameTokenizer, CardEmbeddingModel, Trajectory, TrajectoryStore |
| `game_tokenizer.py` | `GameTokenizer` — Converts `GameState` ↔ fixed-size numpy arrays. Handles variable-length zones with padding/masking. |
| `card_embeddings.py` | `CardEmbeddingModel` — Card name → 128d vector. Text-based (sentence-transformer) + gameplay-based (planned). |
| `state_encoder.py` | `StateEncoder` (V model) — VAE: tokenized features → z ∈ ℝ²⁵⁶. Uses SetEncoder for card zones (permutation-invariant pooling). |
| `dynamics_model.py` | `DynamicsModel` (M model) — MDN-LSTM: P(z'|z,a,h). 5-Gaussian MDN head + done/reward prediction. Temperature τ. |
| `controller.py` | `Controller` (C model) — Linear/MLP: (z,h) → action scores. CMA-ES param interface. |
| `world_model.py` | `WorldModel` — Orchestrator: encode(), predict(), dream(), dream_search(), save/load. |
| `trajectory.py` | `Transition`, `Trajectory`, `TrajectoryStore` — Game trajectory storage with NPZ persistence. |

### Data Sources (`src/world_model/data_sources/`)

| File | Purpose |
|------|---------|
| `seventeen_lands.py` | Parse 17Lands public CSV datasets. Extract card win-rate statistics. |
| `self_play_collector.py` | Hook into game engine to record trajectories during self-play. |
| `mtga_log_parser.py` | Parse MTG Arena's `output_log.txt` for game replay data. |

### Training Pipelines (`src/world_model/training/`)

| File | Purpose |
|------|---------|
| `train_encoder.py` | Train the V model (VAE) with reconstruction + KL loss, β warmup. |
| `train_dynamics.py` | Train the M model on pre-encoded latent sequences. |
| `train_controller.py` | Train the C model via CMA-ES or REINFORCE, inside dreams. |
| `dream_trainer.py` | Full V→M→C iterative training pipeline with increasing dream temperature. |

### Agent Integration (`src/agents/`)

| File | Purpose |
|------|---------|
| `world_model_agent.py` | `WorldModelAgent(MTGAgent)` — Plugs world model into game loop. Direct policy or dream search modes. |
| `base_agent.py` | `MTGAgent` — Abstract base: `decide_action()`, `observe()`, `reset()`. |
| `llm_agent.py` | LLM-powered agent with tool use. |
| `random_agent.py` | Uniform random action selection (baseline). |
| `active_inference.py` | Free Energy Principle belief updates, opponent modeling. |
| `neural_reasoner.py` | GNN + Transformer fusion for board evaluation. |

### Supporting Infrastructure

| Module | Key Files | Purpose |
|--------|-----------|---------|
| `src/engine/` | `game_state.py`, `rules_engine.py`, `game_simulator.py` | Core game engine: state, rules, simulation |
| `src/training/` | `self_play.py`, `rewards.py`, `experience_buffer.py` | AlphaZero-style self-play training |
| `src/integrations/` | `scryfall.py`, `decklist_loader.py` | External API connectors |
| `src/knowledge/` | `knowledge_graph.py`, `combo_database.py` | Card relationship graph |
| `src/judge/` | `judge_agent.py` | Rules arbitration |

---

## 12. Open Questions & Next Steps

### Immediate
- [x] Implement proper reconstruction targets in V model's loss function (done)
- [x] Build the `build_from_trajectories()` method for gameplay-based card embeddings (done with context smoothing)
- [x] Integrate `SelfPlayCollector` hooks into `GameRunner`/priority loop (done)
- [x] Download and test 17Lands public dataset parsing (structure ready; parser added)

### Architecture Decisions
- **LSTM vs Transformer for M model?** LSTM is simpler and works for Ha & Schmidhuber's games. MTG games can be 20+ turns with 5+ priority passes per turn = 100+ steps. Transformer may handle longer sequences better, but attention over structured state (not pixels) is unexplored territory.
- **MDN components:** We default to 5 Gaussians. Is this enough for MTG's branching futures? Empirical tuning needed.
- **Dream temperature schedule:** Default [1.0, 1.05, 1.1, 1.15, 1.2] across training iterations. Need to validate this prevents policy exploitation without making dreams too hard.

### Commander Support — IMPLEMENTED

Commander (4-player, multiplayer EDH) is now fully supported:

- **GameTokenizer** (`src/world_model/game_tokenizer.py`): Rewritten for N-player encoding. `_get_opponent_ids()` returns all opponents. `encode_state()` aggregates opponent features via mean pooling, adds `num_players`, `format_commander` flag, and per-opponent `commander_damage` tracking. This keeps the state representation fixed-size regardless of player count.
- **GameRunner** (`src/orchestrator/game_runner.py`): Already N-player — turn rotation via modulo, APNAP priority passing, all phases support arbitrary player counts.
- **main.py**: `--commander` flag creates 4-player games (Alice, Bob, Charlie, Diana) with 100-card decks and 40 starting life.
- **RLTrainer** (`src/training/rl_trainer.py`): `_run_games()` supports both 2-player and 4-player Commander matchups with ELO tracking.
- **Self-play is the primary data path** — no external Commander game log databases exist (see §9).

### LLM Fusion Agent — IMPLEMENTED

The `LLMFusionAgent` (`src/agents/llm_fusion_agent.py`) combines four decision signals with configurable weights:

| Signal | Weight | Source | What It Provides |
|--------|--------|--------|-----------------|
| **World Model** | 0.40 | Dream rollouts via `WorldModel.dream()` | State-transition planning, "what happens if I do X?" |
| **LLM** | 0.30 | Ollama via `OllamaAgent` | Natural-language strategic reasoning, novel interactions |
| **Knowledge Graph** | 0.15 | Neo4j combo/synergy queries | Long-term card relationship memory, combo detection |
| **Heuristics** | 0.15 | Hand-coded MTG rules of thumb | Mana efficiency, threat assessment, board evaluation |

The fusion process:
1. For each legal action, each signal produces a score in [0, 1]
2. Scores are combined: `final = Σ(weight_i × score_i)`
3. The agent picks the highest-scoring action
4. **Optimization**: If one action is "obviously best" (heuristic score > 0.9 and only one such action), the expensive LLM/WM signals are skipped entirely.

`observe()` keeps the world model's LSTM hidden state updated and feeds observations to the opponent model.

### RL Self-Play Training — IMPLEMENTED

The `RLTrainer` (`src/training/rl_trainer.py`) provides a production-ready self-play loop:

```
for iteration in 1..N:
    1. Run K games between pool agents (2-player or 4-player Commander)
    2. Collect transitions into ExperienceBuffer
    3. Train NeuralReasoningModule on collected data (MSE loss, gradient clipping)
    4. Evaluate agents, update ELO ratings
    5. Periodically run DreamTrainer (world model improvement)
    6. Save checkpoint (model weights, ELO ratings, stats)
```

Agent pool uses **ELO ratings** for matchup selection — stronger agents play each other more often, preventing the system from wasting compute on trivially easy opponents.

### Deploy Pipeline — IMPLEMENTED

`scripts/deploy.py` orchestrates the full pipeline from raw infrastructure to a trained, playing agent:

1. **Check** — Verify Python, PyTorch, Neo4j, Ollama, n10s availability
2. **KG Build** — Ontology → Scryfall import → Combos → GraphSAGE embeddings → SHACL validation
3. **RL Train** — Self-play training loop with ELO tracking
4. **Dream Train** — World model V→M→C iterative improvement
5. **Play** — Demo game with trained agents

CLI: `python scripts/deploy.py --all` or pick stages: `--check`, `--kg`, `--train`, `--dream`, `--play`. Flags: `--commander`, `--fusion`, `--ollama`.

---

## 13. The Knowledge Graph as Long-Term Semantic Memory

This is perhaps the most architecturally important insight in the project: **the Neo4j knowledge graph is not just a database — it is the agent's long-term semantic memory**, analogous to how the hippocampus and neocortex store and retrieve structured knowledge in biological intelligence.

### Why a Graph, Not a Vector Store?

Most AI systems use flat vector databases (FAISS, Pinecone) for retrieval. These work well for "find me something similar to X" but fail at **relational reasoning** — the kind of thinking MTG demands:

> "If I have Doubling Season on the battlefield and play Ajani, Steadfast, does Ajani enter with enough loyalty counters to use his ultimate immediately?"

This question requires traversing a *chain of relationships*:
1. Doubling Season has an effect: "double counters placed on permanents you control"
2. Ajani, Steadfast is a planeswalker → enters with loyalty counters
3. Loyalty counters are counters → Doubling Season applies
4. Ajani's starting loyalty × 2 = enough for ultimate? Check the ultimate cost.

A vector store would need to have seen this exact combination during training. A knowledge graph can **derive it through traversal** — even for combinations it has never explicitly encountered.

### The Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT DECISION LOOP                   │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │  World Model  │   │  LLM (Ollama)│   │ Heuristics │  │
│  │  (Short-term) │   │  (Reasoning) │   │  (Reflex)  │  │
│  └──────┬───────┘   └──────┬───────┘   └─────┬──────┘  │
│         │                  │                  │         │
│         └──────────┬───────┘──────────────────┘         │
│                    │                                    │
│         ┌──────────▼───────────┐                        │
│         │   LLM Fusion Agent   │                        │
│         │  (Signal Combiner)   │                        │
│         └──────────┬───────────┘                        │
│                    │ queries                             │
│         ┌──────────▼───────────┐                        │
│         │   Knowledge Graph    │                        │
│         │   (Long-term Memory) │                        │
│         │                      │                        │
│         │  • Card identities   │                        │
│         │  • Combo paths       │                        │
│         │  • Synergy networks  │                        │
│         │  • Archetype models  │                        │
│         │  • Meta patterns     │                        │
│         │  • Game history      │                        │
│         └──────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

The three memory systems map to cognitive science concepts:

| System | Cognitive Analog | Timescale | What It Stores |
|--------|-----------------|-----------|----------------|
| **World Model** (V+M+C) | Working memory / mental simulation | Seconds (current turn lookahead) | Latent state vectors, predicted transitions, action values |
| **Knowledge Graph** (Neo4j) | Semantic memory / long-term declarative knowledge | Permanent (persists across all games) | Card properties, relationships, combos, archetypes, historical patterns |
| **LLM** (Ollama) | Linguistic reasoning / System 2 thinking | Per-query (stateless) | General MTG strategy, novel card interaction inference, natural language explanations |

### What Makes the KG a Good Long-Term Memory?

#### 1. Structured Relational Knowledge

The KG doesn't just store facts — it stores *relationships between facts* as first-class objects:

```cypher
// "What cards synergize with my commander and are also part of known combos?"
MATCH (cmd:Card {name: $commander})-[:SYNERGIZES_WITH]->(synergy:Card)
MATCH (synergy)-[:PART_OF]->(combo:Combo)
WHERE combo.format = 'commander'
RETURN synergy.name, combo.name, combo.steps
```

This single query performs multi-hop reasoning that would require multiple vector lookups and manual stitching in a flat store.

#### 2. Episodic Accumulation

Every game the agent plays can write back to the KG.

Important architectural rule: deterministic base graph facts (Scryfall +
ontology imports) are treated as immutable. Learned self-play knowledge is
written as append-only extension evidence events.

```cypher
// Append-only learned synergy evidence (no base-edge mutation):
MATCH (a:Card {cardName: $card_a})
MATCH (b:Card {cardName: $card_b})
CREATE (ev:LearnedSynergyEvidence:KGExtensionEvent {
  runId: $run_id,
  source: 'self_play',
  observedAt: datetime(),
  weight: $weight
})
CREATE (a)-[:SUPPORTED_BY]->(ev)
CREATE (ev)-[:SUPPORTED_BY]->(b)

// Append-only per-card outcomes:
MATCH (c:Card {cardName: $card_name})
CREATE (o:LearnedCardOutcome:KGExtensionEvent {
  runId: $run_id,
  source: 'self_play',
  observedAt: datetime(),
  won: $won
})
CREATE (c)-[:HAS_LEARNED_OUTCOME]->(o)
```

Over thousands of games, the KG builds a **stratified experience map**: which strategies work against which archetypes, which combos are reliable vs fragile, which cards over/underperform relative to their EDHREC popularity. This is genuine *learning from experience*, stored in a form that is queryable, explainable, and persistent.

#### 3. Graph Embeddings Bridge Symbolic and Neural

The `scripts/build_embeddings.py` pipeline trains GraphSAGE embeddings on the KG structure:

```
Card nodes → Feature vectors (CMC, colors, type, P/T, EDHREC rank)
    → GraphSAGE aggregates neighborhood features
    → 384-dimensional embedding per card
    → Written back to Neo4j as card.embedding property
    → Also exported to data/card_embeddings/graph_embeddings.npz
```

These embeddings capture **structural similarity** — cards that occupy similar positions in the synergy/combo/archetype graph get similar vectors, even if their text descriptions are very different. A board wipe that combos with graveyard recursion will be embedded near other board wipes with recursion synergy, not just near other board wipes.

The world model's StateEncoder can use these KG-derived embeddings as card features, giving the neural network access to the graph's relational knowledge without needing to query Neo4j at inference time.

#### 4. Inference Chains for Novel Situations

Unlike a lookup table, the KG supports **multi-hop inference** — deriving conclusions about card interactions the system has never explicitly analyzed:

```
Known: Card A SYNERGIZES_WITH Card B
Known: Card B ENABLES Combo C
Known: Combo C WINS_AGAINST Archetype D
Derived: Card A is strategically valuable against Archetype D
```

The `GraphRAG` module (`src/knowledge/graph_rag.py`) implements 6 retrieval strategies specifically designed for this: subgraph expansion, combo path search, archetype context, synergy clustering, meta positioning, and strategic context aggregation. Each strategy is a different "reasoning path" through the graph.

#### 5. Graceful Knowledge Decay and Update

The KG naturally handles knowledge evolution over time:

- **New cards released?** Add nodes and edges. Existing relationships are unaffected.
- **Meta shift makes a combo worse?** Update win-rate properties on combo nodes.
- **Card banned in Commander?** Mark legality property. All downstream queries automatically exclude it.
- **New synergy discovered?** Add an edge. The GraphSAGE embeddings can be re-trained to incorporate it.

This is fundamentally different from how neural networks handle knowledge updates (catastrophic forgetting, full retraining). The KG's symbolic structure means **local updates have local effects** — adding a fact about one card doesn't corrupt knowledge about other cards.

#### 6. Explainability

When the agent makes a decision influenced by the KG, the reasoning is fully traceable:

```
Agent chose "Play Doubling Season" because:
  KG query returned:
    - 3 active combos with cards in hand (Combo: DS + Ajani ultimate)
    - Synergy score 0.87 with current board state
    - Archetype match: "Superfriends" (historically 67% win rate)
  World Model dream rollout:
    - +0.3 expected value over 5 turns
  Combined fusion score: 0.82 (highest among legal actions)
```

This level of explainability is impossible with pure neural approaches. The KG provides the *why*, the world model provides the *what-if*, and the LLM provides the *natural language explanation*.

### KG + Dream Model Synergy

The knowledge graph and world model are complementary, not competing:

| Capability | KG (Long-term) | World Model (Short-term) |
|-----------|----------------|-------------------------|
| "What combos exist?" | ✅ Graph traversal | ❌ Not represented |
| "What happens next turn?" | ❌ No state simulation | ✅ Dream rollouts |
| "Is this card good here?" | ✅ Synergy/archetype context | ✅ Action-value prediction |
| "What did we learn last game?" | ✅ Persistent episodic memory | ❌ Weights only, no episodes |
| "How do opponents play?" | ✅ Archetype + strategy patterns | ✅ Opponent model LSTM |

The KG informs the world model through:
- **Richer card features**: KG embeddings feed into the StateEncoder, giving the VAE access to relational card knowledge
- **Combo-based reward shaping**: +bonus reward when the agent reaches states that are "on track" for a KG-identified combo
- **Strategic priors**: KG archetype analysis biases the Controller's initial action preferences toward historically successful strategies
- **Dream guidance**: When the world model is uncertain (high MDN variance), the KG provides a fallback: "even if I can't simulate this precisely, the KG says this combo is reliable"

---

## 14. Transfer Learning: Standard → Commander

### The Insight

Standard and Commander share ~95% of game rules: phases, priority, combat, state-based actions, mana systems, card types, and spell resolution all work identically. The difference is **multiplayer dynamics** — threat assessment across 3 opponents, political negotiation, and commander-specific mechanics (command zone, commander damage, color identity restriction).

This means a world model trained on Standard already understands:
- How mana curves work
- How combat math resolves
- What card interactions do
- How to sequence plays within a turn

All of this transfers directly to Commander. The only thing the model needs to *learn anew* is how to handle multiple opponents — which is a much smaller learning problem than learning the entire game from scratch.

### Why Transfer Works Here

The `GameTokenizer` already maps any N-player game to a **fixed-size perspective** (self + mean-aggregated opponents). This means the neural network sees the same tensor shape for Standard and Commander:

```
Standard (2 players):
  player_features:   [life, mana(6), hand, lib, bf, lands]  → 11 dims
  opponent_features:  [life, mana(6), hand, lib, bf, lands]  → 11 dims (1 opponent)

Commander (4 players):
  player_features:   [life, mana(6), hand, lib, bf, lands]  → 11 dims
  opponent_features:  mean([opp1, opp2, opp3])               → 11 dims (averaged)
                      + num_players (1), format_commander (1), commander_damage (4)
```

The base StateEncoder (V model) doesn't need architectural changes — only the `MultiplayerAdapter` adds format-specific conditioning through a lightweight residual connection.

### Three-Phase Curriculum

```
Phase 1: STANDARD (50 iterations)
  ┌──────────────────────────────────────┐
  │  V(VAE) + M(LSTM) + C(Linear)       │  All weights trained
  │  2-player games, 20 life, base rules │  LR = 3e-4
  │  → Learns: cards, combat, mana       │
  └──────────────────────────────────────┘
                    │
                    ▼ Load pretrained weights
Phase 2: COMMANDER (30 iterations)
  ┌──────────────────────────────────────┐
  │  V + M (FROZEN first 10 iters)      │  Adapter + C trained
  │  + MultiplayerAdapter (trainable)    │  LR = 1e-4 (lower)
  │  4-player games, 40 life, cmd zone   │
  │  → Learns: multiplayer, politics     │
  └──────────────────────────────────────┘
                    │
                    ▼ Unfreeze all weights
Phase 3: JOINT (20 iterations)
  ┌──────────────────────────────────────┐
  │  All weights trainable               │  LR = 5e-5 (very low)
  │  30% Standard + 70% Commander games  │
  │  → Consolidates: prevents forgetting │
  └──────────────────────────────────────┘
                    │
                    ▼
              Final model (plays both formats)
```

### The Multiplayer Adapter

Instead of fine-tuning the entire V+M stack (which risks catastrophic forgetting), a small adapter module learns to transform opponent features for the multiplayer context:

```python
class MultiplayerAdapter:
    # Input:  opponent_features(11) + [num_players, is_commander]
    # Hidden: 32 neurons (ReLU)
    # Output: 11 dims (Tanh)  → residual connection
    #
    # adapted = original_features + adapter(features, conditioning)
    #
    # Initialized near-zero so the adapter starts as identity.
```

This is inspired by adapter-based transfer learning (Houlsby et al. 2019) — small bottleneck modules added to a frozen pretrained model that learn task-specific adaptations without corrupting the original weights.

### Running Transfer Learning

```bash
# Full curriculum (locally)
python main.py --transfer --rl-iters 100

# On a remote machine
python scripts/push_remote.py user@gpu-server --transfer

# Via deploy script
python scripts/deploy.py --transfer --iters 100
```

---

## 15. Remote Deployment

### Why Remote?

The full stack (Neo4j + Ollama LLM + PyTorch training) needs more resources than a typical development laptop:
- **Neo4j** wants 2-4 GB heap + 2 GB page cache for the full Scryfall card database
- **Ollama** needs 4-8 GB RAM for Mistral/Llama models (more for larger models)
- **RL training** benefits from GPU acceleration for the neural modules

### Architecture

```
┌─────────────────────────────────────────────┐
│  Remote Machine (GPU server / cloud VM)     │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Neo4j  │  │  Ollama  │  │  Trainer   │  │
│  │  :7474  │  │  :11434  │  │  (Python)  │  │
│  │  :7687  │  │          │  │            │  │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘  │
│       │            │              │          │
│       └────────────┼──────────────┘          │
│            Docker network                    │
└─────────────────────────────────────────────┘
          ▲                    │
          │ rsync              │ rsync
          │ (push code)        │ (pull checkpoints)
          │                    ▼
┌─────────────────────────────────────────────┐
│  Local Machine (development laptop)         │
│  - Edit code                                │
│  - Run tests                                │
│  - Play demo games with trained models      │
└─────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Push to remote and start full training
python scripts/push_remote.py user@gpu-server.example.com

# 2. Transfer learning curriculum
python scripts/push_remote.py user@server --transfer

# 3. Just sync code without starting training
python scripts/push_remote.py user@server --sync-only

# 4. Pull trained models back to local machine
python scripts/push_remote.py user@server --pull-checkpoints

# 5. Monitor training on the remote machine
ssh user@server 'cd ~/mtg-agents && docker compose \
  -f docker-compose.yml -f docker-compose.remote.yml logs -f trainer'
```

### Docker Compose Stack

The base `docker-compose.yml` provides Neo4j. The `docker-compose.remote.yml` override adds:
- **Ollama** service for LLM inference (with optional GPU passthrough)
- **Trainer** container built from the project Dockerfile
- Higher memory limits for Neo4j (4 GB heap, 2 GB page cache)
- Shared volume mounts for checkpoints and data

For GPU-accelerated training, uncomment the `deploy.resources` sections in `docker-compose.remote.yml`.

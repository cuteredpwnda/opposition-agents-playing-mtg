# How the World Model + Agents Play Magic: The Gathering

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
- [ ] Implement proper reconstruction targets in V model's loss function
- [ ] Build the `build_from_trajectories()` method for gameplay-based card embeddings
- [ ] Integrate `SelfPlayCollector` hooks into `GameCoordinator`'s game loop
- [ ] Download and test 17Lands public dataset parsing

### Architecture Decisions
- **LSTM vs Transformer for M model?** LSTM is simpler and works for Ha & Schmidhuber's games. MTG games can be 20+ turns with 5+ priority passes per turn = 100+ steps. Transformer may handle longer sequences better, but attention over structured state (not pixels) is unexplored territory.
- **MDN components:** We default to 5 Gaussians. Is this enough for MTG's branching futures? Empirical tuning needed.
- **Dream temperature schedule:** Default [1.0, 1.05, 1.1, 1.15, 1.2] across training iterations. Need to validate this prevents policy exploitation without making dreams too hard.

### Commander Support
- Extend `GameTokenizer` for 4-player games (currently 2-player perspective)
- `PlayerState` list already supports N players in `GameState`
- Commander-specific features: command zone, commander damage tracking, commander tax
- Self-play is the primary data path (no external Commander game logs)

### LLM Fusion
- [ ] Design the LLM → World Model feedback loop
- [ ] Implement "LLM proposes strategy, world model evaluates" pipeline
- [ ] Use LLM for novel card interaction prediction when world model is uncertain
- [ ] Fine-tune card embeddings using LLM-generated card descriptions and comparisons

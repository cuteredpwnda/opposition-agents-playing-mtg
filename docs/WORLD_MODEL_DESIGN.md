# World Model Architecture for MTG

## Executive Summary

We apply the **World Model** framework (Ha & Schmidhuber, 2018) to Magic: The Gathering. Instead of learning from pixel frames, our world model learns compressed representations of structured game states and predicts future state transitions — enabling an agent to plan and "dream" entire games in latent space.

The key insight: MTG has ~28,000 unique cards, complex rules interactions, and hidden information. An LLM alone can't efficiently search the decision tree. A world model lets us:

1. **Compress** any game state into a fixed-size latent vector
2. **Predict** state transitions (what happens after I cast Lightning Bolt?)
3. **Simulate** entire games in latent space (1000x faster than the rules engine)
4. **Train** a compact controller policy entirely inside "dreams"

### Collective Intelligence Through Shared Graph Memory

The world model in this repository is intended to operate in a multi-agent
population loop, not as an isolated policy learner.

1. Multiple agent families generate trajectories through self-play.
2. A KG enrichment pass extracts repeatable card-pair and outcome patterns.
3. Learned evidence is appended (with provenance) to the KG extension layer.
4. Later world-model/controller runs consume that shared graph memory as prior context.

This creates cumulative strategic learning across runs while keeping immutable
card-source facts and ontology-grounded base edges unchanged.

---

## Architecture: V-M-C Adapted for MTG

```
                 ┌──────────────────────────────────────────────┐
                 │            REAL GAME ENGINE                   │
                 │  GameState → Rules Engine → next GameState    │
                 └──────────┬───────────────────────┬───────────┘
                            │ collect trajectories  │
                 ┌──────────▼───────────────────────▼───────────┐
                 │           TRAJECTORY STORE                    │
                 │  [(s₀,a₀,s₁), (s₁,a₁,s₂), ... ]            │
                 │  Sources: self-play, 17Lands, MTGA logs       │
                 └──────────┬───────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   V: State   │  │ M: Dynamics  │  │ C: Controller│
   │   Encoder    │  │   Model      │  │   (Policy)   │
   │              │  │              │  │              │
   │  GameState   │  │ P(zₜ₊₁ |    │  │  aₜ = f(zₜ, │
   │  → zₜ ∈ ℝ²⁵⁶│  │  zₜ, aₜ, hₜ)│  │         hₜ)  │
   │              │  │              │  │              │
   │  [Card Emb]  │  │  [LSTM/TF]  │  │  [Linear/   │
   │  [Board Enc] │  │  [MDN head]  │  │   Small MLP] │
   │  [Mana Enc]  │  │              │  │              │
   └──────────────┘  └──────────────┘  └──────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                 ┌──────────▼───────────────────────────────────┐
                 │            DREAM ENVIRONMENT                  │
                 │  Agent plays entire games in latent space     │
                 │  No rules engine needed — M generates states  │
                 │  Temperature τ controls uncertainty           │
                 └──────────────────────────────────────────────┘
```

---

## V: State Encoder

Unlike Ha & Schmidhuber's VAE on image pixels, we encode **structured** game state.

### Input Features

| Feature Group         | Dimensions | Description                              |
|-----------------------|------------|------------------------------------------|
| Per-card embeddings   | 128 each   | Learned from card name + oracle text      |
| Board state           | ~512       | Cards per zone, per player (pooled)       |
| Player vitals         | ~32        | Life, mana pool, land plays, commander dmg|
| Phase/turn context    | ~16        | One-hot phase + turn number encoding      |
| Stack state           | ~64        | Items on stack (spells pending)           |
| Hidden info belief    | ~64        | Estimated opponent hand composition       |

### Architecture

```python
GameState → GameTokenizer → token sequences
    → Card Embeddings (lookup from pretrained or learned)
    → Set Transformer (pool variable-length card sets)
    → Concatenate [board_enc, player_enc, phase_enc, stack_enc]
    → MLP → μ, σ (VAE bottleneck)
    → z ∈ ℝ²⁵⁶
```

### Training

- **Reconstruction loss**: Decode z back to predict game features (life totals, cards in each zone, etc.)
- **KL divergence**: Regularize toward N(0, I) for smooth latent space
- **Contrastive**: Similar game states → nearby z vectors

---

## M: Dynamics Model (MDN-RNN / Transformer)

Predicts the distribution of the next latent state given current state + action.

### Formulation

$$P(z_{t+1}, d_{t+1} \mid z_t, a_t, h_t)$$

Where:
- $z_t$ = current latent state (from V)
- $a_t$ = action taken (tokenized)
- $h_t$ = hidden state of the recurrent model
- $z_{t+1}$ = predicted next latent state
- $d_{t+1}$ = game-over probability

### Architecture Options

1. **MDN-LSTM** (original World Models): LSTM + Mixture Density Network output
2. **Transformer** (modern): Causal transformer over (z, a) sequences
3. **Hybrid**: Transformer encoder, MDN decoder

### Why MDN?

MTG has discrete, multi-modal outcomes:
- "Will opponent counter my spell?" → two very different next states
- "What will opponent draw?" → affects all subsequent states
- Mixture of Gaussians captures these discrete modes

### Temperature Parameter τ

- τ = 1.0: Standard uncertainty (matches training data)
- τ > 1.0: More stochastic dreams (harder, prevents policy exploits)
- τ < 1.0: More deterministic (easier, but policy may overfit to model errors)

---

## C: Controller

Deliberately simple — a linear model or small MLP:

$$a_t = W_c [z_t; h_t] + b_c$$

### Why Simple?

- V and M absorb game complexity (~5-10M params)
- C has ~1-5K params → trainable with CMA-ES or simple policy gradient
- Prevents the controller from "memorizing" the world model's flaws
- Can also distill LLM decisions into C as a starting policy

---

## Card Embeddings

Foundation of the system — every card needs a fixed-size vector.

### Approach 1: Text-Based (Bootstrap)
- Encode card name + oracle text + type line with a sentence transformer
- Fine-tune on card similarity / synergy pairs from EDHREC

### Approach 2: Gameplay-Based (Learned)
- Cards that appear in similar contexts → similar embeddings
- Train using game trajectories: predict next action given cards in hand/board
- Like Word2Vec but for cards

### Approach 3: Hybrid
- Initialize from text embeddings, fine-tune during world model training
- Best of both worlds — semantic meaning + gameplay utility

---

## Game Tokenizer

Converts game states and actions into token sequences the models can process.

### State Tokens
```
[TURN 5] [PHASE main_1] [ACTIVE P1]
[P1 LIFE 18] [P1 MANA W:2 U:1 R:0 ...]
[P1 HAND] <Lightning Bolt> <Counterspell> <Island>
[P1 BATTLEFIELD] <Snapcaster Mage (tapped)> <Island> <Mountain>
[P2 LIFE 15] [P2 MANA ...]
[P2 BATTLEFIELD] <Goblin Guide> <Mountain>
[GRAVEYARD P1] <Opt>
[STACK] (empty)
```

### Action Tokens
```
[CAST Lightning Bolt targeting Goblin Guide]
[PLAY_LAND Island]
[ATTACK Snapcaster Mage]
[PASS_PRIORITY]
```

---

## Training Data Sources

### 1. 17Lands (Primary — Draft/Limited)
- **What**: Millions of MTG Arena draft games with per-pick and per-game data
- **URL**: https://17lands.com/
- **Data**: Card pick rates, game actions, win rates per card, game replays
- **Access**: Public datasets at https://www.17lands.com/public_datasets
- **Format**: CSV files with draft picks, game data with mulligan/play/draw info
- **Limitation**: Draft/Limited only (not Constructed), no full action-by-action logs

### 2. Self-Play (Primary — Constructed)
- **What**: Our own engine plays thousands of games, recording full trajectories
- **Source**: GameSimulator + RandomAgent / LLMAgent
- **Advantage**: Full action-level detail, any format, any decklist
- **Pipeline**: Run games → extract (state, action, next_state) tuples → store

### 3. MTGA Detailed Logs (Supplementary)
- **What**: MTG Arena writes detailed game logs to disk
- **Path**: `%APPDATA%/../LocalLow/Wizards Of The Coast/MTGA/`
- **Format**: JSON-ish log entries (GRE messages with game state diffs)
- **Content**: Full game actions, state changes, card movements between zones
- **Caveat**: Requires a local MTGA installation and playing games

### 4. XMage Game Logs (Supplementary)
- **What**: Open-source MTG engine (Java) with AI opponents
- **URL**: https://github.com/magefree/mage (2.2k stars, 28k+ cards)
- **Data**: Game logs from AI vs AI matches
- **Advantage**: Full rules enforcement, can generate unlimited training data

### 5. MTGJSON (Card Database)
- **What**: Complete card database in JSON/CSV/SQLite
- **URL**: https://mtgjson.com/
- **Use**: Card metadata for building card embeddings (name, text, types, CMC)

### 6. Scryfall Bulk Data
- **What**: Full card database with oracle text, rulings, legality
- **URL**: https://scryfall.com/docs/api/bulk-data
- **Use**: Card embeddings, oracle text encoding

---

## Training Pipeline

### Phase 1: Collect Trajectories
```
Random/LLM agents play 10,000+ games
    → Record (GameState, Action, next_GameState) for every decision
    → Store as trajectory files (Parquet or SQLite)
```

### Phase 2: Train Card Embeddings
```
All unique cards from trajectories + Scryfall bulk data
    → Sentence transformer on (name + oracle_text + type_line)
    → Fine-tune: cards played together should be close
    → Output: card_name → ℝ¹²⁸ embedding lookup
```

### Phase 3: Train State Encoder (V)
```
GameState → GameTokenizer → V (encoder) → z ∈ ℝ²⁵⁶
z → V (decoder) → reconstructed features
Loss = reconstruction + KL divergence
```

### Phase 4: Train Dynamics Model (M)
```
Sequences of (zₜ, aₜ) → M → P(zₜ₊₁)
Train on full game trajectories (encoded through V)
Loss = negative log-likelihood of actual zₜ₊₁ under predicted distribution
```

### Phase 5: Train Controller (C) in Dreams
```
Roll out M to generate dream games (no rules engine needed)
C takes (zₜ, hₜ) and outputs action
Reward = win/loss + shaped rewards (life advantage, board control)
Optimize C using CMA-ES or policy gradient
```

### Phase 6: Transfer + Iterate
```
Deploy C into real game engine
Collect more trajectories from improved agent
Retrain V, M with new data → retrain C
Repeat
```

---

## Integration With Existing Architecture

### WorldModelAgent (extends MTGAgent)
```python
class WorldModelAgent(MTGAgent):
    """Agent powered by a trained world model."""

    def __init__(self, player_id, world_model):
        self.world_model = world_model  # V + M + C

    async def decide_action(self, game_state, legal_actions):
        # Encode current state
        z = self.world_model.encode(game_state)
        h = self.world_model.get_hidden_state()

        # Option A: Direct policy from controller
        action_probs = self.world_model.controller(z, h)

        # Option B: Lookahead — simulate N futures in latent space
        best_action = self.world_model.dream_search(
            z, h, legal_actions, depth=3, num_rollouts=100
        )

        return best_action
```

### Dream Search (Monte Carlo in Latent Space)
```
For each legal action aₜ:
    For rollout in range(num_rollouts):
        z' = M.predict(zₜ, aₜ, hₜ, τ=1.15)
        Simulate forward N steps using C to pick actions
        Estimate terminal reward
    Score[aₜ] = mean(rewards)
Return argmax Score
```

This is essentially **Monte Carlo Tree Search in latent space** — but runs 100-1000x faster than searching through the actual rules engine.

---

## Module Layout

```
src/world_model/
├── __init__.py
├── state_encoder.py         # V — GameState → z (VAE)
├── dynamics_model.py        # M — predicts P(z_{t+1} | z_t, a_t, h_t)
├── controller.py            # C — maps (z, h) → action distribution
├── world_model.py           # Orchestrates V + M + C
├── game_tokenizer.py        # Converts GameState ↔ token/feature vectors
├── card_embeddings.py       # Card name → ℝ¹²⁸ embedding
├── trajectory.py            # Trajectory dataclasses + storage
├── data_sources/
│   ├── __init__.py
│   ├── seventeen_lands.py   # 17Lands public dataset import
│   ├── self_play_collector.py  # Record games from our engine
│   └── mtga_log_parser.py   # Parse MTG Arena detailed logs
└── training/
    ├── __init__.py
    ├── train_encoder.py     # Train V model
    ├── train_dynamics.py    # Train M model
    ├── train_controller.py  # Train C model (CMA-ES or PG)
    └── dream_trainer.py     # Full dream training loop

src/agents/
├── world_model_agent.py     # MTGAgent subclass using world model
```

---

## Open Questions

1. **Latent space size**: 128? 256? 512? Needs experimentation
2. **Transformer vs LSTM for M**: Transformers are better at long sequences but LSTM is proven for world models. Could try both.
3. **Handling variable action spaces**: MTG has 0-50+ legal actions per decision point. Need masking.
4. **Hidden information**: How much belief state to encode? Active inference module already tracks this — can feed directly into V.
5. **Card embedding cold start**: New cards have no gameplay data. Text embeddings bootstrap, then fine-tune.
6. **Dream fidelity**: How accurate does M need to be? Ha & Schmidhuber showed even imperfect models work if you increase τ.

---

## References

- Ha, D. & Schmidhuber, J. (2018). "Recurrent World Models Facilitate Policy Evolution." NeurIPS 2018. https://worldmodels.github.io/
- Silver, D. et al. (2016). "The Predictron: End-to-End Learning and Planning."
- Schrittwieser, J. et al. (2020). "MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model."
- 17Lands. MTG Arena draft analytics. https://17lands.com/
- MTGJSON. Portable MTG card data. https://mtgjson.com/
- XMage. Open-source MTG game engine. https://github.com/magefree/mage

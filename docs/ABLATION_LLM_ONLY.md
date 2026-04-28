# LLM-Only Ablation: Getting Started

## What This Is

This is the **first meaningful ablation** of the agentic play system: two small language models (2-4B parameters) making real-time decisions in actual MTG games, **without any training or world models**. Pure LLM reasoning vs. the enforced rules.

## Why Start Here?

This ablation answers: **"How good is raw LLM reasoning at MTG?"**

Before we layer in world models, training, active inference, etc., we want a baseline: can two small open-source models play reasonable Magic just by reasoning through the game state?

## Available Models on Your System

Run `ollama list` to see what's installed. Currently available:
- `llama3.2:1b` — tiny, 1B params, quick
- `qwen2.5-coder:1.5b` — coding-optimized, 1.5B params
- `phi:latest` — small, general
- `stable-code:3b-code-q4_0` — code-focused, 3B params  
- `nemotron-3-nano:4b` — general, 4B params

## Running the Ablation

### Single Game
```bash
python examples/ablation_llm_only.py
```
Runs: `llama3.2:1b` vs `qwen2.5-coder:1.5b` (default)

### Custom Match
```bash
python examples/ablation_llm_only.py --model1 phi --model2 stable-code --games 3
```
Runs 3 games: `phi:latest` vs `stable-code:3b-code-q4_0`

### Available Aliases
Short names map to full ollama tags:
- `1b` → `llama3.2:1b`
- `1.5b` → `qwen2.5-coder:1.5b`
- `3b` → `stable-code:3b-code-q4_0`
- `4b` → `nemotron-3-nano:4b`
- `llama`, `qwen`, `phi`, `stable-code`, `nemotron` (full names)

### Options
```bash
python examples/ablation_llm_only.py --help
```
- `--model1 NAME` — First agent's model
- `--model2 NAME` — Second agent's model  
- `--games N` — Number of games (default: 1)
- `--seed SEED` — Reproducibility
- `--verbose` — Detailed logs

## What We're Testing

Each agent:
1. **Sees** the current game state (lands, creatures, hand, mana pool, life totals, etc.)
2. **Reasons** about the best action via LLM inference
3. **Decides** on an action (cast spell, declare attackers, play land, pass)
4. **Executes** the action (rules engine validates and applies it)

The game loop:
- Full turn cycle (untap → draw → main → combat → main 2 → cleanup)
- Priority system for spells/abilities
- Combat resolution with damage  
- Library depletion (deck out) detection
- Game-over detection (20 life lost or deck empty)

**No training.** No world models. No KG. Just raw LLM decision-making at each turn.

## Key Metrics

After running games, you'll see:
- **Win counts** — which model won more
- **Average turns** — game length (shorter = faster play, longer = grinding games)
- **Time per game** — inference latency on your hardware

## Next Steps

Once you understand how pure LLM play works:
1. **Add world model** — Give agents a learned predictor of future states
2. **Add KG reasoning** — Inject Magic-specific knowledge from the ontology
3. **Add active inference** — Model agents as minimizing free energy over beliefs
4. **Train on self-play** — Improve agent policies via RL on recorded games

## Code Structure

- **`src/agents/llm_agent.py`** — `OllamaAgent` class with:
  - `decide_action(game_state, legal_actions)` — main decision method
  - `decide_mulligan(hand, mulligans_taken, max_mulligans)` — opening hand logic
  - `select_bottom_cards(hand, n)` — mulligan card ordering
  - HTTP bridge to local Ollama server

- **`src/orchestrator/game_runner.py`** — Async game executor:
  - Sets up game state, mulligans, draws opening hand
  - Runs turn loop with priority system
  - Calls `agent.decide_action()` for every decision point

- **`src/orchestrator/priority_loop.py`** — MTG priority loop:
  - Gets legal actions for current player via rules engine
  - Calls `agent.decide_action()` to pick one
  - Resolves stack and checks for game end

- **`examples/ablation_llm_only.py`** — The ablation runner (this is what you run)

## Interpreting Results

- **Short games (5-10 turns)** — Aggressive play, quick kills
- **Medium games (15-25 turns)** — Balanced, some resource trading
- **Long games (30+ turns)** — Grindy, both players struggling  
- **Identical win rates** — Models are evenly matched in reasoning ability
- **Lopsided win rates** — One model's reasoning is better (or better-suited for the decks)

## Troubleshooting

### "Model not loaded"
```
Available models: llama3.2:1b, qwen2.5-coder:1.5b, ...
Model XXX not loaded.
```
**Solution:** Add the model via `ollama pull <model-name>` or use an available one.

### "Ollama unavailable"
```
[player1] Ollama unavailable: connection refused
[player1] Using RandomAgent fallback
```
**Solution:** Start Ollama server: `ollama serve` in another terminal (usually runs on `http://localhost:11434`)

### Game times out / loops forever
The priority loop has a safety cap (500 iterations). If you see:
```
Priority loop aborted: exceeded max iterations
```
This means the agents are stuck (e.g., both passing but board state not changing). This is rare and usually indicates a bug in action generation. Report with the game log.

## Next: Adding World Models

Once you're happy with the baseline, we add a learned world model:

```python
from src.agents.llm_fusion_agent import LLMFusionAgent
from src.world_model.world_model import WorldModel

wm = WorldModel.load("checkpoints/wm-medium.pt")
agent = LLMFusionAgent("p1", world_model=wm, model="llama3.2:1b")
```

This agent will:
1. Call LLM for reasoning: "What should I do?"
2. Dream-rollout via world model: "What if I did that?"
3. Fuse both signals: "Here's my best decision"

---

**Happy ablating!** 🧪✨

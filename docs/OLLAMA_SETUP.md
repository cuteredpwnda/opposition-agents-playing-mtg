# Ollama Setup for MTG Agents

This project uses Ollama for LLM-powered agent decision-making in Magic: The Gathering games.

## Installation

1. **Install Ollama**
   - Download from: https://ollama.ai
   - Available for Mac, Linux, and Windows

2. **Pull a model**
   ```bash
   ollama pull gemma4:e2b
   ```
   
   Other recommended models:
   - `ollama pull gemma4:e4b` - Larger Gemma 4 edge variant (better reasoning, ~4 GB VRAM)
   - `ollama pull gemma4:12b` - Mid-tier Gemma 4 (needs ~8 GB VRAM)
   - `ollama pull gemma4:27b` - Full Gemma 4 (best quality, ~16 GB VRAM)
   - `ollama pull mistral` - Alternative open-source model
   - `ollama pull phi` - Lightweight option

3. **Start Ollama server**
   ```bash
   ollama serve
   ```
   
   This runs the API on `http://localhost:11434` by default

### Keeping multiple models loaded in VRAM (recommended for ablations)

By default Ollama unloads a model after ~5 min of idle and only keeps **one**
model resident in VRAM. Ablations that pit two models against each other
will then ping-pong load/unload on every turn, which is brutally slow.

Set these environment variables **before** starting `ollama serve`:

| Variable | Recommended | Purpose |
|----------|-------------|---------|
| `OLLAMA_MAX_LOADED_MODELS` | `3` | Allow 3 models in VRAM simultaneously |
| `OLLAMA_NUM_PARALLEL`      | `2` | Allow 2 concurrent requests per model |
| `OLLAMA_KEEP_ALIVE`        | `30m` | Keep models loaded 30 min after last call |

**Windows (PowerShell, persistent):**
```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '3', 'User')
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL',      '2', 'User')
[System.Environment]::SetEnvironmentVariable('OLLAMA_KEEP_ALIVE',        '30m', 'User')
# Restart Ollama (tray -> Quit, then re-launch) for changes to take effect.
```

**Linux/macOS (`~/.bashrc` or systemd unit):**
```bash
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_KEEP_ALIVE=30m
```

Verify with `ollama ps` — you should see multiple models in the `LOADED`
column once two ablation calls have run.

> Note: [src/agents/llm_agent.py](../src/agents/llm_agent.py) also passes
> `keep_alive: "30m"` per request, but the `OLLAMA_MAX_LOADED_MODELS` cap
> can only be raised at server start.

4. **Install Python dependencies**
   ```bash
   pip install requests
   ```

## Usage

The MTG agents will automatically use Ollama when:
- It's running on localhost:11434
- A model is available

### Run Meta Games with LLM Agents

```bash
python examples/meta_game.py
```

The output will show:
- **[OLLAMA ENABLED - gemma4:e2b]**: LLM is active
- **[LLM] actions**: Decisions made by the language model
- **[Heuristic] actions**: Fallback decisions when LLM is unavailable

### Example Output

```
STANDARD META TOURNAMENT - March 2026
LLM-POWERED AGENTS (Ollama)
======================================================================

Agent Mode: [OLLAMA ENABLED - gemma4:e2b]
Ollama URL: http://localhost:11434

======================================================================
MATCHUP: UR Aggro vs Izzet Control
======================================================================
Playing 2 games...

  Game 1: [LLM] play_card - Both decks have efficient early plays, 
           playing a creature to establish board presence is good.
  
  Game 2: DRAW (Turn 20)
```

## Troubleshooting

### Ollama not available
- Check if `ollama serve` is running
- Verify models are installed with `ollama list`
- Check port 11434 is open

### Slow responses
- `gemma4:2b` is already the smallest variant; ensure sufficient RAM (4GB+)
- Increase timeout: Edit `llm_orchestration.py` OllamaConnector timeout

### Not enough memory
- Pull a quantized model: `ollama pull mistral:7b-instruct-q4_K_M`

## Configuration

Edit parameters in `src/engine/llm_orchestration.py`:

```python
from src.engine.llm_orchestration import OllamaConnector

# Custom configuration
llm = OllamaConnector(
    model="neural-chat",           # Choose model
    base_url="http://localhost:11434",  # Ollama server
    timeout=60                      # Request timeout in seconds
)
```

## How Agents Use LLM

1. **Main Phase Decisions**: "What should I play?"
2. **Combat Phase**: "Which creatures should attack?"
3. **Reasoning Chain**: Decisions are recorded with reasoning to the knowledge graph

The LLM evaluates:
- Current hand with mana costs
- Board state (your creatures vs opponent's)
- Life totals
- Strategy (aggressive, control, combo, balanced)

## Performance Notes

- First request takes longer (model loading)
- Subsequent requests are faster
- GPU support improves performance significantly
- Consider running Ollama with CUDA or Metal for acceleration


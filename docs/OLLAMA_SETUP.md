# Ollama Setup for MTG Agents

This project uses Ollama for LLM-powered agent decision-making in Magic: The Gathering games.

## Installation

1. **Install Ollama** (version 0.3 or later recommended for all models)
   - Download from: https://ollama.ai
   - Available for Mac, Linux, and Windows

2. **Pull a model**
   ```bash
   ollama pull llama3.2:1b
   ollama pull qwen2.5-coder:1.5b
   ```
   
   Other models:
   - `ollama pull mistral` - Alternative open-source model
   - `ollama pull lfm2.5:8b` - Liquid Foundation Model 2.5 (strong reasoning, ~6 GB VRAM)
   - `ollama pull phi` - Lightweight option

Gemma is excluded from the current comparison: its Commander screen produced
timeouts/HTTP 500 responses and no accepted model choices. That is a serving
qualification failure, not evidence that Gemma is intrinsically weak.

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

### Local vLLM Docker startup

The phase-rs `--picker vllm` path uses a local OpenAI-compatible
`/v1/chat/completions` endpoint with strict JSON output and no random fallback.
Strict Ollama and vLLM requests enumerate eligible original action indices
in a JSON schema and validate the response; unavailable services and invalid
answers fail the game explicitly. The ablation manifest omits `--vllm-api-key`,
but command-line arguments may still be visible to local processes.
For a 12 GiB GPU, do not leave vLLM at a memory target above the free VRAM.
For example:

```powershell
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest `
  --model Qwen/Qwen3-0.6B `
  --gpu-memory-utilization 0.85 `
  --max-model-len 8192
```

The observed startup failure requested 11.04 GiB while only 10.98 GiB was
free. It occurred before model execution. Lowering GPU utilization fixes that
specific check; reducing the default 40,960-token context also avoids reserving
an unnecessarily large KV cache for the MTG decision prompt.

Liquid d1 is not served through this generative endpoint. Its released
`system_one` classifier head runs through the optional `liquid_d1` dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[liquid_d1]"
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
  --picker liquid_d1 --liquid-d1-model LiquidAI/d1-omni-600M
```

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

# Ollama Setup for MTG Agents

This project uses Ollama for LLM-powered agent decision-making in Magic: The Gathering games.

## Installation

1. **Install Ollama**
   - Download from: https://ollama.ai
   - Available for Mac, Linux, and Windows

2. **Pull a model**
   ```bash
   ollama pull mistral
   ```
   
   Other recommended models:
   - `ollama pull neural-chat` - Optimized for conversations
   - `ollama pull llama2` - General purpose
   - `ollama pull phi` - Lightweight option

3. **Start Ollama server**
   ```bash
   ollama serve
   ```
   
   This runs the API on `http://localhost:11434` by default

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
- **[OLLAMA ENABLED - mistral]**: LLM is active
- **[LLM] actions**: Decisions made by the language model
- **[Heuristic] actions**: Fallback decisions when LLM is unavailable

### Example Output

```
STANDARD META TOURNAMENT - March 2026
LLM-POWERED AGENTS (Ollama)
======================================================================

Agent Mode: [OLLAMA ENABLED - mistral]
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
- Try a smaller model: `ollama pull phi`
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


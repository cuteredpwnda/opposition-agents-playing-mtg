# Development & Deployment Guide

> **Status (April 2026):** Engine and orchestrator are stable for two-player
> games. Run `python examples/demo_game_simple.py` for a complete match
> (now defaults to up to 40 turns with deterministic timeout tie-breakers).
> Mulligan behaviour is configurable via `GameSimulator.setup_game(
> mulligan_enabled=..., max_mulligans=...)` and `GameConfig.mulligan_enabled`
> / `GameConfig.max_mulligans`.

## Local Development Setup

### Prerequisites

- Python 3.11+
- Neo4j Community Edition 5.x (for knowledge graph)
- Docker & Docker Compose (recommended)

### Initial Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For ML features (optional):
```bash
pip install -r requirements-ml.txt
```

4. Start Neo4j:
```bash
docker-compose up -d neo4j
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Server Deployment

For production LLM agent execution, deploy as a containerized service:

### Option 1: Docker Deployment

```bash
docker build -t opposition-agents:latest .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e NEO4J_URI=bolt://neo4j:7687 \
           -p 8000:8000 \
           opposition-agents:latest
```

### Option 2: Docker Compose (Full Stack)

```bash
docker-compose -f docker-compose.prod.yml up
```

This starts:
- Neo4j 5.x with n10s plugin
- FastAPI server on port 8000
- Redis for caching (optional)

### Option 3: Kubernetes Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## API Server for Remote Execution

### Starting the Server

The API server allows remote LLM agent execution:

```bash
# Install server dependencies
pip install fastapi uvicorn

# Start server (development)
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Start server (production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.server:app \
         --bind 0.0.0.0:8000
```

### API Endpoints

```
POST /api/v1/games/start
   - Create a new game
   - Request: {"format": "commander", "max_turns": 100}
   - Response: {"game_id": "...", "state": {...}}

POST /api/v1/games/{game_id}/decide
   - Get agent decision for a game state
   - Request: {"player_id": "...", "legal_actions": [...]}
   - Response: {"action": {...}, "reasoning": "..."}

GET /api/v1/games/{game_id}
   - Get current game state

POST /api/v1/games/{game_id}/step
   - Execute one turn
   - Request: {"action": {...}}
   - Response: {"new_state": {...}, "events": [...]}

POST /api/v1/judge/rule-on
   - Get judge decision on a rule question
   - Request: {"situation": "Player casts...", "game_state": {...}}
   - Response: {"ruling": "...", "confidence": 0.95, "sources": [...]}

POST /api/v1/knowledge/detect-combos
   - Detect playable combos
   - Request: {"hand": [...], "battlefield": [...]}
   - Response: {"combos": [...], "near_combos": [...]}
```

### Example Client Usage

```python
import httpx
import asyncio

async def run_remote_game():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Start game
        resp = await client.post("/api/v1/games/start", 
            json={"format": "commander", "max_turns": 50})
        game_id = resp.json()["game_id"]
        
        # Play turns
        for turn in range(50):
            # Get legal actions
            resp = await client.get(f"/api/v1/games/{game_id}")
            state = resp.json()
            
            # Ask agent for decision
            resp = await client.post(f"/api/v1/games/{game_id}/decide",
                json={"player_id": "player_1", 
                      "legal_actions": state["legal_actions"]})
            action = resp.json()["action"]
            
            # Execute action
            resp = await client.post(f"/api/v1/games/{game_id}/step",
                json={"action": action})
            new_state = resp.json()["new_state"]
            
            if new_state.get("game_over"):
                print(f"Game over! Winner: {new_state['winner']}")
                break

asyncio.run(run_remote_game())
```

## LLM Integration

### OpenAI Models

Set in `.env`:
```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
```

### Other LLM Providers

Modify `src/config.py` to use alternative providers:

```python
from langchain.chat_models import ChatAnthropic
# or ChatGooglePalm, ChatCohere, etc.

llm = ChatAnthropic(model="claude-2", temperature=0.7)
```

### Prompts & Reasoning

Agent decision-making uses a system prompt in `src/agents/llm_agent.py`:
- Context: Current game state, legal actions, hand contents
- Strategy: Win condition, opponent modeling, combo analysis
- Output: Action index selection with reasoning

For more detailed reasoning, enable extended output:
```env
ENABLE_REASONING_TRACE=true
```

## Performance Optimization

### Caching

Card data is cached in SQLite:
```bash
python -m src.integrations.card_cache --rebuild
```

### Database Indexes

Neo4j indexes for fast queries:
```cypher
CREATE INDEX idx_card_name FOR (c:Card) ON (c.name);
CREATE INDEX idx_combo_pieces FOR (co:Combo) ON (co.pieces);
CREATE FULLTEXT INDEX idx_oracle_text FOR (c:Card) ON (c.oracle_text);
```

### Vectorization

Pre-compute embeddings:
```bash
python -m src.training.build_embeddings --model sentence-transformers/all-MiniLM-L6-v2
```

## Monitoring

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Game started")
```

### Metrics

Via Prometheus (optional):
```bash
pip install prometheus-client
```

Metrics endpoint: `http://localhost:8000/metrics`

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_engine.py::test_legal_actions
```

## Debugging

### Enable verbose logging
```env
LOG_LEVEL=DEBUG
```

### Profile a game
```bash
python -m cProfile -s cumulative -m src.orchestrator.game_runner > profile.txt
```

### Remote debugger
```bash
pip install remote-pdb
# In code: import remote_pdb; remote_pdb.set_trace()
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Lint (black, isort, flake8)
- Type check (mypy)
- Test (pytest)
- Build Docker image
- Deploy to staging

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: black --check src/
      - run: pytest tests/
```

## Performance Benchmarks

Expected performance on standard hardware:

| Task | Time | Notes |
|------|------|-------|
| Legal action generation | 5-10ms | 50-200 actions |
| Single game (40 turns) | 5-30s | LLM+KB queries |
| SBA checking | 1-2ms | All state checks |
| Combo detection | 10-50ms | Cypher query |
| Archetype inference | 20-100ms | Graph traversal |

## Troubleshooting

### Neo4j Connection Errors
```bash
# Check Neo4j is running
docker ps | grep neo4j

# View Neo4j logs
docker logs neo4j

# Rebuild container
docker-compose down neo4j && docker-compose up neo4j
```

### LLM API Errors
- Check OPENAI_API_KEY is set
- Verify API quota hasn't exceeded
- Check rate limits and retry logic

### Memory Issues
- Reduce batch size (TRAINING_BATCH_SIZE)
- Use smaller models (gpt-3.5-turbo instead of gpt-4)
- Limit knowledge graph queries (performance limits in kg.py)

---

For questions or issues, see the main README.md or open an issue on GitHub.

# Multi-stage build for MTG Agent training
# Stage 1: Base Python + dependencies
# Stage 2: Full training image with Neo4j client + Ollama-ready
#
# Usage:
#   docker build -t mtg-agents .
#   docker compose -f docker-compose.yml -f docker-compose.remote.yml up -d

FROM python:3.12-slim AS base

WORKDIR /app

# System deps for torch and neo4j driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer)
COPY requirements.txt requirements-ml.txt requirements-ontology.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir -r requirements-ontology.txt || true

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY main.py ./
COPY neo4j/ neo4j/

# Default: run the deploy pipeline
ENTRYPOINT ["python", "scripts/deploy.py"]
CMD ["--all"]

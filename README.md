# Scientis — Agentic Scientific Discovery System

A stateful, multimodal, retrieval-grounded agent platform for cross-paper
evidence synthesis, mechanism induction, and hypothesis generation.

## Architecture

```
                  PDFs / images / supplements
                           │
                   ┌───────▼────────┐
                   │  INGESTION      │  S3 + metadata
                   └───────┬────────┘
                           │
                   ┌───────▼────────┐
                   │  UNDERSTANDING  │  Sections, figures, claims
                   └───────┬────────┘
                           │
                   ┌───────▼────────┐
                   │  EVIDENCE       │  Neo4j graph + vector index
                   └───────┬────────┘
                           │
                   ┌───────▼────────┐
                   │  REASONING      │  LangGraph agent
                   └───────┬────────┘
                           │
                   ┌───────▼────────┐
                   │  PRESENTATION   │  Stories, visuals, exports
                   └────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Orchestration | LangGraph |
| Graph | Neo4j (property graph + vector index) |
| Relational | PostgreSQL (metadata, jobs, audit) |
| Object Store | S3-compatible (MinIO for local) |
| Queue | Redis |
| Model Serving | vLLM (local) + GPT-4o mini + Gemini Flash |

## Quickstart

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Run API
uvicorn scientis.main:app --reload --port 8080
```

### Ingest a paper

```bash
curl -X POST http://localhost:8080/v1/papers \
  -F "file=@paper.pdf" \
  -F "metadata=$(cat meta.json)"
```

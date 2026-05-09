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

## Agent Workflow

```
question → expand_query → retrieve_context → compile_evidence →
induce_mechanism → check_contradictions → human_review → publish
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Orchestration | LangGraph (stateful, streaming, human-in-the-loop) |
| Graph | Neo4j (property graph + vector index) |
| Relational | PostgreSQL (metadata, jobs, audit) |
| Object Store | S3-compatible (MinIO for local) |
| Queue | Redis |
| Model Serving | vLLM (local) + GPT-4o mini + Gemini Flash |

## API

| Endpoint | Description |
|---|---|
| `GET /v1/health` | Health check |
| `POST /v1/papers` | Upload PDF for ingestion |
| `GET /v1/papers/{id}` | Get paper status |
| `GET /v1/papers` | List all papers |
| `POST /v1/questions` | Submit a discovery question |
| `GET /v1/results/{session_id}` | Get discovery results |
| `POST /v1/reviews` | Submit human review decisions |
| `POST /v1/hypotheses/generate` | Generate hypotheses from papers |
| `POST /v1/exports/report` | Export markdown report |
| `POST /v1/exports/slides` | Export slides |

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

### Ask a discovery question

```bash
curl -X POST http://localhost:8080/v1/questions \
  -H "Content-Type: application/json" \
  -d '{"question": "What shared mechanisms explain AD, PD, ALS, and FTD?"}'
```

## How it works

1. **Ingestion**: PDF uploaded → stored in S3, metadata in Postgres
2. **Understanding**: pymupdf extracts text/figures → LLM extracts claims with evidence spans
3. **Evidence**: Claims and entities stored in Neo4j graph with support/contradiction edges
4. **Reasoning**: LangGraph agent retrieves, compares, induces mechanisms, checks contradictions
5. **Review**: Human-in-the-loop gate for novel/high-impact hypotheses
6. **Export**: Reports and slide decks generated from evidence graph

## Project Structure

```
src/scientis/
├── main.py              FastAPI app factory
├── config.py             pydantic-settings
├── api/                  REST endpoints
│   ├── papers.py         Paper CRUD
│   ├── questions.py      Discovery questions + review
│   └── exports.py        Report/slide generation
├── models/               Pydantic domain models
├── llm/                  LLM client abstraction + structured output schemas
├── services/
│   ├── ingestion.py      PDF → object store
│   ├── parsing.py        pymupdf text/figure extraction
│   ├── understanding.py  LLM claim extraction
│   ├── graph_service.py  Neo4j CRUD + subgraph queries
│   ├── retrieval.py      BM25 + vector + graph hybrid retrieval
│   └── events.py         Internal event bus
├── agent/
│   ├── graph.py          LangGraph workflow definition
│   ├── tools.py          Agent tools (expand, retrieve, compile, induce, contradict)
│   ├── state.py          Agent state schema
│   └── runner.py         Workflow runner with streaming + checkpoints
├── graph/                Neo4j connection + schema
└── storage/              S3-compatible object store
```

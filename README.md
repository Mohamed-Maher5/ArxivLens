# ArxivLens

ArxivLens is a multimodal RAG app for ArXiv papers. It lets you:

- search papers from ArXiv
- ingest a paper into a per-paper local Qdrant collection
- generate local embeddings with `BAAI/bge-m3`
- use Gemma for chat, reranking, contextualization, and vision descriptions
- talk to the indexed paper through a FastAPI backend and Streamlit frontend

## Current Architecture

1. `app/ingestion`
   Fetches the paper, downloads the PDF, parses pages/images/tables, and adds figure descriptions.
2. `app/indexing`
   Chunks parsed content, creates dense+sparse vectors, and stores them in Qdrant.
3. `app/retrieval`
   Runs hybrid retrieval and LLM reranking against a single local Qdrant paper collection.
4. `app/generation`
   Routes the request, contextualizes it, retrieves evidence, and produces the final answer.
5. `app/api`
   Exposes `/health`, `/papers`, `/ingest`, and `/chat`.
6. `streamlit_app.py`
   Acts as a thin HTTP client for the FastAPI backend.

## Quick Start

```bash
cp .env.example .env
```

Run Qdrant locally:

```bash
docker run -d --name arxivlens-qdrant -p 6333:6333 qdrant/qdrant:latest
```

Install dependencies and dev tools:

```bash
uv sync --all-groups
```

Start the backend:

```bash
uv run python run_api.py
```

Start the frontend:

```bash
uv run streamlit run streamlit_app.py
```

Optional CLI demo:

```bash
uv run python cli_demo.py
```

## Environment

Minimal local setup:

```env
GOOGLE_API_KEY=
QDRANT_URL=http://localhost:6333
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=arxiv-lens
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Notes:

- `GOOGLE_API_KEY` is needed for live Gemma text and vision calls.
- This project is currently designed around local Qdrant, not Qdrant Cloud.
- The project assumes an open local Qdrant instance on `localhost:6333`.
- The local embedding flow does not use Hugging Face auth tokens.

## Tests

Run the fast automated checks with:

```bash
uv run --group dev pytest -q
```

The current test suite is intentionally fast and focuses on:

- pipeline orchestration behavior
- vector store collection/payload behavior
- API route smoke coverage

## Project Layout

```text
app/
  api/
  core/
  generation/
  indexing/
  ingestion/
  interfaces/
  llm/
  models/
  orchestration/
  retrieval/
tests/
  integration/
  unit/
run_api.py
streamlit_app.py
cli_demo.py
```

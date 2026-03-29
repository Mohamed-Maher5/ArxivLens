# ArxivLens 🔬

**Multimodal RAG for ArXiv papers** — reads text, figures & charts to answer research questions with full citations.

ArxivLens lets you search ArXiv, ingest a paper (PDF + figures), index it into a vector database, and then have a full conversational Q&A session grounded in that paper's content. It combines hybrid dense+sparse retrieval, LLM-based reranking, and multimodal figure understanding into a single pipeline.

---

## Table of Contents

- [Architecture & Flow](#architecture--flow)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [LangSmith Tracing](#langsmith-tracing)
- [Project Structure](#project-structure)

---

## Architecture & Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│               INGESTION PIPELINE                │
│                                                 │
│  ArxivFetcher → download PDF from ArXiv         │
│  PDFParser    → extract text / tables / images  │
│  VisionProcessor → Groq Vision → describe figs  │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│               INDEXING PIPELINE                 │
│                                                 │
│  Chunker   → abstract / content / figure /      │
│              table / references chunks          │
│  Embedder  → BGE-M3 via HuggingFace API         │
│              dense vector + sparse (BM25-style) │
│  VectorStore → Qdrant Cloud (per-paper coll.)   │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│               GENERATION PIPELINE               │
│                                                 │
│  Intent Classifier (Groq)                       │
│      │                                          │
│      ├── CHAT → Qwen3-8B (HuggingFace)          │
│      │                                          │
│      └── TASK                                   │
│           │                                     │
│           ├── Contextualize query (Ollama/phi3) │
│           ├── HybridRetriever → RRF fusion      │
│           ├── Reranker (Groq) > 6.0 threshold   │
│           │                                     │
│           ├── chunks found →                    │
│           │     Top-3 answer (Qwen3-8B / HF)   │
│           └── no chunks →                       │
│                 General knowledge (Qwen3-8B)    │
└─────────────────────────────────────────────────┘
```
<br>
<img width="769" height="937" alt="Screenshot from 2026-03-29 21-04-01" src="https://github.com/user-attachments/assets/899bca0b-aa99-4d57-a96a-09ff76166dfb" />
<br>
---

## Tech Stack

| Role | Model / Service |
|---|---|
| PDF parsing & figure extraction | PyMuPDF |
| Figure understanding (vision) | Groq — `llama-4-scout-17b-16e-instruct` |
| Intent classification | Groq — `llama-3.1-8b-instant` |
| Reranking | Groq — `llama-3.1-8b-instant` |
| Query contextualization | Ollama — `phi3` (local) |
| History summarization | Ollama — `phi3` (local) |
| Dense + sparse embeddings | HuggingFace API — `BAAI/bge-m3` |
| Answer generation (Q&A + chat) | HuggingFace API — `Qwen/Qwen3-8B` |
| Vector database | Qdrant Cloud |
| Observability (optional) | LangSmith |
| UI | Streamlit |

---

## Prerequisites

Before you start, make sure you have the following installed and ready:

**Runtime:**
- Python 3.11 (recommended; 3.11–3.13 supported per `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [Ollama](https://ollama.com/) — for running `phi3` locally

**Accounts & API keys (all free tiers work):**
- [Groq](https://console.groq.com/) — vision + classification + reranking
- [HuggingFace](https://huggingface.co/settings/tokens) — embeddings + generation
- [Qdrant Cloud](https://cloud.qdrant.io/) — vector database (free cluster available)
- [LangSmith](https://smith.langchain.com/) — pipeline observability and tracing

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/arxivlens.git
cd arxivlens
```

### 2. Install dependencies with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync
```

### 3. Pull the Ollama model

Make sure Ollama is running, then pull phi3:

```bash
ollama pull phi3
ollama serve   # keep this running in a background terminal
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Then fill in your keys — see the section below
```

---

## Environment Variables

Open `.env` and fill in the following:

```env
# ── Groq ────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key

# ── HuggingFace ─────────────────────────────────
HUGGINGFACE_API_KEY=your_hf_token

# ── Qdrant Cloud ────────────────────────────────
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# ── Ollama (local) ──────────────────────────────
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3

# ── Models ──────────────────────────────────────
BGE_MODEL_NAME=BAAI/bge-m3
HF_MODEL=Qwen/Qwen3-8B
GROQ_CLASSIFIER_MODEL=llama-3.1-8b-instant
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# ── Retrieval settings ──────────────────────────
TOP_K_RETRIEVAL=10
TOP_K_RERANK=3
RERANK_SCORE_THRESHOLD=6.0
CHUNK_SIZE=256
CHUNK_OVERLAP=50
MAX_HISTORY=6

# ── LangSmith ───────────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=arxiv-lens
```

---

Search result limit — When ArXiv returns papers for a query, the app fetches a maximum of 5 by default. You can increase or decrease this by changing the max_results default in app/ingestion/arxiv_fetcher.py:

python
def search_papers(self, query: str, max_results: int = 5) -> list[Paper]:

Change 5 to however many results you want shown. Keep in mind that a higher number means more papers to scroll through before selecting one, and ArXiv may rate-limit aggressive requests.

## Running the Project

There are two ways to run ArxivLens:

### Option A — Streamlit UI (recommended)

```bash
uv run streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser. The UI lets you search ArXiv, pick a paper, trigger ingestion, and chat — all from one interface.

### Option B — CLI (end-to-end test / development)

```bash
uv run python test_project.py
```

This walks you through the full pipeline interactively in your terminal:

1. Enter a search query → up to 5 papers are shown
2. Select a paper by number
3. Ingestion + indexing runs automatically
4. Chat with the paper in a REPL loop — type `exit` to quit

**Example session:**

```
Enter your search query: attention mechanism

[1] Attention Is All You Need
    ID: 1706.03762v5
    Authors: Ashish Vaswani et al.
    Published: 2017-06-12

Select paper [1-5]: 1

⬇️  Fetching and parsing paper 1706.03762v5...
   ✅ Parsed: 15 pages
✂️  Chunking paper...
   ✅ Created 87 chunks
🔢 Embedding chunks...
   ✅ Embedded 87 chunks
💾 Storing in Qdrant...
   ✅ Indexed to collection: paper_1706_03762v5

You: What is multi-head attention?
ArxivLens: Multi-head attention allows the model to jointly attend to
information from different representation subspaces...
[Sources: 3 chunks]
  - Attention Is All You Need, p.4
  - Attention Is All You Need, p.5
```

---

## LangSmith Tracing

ArxivLens uses LangSmith for full pipeline observability. It is **required** — the pipeline is instrumented with `@traceable` decorators and tracing must be active for the app to run correctly.

**Setup:**

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Go to Settings → API Keys and generate a key
3. Make sure your `.env` has all three LangSmith variables set:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=arxivlens
```

Traces will appear in your LangSmith dashboard under the `arxivlens` project. Each run captures the full chain as named spans:

- `arxivlens_pipeline` — full chain run
- `classify_intent` — CHAT vs TASK routing
- `contextualize_query` — query rewriting via phi3
- `retrieve_chunks` — hybrid RRF search
- `rerank_chunks` — Groq answerability scoring

---

## Project Structure

```
arxivlens/
├── streamlit_app.py          # Streamlit UI entry point
├── test_project.py           # CLI end-to-end test
├── pyproject.toml
├── .env.example
│
├── app/
│   ├── core/
│   │   ├── settings.py       # Pydantic settings (reads .env)
│   │   ├── logger.py         # Loguru setup (stdout + file)
│   │   └── exceptions.py     # Custom exception hierarchy
│   │
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py  # ArXiv search + PDF download
│   │   ├── pdf_parser.py     # PyMuPDF text/table/image extraction
│   │   └── vision_processor.py # Groq vision → figure descriptions
│   │
│   ├── indexing/
│   │   ├── chunker.py        # Chunk creation by type
│   │   ├── embedder.py       # BGE-M3 dense + sparse vectors
│   │   └── vector_store.py   # Qdrant client, per-paper collections
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Dense + sparse RRF fusion
│   │   ├── reranker.py          # Groq answerability scoring
│   │   └── cache.py             # In-memory query cache
│   │
│   ├── generation/
│   │   ├── pipeline.py       # Main orchestrator (intent → retrieve → generate)
│   │   ├── adaptive_rag.py   # LLM call wrappers (Groq + HF)
│   │   └── prompts.py        # All LangChain prompt templates
│   │
│   └── models/
│       └── schemas.py        # Pydantic models (Paper, Chunk, QueryResult, Message)
│
├── data/
│   ├── raw/                  # Downloaded PDFs
│   └── processed/            # Parsed paper JSON files
│
└── logs/
    └── arxivlens.log         # Debug log (10 MB rotation, 7 day retention)
```

---

## Notes

- Each paper gets its own Qdrant collection (`paper_{arxiv_id}`), so multiple papers can be indexed independently without collision.
- History is managed automatically: up to 6 messages are kept in full; older messages are summarized by phi3 into 2–3 sentences to keep context without overflowing the prompt.
- The reranker uses an **answerability** score (not similarity) — a chunk must score above 6.0/10 to be used in generation. This significantly reduces hallucination from tangentially related chunks.




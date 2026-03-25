import json
import requests
from pathlib import Path
from app.core.logger import logger
from app.core.exceptions import PipelineError
from app.core.settings import settings
from app.models.schemas import QueryResult, Chunk, Message
from app.generation.adaptive_rag import AdaptiveRAG
from app.generation import prompts
from app.retrieval import retrieve
from app.retrieval.reranker import Reranker

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available, tracing disabled")


DATA_PROCESSED = Path("data/processed")
ABSTRACT_MAX_CHARS = 200


class Pipeline:

    def __init__(self):
        self.adaptive_rag = AdaptiveRAG()
        self.reranker = Reranker()
        logger.info("Pipeline initialized")

    def run(self, question: str, history: list[Message] = None) -> QueryResult:
        if LANGSMITH_AVAILABLE and settings.langchain_tracing_v2:
            return self._run_traced(question, history)
        return self._run_pipeline(question, history)

    @traceable(run_type="chain", name="arxivlens_pipeline")
    def _run_traced(self, question: str, history: list[Message] = None) -> QueryResult:
        return self._run_pipeline(question, history)

    def _run_pipeline(self, question: str, history: list[Message] = None) -> QueryResult:
        logger.info(f"Pipeline running for: {question[:50]}...")
        try:
            if history is None:
                history = []

            managed_history = self._manage_history(history)
            intent = self._classify_intent(question)

            if intent == "chat":
                return self._handle_chat(question)

            contextualized = self._contextualize(question, managed_history)
            chunks = self._retrieve_chunks(contextualized)

            if not chunks:
                metadata = self._load_paper_metadata()
                return self._no_papers_response(question, managed_history, metadata)

            filtered_chunks = self._filter_chunks(chunks)

            if not filtered_chunks:
                # Chunks exist but all scored below score_threshold (0.25)
                metadata = self._load_paper_metadata(chunks)
                return self._no_papers_response(question, managed_history, metadata)

            reranked = self._rerank_chunks(contextualized, filtered_chunks)

            if not reranked:
                # Chunks passed vector threshold but all failed reranker (< 6.0)
                metadata = self._load_paper_metadata(filtered_chunks)
                return self._no_papers_response(question, managed_history, metadata)

            return self._generate_paper_answer(
                question, contextualized, reranked, managed_history
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline failed: {e}")

    # ─────────────────────────────────────────

    @traceable(run_type="llm", name="classify_intent")
    def _classify_intent(self, question: str) -> str:
        return self.adaptive_rag.classify_intent(question)

    @traceable(run_type="llm", name="manage_history")
    def _manage_history(self, history: list[Message]) -> list[Message]:
        if len(history) <= settings.max_history:
            return history
        
        old_messages = history[:-settings.max_history]
        recent_messages = history[-settings.max_history:]
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])

        try:
            prompt = prompts.HISTORY_SUMMARY_PROMPT.format(text=old_text)
            summary = self._ollama(prompt, max_tokens=150)

            if summary:
                return [Message(
                    role="system",
                    content=f"Earlier conversation summary: {summary}"
                )] + recent_messages
            return recent_messages
        except Exception:
            return recent_messages

    @traceable(run_type="llm", name="contextualize_query")
    def _contextualize(self, question: str, history: list[Message]) -> str:
        if not history:
            return question

        try:
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
            prompt = prompts.CONTEXTUALIZATION_PROMPT.format(
                history=history_text,
                question=question
            )
            result = self._ollama(prompt, max_tokens=100)
            return result if result and len(result) > 10 else question
        except Exception:
            return question

    @traceable(run_type="retriever", name="retrieve_chunks")
    def _retrieve_chunks(self, contextualized: str) -> list:
        return retrieve(contextualized)

    @traceable(run_type="chain", name="filter_chunks")
    def _filter_chunks(self, chunks: list) -> list:
        return [c for c in chunks if c.get('score', 0.0) >= settings.score_threshold]

    @traceable(run_type="llm", name="rerank_chunks")
    def _rerank_chunks(self, contextualized: str, chunks: list) -> list:
        return self.reranker.rerank(contextualized, chunks)

    @traceable(run_type="llm", name="generate_paper_answer")
    def _generate_paper_answer(self, question, contextualized, chunks, history):
        context = self._build_context(chunks)

        history_text = ""
        if history:
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history])

        result = self.adaptive_rag.generate_from_paper_with_history(
            contextualized, context, history_text
        )

        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=self._chunks_to_schema(chunks),
            confidence=result["confidence"],
            contextualized_query=contextualized
        )

    # ─────────────────────────────────────────

    def _handle_chat(self, question: str) -> QueryResult:
        answer = self.adaptive_rag.generate_chat(question)
        return QueryResult(
            question=question,
            answer=answer,
            sources=[],
            confidence="HIGH",
            contextualized_query=question
        )

    def _load_paper_metadata(self, chunks: list[dict] = None) -> str:
        """
        Build a metadata string (title, authors, truncated abstract) for the
        fallback prompt.

        Strategy:
        1. If chunks were retrieved, extract paper_id from the first chunk and
           load the corresponding processed JSON file for the full abstract.
        2. If no chunks (nothing retrieved at all), scan data/processed/ for
           the most recently modified file as a best-effort heuristic.
        3. If no processed file found, return an empty string so the fallback
           prompt still works gracefully.
        """
        paper_id = None

        # Try to get paper_id from chunks payload
        if chunks:
            paper_id = chunks[0].get("paper_id")

        if paper_id:
            processed_path = DATA_PROCESSED / f"{paper_id}.json"
            if processed_path.exists():
                return self._format_metadata_from_file(processed_path)
            logger.warning(f"[METADATA] Processed file not found for paper_id={paper_id}")

        # Fallback: pick most recently modified processed file
        processed_files = sorted(
            DATA_PROCESSED.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if processed_files:
            logger.info(f"[METADATA] Using most recent processed file: {processed_files[0].name}")
            return self._format_metadata_from_file(processed_files[0])

        logger.warning("[METADATA] No processed files found — metadata will be empty")
        return ""

    def _format_metadata_from_file(self, path: Path) -> str:
        """Read a processed JSON and return a formatted metadata string."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "Unknown title")
            authors = data.get("authors", [])
            abstract = data.get("abstract", "")
            published = data.get("published", "")

            # Truncate abstract to ABSTRACT_MAX_CHARS
            abstract_snippet = abstract[:ABSTRACT_MAX_CHARS]
            if len(abstract) > ABSTRACT_MAX_CHARS:
                abstract_snippet += "..."

            authors_str = ", ".join(authors) if authors else "Unknown authors"

            metadata_lines = [
                f"Title: {title}",
                f"Authors: {authors_str}",
            ]
            if published:
                metadata_lines.append(f"Published: {published}")
            metadata_lines.append(f"Abstract (excerpt): {abstract_snippet}")

            return "\n".join(metadata_lines)

        except Exception as e:
            logger.warning(f"[METADATA] Failed to read {path}: {e}")
            return ""

    def _no_papers_response(
        self,
        question: str,
        history: list[Message],
        metadata: str = "",
    ) -> QueryResult:
        """Answer from model knowledge when no paper chunks pass thresholds."""
        history_text = (
            "\n".join([f"{m.role}: {m.content}" for m in history])
            if history else ""
        )

        answer = self.adaptive_rag._hf_with_prompt(
            prompts.MODEL_KNOWLEDGE_FALLBACK_PROMPT,
            {
                "metadata": metadata if metadata else "No paper metadata available.",
                "history": history_text,
                "question": question,
            },
            max_tokens=512
        )

        return QueryResult(
            question=question,
            answer=answer,
            sources=[],
            confidence="MEDIUM",
            contextualized_query=question
        )

    # ─────────────────────────────────────────

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = requests.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return ""
        except Exception:
            return ""

    def _build_context(self, chunks: list[dict]) -> str:
        return "\n\n".join([
            f"[{i}] {chunk.get('content', '')}"
            for i, chunk in enumerate(chunks, 1)
        ])

    def _chunks_to_schema(self, chunks: list[dict]) -> list[Chunk]:
        result = []
        for chunk in chunks:
            try:
                result.append(Chunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    paper_id=chunk.get("paper_id", ""),
                    paper_title=chunk.get("paper_title", ""),
                    authors=chunk.get("authors", []),
                    content=chunk.get("content", ""),
                    chunk_type=chunk.get("chunk_type", "content"),
                    page_number=chunk.get("page_number"),
                ))
            except Exception:
                continue
        return result
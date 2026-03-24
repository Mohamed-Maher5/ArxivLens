import requests
import os
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


class Pipeline:

    def __init__(self):
        self.adaptive_rag = AdaptiveRAG()
        self.reranker = Reranker()
        logger.info("Pipeline initialized")

    def run(self, question: str, history: list[Message] = None) -> QueryResult:
        """Main pipeline entry point with optional LangSmith tracing."""
        # Wrap with LangSmith trace if available
        if LANGSMITH_AVAILABLE and os.getenv("LANGCHAIN_TRACING_V2") == "true":
            return self._run_traced(question, history)
        return self._run_pipeline(question, history)

    @traceable(run_type="chain", name="arxivlens_pipeline")
    def _run_traced(self, question: str, history: list[Message] = None) -> QueryResult:
        """Traced version of pipeline."""
        return self._run_pipeline(question, history)

    def _run_pipeline(self, question: str, history: list[Message] = None) -> QueryResult:
        """Core pipeline logic."""
        logger.info(f"Pipeline running for: {question[:50]}...")
        try:
            if history is None:
                history = []

            # ── Step 1: Manage history ─────────────────────────────────────
            managed_history = self._manage_history(history)

            # ── Step 2: Classify intent ──────────────────────────────────
            intent = self._classify_intent(question)

            if intent == "chat":
                return self._handle_chat(question)

            # ── TASK path ─────────────────────────────────────────────────
            contextualized = self._contextualize(question, managed_history)
            chunks = self._retrieve_chunks(contextualized)
            
            if not chunks:
                return self._no_papers_response(question, contextualized)

            # Filter by threshold >= 0.25
            filtered_chunks = self._filter_chunks(chunks)
            
            if not filtered_chunks:
                return self._web_search_fallback(contextualized, chunks, "retrieval_low_score")

            # Rerank with threshold >= 6.0
            reranked = self._rerank_chunks(contextualized, filtered_chunks)
            
            if not reranked:
                return self._web_search_fallback(contextualized, filtered_chunks, "rerank_low_score")

            # Generate paper-based answer
            return self._generate_paper_answer(
                question, contextualized, reranked, managed_history
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline failed: {e}")

    # ── Individual Steps (for tracing granularity) ─────────────────────────

    @traceable(run_type="llm", name="classify_intent")
    def _classify_intent(self, question: str) -> str:
        """Classify user intent."""
        return self.adaptive_rag.classify_intent(question)

    @traceable(run_type="llm", name="manage_history")
    def _manage_history(self, history: list[Message]) -> list[Message]:
        """Summarize and manage conversation history."""
        if len(history) <= settings.max_history:
            return history
        
        old_messages = history[:-settings.max_history]
        recent_messages = history[-settings.max_history:]
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
        
        logger.info(f"[HISTORY] Summarizing {len(old_messages)} old messages")
        try:
            prompt = prompts.HISTORY_SUMMARY_PROMPT.format(text=old_text)
            summary = self._ollama(prompt, max_tokens=150)
            
            if summary:
                return [Message(
                    role="system",
                    content=f"Earlier conversation summary: {summary}"
                )] + recent_messages
            return recent_messages
        except Exception as e:
            logger.warning(f"[HISTORY] Failed: {e}")
            return recent_messages

    @traceable(run_type="llm", name="contextualize_query")
    def _contextualize(self, question: str, history: list[Message]) -> str:
        """Rewrite query to be self-contained."""
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
        except Exception as e:
            logger.warning(f"[CONTEXT] Failed: {e}")
            return question

    @traceable(run_type="retriever", name="retrieve_chunks")
    def _retrieve_chunks(self, contextualized: str) -> list:
        """Retrieve chunks from vector store."""
        chunks = retrieve(contextualized)
        logger.info(f"[RETRIEVE] Got {len(chunks)} chunks")
        return chunks

    @traceable(run_type="chain", name="filter_chunks")
    def _filter_chunks(self, chunks: list) -> list:
        """Filter chunks by score >= 0.25."""
        filtered = [c for c in chunks if c.get('score', 0.0) >= settings.score_threshold]
        logger.info(f"[FILTER] {len(filtered)} chunks >= {settings.score_threshold}")
        return filtered

    @traceable(run_type="llm", name="rerank_chunks")
    def _rerank_chunks(self, contextualized: str, chunks: list) -> list:
        """Rerank chunks with Groq, keep only >= 6.0."""
        reranked = self.reranker.rerank(contextualized, chunks)
        logger.info(f"[RERANK] {len(reranked)} chunks >= 6.0")
        return reranked

    @traceable(run_type="tool", name="web_search")
    def _web_search_fallback(self, contextualized: str, chunks: list, reason: str) -> QueryResult:
        """Fallback to web search when paper chunks insufficient."""
        logger.info(f"[FALLBACK] {reason}, triggering web search")
        
        paper_titles = list(set([c.get('paper_title', 'Unknown') for c in chunks]))
        web_results = self.adaptive_rag.search_web(contextualized, paper_titles)
        result = self.adaptive_rag.generate_web_augmented(
            contextualized, web_results, paper_titles
        )
        
        return QueryResult(
            question=contextualized,
            answer=result["answer"],
            sources=[],
            confidence=result["confidence"],
            contextualized_query=contextualized
        )

    @traceable(run_type="llm", name="generate_paper_answer")
    def _generate_paper_answer(self, question: str, contextualized: str, 
                               chunks: list, history: list[Message]) -> QueryResult:
        """Generate answer from paper chunks with history."""
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

    # ── Simple handlers ─────────────────────────────────────────────────────

    def _handle_chat(self, question: str) -> QueryResult:
        """Handle chat intent."""
        logger.info("Chat path selected — no retrieval")
        answer = self.adaptive_rag.generate_chat(question)
        return QueryResult(
            question=question,
            answer=answer,
            sources=[],
            confidence="HIGH",
            contextualized_query=question
        )

    def _no_papers_response(self, question: str, contextualized: str) -> QueryResult:
        """Return when no papers indexed."""
        return QueryResult(
            question=question,
            answer="NO_PAPERS_INDEXED",
            sources=[],
            confidence="LOW",
            contextualized_query=contextualized
        )

    # ── Helper methods (unchanged) ──────────────────────────────────────────

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """Direct Ollama call for phi3."""
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
        """Format chunks into context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] From '{chunk.get('paper_title', 'Unknown')}' "
                f"(page {chunk.get('page_number', 'N/A')}, "
                f"type: {chunk.get('chunk_type', 'content')}, "
                f"relevance: {chunk.get('rerank_score', chunk.get('score', 0)):.1f}):\n"
                f"{chunk.get('content', '')}"
            )
        return "\n\n".join(parts)

    def _chunks_to_schema(self, chunks: list[dict]) -> list[Chunk]:
        """Convert chunk dicts to Chunk schema objects."""
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
                    caption=chunk.get("caption"),
                    figure_description=chunk.get("figure_description")
                ))
            except Exception:
                continue
        return result
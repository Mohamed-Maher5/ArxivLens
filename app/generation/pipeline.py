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
        if LANGSMITH_AVAILABLE and os.getenv("LANGCHAIN_TRACING_V2") == "true":
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

            # ❌ No fallback anymore
            if not chunks:
                return self._no_papers_response(question, contextualized)

            filtered_chunks = self._filter_chunks(chunks)

            # ❌ No web fallback → just stop
            if not filtered_chunks:
                return self._no_papers_response(question, contextualized)

            reranked = self._rerank_chunks(contextualized, filtered_chunks)

            # ❌ No web fallback → just stop
            if not reranked:
                return self._no_papers_response(question, contextualized)

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

    def _no_papers_response(self, question: str, history: list[Message]) -> QueryResult:
        """Answer from model knowledge when no paper chunks exist."""
        history_text = "\n".join([f"{m.role}: {m.content}" for m in history]) if history else ""
        
        answer = self.adaptive_rag._hf_with_prompt(
            prompts.MODEL_KNOWLEDGE_FALLBACK_PROMPT,
            {"history": history_text, "question": question},
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
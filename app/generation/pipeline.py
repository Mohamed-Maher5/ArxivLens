import re
import requests
from app.core.logger import logger
from app.core.exceptions import PipelineError
from app.core.settings import settings
from app.models.schemas import QueryResult, Chunk, Message
from app.generation.adaptive_rag import AdaptiveRAG
from app.retrieval import retrieve
from app.retrieval.reranker import Reranker


class Pipeline:

    def __init__(self):
        self.adaptive_rag = AdaptiveRAG()
        self.reranker = Reranker()
        logger.info("Pipeline initialized")

    def run(self, question: str, history: list[Message] = None) -> QueryResult:
        logger.info(f"Pipeline running for: {question[:50]}...")
        try:
            if history is None:
                history = []

            # ── Step 1: Manage history (phi3 via Ollama) ──────────────────
            managed_history = self._manage_history(history)

            # ── Step 2: Classify intent (Groq llama-3.1-8b-instant) ───────
            intent = self.adaptive_rag.classify_intent(question)

            if intent == "chat":
                # CHAT path — respond naturally, no retrieval needed
                logger.info("Chat path selected — no retrieval")
                answer = self.adaptive_rag.generate_chat(question)
                return QueryResult(
                    question=question,
                    answer=answer,
                    sources=[],
                    confidence="HIGH",
                    contextualized_query=question
                )

            # ── TASK path ──────────────────────────────────────────────────

            # ── Step 3: Contextualize query (phi3 via Ollama) ─────────────
            contextualized = self._contextualize(question, managed_history)
            logger.info(f"[CONTEXT] Result: {contextualized[:80]}")

            # ── Step 4: HyDE (Groq llama-3.1-8b-instant) ─────────────────
            hypothetical = self.adaptive_rag.generate_hypothetical(contextualized)
            logger.info(f"[HYDE] Using for retrieval: {hypothetical[:80]}...")

            # ── Step 5: Retrieve top 10 chunks with scores ────────────────
            chunks = retrieve(hypothetical)
            logger.info(f"[RETRIEVE] Got {len(chunks)} chunks")
            if chunks:
                scores = [c.get('score', 0.0) for c in chunks]
                logger.info(f"[RETRIEVE] Scores: min={min(scores):.4f} max={max(scores):.4f} avg={sum(scores)/len(scores):.4f}")

            if not chunks:
                return QueryResult(
                    question=question,
                    answer="NO_PAPERS_INDEXED",
                    sources=[],
                    confidence="LOW",
                    contextualized_query=contextualized
                )

            # ── Step 6: Score threshold check ─────────────────────────────
            best_score = max(c.get("score", 0.0) for c in chunks)
            logger.info(
                f"[THRESHOLD] Best score: {best_score:.4f} | "
                f"Threshold: {settings.score_threshold} | "
                f"Path: {'PAPER' if best_score >= settings.score_threshold else 'GENERIC'}"
            )

            if best_score < settings.score_threshold:
                # Below threshold — generic answer from model knowledge
                result = self.adaptive_rag.generate_generic(contextualized)
                return QueryResult(
                    question=question,
                    answer=result["answer"],
                    sources=[],
                    confidence=result["confidence"],
                    contextualized_query=contextualized
                )

            # ── Step 7: Rerank top 10 → top 3 (Groq llama-3.1-8b-instant) ─
            # Rerank against original contextualized question (not HyDE)
            reranked = self.reranker.rerank(contextualized, chunks)
            if not reranked:
                reranked = chunks[:settings.top_k_rerank]
            logger.info(f"[RERANK] Final chunks: {len(reranked)}")
            for i, c in enumerate(reranked, 1):
                logger.info(
                    f"[RERANK] Chunk {i}: '{c.get('paper_title','?')[:40]}' "
                    f"page={c.get('page_number','?')} "
                    f"type={c.get('chunk_type','?')} "
                    f"score={c.get('score',0.0):.4f}"
                )

            # ── Step 8: Build context with metadata proof ─────────────────
            context = self._build_context(reranked)

            # ── Step 9: Generate paper-based answer (Qwen3-8B via HF) ─────
            result = self.adaptive_rag.generate_from_paper(contextualized, context)

            return QueryResult(
                question=question,
                answer=result["answer"],
                sources=self._chunks_to_schema(reranked),
                confidence=result["confidence"],
                contextualized_query=contextualized
            )

        except Exception as e:
            raise PipelineError(f"Pipeline failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _manage_history(self, history: list[Message]) -> list[Message]:
        """
        Summarize old messages beyond max_history using phi3 (Ollama).
        Free call — no API quota consumed.
        """
        if len(history) <= settings.max_history:
            return history
        old_messages = history[:-settings.max_history]
        recent_messages = history[-settings.max_history:]
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
        logger.info(f"[HISTORY] Summarizing {len(old_messages)} old messages using phi3 (Ollama)")
        try:
            prompt = (
                f"Summarize this conversation in 2-3 sentences. "
                f"Keep key research topics and important context:\n\n{old_text}"
            )
            summary = self._ollama(prompt, max_tokens=150)
            if summary:
                logger.info(f"[HISTORY] Summary: {summary[:100]}")
                return [Message(
                    role="system",
                    content=f"Earlier conversation summary: {summary}"
                )] + recent_messages
            else:
                logger.warning("[HISTORY] Ollama returned empty — using recent only")
                return recent_messages
        except Exception as e:
            logger.warning(f"[HISTORY] Failed: {e} — using recent only")
            return recent_messages

    def _contextualize(self, question: str, history: list[Message]) -> str:
        """
        Rewrite follow-up questions to be self-contained using phi3 (Ollama).
        Free call — no API quota consumed.
        Returns original question if no history or Ollama fails.
        """
        if not history:
            logger.debug("[CONTEXT] No history — returning question as-is")
            return question
        logger.info(f"[CONTEXT] Contextualizing using phi3 (Ollama)")
        try:
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
            prompt = (
                f"Rewrite this question to be completely self-contained using the conversation history.\n"
                f"If the question already makes sense alone return it unchanged.\n"
                f"Return ONLY the rewritten question — no explanation.\n\n"
                f"History:\n{history_text}\n\n"
                f"Question: {question}\n\n"
                f"Rewritten question:"
            )
            result = self._ollama(prompt, max_tokens=100)
            if result:
                logger.debug(f"[CONTEXT] Rewritten: {result[:100]}")
                return result
            return question
        except Exception as e:
            logger.warning(f"[CONTEXT] Failed: {e} — using original question")
            return question

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """Direct Ollama call for phi3."""
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

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Format reranked chunks into context string for LLM.
        Includes paper title, page, chunk type and score as metadata proof.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] From '{chunk.get('paper_title', 'Unknown')}' "
                f"(page {chunk.get('page_number', 'N/A')}, "
                f"type: {chunk.get('chunk_type', 'content')}, "
                f"score: {chunk.get('score', 0.0):.4f}):\n"
                f"{chunk.get('content', '')}"
            )
        return "\n\n".join(parts)

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
                    caption=chunk.get("caption"),
                    figure_description=chunk.get("figure_description")
                ))
            except Exception:
                continue
        return result
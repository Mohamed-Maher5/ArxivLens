import requests
from app.core.logger import logger
from app.core.exceptions import PipelineError
from app.core.settings import settings
from app.models.schemas import QueryResult, Chunk, Message
from app.generation.adaptive_rag import AdaptiveRAG
from app.generation import prompts
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
            # Summarizes old messages beyond last 5 using phi3 (free)
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
            # Rewrites follow-up questions to be self-contained
            contextualized = self._contextualize(question, managed_history)
            logger.info(f"[CONTEXT] Result: {contextualized[:80]}")

            # ── Step 4: Retrieve chunks (direct retrieval, NO HyDE) ───────
            chunks = retrieve(contextualized)
            logger.info(f"[RETRIEVE] Got {len(chunks)} chunks")
            
            if not chunks:
                # No papers indexed at all
                return QueryResult(
                    question=question,
                    answer="NO_PAPERS_INDEXED",
                    sources=[],
                    confidence="LOW",
                    contextualized_query=contextualized
                )

            # ── Step 5: Filter by threshold >= 0.25 ───────────────────────
            # Keep raw chunks before filtering to extract paper titles for web search
            raw_chunks = chunks.copy()
            
            # Only keep chunks with similarity score >= 0.25
            chunks = [c for c in chunks if c.get('score', 0.0) >= settings.score_threshold]
            logger.info(f"[FILTER] {len(chunks)} chunks >= {settings.score_threshold} threshold")

            if not chunks:
                # ── WEB SEARCH FALLBACK: No chunks >= 0.25 ─────────────────
                logger.info("[FALLBACK] No chunks >= 0.25, triggering web search with paper names")
                
                # Extract paper titles from raw chunks (before filtering)
                paper_titles = list(set([c.get('paper_title', 'Unknown') for c in raw_chunks]))
                logger.info(f"[FALLBACK] Paper titles from low-relevance chunks: {paper_titles}")
                
                # Search web with question + paper names
                web_results = self.adaptive_rag.search_web(contextualized, paper_titles)
                
                # Generate answer explaining papers don't cover this, but web results exist
                result = self.adaptive_rag.generate_web_augmented(
                    contextualized, web_results, paper_titles
                )
                
                return QueryResult(
                    question=question,
                    answer=result["answer"],
                    sources=[],  # No paper sources since they weren't relevant enough (score < 0.25)
                    confidence=result["confidence"],
                    contextualized_query=contextualized
                )

            # ── Step 6: Rerank with Groq (returns only chunks >= 6.0) ─────
            # Reranker filters out anything below 6.0 score
            reranked = self.reranker.rerank(contextualized, chunks)
            
            if not reranked:
                # ── WEB SEARCH FALLBACK: No chunks >= 6.0 after reranking ─
                logger.info("[FALLBACK] No chunks >= 6.0 after reranking, triggering web search")
                
                # Get paper titles from the retrieved chunks (for context)
                paper_titles = list(set([c.get('paper_title', 'Unknown') for c in chunks]))
                logger.info(f"[FALLBACK] Paper titles for context: {paper_titles}")
                
                # Search web with question + paper names
                web_results = self.adaptive_rag.search_web(contextualized, paper_titles)
                
                # Generate answer explaining papers don't cover this, but web results exist
                result = self.adaptive_rag.generate_web_augmented(
                    contextualized, web_results, paper_titles
                )
                
                return QueryResult(
                    question=question,
                    answer=result["answer"],
                    sources=[],  # No paper sources since they weren't relevant enough (score < 6)
                    confidence=result["confidence"],
                    contextualized_query=contextualized
                )

            # ── Step 7: Build context with metadata ───────────────────────
            # We have chunks >= 6.0 — proceed with paper-based answer
            context = self._build_context(reranked)
            logger.info(f"[CONTEXT] Built context from {len(reranked)} chunks")

            # ── Step 8: Generate paper-based answer with history ──────────
            # Prepare history text (last 5 messages + summary if exists)
            history_text = ""
            if managed_history:
                history_text = "\n".join([f"{m.role}: {m.content}" for m in managed_history])
                logger.debug(f"[HISTORY] Including {len(managed_history)} messages")

            # Generate with paper context + conversation history
            result = self.adaptive_rag.generate_from_paper_with_history(
                contextualized, context, history_text
            )

            return QueryResult(
                question=question,
                answer=result["answer"],
                sources=self._chunks_to_schema(reranked),
                confidence=result["confidence"],
                contextualized_query=contextualized
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline failed: {e}")

    # ── Helper Methods ───────────────────────────────────────────────────────

    def _manage_history(self, history: list[Message]) -> list[Message]:
        """
        Summarize old messages beyond max_history (5) using phi3 (Ollama).
        Returns last 5 messages plus summary of older ones at start.
        """
        if len(history) <= settings.max_history:
            return history
        
        old_messages = history[:-settings.max_history]
        recent_messages = history[-settings.max_history:]
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
        
        logger.info(f"[HISTORY] Summarizing {len(old_messages)} old messages using phi3")
        try:
            # Use prompt from prompts.py
            prompt = prompts.HISTORY_SUMMARY_PROMPT.format(text=old_text)
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
        If no history or Ollama fails, returns original question.
        """
        if not history:
            logger.debug("[CONTEXT] No history — returning question as-is")
            return question
        
        logger.info(f"[CONTEXT] Contextualizing using phi3")
        try:
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
            
            # Use prompt from prompts.py
            prompt = prompts.CONTEXTUALIZATION_PROMPT.format(
                history=history_text,
                question=question
            )
            
            result = self._ollama(prompt, max_tokens=100)
            
            if result and len(result) > 10:  # Basic validation
                logger.info(f"[CONTEXT] Rewritten: {result[:100]}")
                return result
            return question
            
        except Exception as e:
            logger.warning(f"[CONTEXT] Failed: {e} — using original question")
            return question

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
        """
        Format reranked chunks into context string for LLM.
        Includes paper title, page, chunk type and score as metadata.
        """
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
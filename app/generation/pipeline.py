# /mnt/hdd/projects/ArxivLens/app/generation/pipeline.py
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
from app.indexing.vector_store import collection_name_from_paper_id
from app.core.settings import settings

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available, tracing disabled")


DATA_PROCESSED = Path("data/processed")

class Pipeline:

    def __init__(self):
        self.adaptive_rag = AdaptiveRAG()
        self.reranker = Reranker()
        logger.info("Pipeline initialized")

    def run(self, question: str, history: list[Message] = None, paper_id: str = None) -> QueryResult:
        """
        Run the full RAG pipeline with new routing logic:
        1. Classify intent (CHAT vs TASK)
        2. CHAT -> direct response with history
        3. TASK -> contextualize -> retrieve -> check chunks -> rerank -> 
                   if chunks: top-3 generation
                   if no chunks: general knowledge fallback
        """
        if LANGSMITH_AVAILABLE and settings.langchain_tracing_v2:
            return self._run_traced(question, history, paper_id)
        return self._run_pipeline(question, history, paper_id)

    @traceable(run_type="chain", name="arxivlens_pipeline")
    def _run_traced(self, question: str, history: list[Message] = None, paper_id: str = None) -> QueryResult:
        return self._run_pipeline(question, history, paper_id)

    def _run_pipeline(self, question: str, history: list[Message] = None, paper_id: str = None) -> QueryResult:
        logger.info(f"Pipeline running for: {question[:50]}...")
        
        try:
            if history is None:
                history = []

            # Step 1: Manage history - get BOTH summary and recent messages
            summary, recent_messages = self._manage_history(history)
            history_text = self._format_history_for_prompt(summary, recent_messages)
            logger.info(f"[PIPELINE] History: summary={bool(summary)}, recent={len(recent_messages)}")

            # Step 2: Classify intent
            intent = self._classify_intent(question)
            logger.info(f"[PIPELINE] Intent classified as: {intent}")

            # Step 3: Route based on intent
            if intent == "chat":
                return self._handle_chat(question, history_text)

            # TASK path: Attempt paper-based answer
            return self._handle_task(question, history_text, history, paper_id)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline failed: {e}")

    # ── Intent & History Handling ─────────────────────────────────────────

    @traceable(run_type="llm", name="classify_intent")
    def _classify_intent(self, question: str) -> str:
        return self.adaptive_rag.classify_intent(question)

    def _manage_history(self, history: list[Message]) -> tuple[str, list[Message]]:
        """
        Manage conversation history.
        Returns: (summary_text, recent_messages_list)
        
        - If <= 6 messages: no summary, return all as recent
        - If > 6 messages: summarize old (everything except last 6), keep last 6 full
        - Both summary and recent are sent to generation
        """
        if len(history) <= settings.max_history:
            # No summarization needed, return empty summary and all messages
            return "", history

        # More than 6: summarize old, keep last 6
        old_messages = history[:-settings.max_history]   # Everything BEFORE last 6
        recent_messages = history[-settings.max_history:]  # Last 6 (KEEP FULL)
        
        # Summarize old messages into 2 sentences
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
        
        try:
            summary = self._ollama_summarize(old_text, max_tokens=100)
            if summary:
                logger.info(f"[HISTORY] Summarized {len(old_messages)} messages into: {summary[:80]}...")
            else:
                summary = f"Earlier conversation with {len(old_messages)} messages."
        except Exception as e:
            logger.warning(f"[HISTORY] Summarization failed: {e}")
            summary = f"Earlier conversation with {len(old_messages)} messages."
        
        # Return BOTH: summary text AND recent messages (last 6)
        return summary, recent_messages

    def _format_history_for_prompt(self, summary: str, recent_messages: list[Message]) -> str:
        """Format both summary and recent messages for prompt injection."""
        parts = []
        
        if summary:
            parts.append(f"Earlier conversation summary: {summary}")
        
        if recent_messages:
            parts.append("Recent messages:")
            parts.extend([f"{m.role}: {m.content}" for m in recent_messages])
        
        return "\n".join(parts) if parts else "No previous conversation."

    def _format_history(self, history: list[Message]) -> str:
        """Format history messages for prompt injection."""
        if not history:
            return "No previous conversation."
        return "\n".join([f"{m.role}: {m.content}" for m in history])

    def _ollama_summarize(self, text: str, max_tokens: int = 150) -> str:
        """Summarize text using Ollama phi3."""
        try:
            prompt = prompts.HISTORY_SUMMARY_PROMPT.format(text=text)
            return self._ollama(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.warning(f"[OLLAMA] Summarization error: {e}")
            return ""

    # ── CHAT Handler ─────────────────────────────────────────

    def _handle_chat(self, question: str, history_text: str) -> QueryResult:
        """Handle casual chat with history context."""
        logger.info("[PIPELINE] Handling as CHAT")
        
        answer = self.adaptive_rag.generate_chat_with_history(question, history_text)
        
        return QueryResult(
            question=question,
            answer=answer,
            sources=[],
            contextualized_query=question
        )

    # ── TASK Handler ─────────────────────────────────────────
    def _handle_task(
        self,
        question: str,
        history_text: str,
        full_history: list[Message],
        paper_id: str = None
    ) -> QueryResult:
        """Handle task/research questions with paper retrieval attempt."""
        logger.info("[PIPELINE] Handling as TASK")

        # Step 1: Contextualize query using full history
        contextualized = self._contextualize(question, full_history)
        logger.info(f"[PIPELINE] Contextualized: {contextualized[:60]}...")

        # Step 2: Resolve collection
        collection, resolved_paper_id = self._resolve_collection(paper_id)

        # Step 3: Load metadata
        metadata = self.get_paper_metadata(resolved_paper_id) if resolved_paper_id else ""

        # Step 4: Retrieve chunks (if collection exists)
        chunks = []
        if collection:
            chunks = self._retrieve_chunks(contextualized, collection)

        # Step 5: Rerank
        if chunks:
            chunks = self._rerank_chunks(contextualized, chunks)
            chunks = chunks[:3] 

        # Step 6: Route to generator
        if chunks:
            return self._generate_paper_answer(contextualized, chunks, history_text)
        else:
            logger.info("[PIPELINE] No chunks passed threshold, falling back to general knowledge")
            return self._general_knowledge_response(contextualized, history_text, metadata)

    def _resolve_collection(self, paper_id: str = None) -> tuple[str | None, str | None]:
        """Resolve paper collection name from explicit ID or disk inference."""
        if paper_id:
            collection = collection_name_from_paper_id(paper_id)
            logger.info(f"[PIPELINE] Using explicit paper_id={paper_id}")
            return collection, paper_id
        
        # Try to infer from disk
        inferred_id = self._infer_paper_id_from_disk()
        if inferred_id:
            collection = collection_name_from_paper_id(inferred_id)
            logger.info(f"[PIPELINE] Inferred paper_id={inferred_id} from disk")
            return collection, inferred_id
        
        return None, None

    @traceable(run_type="llm", name="contextualize_query")
    def _contextualize(self, question: str, history: list[Message]) -> str:
        """Rewrite question to be self-contained given history."""
        if not history:
            return question
        
        try:
            history_text = self._format_history(history)
            prompt = prompts.CONTEXTUALIZATION_PROMPT.format(
                history=history_text,
                question=question
            )
            result = self._ollama(prompt, max_tokens=100)
            return result if result and len(result) > 10 else question
            
        except Exception as e:
            logger.warning(f"[CONTEXT] Contextualization failed: {e}")
            return question

    @traceable(run_type="retriever", name="retrieve_chunks")
    def _retrieve_chunks(self, contextualized: str, collection_name: str) -> list:
        """Retrieve chunks from vector store."""
        chunks = retrieve(contextualized, collection_name)
        logger.info(f"[RETRIEVE] Got {len(chunks)} chunks")
        return chunks

    @traceable(run_type="llm", name="rerank_chunks")
    def _rerank_chunks(self, contextualized: str, chunks: list) -> list:
        """
        Rerank chunks and return only those passing threshold.
        Reranker already filters by score > 2.0
        """
        reranked = self.reranker.rerank(contextualized, chunks)
        logger.info(f"[RERANK] {len(reranked)} chunks passed threshold")
        return reranked

    def _generate_paper_answer(
        self,
        question: str,
        chunks: list[dict],
        history_text: str
    ) -> QueryResult:
        """Generate answer from top paper chunks."""
        result = self.adaptive_rag.generate_from_paper_top3(
            question, chunks, history_text
        )
        
        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=self._chunks_to_schema(chunks),
            contextualized_query=question
        )

    def _general_knowledge_response(
        self,
        question: str,
        history_text: str,
        metadata: str
    ) -> QueryResult:
        """Generate answer from general knowledge with optional metadata."""
        result = self.adaptive_rag.generate_general_knowledge(
            question, metadata, history_text
        )
        
        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=[],  # No specific chunks to cite
            contextualized_query=question
        )

    # ── Metadata & Utilities ─────────────────────────────────────────

    def _infer_paper_id_from_disk(self) -> str | None:
        """Scan data/processed/ for most recent JSON file."""
        if not DATA_PROCESSED.exists():
            logger.warning(f"[INFER] Directory not found: {DATA_PROCESSED}")
            return None
        
        json_files = sorted(
            DATA_PROCESSED.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not json_files:
            logger.info("[INFER] No processed papers found on disk")
            return None
            
        paper_id = json_files[0].stem
        logger.info(f"[INFER] Inferred paper_id from disk: {paper_id}")
        return paper_id

    def get_paper_metadata(self, paper_id: str) -> str:
        """Load paper metadata from processed JSON file."""
        if not paper_id:
            return ""
        
        file_path = DATA_PROCESSED / f"{paper_id}.json"

        if not file_path.exists():
            logger.warning(f"[METADATA] File not found: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "Unknown title")
            authors = data.get("authors", [])
            abstract = data.get("abstract", "")
            published = data.get("published", "")

            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += f" et al."
            else:
                authors_str = str(authors) if authors else "Unknown authors"

            abstract_display = abstract[:500]
            if len(abstract) > 500:
                abstract_display += "..."

            lines = [
                f"Title: {title}",
                f"Authors: {authors_str}",
            ]
            if published:
                lines.append(f"Published: {published}")
            lines.append(f"Abstract: {abstract_display}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[METADATA] Error reading {file_path}: {e}")
            return ""

    # ── Ollama & Helpers ─────────────────────────────────────────

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """Call Ollama API for local LLM tasks."""
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
            
            logger.warning(f"[OLLAMA] Non-200 status: {response.status_code}")
            return ""
            
        except requests.exceptions.Timeout:
            logger.warning("[OLLAMA] Request timed out")
            return ""
        except requests.exceptions.ConnectionError:
            logger.warning("[OLLAMA] Connection failed - is Ollama running?")
            return ""
        except Exception as e:
            logger.warning(f"[OLLAMA] Error: {e}")
            return ""

    def _chunks_to_schema(self, chunks: list[dict]) -> list[Chunk]:
        """Convert raw chunks to schema Chunk objects."""
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
            except Exception as e:
                logger.warning(f"[SCHEMA] Failed to convert chunk: {e}")
                continue
        return result
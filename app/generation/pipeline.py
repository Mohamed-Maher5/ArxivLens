from langchain_groq import ChatGroq
from app.core.logger import logger
from app.core.exceptions import PipelineError
from app.core.settings import settings
from app.models.schemas import QueryResult, Chunk, Message
from app.generation.prompts import (
    CONTEXTUALIZATION_PROMPT,
    COVERAGE_PROMPT
)
from app.generation.adaptive_rag import AdaptiveRAG
from app.retrieval import retrieve
from app.retrieval.reranker import Reranker


class Pipeline:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model
        )
        self.adaptive_rag = AdaptiveRAG()
        self.reranker = Reranker()
        logger.info("Pipeline initialized")

    def run(self, question: str, history: list[Message] = None) -> QueryResult:
        logger.info(f"Pipeline running for: {question[:50]}...")
        try:
            if history is None:
                history = []

            # Step 1 — Manage history
            managed_history = self._manage_history(history)

            # Step 2 — Contextualize
            contextualized = self._contextualize(question, managed_history)
            logger.info(f"Contextualized: {contextualized[:60]}...")

            # Step 3 — Retrieve 10 chunks
            chunks = retrieve(contextualized)
            logger.info(f"Retrieved {len(chunks)} chunks")

            if not chunks:
                return QueryResult(
                    question=question,
                    answer="NO_PAPERS_INDEXED",
                    sources=[],
                    confidence="LOW",
                    contextualized_query=contextualized
                )

            # Step 4 — LLM coverage check
            context = self._build_context(chunks)
            covered = self._check_coverage(contextualized, context)

            if not covered:
                logger.warning("Question not covered by indexed papers")
                return QueryResult(
                    question=question,
                    answer="NOT_COVERED",
                    sources=self._chunks_to_schema(chunks),
                    confidence="LOW",
                    contextualized_query=contextualized
                )

            # Step 5 — Route
            route = self.adaptive_rag.route(contextualized, context)
            logger.info(f"Route: {route}")

            if route == "simple":
                return self._simple_path(
                    question, contextualized, chunks, context
                )
            else:
                return self._complex_path(
                    question, contextualized, chunks
                )

        except Exception as e:
            raise PipelineError(f"Pipeline failed: {e}")

    def _simple_path(
        self,
        question: str,
        contextualized: str,
        chunks: list[dict],
        context: str
    ) -> QueryResult:
        logger.info("Simple path — generating with 10 chunks")
        result = self.adaptive_rag.generate(contextualized, context)
        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=self._chunks_to_schema(chunks),
            confidence=result["confidence"],
            contextualized_query=contextualized
        )

    def _complex_path(
        self,
        question: str,
        contextualized: str,
        chunks: list[dict]
    ) -> QueryResult:
        logger.info("Complex path — reranking to 3 chunks")
        reranked = self.reranker.rerank(contextualized, chunks)
        if not reranked:
            reranked = chunks[:3]
        context = self._build_context(reranked)
        result = self.adaptive_rag.generate(contextualized, context)
        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=self._chunks_to_schema(reranked),
            confidence=result["confidence"],
            contextualized_query=contextualized
        )

    def _check_coverage(self, question: str, context: str) -> bool:
        try:
            chain = COVERAGE_PROMPT | self.llm
            result = chain.invoke({
                "question": question,
                "context": context
            })
            coverage = result.content.strip().upper()
            logger.info(f"Coverage check: {coverage}")
            return coverage == "COVERED"
        except Exception:
            logger.warning("Coverage check failed — defaulting to covered")
            return True

    def _manage_history(self, history: list[Message]) -> list[Message]:
        if len(history) <= settings.max_history:
            return history
        old_messages = history[:-settings.max_history]
        recent_messages = history[-settings.max_history:]
        old_text = "\n".join([
            f"{m.role}: {m.content}"
            for m in old_messages
        ])
        try:
            summary_response = self.llm.invoke(
                f"Summarize this conversation in 2-3 sentences. "
                f"Keep key research topics and important context:\n\n{old_text}"
            )
            summary = summary_response.content.strip()
            logger.info("Old history summarized successfully")
            summary_message = Message(
                role="system",
                content=f"Earlier conversation summary: {summary}"
            )
            return [summary_message] + recent_messages
        except Exception:
            logger.warning("History summarization failed — using recent only")
            return recent_messages

    def _contextualize(self, question: str, history: list[Message]) -> str:
        if not history:
            return question
        try:
            history_text = "\n".join([
                f"{m.role}: {m.content}"
                for m in history
            ])
            chain = CONTEXTUALIZATION_PROMPT | self.llm
            result = chain.invoke({
                "history": history_text,
                "question": question
            })
            return result.content.strip()
        except Exception:
            return question

    def _build_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            part = (
                f"[{i}] From '{chunk.get('paper_title', 'Unknown')}' "
                f"(page {chunk.get('page_number', 'N/A')}, "
                f"type: {chunk.get('chunk_type', 'content')}):\n"
                f"{chunk.get('content', '')}"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

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
from app.core.exceptions import PipelineError, RetrievalError
from app.core.logger import logger
from app.core.settings import settings
from app.dependencies import get_llm_client
from app.models.schemas import Chunk, QueryResult
from app.orchestration.collection_resolver import CollectionResolver
from app.orchestration.context_manager import ContextManager
from app.orchestration.query_router import QueryRouter
from app.retrieval import retrieve
from app.retrieval.reranker import Reranker

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Pipeline:
    def __init__(
        self,
        llm_client=None,
        reranker=None,
        context_manager: ContextManager | None = None,
        collection_resolver: CollectionResolver | None = None,
        query_router: QueryRouter | None = None,
        retrieve_fn=None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.reranker = reranker or Reranker(self.llm_client)
        self.context_manager = context_manager or ContextManager(self.llm_client)
        self.collection_resolver = collection_resolver or CollectionResolver()
        self.query_router = query_router or QueryRouter(self.llm_client)
        self.retrieve_fn = retrieve_fn or retrieve
        logger.info('Pipeline initialized')

    def run(self, question: str, arxiv_id: str | None = None) -> QueryResult:
        if settings.langchain_tracing_v2:
            return self._run_traced(question, arxiv_id)
        return self._run_pipeline(question, arxiv_id)

    @traceable(run_type='chain', name='arxivlens_pipeline')
    def _run_traced(self, question: str, arxiv_id: str | None = None) -> QueryResult:
        return self._run_pipeline(question, arxiv_id)

    def _run_pipeline(self, question: str, arxiv_id: str | None = None) -> QueryResult:
        logger.info(f'Pipeline running for: {question[:50]}...')
        try:
            intent = self._classify_intent(question)
            logger.info(f'[PIPELINE] Intent classified as: {intent}')

            if intent == 'chat':
                return self._handle_chat(question)
            return self._handle_task(question, arxiv_id)
        except Exception as error:
            logger.error(f'Pipeline failed: {error}')
            raise PipelineError(f'Pipeline failed: {error}') from error

    @traceable(run_type='llm', name='classify_intent')
    def _classify_intent(self, question: str) -> str:
        return self.query_router.route(question)

    def _handle_chat(self, question: str) -> QueryResult:
        logger.info('[PIPELINE] Handling as CHAT')
        answer = self.llm_client.generate_chat_response(question)
        return QueryResult(
            question=question,
            answer=answer,
            sources=[],
            contextualized_query=question,
        )

    def _handle_task(self, question: str, arxiv_id: str | None = None) -> QueryResult:
        logger.info('[PIPELINE] Handling as TASK')
        contextualized = self._contextualize(question)
        collection, resolved_arxiv_id = self._resolve_collection(arxiv_id)
        metadata = self.get_paper_metadata(resolved_arxiv_id) if resolved_arxiv_id else ''

        chunks: list[dict] = []
        if collection:
            try:
                chunks = self._retrieve_chunks(contextualized, collection)
            except RetrievalError as error:
                logger.warning(
                    f'[PIPELINE] Retrieval failed for {collection}: {error}. '
                    'Falling back to metadata/general knowledge.'
                )

        if chunks:
            chunks = self._rerank_chunks(contextualized, chunks)[:3]

        if chunks:
            return self._generate_paper_answer(contextualized, chunks)

        logger.info('[PIPELINE] No chunks passed threshold, falling back to general knowledge')
        return self._general_knowledge_response(contextualized, metadata)

    def _resolve_collection(self, arxiv_id: str | None = None) -> tuple[str | None, str | None]:
        return self.collection_resolver.resolve(arxiv_id)

    @traceable(run_type='llm', name='contextualize_query')
    def _contextualize(self, question: str) -> str:
        return self.context_manager.contextualize(question)

    @traceable(run_type='retriever', name='retrieve_chunks')
    def _retrieve_chunks(self, contextualized: str, collection_name: str) -> list[dict]:
        chunks = self.retrieve_fn(contextualized, collection_name)
        logger.info(f'[RETRIEVE] Got {len(chunks)} chunks')
        return chunks

    @traceable(run_type='llm', name='rerank_chunks')
    def _rerank_chunks(self, contextualized: str, chunks: list[dict]) -> list[dict]:
        reranked = self.reranker.rerank(contextualized, chunks)
        logger.info(f'[RERANK] {len(reranked)} chunks passed threshold')
        return reranked

    def _generate_paper_answer(self, question: str, chunks: list[dict]) -> QueryResult:
        result = self.llm_client.generate_from_paper_top3(question, chunks)
        return QueryResult(
            question=question,
            answer=result['answer'],
            sources=self._chunks_to_schema(chunks),
            contextualized_query=question,
        )

    def _general_knowledge_response(self, question: str, metadata: str) -> QueryResult:
        result = self.llm_client.generate_general_knowledge(question, metadata)
        return QueryResult(
            question=question,
            answer=result['answer'],
            sources=[],
            contextualized_query=question,
        )

    def _infer_arxiv_id_from_disk(self) -> str | None:
        return self.collection_resolver.infer_arxiv_id_from_disk()

    def get_paper_metadata(self, arxiv_id: str) -> str:
        return self.collection_resolver.get_paper_metadata(arxiv_id)

    def _chunks_to_schema(self, chunks: list[dict]) -> list[Chunk]:
        result: list[Chunk] = []
        for chunk in chunks:
            try:
                result.append(
                    Chunk(
                        chunk_id=chunk.get('chunk_id', ''),
                        arxiv_id=chunk.get('arxiv_id', ''),
                        paper_title=chunk.get('paper_title', ''),
                        authors=chunk.get('authors', []),
                        content=chunk.get('content', ''),
                        chunk_type=chunk.get('chunk_type', 'content'),
                        page_number=chunk.get('page_number'),
                    )
                )
            except Exception as error:
                logger.warning(f'[SCHEMA] Failed to convert chunk: {error}')
        return result

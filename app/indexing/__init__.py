from app.indexing.chunker import Chunker
from app.indexing.embedder import Embedder
from app.indexing.vector_store import VectorStore
from app.core.logger import logger
from app.core.exceptions import ArxivLensException
from app.models.schemas import Chunk


def index_paper(parsed_result: dict) -> list[Chunk]:
    paper_id = parsed_result["paper_id"]
    chunker = Chunker()
    embedder = Embedder()
    # VectorStore now receives paper_id to derive the per-paper collection name.
    store = VectorStore(paper_id=paper_id)
    try:
        logger.info(f"Starting indexing for: {paper_id}")
        chunks = chunker.chunk(parsed_result)
        embedded = embedder.embed_chunks(chunks)
        store.store(embedded)
        logger.info(f"Indexing complete: {len(chunks)} chunks stored")
        return chunks
    except ArxivLensException:
        raise
    except Exception as e:
        raise ArxivLensException(f"Indexing failed: {e}")
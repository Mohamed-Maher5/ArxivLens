# /mnt/hdd/projects/ArxivLens/app/ingestion/test_retrieval.py
from app.retrieval import retrieve, rerank_chunks
from app.core.logger import logger

QUERY = "machine learning applications in NLP"


def test_retrieval(query: str):
    """Test the normal retrieval pipeline."""
    try:
        results = retrieve(query)
        print(f"Retrieved {len(results)} chunks for query: '{query}'")
        for i, chunk in enumerate(results[:5], 1):  # top 5 for brevity
            print(f"{i:02d}: score={chunk.get('score',0):.4f} | "
                  f"type={chunk.get('chunk_type','?')} | "
                  f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}...")
        print("✅ Retrieval test completed")
        return results
    except Exception as e:
        logger.error(f"Retrieval test failed: {e}")
        print(f"❌ Retrieval test failed: {e}")
        return []


def test_rerank(query: str, chunks: list[dict]):
    """Test reranking on already retrieved chunks."""
    try:
        if not chunks:
            print("❌ No chunks to rerank")
            return []

        reranked = rerank_chunks(query, chunks)
        print(f"\nReranked {len(reranked)} chunks for query: '{query}'")
        for i, chunk in enumerate(reranked[:5], 1):  # top 5 for brevity
            print(f"{i:02d}: score={chunk.get('score',0):.4f} | "
                  f"rerank_score={chunk.get('rerank_score',0):.2f} | "
                  f"type={chunk.get('chunk_type','?')} | "
                  f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}...")
        print("✅ Rerank test completed")
        return reranked
    except Exception as e:
        logger.error(f"Rerank test failed: {e}")
        print(f"❌ Rerank test failed: {e}")
        return []


if __name__ == "__main__":
    retrieved_chunks = test_retrieval(QUERY)
    test_rerank(QUERY, retrieved_chunks)
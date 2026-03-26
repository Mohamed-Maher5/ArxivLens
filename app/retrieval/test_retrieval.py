# /mnt/hdd/projects/ArxivLens/app/ingestion/test_retrieval.py
from app.retrieval import retrieve, rerank_chunks
from app.core.logger import logger

QUERY = "what is the attension mechanism "
COLLECTION = "paper_2105_02723v1"


def test_retrieval(query: str, collection_name: str):
    """Test the normal retrieval pipeline."""
    try:
        results = retrieve(query, collection_name)
        print(f"\n🔍 Retrieved {len(results)} chunks for query: '{query}'")
        print(f"   Collection: {collection_name}")
        print("-" * 80)
        
        for i, chunk in enumerate(results[:5], 1):
            print(f"{i:02d}: score={chunk.get('score',0):.4f} | "
                  f"type={chunk.get('chunk_type','?')} | "
                  f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}...")
        print("-" * 80)
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
        print(f"\n🔄 Reranked {len(reranked)} chunks for query: '{query}'")
        print("-" * 80)
        
        for i, chunk in enumerate(reranked[:5], 1):
            print(f"{i:02d}: score={chunk.get('score',0):.4f} | "
                  f"rerank_score={chunk.get('rerank_score',0):.2f} | "
                  f"type={chunk.get('chunk_type','?')} | "
                  f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}...")
        print("-" * 80)
        print("✅ Rerank test completed")
        return reranked
        
    except Exception as e:
        logger.error(f"Rerank test failed: {e}")
        print(f"❌ Rerank test failed: {e}")
        return []


if __name__ == "__main__":
    retrieved_chunks = test_retrieval(QUERY, COLLECTION)
    test_rerank(QUERY, retrieved_chunks)
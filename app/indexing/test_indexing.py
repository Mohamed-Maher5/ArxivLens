# /mnt/hdd/projects/ArxivLens/app/ingestion/test_indexing.py
import json
from pathlib import Path
from app.indexing.chunker import Chunker
from app.indexing.embedder import Embedder
from app.indexing.vector_store import VectorStore
from app.core.logger import logger

# Path to a parsed paper JSON (from your ingestion test)
PARSED_JSON = Path("/mnt/hdd/projects/ArxivLens/data/processed/2006.16189v4.json")  # <- replace with a real file

def test_indexing(parsed_file: Path):
    if not parsed_file.exists():
        logger.error(f"Parsed paper not found: {parsed_file}")
        return

    # Load the parsed paper
    with open(parsed_file, "r") as f:
        parsed_result = json.load(f)

    # Initialize components
    chunker = Chunker()
    embedder = Embedder()
    store = VectorStore(parsed_result['paper_id'])
    try:
        logger.info(f"Starting test indexing for paper: {parsed_result['paper_id']}")

        # 1️⃣ Chunk the paper
        chunks = chunker.chunk(parsed_result)
        logger.info(f"Chunking complete: {len(chunks)} chunks created")

        # 2️⃣ Embed the chunks
        embedded_chunks = embedder.embed_chunks(chunks)
        logger.info(f"Embedding complete: {len(embedded_chunks)} chunks embedded")

        # 3️⃣ Store in Qdrant
        store.store(embedded_chunks)
        logger.info("All chunks stored successfully in Qdrant")

    except Exception as e:
        logger.error(f"Indexing test failed: {e}")
        print(f"❌ Indexing test failed: {e}")


if __name__ == "__main__":
    test_indexing(PARSED_JSON)
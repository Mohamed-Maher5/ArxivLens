import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.logger import logger
from app.core.settings import settings
from app.indexing.vector_store import collection_name_from_arxiv_id


DATA_PROCESSED = Path("data/processed")


class CollectionResolver:
    """Owns arXiv ID inference, collection naming, and processed metadata lookup."""

    def __init__(
        self,
        data_processed: Path | None = None,
        qdrant_client: QdrantClient | None = None,
    ):
        self.data_processed = data_processed or DATA_PROCESSED
        self.qdrant_client = qdrant_client or QdrantClient(url=settings.qdrant_url)

    def resolve(self, arxiv_id: str | None = None) -> tuple[str | None, str | None]:
        if arxiv_id:
            collection = collection_name_from_arxiv_id(arxiv_id)
            logger.info(f"[PIPELINE] Using explicit arxiv_id={arxiv_id}")
            if self.collection_exists(collection):
                return collection, arxiv_id
            logger.warning(
                f"[PIPELINE] Collection {collection} not found in local Qdrant; "
                "falling back to metadata/general knowledge."
            )
            return None, arxiv_id

        inferred_id = self.infer_arxiv_id_from_disk()
        if inferred_id:
            collection = collection_name_from_arxiv_id(inferred_id)
            logger.info(f"[PIPELINE] Inferred arxiv_id={inferred_id} from disk")
            return collection, inferred_id

        return None, None

    def infer_arxiv_id_from_disk(self) -> str | None:
        if not self.data_processed.exists():
            logger.warning(f"[INFER] Directory not found: {self.data_processed}")
            return None

        json_files = sorted(
            self.data_processed.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            logger.info("[INFER] No processed papers found on disk")
            return None

        for file_path in json_files:
            arxiv_id = file_path.stem
            collection = collection_name_from_arxiv_id(arxiv_id)
            if self.collection_exists(collection):
                logger.info(f"[INFER] Inferred arxiv_id from disk: {arxiv_id}")
                return arxiv_id

        logger.info("[INFER] No processed papers have a matching local Qdrant collection")
        return None

    def collection_exists(self, collection_name: str) -> bool:
        try:
            existing = {item.name for item in self.qdrant_client.get_collections().collections}
            return collection_name in existing
        except Exception as error:
            logger.warning(f"[INFER] Failed to check Qdrant collections: {error}")
            return False

    def get_paper_metadata(self, arxiv_id: str | None) -> str:
        if not arxiv_id:
            return ""

        file_path = self.data_processed / f"{arxiv_id}.json"
        if not file_path.exists():
            logger.warning(f"[METADATA] File not found: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            title = data.get("title", "Unknown title")
            authors = data.get("authors", [])
            abstract = data.get("abstract", "")
            published = data.get("published", "")

            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += " et al."
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
        except Exception as error:
            logger.error(f"[METADATA] Error reading {file_path}: {error}")
            return ""

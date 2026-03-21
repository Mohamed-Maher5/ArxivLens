import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logger import logger
from app.core.exceptions import ChunkingError
from app.core.settings import settings
from app.models.schemas import Chunk


class Chunker:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        logger.info("Chunker initialized")

    def chunk(self, parsed_result: dict) -> list[Chunk]:
        logger.info(f"Chunking paper: {parsed_result['paper_id']}")
        try:
            chunks = []
            paper_id = parsed_result["paper_id"]
            title = parsed_result["title"]
            authors = parsed_result["authors"]

            chunks.append(self._make_abstract_chunk(parsed_result))

            for page in parsed_result.get("pages", []):
                page_chunks = self._chunk_content(
                    page["text"],
                    paper_id,
                    title,
                    authors,
                    page["page_number"]
                )
                chunks.extend(page_chunks)

            for image in parsed_result.get("images", []):
                if image.get("description"):
                    chunks.append(self._make_figure_chunk(
                        image, paper_id, title, authors
                    ))

            for table in parsed_result.get("tables", []):
                chunks.append(self._make_table_chunk(
                    table, paper_id, title, authors
                ))

            if parsed_result.get("references"):
                chunks.append(self._make_references_chunk(
                    parsed_result["references"],
                    paper_id, title, authors
                ))

            logger.info(f"Created {len(chunks)} chunks for {paper_id}")
            return chunks

        except Exception as e:
            raise ChunkingError(f"Chunking failed for {parsed_result['paper_id']}: {e}")

    def _make_abstract_chunk(self, parsed_result: dict) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=parsed_result["paper_id"],
            paper_title=parsed_result["title"],
            authors=parsed_result["authors"],
            content=parsed_result["abstract"],
            chunk_type="abstract",
            page_number=1
        )

    def _chunk_content(self, text: str, paper_id: str, title: str,
                       authors: list, page_number: int) -> list[Chunk]:
        chunks = []
        texts = self.splitter.split_text(text)
        for t in texts:
            if t.strip():
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    paper_title=title,
                    authors=authors,
                    content=t.strip(),
                    chunk_type="content",
                    page_number=page_number
                ))
        return chunks

    def _make_figure_chunk(self, image: dict, paper_id: str,
                           title: str, authors: list) -> Chunk:
        content = image["description"]
        if image.get("caption"):
            content = f"Caption: {image['caption']}\n\n{content}"
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=paper_id,
            paper_title=title,
            authors=authors,
            content=content,
            chunk_type="figure",
            page_number=image.get("page_number"),
            caption=image.get("caption"),
            figure_description=image["description"]
        )

    def _make_table_chunk(self, table: dict, paper_id: str,
                          title: str, authors: list) -> Chunk:
        content = table["content"]
        if table.get("caption"):
            content = f"Caption: {table['caption']}\n\n{content}"
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=paper_id,
            paper_title=title,
            authors=authors,
            content=content,
            chunk_type="table",
            page_number=table.get("page_number"),
            caption=table.get("caption")
        )

    def _make_references_chunk(self, references: str, paper_id: str,
                                title: str, authors: list) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=paper_id,
            paper_title=title,
            authors=authors,
            content=references,
            chunk_type="references"
        )
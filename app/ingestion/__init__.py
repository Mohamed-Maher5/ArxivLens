import json
from pathlib import Path
from app.core.logger import logger
from app.core.exceptions import ArxivLensException
from app.ingestion.arxiv_fetcher import ArxivFetcher
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.vision_processor import VisionProcessor
from app.models.schemas import Paper


DATA_PROCESSED = Path("data/processed")


def ingest_paper(paper: Paper) -> dict:
    fetcher = ArxivFetcher()
    parser = PDFParser()
    vision = VisionProcessor()
    try:
        logger.info(f"Starting ingestion for: {paper.paper_id}")
        paper = fetcher.download_pdf(paper)
        parsed = parser.parse(paper)
        parsed = vision.process(parsed)
        output_path = DATA_PROCESSED / f"{paper.paper_id}.json"
        with open(output_path, "w") as f:
            json.dump(parsed, f, indent=2)
        logger.info(f"Ingestion complete: {paper.paper_id}")
        return parsed
    except ArxivLensException:
        raise
    except Exception as e:
        raise ArxivLensException(f"Ingestion failed for {paper.paper_id}: {e}")
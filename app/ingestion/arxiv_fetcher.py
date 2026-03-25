import arxiv
from pathlib import Path
from app.core.logger import logger
from app.core.exceptions import ArxivFetchError
from app.core.settings import settings
from app.models.schemas import Paper
import time

DATA_RAW = Path("data/raw")


class ArxivFetcher:

    def __init__(self):
        self.client = arxiv.Client()
        logger.info("ArxivFetcher initialized")

    def search_papers(self, query: str, max_results: int = 5) -> list[Paper]:
        logger.info(f"Searching ArXiv for: {query}")
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            papers = []
            for result in self.client.results(search):
                # ABSTRACT STORED AS-IS (no summarization)
                paper = Paper(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary, 
                    published=str(result.published.date())
                )
 
                papers.append(paper)
                logger.info(f"Found: {paper.title[:60]}...")
            logger.info(f"Total papers found: {len(papers)}")
            return papers
        except Exception as e:
            raise ArxivFetchError(f"Search failed for query '{query}': {e}")

    def download_pdf(self, paper: Paper, max_retries: int = 3) -> Paper:
        logger.info(f"Attempting download for: {paper.paper_id}")
        pdf_path = DATA_RAW / f"{paper.paper_id}.pdf"
        
        for attempt in range(max_retries):
            try:
                search = arxiv.Search(id_list=[paper.paper_id])
                result = next(self.client.results(search))
                
                result.download_pdf(
                    dirpath=str(DATA_RAW),
                    filename=f"{paper.paper_id}.pdf"
                )
                
                paper.pdf_path = str(pdf_path)
                logger.info(f"PDF saved successfully on attempt {attempt + 1}")
                return paper

            except Exception as e:
                if "429" in str(e):
                    wait_time = (attempt + 1) * 5  # Wait 5s, then 10s...
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Non-429 error during download: {e}")
                    raise ArxivFetchError(f"Download failed: {e}")

        raise ArxivFetchError("Download failed after multiple retries due to ArXiv rate limits.")

    def fetch_by_id(self, paper_id: str) -> Paper:
        logger.info(f"Fetching paper by ID: {paper_id}")
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(self.client.results(search))
            # ABSTRACT STORED AS-IS (no summarization)
            paper = Paper(
                paper_id=paper_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,  # Raw abstract from arXiv
                published=str(result.published.date())
            )

            logger.info(f"Fetched: {paper.title[:60]}...")
            return paper
        except Exception as e:
            raise ArxivFetchError(f"Fetch by ID failed for {paper_id}: {e}")
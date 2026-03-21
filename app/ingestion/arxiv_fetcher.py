import arxiv
import requests
from pathlib import Path
from app.core.logger import logger
from app.core.exceptions import ArxivFetchError
from app.core.settings import settings
from app.models.schemas import Paper


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
                paper = Paper(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    published=str(result.published.date())
                )
                paper.summary = self.summarize_abstract(result.summary)
                papers.append(paper)
                logger.info(f"Found: {paper.title[:60]}...")
            logger.info(f"Total papers found: {len(papers)}")
            return papers
        except Exception as e:
            raise ArxivFetchError(f"Search failed for query '{query}': {e}")

    def download_pdf(self, paper: Paper) -> Paper:
        logger.info(f"Downloading PDF for: {paper.paper_id}")
        try:
            search = arxiv.Search(id_list=[paper.paper_id])
            result = next(self.client.results(search))
            pdf_path = DATA_RAW / f"{paper.paper_id}.pdf"
            result.download_pdf(
                dirpath=str(DATA_RAW),
                filename=f"{paper.paper_id}.pdf"
            )
            paper.pdf_path = str(pdf_path)
            logger.info(f"PDF saved to: {pdf_path}")
            return paper
        except Exception as e:
            raise ArxivFetchError(f"Download failed for {paper.paper_id}: {e}")

    def fetch_by_id(self, paper_id: str) -> Paper:
        logger.info(f"Fetching paper by ID: {paper_id}")
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(self.client.results(search))
            paper = Paper(
                paper_id=paper_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=str(result.published.date())
            )
            paper.summary = self.summarize_abstract(result.summary)
            logger.info(f"Fetched: {paper.title[:60]}...")
            return paper
        except Exception as e:
            raise ArxivFetchError(f"Fetch by ID failed for {paper_id}: {e}")

    def summarize_abstract(self, abstract: str) -> str:
        try:
            response = requests.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_summarizer_model,
                    "prompt": f"""Summarize this abstract in one line of 10 words as a maximum for you :

                    {abstract}""",
                    "stream": False
                    },
                    timeout=60
                )
            if response.status_code == 200:
                summary = response.json()["response"].strip()
                logger.info("Abstract summarized successfully")
                return summary
            else:
                logger.warning(f"Ollama error: {response.status_code} — {response.text}")
                return abstract
        except requests.exceptions.Timeout:
            logger.warning("Ollama timeout — using raw abstract")
            return abstract
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e} — using raw abstract")
            return abstract
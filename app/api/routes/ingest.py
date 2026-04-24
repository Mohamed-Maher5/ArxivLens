from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.indexing import index_paper
from app.ingestion import ingest_paper
from app.ingestion.arxiv_fetcher import ArxivFetcher


router = APIRouter(tags=['ingest'])


class IngestRequest(BaseModel):
    arxiv_id: str | None = None
    query: str | None = None


@router.post('/ingest')
def ingest_route(payload: IngestRequest) -> dict:
    try:
        fetcher = ArxivFetcher()
        if payload.arxiv_id:
            paper = fetcher.fetch_by_id(payload.arxiv_id)
        elif payload.query:
            papers = fetcher.search_papers(payload.query, max_results=1)
            if not papers:
                raise ValueError('No papers found for the provided query.')
            paper = papers[0]
        else:
            raise ValueError('Provide either arxiv_id or query.')

        parsed = ingest_paper(paper)
        chunks = index_paper(parsed)

        return {
            'status': 'completed',
            'arxiv_id': parsed['arxiv_id'],
            'paper': paper.model_dump(),
            'chunk_count': len(chunks),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

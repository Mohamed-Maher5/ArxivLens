import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from qdrant_client import QdrantClient

from app.core.settings import settings
from app.ingestion.arxiv_fetcher import ArxivFetcher
from app.models.schemas import Paper


router = APIRouter(prefix='/papers', tags=['papers'])

DATA_PROCESSED = Path('data/processed')


def _paper_from_json(file_path: Path) -> Paper | None:
    try:
        with file_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        return Paper(
            arxiv_id=data['arxiv_id'],
            title=data.get('title', 'Unknown title'),
            authors=data.get('authors', []),
            abstract=data.get('abstract', ''),
            published=data.get('published', ''),
            processed=True,
        )
    except Exception:
        return None


def _load_processed_papers() -> list[Paper]:
    papers: list[Paper] = []
    if not DATA_PROCESSED.exists():
        return papers

    for file_path in sorted(DATA_PROCESSED.glob('*.json'), reverse=True):
        paper = _paper_from_json(file_path)
        if paper is not None:
            papers.append(paper)
    return papers


def _load_qdrant_papers() -> list[Paper]:
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collections = client.get_collections().collections
    except Exception:
        return []

    papers: list[Paper] = []
    for collection in collections:
        name = collection.name
        if not name.startswith('paper_'):
            continue

        arxiv_id = name.removeprefix('paper_').replace('_', '.')
        processed_path = DATA_PROCESSED / f'{arxiv_id}.json'
        paper = _paper_from_json(processed_path) if processed_path.exists() else None
        if paper is not None:
            papers.append(paper)
            continue

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=arxiv_id,
                authors=[],
                abstract='',
                published='',
                processed=True,
            )
        )

    return papers


@router.get('')
def list_papers(
    q: str | None = Query(default=None),
    max_results: int = Query(default=5, ge=1, le=20),
) -> list[dict]:
    if q:
        papers = ArxivFetcher().search_papers(q, max_results=max_results)
    else:
        papers = _load_qdrant_papers() or _load_processed_papers()
    return [paper.model_dump() for paper in papers]


@router.get('/{arxiv_id}')
def get_paper(arxiv_id: str) -> dict:
    processed_path = DATA_PROCESSED / f'{arxiv_id}.json'
    if processed_path.exists():
        with processed_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        data['processed'] = True
        return data

    try:
        paper = ArxivFetcher().fetch_by_id(arxiv_id)
        return paper.model_dump()
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

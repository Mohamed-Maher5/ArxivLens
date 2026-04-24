from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.dependencies import get_llm_client
from app.generation.pipeline import Pipeline


router = APIRouter(tags=['chat'])


class ChatRequest(BaseModel):
    message: str
    arxiv_id: str | None = None


@router.post('/chat')
def chat_route(payload: ChatRequest) -> dict:
    try:
        llm = get_llm_client()
        result_obj = Pipeline(llm_client=llm).run(payload.message, arxiv_id=payload.arxiv_id)
        return {
            'question': result_obj.question,
            'answer': result_obj.answer,
            'contextualized_query': result_obj.contextualized_query,
            'sources': [
                {
                    'paper_title': source.paper_title,
                    'chunk_type': source.chunk_type,
                    'page_number': source.page_number,
                    'content': source.content[:200],
                }
                for source in result_obj.sources
            ],
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

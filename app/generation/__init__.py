from app.generation.pipeline import Pipeline
from app.models.schemas import Message


def run_pipeline(question: str, history: list = None) -> dict:
    pipeline = Pipeline()
    result = pipeline.run(question, history)
    return {
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence,
        "contextualized_query": result.contextualized_query,
        "sources": [
            {
                "paper_title": s.paper_title,
                "chunk_type": s.chunk_type,
                "page_number": s.page_number,
                "content": s.content[:200]
            }
            for s in result.sources
        ]
    }
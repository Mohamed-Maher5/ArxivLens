# /mnt/hdd/projects/ArxivLens/app/generation/__init__.py
from app.generation.pipeline import Pipeline
from app.models.schemas import Message
from app.core.logger import logger


def run_pipeline(question: str, history: list = None, paper_id: str = None) -> dict:
    """
    Module to run the generation pipeline and return structured results.
    
    Args:
        question: The user's question
        history: Optional list of Message objects for conversation history
        paper_id: Optional specific paper ID to scope the search
    """
    if history is None:
        history = []
    
    try:
        pipeline = Pipeline()
        result = pipeline.run(question, history, paper_id)
        
        return {
            "question": result.question,
            "answer": result.answer,
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
        
    except Exception as e:
        logger.error(f"[RUN_PIPELINE] Pipeline execution failed: {e}")
        return {
            "question": question,
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "contextualized_query": question,
            "sources": []
        }
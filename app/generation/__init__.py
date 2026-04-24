from app.core.logger import logger


def run_pipeline(question: str, arxiv_id: str | None = None) -> dict:
    """Run the generation pipeline and return structured results."""
    try:
        from app.generation.pipeline import Pipeline

        pipeline = Pipeline()
        result = pipeline.run(question, arxiv_id)
        return {
            'question': result.question,
            'answer': result.answer,
            'contextualized_query': result.contextualized_query,
            'sources': [
                {
                    'paper_title': source.paper_title,
                    'chunk_type': source.chunk_type,
                    'page_number': source.page_number,
                    'content': source.content[:200],
                }
                for source in result.sources
            ],
        }
    except Exception as error:
        logger.error(f'[RUN_PIPELINE] Pipeline execution failed: {error}')
        return {
            'question': question,
            'answer': f'Sorry, I encountered an error: {error}',
            'contextualized_query': question,
            'sources': [],
        }

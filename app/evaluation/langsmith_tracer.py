"""Simple LangSmith tracing wrapper."""

import os
from contextlib import contextmanager
from langsmith import Client, traceable
from app.core.logger import logger


# Initialize LangSmith client
langsmith_client = Client()


@contextmanager
def trace_pipeline_run(run_name: str = "pipeline_run"):
    """Simple context manager to trace a pipeline run."""
    try:
        logger.info(f"[LANGSMITH] Starting trace: {run_name}")
        yield langsmith_client
    except Exception as e:
        logger.warning(f"[LANGSMITH] Tracing error: {e}")
        yield None


def trace_step(step_name: str):
    """Decorator to trace individual pipeline steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Log to LangSmith if available
                if os.getenv("LANGCHAIN_TRACING_V2") == "true":
                    with traceable(run_type="chain", name=step_name):
                        result = func(*args, **kwargs)
                        return result
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[LANGSMITH] Step tracing failed: {e}")
                return func(*args, **kwargs)
        return wrapper
    return decorator
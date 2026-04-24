import app.generation.prompts as prompts
from app.generation.llm_client import LLMClient


class _FakeGemmaModels:
    def generate_content(self, **kwargs):
        class _Response:
            text = "8.5"

        return _Response()


class _FakeGemmaClient:
    def __init__(self):
        self.models = _FakeGemmaModels()


def test_rerank_prompt_exists():
    assert hasattr(prompts, "RERANK_PROMPT")
    rendered = prompts.RERANK_PROMPT.format(query="q", content="c")
    assert "0 to 10" in rendered


def test_score_chunk_parses_numeric_response():
    client = LLMClient(gemma_client=_FakeGemmaClient())

    score = client.score_chunk(
        "What is the text classification method?",
        "This chunk describes the paper's short text classification method.",
    )

    assert score == 8.5

from app.generation.pipeline import Pipeline


class _FakeLLM:
    def __init__(self, *, intent="task", contextualized="contextualized question"):
        self.intent = intent
        self.contextualized = contextualized

    def classify_intent(self, message: str) -> str:
        return self.intent

    def contextualize_query(self, question: str, max_tokens: int = 100) -> str:
        return self.contextualized

    def score_chunk(self, query: str, content: str) -> float:
        return 9.0

    def generate_chat_response(self, message: str) -> str:
        return f"chat:{message}"

    def generate_paper_answer(self, question: str, chunks: list[dict]) -> dict:
        return {"answer": f"paper:{question}", "sources": chunks}

    def generate_general_knowledge(self, question: str, metadata: str) -> dict:
        return {"answer": f"general:{question}|{metadata}"}

    def describe_image(self, image_b64: str, caption: str = "") -> str:
        return "image"

    def generate_from_paper_top3(self, question: str, chunks: list[dict]) -> dict:
        return self.generate_paper_answer(question, chunks)


class _FakeResolver:
    def __init__(self, collection="paper_1", arxiv_id="1706.03762", metadata="paper metadata"):
        self.collection = collection
        self.arxiv_id = arxiv_id
        self.metadata = metadata

    def resolve(self, arxiv_id=None):
        return self.collection, self.arxiv_id

    def get_paper_metadata(self, arxiv_id: str) -> str:
        return self.metadata

    def infer_arxiv_id_from_disk(self):
        return self.arxiv_id


class _PassthroughReranker:
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        return chunks


def test_pipeline_handles_chat_without_retrieval():
    pipeline = Pipeline(
        llm_client=_FakeLLM(intent="chat"),
        reranker=_PassthroughReranker(),
        collection_resolver=_FakeResolver(collection=None, arxiv_id=None, metadata=""),
        retrieve_fn=lambda *_: [],
    )

    result = pipeline.run("hello there")

    assert result.answer == "chat:hello there"
    assert result.sources == []


def test_pipeline_returns_paper_grounded_answer_when_chunks_exist():
    chunk = {
        "chunk_id": "c1",
        "arxiv_id": "1706.03762",
        "paper_title": "Attention Is All You Need",
        "authors": ["A. Vaswani"],
        "content": "The model uses multi-head attention.",
        "chunk_type": "content",
        "page_number": 4,
    }
    pipeline = Pipeline(
        llm_client=_FakeLLM(),
        reranker=_PassthroughReranker(),
        collection_resolver=_FakeResolver(),
        retrieve_fn=lambda *_: [chunk],
    )

    result = pipeline.run("How does attention work?", arxiv_id="1706.03762")

    assert result.answer == "paper:contextualized question"
    assert len(result.sources) == 1
    assert result.sources[0].paper_title == "Attention Is All You Need"


def test_pipeline_falls_back_to_general_knowledge_without_chunks():
    pipeline = Pipeline(
        llm_client=_FakeLLM(),
        reranker=_PassthroughReranker(),
        collection_resolver=_FakeResolver(metadata="transformer paper"),
        retrieve_fn=lambda *_: [],
    )

    result = pipeline.run("Summarize the paper", arxiv_id="1706.03762")

    assert result.answer == "general:contextualized question|transformer paper"

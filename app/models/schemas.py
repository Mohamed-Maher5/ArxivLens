from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_path: Optional[str] = None
    processed: bool = False

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    content: str
    chunk_type: str
    page_number: Optional[int] = None
    figure_description: Optional[str] = None

class QueryResult(BaseModel):
    question: str
    answer: str
    sources: list[Chunk]
    confidence: str
    contextualized_query: str
    timestamp: datetime = Field(default_factory=datetime.now)

class EvalResult(BaseModel):
    question: str
    answer: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
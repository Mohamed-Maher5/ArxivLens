import os
import re
from typing import Any

import requests
from langchain_core.prompts import ChatPromptTemplate

import app.generation.prompts as prompts
from app.core.logger import logger
from app.core.settings import settings
from app.interfaces.llm_provider import LLMProvider


class GemmaClient(LLMProvider):
    """Unified text+vision LLM client with a stub-friendly Gemma fallback."""

    def __init__(self, gemma_client: Any = None):
        self.api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self.client = gemma_client or self._build_gemma_client()
        logger.info("GemmaClient initialized")

    def _build_gemma_client(self):
        if not self.api_key or self.api_key.lower() == "dummy":
            return None
        return object()

    def _model_url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemma_model}:generateContent?key={self.api_key}"
        )

    def _prompt_to_text(self, prompt_template: ChatPromptTemplate, variables: dict) -> str:
        messages = prompt_template.format_messages(**variables)
        return "\n\n".join(
            f"{'SYSTEM' if msg.type == 'system' else 'USER'}:\n{msg.content}"
            for msg in messages
        )

    def _extract_first_number(self, text: str) -> float:
        match = re.search(r"\d+(\.\d+)?", text)
        score = float(match.group()) if match else 0.0
        return min(10.0, max(0.0, score))

    def _generate_text(
        self,
        prompt_template: ChatPromptTemplate,
        variables: dict,
        *,
        max_tokens: int = 1024,
        fallback: str = "",
    ) -> str:
        prompt_text = self._prompt_to_text(prompt_template, variables)

        if self.client is None:
            return fallback

        try:
            response = requests.post(
                self._model_url(),
                json={
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": max_tokens,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return self._clean(text) or fallback
        except Exception as error:
            logger.warning(f"[GEMMA] Text generation failed, using fallback: {error}")
            return fallback

    def _clean(self, text: str) -> str:
        return re.sub(r"thinking.*?/thinking", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _stub_classify_intent(self, message: str) -> str:
        lowered = message.lower().strip()
        task_markers = (
            "what",
            "who",
            "how",
            "why",
            "explain",
            "summarize",
            "method",
            "paper",
            "result",
            "dataset",
            "classification",
            "?",
        )
        if any(marker in lowered for marker in task_markers):
            return "task"
        return "chat"

    def _stub_score_chunk(self, query: str, content: str) -> float:
        query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
        content_terms = set(re.findall(r"[a-z0-9]+", content.lower()))
        if not query_terms:
            return 0.0
        overlap = len(query_terms & content_terms) / len(query_terms)
        return min(10.0, round(overlap * 10, 1))

    def classify_intent(self, message: str) -> str:
        if self.client is None:
            return self._stub_classify_intent(message)

        result = self._generate_text(
            prompts.INTENT_PROMPT,
            {"message": message},
            max_tokens=5,
            fallback="TASK",
        )
        first_word = result.upper().split()[0] if result.split() else "TASK"
        return "chat" if first_word == "CHAT" else "task"

    def contextualize_query(self, question: str, max_tokens: int = 100) -> str:
        return self._generate_text(
            prompts.CONTEXTUALIZATION_PROMPT,
            {"question": question},
            max_tokens=max_tokens,
            fallback=question,
        )

    def generate_chat_response(self, message: str) -> str:
        return self._generate_text(
            prompts.CHAT_PROMPT,
            {"message": message},
            max_tokens=256,
            fallback="Hi! I'm ArxivLens. Ask me about a paper.",
        )

    def generate_paper_answer(self, question: str, chunks: list[dict]) -> dict:
        formatted_chunks = self._format_chunks_for_prompt(chunks[:3])
        answer = self._generate_text(
            prompts.PAPER_ANSWER_TOP3_PROMPT,
            {"question": question, "chunks": formatted_chunks},
            max_tokens=1024,
            fallback="I couldn't generate an answer from the paper chunks.",
        )
        return {"answer": answer, "source": "paper"}

    def generate_general_knowledge(self, question: str, metadata: str) -> dict:
        answer = self._generate_text(
            prompts.GENERAL_KNOWLEDGE_PROMPT,
            {"metadata": metadata, "question": question},
            max_tokens=1024,
            fallback="I don't have specific information to answer this question.",
        )
        return {"answer": answer, "source": "general_knowledge"}

    def score_chunk(self, query: str, content: str) -> float:
        if self.client is None:
            return self._stub_score_chunk(query, content)

        result = self._generate_text(
            prompts.RERANK_PROMPT,
            {"query": query, "content": content[:400]},
            max_tokens=8,
            fallback="0",
        )
        return self._extract_first_number(result)

    def describe_image(self, image_b64: str, caption: str = "") -> str:
        fallback = "Figure description unavailable."
        caption_text = caption or "No caption available."
        prompt = (
            "You are analyzing a figure from an academic research paper.\n"
            f"Caption: {caption_text}\n"
            "Describe this figure in detail including:\n"
            "- What type of figure it is\n"
            "- What data or information it shows\n"
            "- Key values, trends, or findings visible\n"
            "- Any labels, axes, or legends present\n"
            "Be precise and technical."
        )

        if self.client is None:
            return fallback

        try:
            response = requests.post(
                self._model_url(),
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": image_b64,
                                    }
                                },
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 512,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return self._clean(text) or fallback
        except Exception as error:
            logger.warning(f"[GEMMA] Vision generation failed, using fallback: {error}")
            return fallback

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        formatted: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            title = chunk.get("paper_title", "Unknown paper")
            page = chunk.get("page_number", "?")
            content = chunk.get("content", "")
            formatted.append(f"[Chunk {index}] {title} | p.{page}\n{content}")
        return "\n\n".join(formatted)

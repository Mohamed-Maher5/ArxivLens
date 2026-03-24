import re
import requests
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger import logger
from app.core.settings import settings
from app.generation import prompts


class AdaptiveRAG:

    def __init__(self):
        self.groq = Groq(api_key=settings.groq_api_key)
        self.hf = InferenceClient(api_key=settings.huggingface_api_key)
        logger.info("AdaptiveRAG initialized")

    # ── API Helpers ─────────────────────────────────────────

    def _groq(self, system: str, user: str, max_tokens: int = 20) -> str:
        response = self.groq.chat.completions.create(
            model=settings.groq_classifier_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = requests.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            # ADD THIS LINE:
            logger.warning(f"[OLLAMA] Non-200 status: {response.status_code}")
            return ""
        except Exception as e:
            # CHANGE THIS LINE to include error details:
            logger.warning(f"[OLLAMA] Error: {e}")
            return ""

    def _hf_with_prompt(self, prompt_template: ChatPromptTemplate, variables: dict, max_tokens: int = 1024) -> str:
        messages = prompt_template.format_messages(**variables)

        hf_messages = []
        for msg in messages:
            role = "system" if msg.type == "system" else "user"
            hf_messages.append({"role": role, "content": msg.content})

        response = self.hf.chat.completions.create(
            model=settings.hf_model,
            messages=hf_messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._clean(raw)

    # ── Intent Classification ─────────────────────────

    def classify_intent(self, message: str) -> str:
        try:
            messages = prompts.INTENT_PROMPT.format_messages(message=message)

            system_msg = ""
            user_msg = ""
            for msg in messages:
                if msg.type == "system":
                    system_msg = msg.content
                elif msg.type == "human":
                    user_msg = msg.content

            result = self._groq(system_msg, user_msg, max_tokens=5)
            first_word = result.upper().split()[0] if result.split() else "TASK"
            return "chat" if first_word == "CHAT" else "task"

        except Exception:
            return "task"

    # ── Reranker Scoring ─────────────────────────

    def score_chunk(self, query: str, content: str) -> float:
        try:
            user_content = (
                f"Query: {query}\n"
                f"Chunk: {content[:400]}\n\n"
                f"Score from 0 to 10:"
            )

            response = self.groq.chat.completions.create(
                model=settings.groq_classifier_model,
                messages=[
                    {"role": "system", "content": prompts.RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=5,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()
            match = re.search(r"\d+(\.\d+)?", raw)
            return float(match.group()) if match else 0.0

        except Exception:
            return 0.0

    # ── Generation ─────────────────────────

    def generate_chat(self, message: str) -> str:
        try:
            return self._hf_with_prompt(
                prompts.CHAT_PROMPT,
                {"message": message},
                max_tokens=256
            )
        except Exception:
            return "Hi! I'm ArxivLens. Ask me about a paper."

    def generate_from_paper(self, question: str, context: str) -> dict:
        try:
            answer = self._hf_with_prompt(
                prompts.PAPER_ANSWER_PROMPT,
                {"context": context, "question": question},
                max_tokens=1024
            )
            return {
                "answer": answer,
                "confidence": self._parse_confidence(answer),
                "source": "paper"
            }
        except Exception:
            return {"answer": "Error", "confidence": "LOW", "source": "paper"}

    def generate_from_paper_with_history(self, question: str, context: str, history_text: str) -> dict:
        try:
            answer = self._hf_with_prompt(
                prompts.PAPER_ANSWER_WITH_HISTORY_PROMPT,
                {
                    "context": context,
                    "question": question,
                    "history": history_text
                },
                max_tokens=1024
            )
            return {
                "answer": answer,
                "confidence": self._parse_confidence(answer),
                "source": "paper"
            }
        except Exception:
            return {"answer": "Error", "confidence": "LOW", "source": "paper"}

    # ── Helpers ─────────────────────────

    def _clean(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_confidence(self, answer: str) -> str:
        match = re.search(r"(HIGH|MEDIUM|LOW)", answer, re.IGNORECASE)
        return match.group(1).upper() if match else "MEDIUM"
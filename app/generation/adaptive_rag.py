import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from huggingface_hub import InferenceClient
from app.core.logger import logger
from app.core.settings import settings


class AdaptiveRAG:

    def __init__(self):
        # Groq — intent classification, HyDE, reranker scoring
        self.groq = Groq(api_key=settings.groq_api_key)
        # HuggingFace — final answer generation only
        self.hf = InferenceClient(api_key=settings.huggingface_api_key)
        logger.info("AdaptiveRAG initialized")

    # ── Groq helper ───────────────────────────────────────────────────────────

    def _groq(self, system: str, user: str, max_tokens: int = 20) -> str:
        """
        Call Groq llama-3.1-8b-instant.
        Used for: intent classification, HyDE, reranker scoring.
        Fast and cheap — ideal for short outputs.
        """
        response = self.groq.chat.completions.create(
            model=settings.groq_classifier_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    # ── Ollama helper ─────────────────────────────────────────────────────────

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Call phi3 via Ollama local server.
        Used for: query contextualization, history summarization.
        Free — no API quota consumed.
        Falls back to empty string on any failure.
        """
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
            logger.warning(f"Ollama error {response.status_code}: {response.text}")
            return ""
        except requests.exceptions.Timeout:
            logger.warning("Ollama timeout")
            return ""
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e}")
            return ""

    # ── HuggingFace helper ────────────────────────────────────────────────────

    def _hf(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """
        Call Qwen/Qwen3-8B via HuggingFace Inference API.
        Used for: final answer generation only (generic + paper-based + chat).
        Strips Qwen3 <think>...</think> blocks before returning.
        """
        response = self.hf.chat.completions.create(
            model=settings.hf_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._clean(raw)

    # ── Intent classification (Groq) ──────────────────────────────────────────

    def classify_intent(self, message: str) -> str:
        """
        Classify user message as CHAT or TASK using Groq.
        Outputs only 1 word — very cheap.
        Returns 'chat' or 'task'.
        """
        try:
            logger.debug(f"[INTENT] Input: {message[:80]}")
            system = """You are an intent classifier for an academic paper QA system.
Classify the user message as either CHAT or TASK.
Output ONLY one word — either CHAT or TASK.

- CHAT → casual conversation, greetings, small talk, opinions, feelings
- TASK → questions about a paper, research, science, methodology, results, authors

Examples:
"hi how are you" → CHAT
"who are the authors?" → TASK
"what are the limitations?" → TASK
"thanks!" → CHAT
"what is machine learning?" → TASK"""

            result = self._groq(system, message, max_tokens=5)
            logger.debug(f"[INTENT] Raw output: '{result}'")
            first_word = result.upper().split()[0] if result.split() else "TASK"
            intent = "chat" if first_word == "CHAT" else "task"
            logger.info(f"[INTENT] Classified as: {intent.upper()} (model: {settings.groq_classifier_model})")
            return intent
        except Exception as e:
            logger.warning(f"[INTENT] Failed: {e} — defaulting to TASK")
            return "task"

    # ── HyDE (Groq) ───────────────────────────────────────────────────────────

    def generate_hypothetical(self, question: str) -> str:
        """
        HyDE via Groq: Generate a hypothetical academic passage
        that would answer the question. Embed this instead of the
        raw question for better semantic search in Qdrant.
        Falls back to original question on failure.
        """
        try:
            logger.debug(f"[HYDE] Input question: {question[:80]}")
            system = """You are a research paper assistant.
Write a short hypothetical passage (3-5 sentences) that looks like it comes
from an academic paper and directly answers the given question.
Use academic language with specific technical details.
Return ONLY the passage — no preamble, no explanation."""

            result = self._groq(
                system,
                f"Question: {question}\n\nHypothetical passage:",
                max_tokens=150
            )
            if not result:
                logger.warning(f"[HYDE] Empty result — falling back to original question")
                return question
            logger.info(f"[HYDE] Generated (model: {settings.groq_classifier_model}): {result[:120]}...")
            return result
        except Exception as e:
            logger.warning(f"[HYDE] Failed: {e} — falling back to original question")
            return question

    # ── Reranker scoring (Groq) ───────────────────────────────────────────────

    def score_chunk(self, query: str, content: str) -> float:
        """
        Score relevance of a single chunk to the query using Groq.
        Returns float 0.0–10.0.
        """
        try:
            response = self.groq.chat.completions.create(
                model=settings.groq_classifier_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Score the relevance of this chunk to the query.\n"
                        f"Query: {query}\n"
                        f"Chunk: {content[:400]}\n\n"
                        f"Respond with ONLY a single number between 0 and 10."
                    )
                }],
                max_tokens=5,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r"\d+(\.\d+)?", raw)
            return min(10.0, max(0.0, float(match.group()))) if match else 0.0
        except Exception:
            return 0.0

    # ── Generation (HuggingFace Qwen3-8B) ────────────────────────────────────

    def generate_chat(self, message: str) -> str:
        """
        Casual conversational response via Qwen3-8B.
        CHAT intent only — no retrieval context.
        """
        try:
            logger.debug(f"[CHAT] Input: {message[:80]}")
            system = """You are a friendly and helpful research assistant called ArxivLens.
You help researchers explore and understand academic papers.
When users chat casually, respond naturally and warmly.
Keep responses concise and conversational."""

            answer = self._hf(system, message, max_tokens=256)
            logger.info(f"[CHAT] Generated (model: {settings.hf_model}): {answer[:120]}...")
            return answer
        except Exception as e:
            logger.error(f"[CHAT] Failed: {e}")
            return "Hi! I'm ArxivLens, here to help you explore academic papers. What would you like to know?"

    def generate_generic(self, question: str) -> dict:
        """
        Answer from general model knowledge via Qwen3-8B.
        Used when best retrieval score < threshold.
        Always includes disclaimer — NOT from indexed papers.
        """
        logger.info(f"[GENERIC] Generating for: {question[:80]}")
        try:
            system = """You are a knowledgeable research assistant.
The question is NOT covered by the indexed academic papers.
Answer from your general knowledge but be honest about it.

Rules:
1. Start with: "Note: This answer is based on general knowledge, not the indexed papers."
2. Answer accurately and helpfully
3. Never pretend the answer comes from a specific indexed paper

End every response with exactly these two lines:
**Confidence:** MEDIUM
**Reason:** Answer based on general knowledge, not from indexed papers."""

            answer = self._hf(system, f"Question: {question}", max_tokens=1024)
            confidence = self._parse_confidence(answer)
            logger.info(f"[GENERIC] Done (model: {settings.hf_model}) | Confidence: {confidence}")
            logger.debug(f"[GENERIC] Answer preview: {answer[:200]}")
            return {"answer": answer, "confidence": confidence, "source": "generic"}
        except Exception as e:
            logger.error(f"[GENERIC] Failed: {e}")
            return {
                "answer": "I couldn't generate an answer at this time.",
                "confidence": "LOW",
                "source": "generic"
            }

    def generate_from_paper(self, question: str, context: str) -> dict:
        """
        Answer strictly from indexed paper context via Qwen3-8B.
        Used when best retrieval score >= threshold.
        Cites paper title and page for every claim.
        """
        logger.info(f"[PAPER] Generating for: {question[:80]}")
        logger.debug(f"[PAPER] Context preview: {context[:300]}")
        try:
            system = """You are an expert research assistant helping users understand academic papers.

Rules:
1. Answer based ONLY on the provided paper context
2. Cite every claim with [paper title, page N]
3. Never hallucinate or add information not in context
4. If context is insufficient say exactly what is missing
5. If the question asks about figures or charts describe what the figure shows

End every response with exactly these two lines:
**Confidence:** HIGH
**Reason:** [one sentence explaining why]

Where confidence is HIGH / MEDIUM / LOW."""

            user = f"Paper context:\n{context}\n\nQuestion: {question}"
            answer = self._hf(system, user, max_tokens=1024)
            confidence = self._parse_confidence(answer)
            logger.info(f"[PAPER] Done (model: {settings.hf_model}) | Confidence: {confidence}")
            logger.debug(f"[PAPER] Answer preview: {answer[:200]}")
            return {"answer": answer, "confidence": confidence, "source": "paper"}
        except Exception as e:
            logger.error(f"[PAPER] Failed: {e}")
            return {"answer": "INADEQUATE", "confidence": "LOW", "source": "paper"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Strip <think>...</think> blocks Qwen3 emits in thinking mode."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_confidence(self, answer: str) -> str:
        """
        Extract confidence from structured field:
            **Confidence:** HIGH / MEDIUM / LOW
        Tries 3 patterns. Only searches last 3 lines for fallback
        to avoid false positives in the answer body.
        """
        match = re.search(
            r"\*{0,2}Confidence\*{0,2}\s*:\s*(HIGH|MEDIUM|LOW)",
            answer, re.IGNORECASE
        )
        if match:
            level = match.group(1).upper()
            logger.info(f"Parsed confidence: {level}")
            return level

        for line in answer.splitlines():
            if re.match(r"^(HIGH|MEDIUM|LOW)$", line.strip(), re.IGNORECASE):
                level = line.strip().upper()
                logger.info(f"Parsed confidence from standalone line: {level}")
                return level

        last_lines = "\n".join(answer.splitlines()[-3:])
        match = re.search(r"\b(HIGH|MEDIUM|LOW)\b", last_lines, re.IGNORECASE)
        if match:
            level = match.group(1).upper()
            logger.info(f"Parsed confidence from answer tail: {level}")
            return level

        logger.warning("Confidence field not found — defaulting to HIGH")
        return "HIGH"
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger import logger
from app.core.settings import settings
from app.generation import prompts


class AdaptiveRAG:

    def __init__(self):
        # Groq — intent classification, reranker scoring
        self.groq = Groq(api_key=settings.groq_api_key)
        # HuggingFace — final answer generation only
        self.hf = InferenceClient(api_key=settings.huggingface_api_key)
        logger.info("AdaptiveRAG initialized")

    # ── API Helpers ───────────────────────────────────────────────────────────

    def _groq(self, system: str, user: str, max_tokens: int = 20) -> str:
        """Call Groq llama-3.1-8b-instant. Fast and cheap for short outputs."""
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

    def _ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """Call phi3 via Ollama local server. Free - no API quota."""
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

    def _hf_with_prompt(self, prompt_template: ChatPromptTemplate, variables: dict, max_tokens: int = 1024) -> str:
        """Generate using HuggingFace with a langchain prompt template."""
        # Format the prompt
        messages = prompt_template.format_messages(**variables)
        
        # Convert to HF format
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

    # ── Web Search ────────────────────────────────────────────────────────────

    def search_web(self, query: str, paper_titles: list[str]) -> str:
        """
        Search web using SerpAPI when paper doesn't cover the topic.
        Includes paper names in search for context.
        """
        if not settings.serpapi_api_key:
            logger.warning("No SERPAPI key configured, skipping web search")
            return "Web search not available - no API key configured."
        
        # Construct search query with paper names for context
        paper_context = " ".join([f'"{title}"' for title in paper_titles[:3]])
        search_query = f"{query} {paper_context}"
        
        logger.info(f"[WEB SEARCH] Query: {search_query[:100]}...")
        
        try:
            params = {
                "q": search_query,
                "api_key": settings.serpapi_api_key,
                "engine": "google",
                "num": 5
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            data = response.json()
            
            # Extract organic results
            results = []
            for result in data.get("organic_results", [])[:3]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                results.append(f"Source: {title}\nURL: {link}\nSummary: {snippet}")
            
            web_text = "\n\n".join(results) if results else "No relevant web results found."
            logger.info(f"[WEB SEARCH] Found {len(results)} results")
            return web_text
            
        except Exception as e:
            logger.error(f"[WEB SEARCH] Failed: {e}")
            return "Web search unavailable due to error."

    # ── Intent Classification ─────────────────────────────────────────────────

    def classify_intent(self, message: str) -> str:
        """Classify as CHAT or TASK using Groq."""
        try:
            logger.debug(f"[INTENT] Input: {message[:80]}")
            
            # Use format_messages() to get message objects with .type attribute
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
            intent = "chat" if first_word == "CHAT" else "task"
            logger.info(f"[INTENT] Classified as: {intent.upper()}")
            return intent
        except Exception as e:
            logger.warning(f"[INTENT] Failed: {e} — defaulting to TASK")
            return "task"

    # ── Reranker Scoring ─────────────────────────────────────────────────────

    def score_chunk(self, query: str, content: str) -> float:
        """Score relevance 0-10 using Groq. Called by Reranker."""
        try:
            user_content = (
                f"Score the relevance of this chunk to the query.\n"
                f"Query: {query}\n"
                f"Chunk: {content[:400]}\n\n"
                f"Respond with ONLY a single number between 0 and 10."
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
            return min(10.0, max(0.0, float(match.group()))) if match else 0.0
        except Exception:
            return 0.0

    # ── Generation Methods ────────────────────────────────────────────────────

    def generate_chat(self, message: str) -> str:
        """Casual conversational response via Qwen3-8B."""
        try:
            logger.debug(f"[CHAT] Input: {message[:80]}")
            answer = self._hf_with_prompt(
                prompts.CHAT_PROMPT,
                {"message": message},
                max_tokens=256
            )
            logger.info(f"[CHAT] Generated: {answer[:120]}...")
            return answer
        except Exception as e:
            logger.error(f"[CHAT] Failed: {e}")
            return "Hi! I'm ArxivLens, here to help you explore academic papers. What would you like to know?"


    def generate_from_paper(self, question: str, context: str) -> dict:
        """Answer strictly from indexed paper context (no history)."""
        logger.info(f"[PAPER] Generating for: {question[:80]}")
        logger.debug(f"[PAPER] Context preview: {context[:300]}")
        try:
            answer = self._hf_with_prompt(
                prompts.PAPER_ANSWER_PROMPT,
                {"context": context, "question": question},
                max_tokens=1024
            )
            confidence = self._parse_confidence(answer)
            logger.info(f"[PAPER] Done | Confidence: {confidence}")
            return {"answer": answer, "confidence": confidence, "source": "paper"}
        except Exception as e:
            logger.error(f"[PAPER] Failed: {e}")
            return {"answer": "Error generating answer from papers.", "confidence": "LOW", "source": "paper"}

    def generate_from_paper_with_history(self, question: str, context: str, history_text: str) -> dict:
        """
        Answer from paper context WITH conversation history.
        Used when chunks >= 6.0 after reranking.
        """
        logger.info(f"[PAPER+HISTORY] Generating for: {question[:80]}")
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
            confidence = self._parse_confidence(answer)
            logger.info(f"[PAPER+HISTORY] Done | Confidence: {confidence}")
            return {"answer": answer, "confidence": confidence, "source": "paper"}
        except Exception as e:
            logger.error(f"[PAPER+HISTORY] Failed: {e}")
            return {"answer": "Error generating answer.", "confidence": "LOW", "source": "paper"}

    def generate_web_augmented(self, question: str, web_results: str, paper_titles: list[str]) -> dict:
        """
        Generate answer when paper doesn't cover topic (either < 0.25 or < 6.0).
        Unified fallback for both cases.
        """
        logger.info(f"[WEB] Generating web-augmented answer for: {question[:80]}")
        try:
            answer = self._hf_with_prompt(
                prompts.WEB_AUGMENTED_PROMPT,
                {
                    "question": question,
                    "web_results": web_results,
                    "paper_titles": ", ".join(paper_titles)
                },
                max_tokens=1024
            )
            confidence = self._parse_confidence(answer)
            logger.info(f"[WEB] Done | Confidence: {confidence}")
            return {
                "answer": answer,
                "confidence": confidence,
                "source": "web_search"
            }
        except Exception as e:
            logger.error(f"[WEB] Failed: {e}")
            return {
                "answer": "The indexed papers don't cover this topic, and I couldn't retrieve web search results.",
                "confidence": "LOW",
                "source": "web_search"
            }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Strip <think>...</think> blocks Qwen3 emits."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_confidence(self, answer: str) -> str:
        """Extract confidence from structured field: **Confidence:** HIGH/MEDIUM/LOW"""
        match = re.search(
            r"\*{0,2}Confidence\*{0,2}\s*:\s*(HIGH|MEDIUM|LOW)",
            answer, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        for line in answer.splitlines():
            if re.match(r"^(HIGH|MEDIUM|LOW)$", line.strip(), re.IGNORECASE):
                return line.strip().upper()

        last_lines = "\n".join(answer.splitlines()[-3:])
        match = re.search(r"\b(HIGH|MEDIUM|LOW)\b", last_lines, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        logger.warning("Confidence field not found — defaulting to MEDIUM")
        return "MEDIUM"
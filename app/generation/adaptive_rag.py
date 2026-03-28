# /mnt/hdd/projects/ArxivLens/app/generation/adaptive_rag.py
import re
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger import logger
from app.core.settings import settings
from app.core.exceptions import RetrievalError
from app.generation import prompts


class AdaptiveRAG:

    def __init__(self):
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.hf = InferenceClient(api_key=settings.huggingface_api_key)
        logger.info("AdaptiveRAG initialized")

    # ── API Helpers ─────────────────────────────────────────

    def _call_groq(self, system: str, user: str, max_tokens: int = 20) -> str:
        """Call Groq API for classification/scoring tasks."""
        response = self.groq_client.chat.completions.create(
            model=settings.groq_classifier_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    def _hf_with_prompt(self, prompt_template: ChatPromptTemplate, variables: dict, max_tokens: int = 1024) -> str:
        """Call HuggingFace API with formatted prompt."""
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
        """Classify user message as 'chat' or 'task'."""
        try:
            messages = prompts.INTENT_PROMPT.format_messages(message=message)

            system_msg = ""
            user_msg = ""
            for msg in messages:
                if msg.type == "system":
                    system_msg = msg.content
                elif msg.type == "human":
                    user_msg = msg.content

            result = self._call_groq(system_msg, user_msg, max_tokens=5)
            first_word = result.upper().split()[0] if result.split() else "TASK"
            return "chat" if first_word == "CHAT" else "task"

        except Exception as e:
            logger.warning(f"[INTENT] Classification failed: {e}, defaulting to 'task'")
            return "task"

    # ── Reranker Scoring ─────────────────────────

    def score_chunk(self, query: str, content: str) -> float:
        """Score chunk relevance 0-10 using Groq."""
        try:
            user_content = (
                f"Query: {query}\n"
                f"Chunk: {content[:400]}\n\n"
                f"Score from 0 to 10:"
            )

            response = self.groq_client.chat.completions.create(
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
            score = float(match.group()) if match else 0.0
            return min(10.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"[SCORE] Chunk scoring failed: {e}")
            return 0.0

    # ── Generation with History ─────────────────────────

    def generate_chat_with_history(self, message: str, history_text: str) -> str:
        """Generate casual chat response with conversation history."""
        try:
            return self._hf_with_prompt(
                prompts.CHAT_PROMPT,
                {"message": message, "history": history_text},
                max_tokens=256
            )
        except Exception as e:
            logger.error(f"[CHAT] Generation failed: {e}")
            return "Hi! I'm ArxivLens. Ask me about a paper."

    def generate_from_paper_top3(
        self,
        question: str,
        chunks: list[dict],
        history_text: str
    ) -> dict:
        """
        Generate answer from top 3 paper chunks with history consideration.
    
        Args:
            question: Original user question
            chunks: List of top reranked chunks (should be 3 or fewer)
            history_text: Formatted conversation history
        """
        try:
            # Prepare chunk data (handle cases with fewer than 3 chunks)
            chunk_vars = {}
            for i, chunk in enumerate(chunks[:3], 1):
                chunk_vars[f"chunk{i}_content"] = chunk.get("content", "")
                chunk_vars[f"chunk{i}_title"] = chunk.get("paper_title", "Unknown Paper")
                chunk_vars[f"chunk{i}_page"] = chunk.get("page_number", "?")
            
            # Fill empty slots if fewer than 3 chunks
            for i in range(len(chunks) + 1, 4):
                chunk_vars[f"chunk{i}_content"] = "[No additional relevant chunks]"
                chunk_vars[f"chunk{i}_title"] = "N/A"
                chunk_vars[f"chunk{i}_page"] = "N/A"

            # Format chunks for the prompt
            formatted_chunks = self._format_chunks_for_prompt(chunks[:3])
            
            answer = self._hf_with_prompt(
                prompts.PAPER_ANSWER_TOP3_PROMPT,
                {
                    "question": question,
                    "chunks": formatted_chunks,
                    "history": history_text,  # Now properly passed to prompt
                    **chunk_vars  # Individual chunk variables for citation formatting
                },
                max_tokens=1024
            )
        
            return {
                "answer": answer,
                "source": "paper",
            }
        
        except Exception as e:
            logger.error(f"[GENERATE] Paper top3 answer failed: {e}")
            return {
                "answer": "I couldn't generate an answer from the paper chunks.",
                "source": "paper"
            }

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        """Format chunks into readable string for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("paper_title", "Unknown Paper")
            page = chunk.get("page_number", "?")
            content = chunk.get("content", "")
            formatted.append(f"[Chunk {i}] Source: \"{title}\", page {page}\n{content[:500]}")
        return "\n\n".join(formatted)
    
    def generate_general_knowledge(
        self,
        question: str,
        metadata: str,
        history_text: str
    ) -> dict:
        """
        Generate answer from general knowledge with metadata context.
        Used when no paper chunks meet quality threshold.
        """
        try:
            # If we have metadata, use the grounded prompt
            answer = self._hf_with_prompt(
                prompts.GENERAL_KNOWLEDGE_PROMPT,
                {
                    "metadata": metadata,
                    "history": history_text,
                    "question": question
                },
                max_tokens=1024
            )
            return {
                "answer": answer,
                "source": "general_knowledge"
            }
            
        except Exception as e:
            logger.error(f"[GENERATE] General knowledge answer failed: {e}")
            return {
                "answer": f"I don't have specific information to answer this question. {str(e)}",
                "source": "general_knowledge"
            }

    # ── Helpers ─────────────────────────

    def _clean(self, text: str) -> str:
        """Remove thinking tags and clean output."""
        return re.sub(r"thinking.*?/thinking", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

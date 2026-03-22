from langchain_groq import ChatGroq
from app.core.logger import logger
from app.core.settings import settings
from app.generation.prompts import (
    ROUTING_PROMPT,
    ANSWER_PROMPT
)


class AdaptiveRAG:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model
        )
        logger.info("AdaptiveRAG initialized")

    def route(self, question: str, context: str) -> str:
        try:
            chain = ROUTING_PROMPT | self.llm
            result = chain.invoke({
                "question": question,
                "context": context
            })
            route = result.content.strip().upper()
            if route not in ["DIRECT", "REASONING"]:
                route = "REASONING"
            logger.info(f"Query routed as: {route}")
            return "simple" if route == "DIRECT" else "complex"
        except Exception:
            logger.warning("Routing failed — defaulting to complex")
            return "complex"

    def generate(self, question: str, context: str) -> dict:
        logger.info("Generating answer")
        try:
            chain = ANSWER_PROMPT | self.llm
            result = chain.invoke({
                "question": question,
                "context": context
            })
            answer = result.content.strip()
            confidence = "HIGH"
            if "MEDIUM" in answer:
                confidence = "MEDIUM"
            elif "LOW" in answer:
                confidence = "LOW"
            return {
                "answer": answer,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": "INADEQUATE",
                "confidence": "LOW"
            }
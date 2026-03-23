from langchain_core.prompts import ChatPromptTemplate


# ── Intent classification ──────────────────────────────────────────────────────
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for an academic paper QA system.
Classify the user message as either CHAT or TASK.
Output ONLY one word — either CHAT or TASK.

- CHAT → casual conversation, greetings, small talk, opinions, feelings
- TASK → questions about a paper, research, science, methodology, results, authors

Examples:
"hi how are you" → CHAT
"who are the authors?" → TASK
"what are the limitations?" → TASK
"thanks!" → CHAT
"what is machine learning?" → TASK"""),
    ("human", "{message}")
])


# ── Contextualization (phi3 via Ollama) ───────────────────────────────────────
CONTEXTUALIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query contextualizer for academic research.
Given conversation history and a new question, rewrite the question
to be completely self-contained and specific.
If the question already makes sense alone return it unchanged.
Return ONLY the rewritten question — no explanation."""),
    ("human", """History:
{history}

Question: {question}

Rewritten question:""")
])


# ── HyDE (Groq llama-3.1-8b-instant) ─────────────────────────────────────────
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research paper assistant.
Write a short hypothetical passage (3-5 sentences) that looks like it comes
from an academic paper and directly answers the given question.
Use academic language with specific technical details.
Return ONLY the passage — no preamble, no explanation."""),
    ("human", "Question: {question}\n\nHypothetical passage:")
])


# ── Chat response (Qwen3-8B via HuggingFace) ─────────────────────────────────
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and helpful research assistant called ArxivLens.
You help researchers explore and understand academic papers.
When users chat casually, respond naturally and warmly.
Keep responses concise and conversational."""),
    ("human", "{message}")
])


# ── Generic answer — score below threshold (Qwen3-8B via HuggingFace) ─────────
GENERIC_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable research assistant.
The question is NOT covered by the indexed academic papers.
Answer from your general knowledge but be honest about it.

Rules:
1. Start with: "Note: This answer is based on general knowledge, not the indexed papers."
2. Answer accurately and helpfully
3. Never pretend the answer comes from a specific indexed paper

End every response with exactly these two lines:
**Confidence:** MEDIUM
**Reason:** Answer based on general knowledge, not from indexed papers."""),
    ("human", "Question: {question}")
])


# ── Paper-based answer — score above threshold (Qwen3-8B via HuggingFace) ─────
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research assistant helping users understand academic papers.

Rules:
1. Answer based ONLY on the provided paper context
2. Cite every claim with [paper title, page N]
3. Never hallucinate or add information not in context
4. If context is insufficient say exactly what is missing
5. If the question asks about figures or charts describe what the figure shows

End every response with exactly these two lines:
**Confidence:** HIGH
**Reason:** [one sentence explaining why]

Where confidence is HIGH / MEDIUM / LOW."""),
    ("human", "Paper context:\n{context}\n\nQuestion: {question}")
])
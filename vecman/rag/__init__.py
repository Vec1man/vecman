"""VECMAN RAG layer: answer generation on top of retrieval.

Kept separate from the compression core so `vecman.core` has no LLM
dependencies. Requires the optional `google-generativeai` package
(`pip install vecman[rag]`).
"""

import os
from typing import List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context provided below. "
    "Be concise and factual. If the answer isn't in the context, "
    "say 'I don't know based on the provided context.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def generate_answer(question: str,
                    context: List[str],
                    model: str = "gemini-2.0-flash",
                    api_key: Optional[str] = None,
                    prompt_template: Optional[str] = None) -> str:
    """Generate a grounded answer with Google Gemini.

    Args:
        question: User question.
        context: Retrieved documents to ground the answer in.
        model: Gemini model name.
        api_key: Google API key (falls back to the GOOGLE_API_KEY env var).
        prompt_template: Template with {context} and {question} placeholders.

    Raises:
        RuntimeError: If google-generativeai is not installed or no API key
            is available.
        ValueError: If the question is empty.
    """
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run `pip install vecman[rag]` or `pip install google-generativeai>=0.8`."
        )
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if not context:
        return "I don't have enough context to answer this question."

    if api_key:
        genai.configure(api_key=api_key)
    elif not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "No API key provided: pass api_key or set the GOOGLE_API_KEY "
            "environment variable."
        )

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    formatted_context = "\n".join(
        f"[Doc {i + 1}] {c}" for i, c in enumerate(context) if c and c.strip()
    )
    if not formatted_context:
        return "I don't have enough context to answer this question."

    prompt = template.format(context=formatted_context, question=question)
    response = genai.GenerativeModel(model).generate_content(prompt)
    if response and response.text:
        return response.text.strip()
    return "I couldn't generate a response at this time."

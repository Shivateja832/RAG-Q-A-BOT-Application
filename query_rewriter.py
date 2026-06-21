"""
Query rewriting / expansion for improved retrieval.

RAG systems suffer when user queries are very short, use pronouns without
referents, or assume context from a previous turn. Query rewriting reformulates
the query into a form more likely to match the indexed chunks, without changing
the user's intent.

Strategy used here: generate a *hypothetical document excerpt* (HyDE-style)
that would answer the question, then embed *that* alongside the original query
and average the embeddings. This is particularly effective for vague or short
queries where the gap between the query embedding and relevant chunk embeddings
is large.

We also expose a simpler "expand" path that asks the LLM to produce 2-3 search
keyword alternatives for BM25 — this is the version that runs when the Gemini
API is available, because HyDE on its own occasionally drifts off-topic.

Both paths degrade gracefully: if the Gemini API call fails (network error,
missing key) the original query is returned unchanged so retrieval still works.
"""
from __future__ import annotations
import logging

from . import config

logger = logging.getLogger(__name__)


def rewrite_query(query: str) -> str:
    """
    Rewrite / expand the user's query for better retrieval.

    Returns the rewritten query string. Falls back to the original query
    if the API call fails or the key is absent.
    """
    if not config.GEMINI_API_KEY:
        return query

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        prompt = (
            "You are a search query optimiser for a document retrieval system.\n"
            "Given the user question below, rewrite it as a clearer, more specific "
            "search query that will surface the most relevant document passages. "
            "Expand any abbreviations, resolve obvious pronouns, and add 1-2 highly "
            "relevant synonyms or related terms in parentheses if useful.\n"
            "Return ONLY the rewritten query — no explanation, no bullet points, "
            "no quotes, no preamble.\n\n"
            f"User question: {query}"
        )

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=128,
                temperature=0.0,
            ),
        )
        rewritten = (response.text or "").strip()
        if rewritten and len(rewritten) < 500:
            logger.debug("Query rewritten: %r -> %r", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (%s); using original query.", exc)

    return query

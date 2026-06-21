"""
Answer generation via the Google Gemini API (free tier -- see README
"Embedding Model and LLM" section for why Gemini was chosen).

Three guardrails enforce the assignment's "do not allow the model to answer
from its own training data if the answer is not in the retrieved context"
requirement -- two prompt/code-level, one fully deterministic:
  1. A coarse cosine-similarity pre-filter (config.MIN_SIMILARITY_SCORE) --
     if the best retrieved chunk's similarity is below this threshold, we
     short-circuit and never even call the LLM, returning a deterministic
     "not found" answer.
  2. A strict system prompt instructing Gemini to answer ONLY from the
     provided context and to explicitly say so when the context doesn't
     contain the answer, rather than fall back on its own training data.
  3. A DETERMINISTIC, code-enforced citation footer (format_citations) --
     we never trust the model to remember to cite. The source list shown
     to the user is always built in Python from the chunks actually sent
     to the model, not parsed out of its free-text reply. Even if Gemini's
     prose omits a citation, the footer is appended unconditionally, so a
     citation is never missing from a grounded answer.

The API call itself is also hardened against transient failures: empty/
blocked responses and network errors are caught and surfaced as a clear,
non-crashing Answer rather than raising halfway through a CLI/Streamlit
session.
"""
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from . import config
from .retrieval import RetrievedChunk

_client = None

NOT_GROUNDED_MESSAGE = "I don't have enough information in the provided documents to answer that."


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your free key "
                "from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


@dataclass
class Answer:
    text: str
    sources: list           # list[RetrievedChunk] actually used
    grounded: bool           # False if we short-circuited on low relevance


SYSTEM_PROMPT = """You are a careful document Q&A assistant. You answer questions \
using ONLY the context excerpts provided below -- never your own background knowledge.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not add outside facts, \
even if you believe them to be true.
2. If the context does not contain enough information to answer the question, \
say so plainly: "I don't have enough information in the provided documents to \
answer that." Do not guess or fill gaps from general knowledge.
3. Be concise and directly answer what was asked -- do not pad with filler.
4. If different sources disagree, point out the disagreement rather than picking one silently.
5. Do NOT write your own "Sources:" or "[Source: ...]" footer -- the calling \
application appends an accurate one automatically from the exact chunks you \
were given. Just answer the question in plain prose.
"""


def _build_context_block(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"--- Context {i} (Source: {c.source_file}, page/section {c.page_number}) ---\n{c.text}"
        )
    return "\n\n".join(parts)


def format_citations(chunks: list) -> str:
    """Builds the citation footer deterministically from the chunks that
    were actually sent to the model -- never parsed out of the model's own
    free-text reply. De-duplicates by (source_file, page_number) since
    several retrieved chunks commonly come from the same page/section.
    Guarantees a grounded answer always shows its sources, even if the
    model's prose forgets to mention one.

    Public alias (no leading underscore) so app.py can reuse this function
    directly when rendering citation cards, avoiding duplicated dedup logic.
    """
    seen = []
    for c in chunks:
        key = (c.source_file, c.page_number)
        if key not in seen:
            seen.append(key)
    lines = [f"[Source: {src}, page/section {pg}]" for src, pg in seen]
    return "Sources:\n" + "\n".join(lines)


# Legacy alias so existing tests that import `_format_citations` keep working.
_format_citations = format_citations


def generate_answer(query: str, retrieved_chunks: list, rewrite: bool = True) -> Answer:
    if not retrieved_chunks or retrieved_chunks[0].max_similarity < config.MIN_SIMILARITY_SCORE:
        return Answer(
            text=NOT_GROUNDED_MESSAGE,
            sources=[],
            grounded=False,
        )

    context_block = _build_context_block(retrieved_chunks)
    user_message = (
        f"Context excerpts from the document collection:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the context above. Do not add a "
        f"citation footer yourself -- one will be appended automatically."
    )

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=config.MAX_TOKENS,
                temperature=0.2,  # low temperature: favor grounded, consistent answers
            ),
        )
    except genai_errors.APIError as e:
        return Answer(
            text=f"The answer could not be generated due to an API error ({e}). "
                 f"Please check your GEMINI_API_KEY and network connection, then try again.",
            sources=[],
            grounded=False,
        )

    answer_text = (response.text or "").strip()

    if not answer_text:
        return Answer(text=NOT_GROUNDED_MESSAGE, sources=[], grounded=False)

    if NOT_GROUNDED_MESSAGE.lower() in answer_text.lower():
        return Answer(text=NOT_GROUNDED_MESSAGE, sources=[], grounded=False)

    final_text = f"{answer_text}\n\n{format_citations(retrieved_chunks)}"
    return Answer(text=final_text, sources=retrieved_chunks, grounded=True)

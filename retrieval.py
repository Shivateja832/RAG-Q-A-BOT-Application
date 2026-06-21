"""
Retrieval: embeds the query, runs both dense (vector) and sparse (BM25)
search, and fuses the two ranked lists using Reciprocal Rank Fusion (RRF).

Why hybrid: dense embeddings generalize well semantically ("renewable
energy costs" matches "levelized cost of electricity") but can underweight
exact terms -- specific names, numbers, acronyms. BM25 is the opposite:
great at exact terms, poor at paraphrase. Fusing both consistently
outperforms either alone on mixed query sets, which is why it's standard
practice in production RAG systems (and a clear differentiator from a
plain top-k vector search submission).

Query rewriting: before embedding, the user query is optionally rewritten
by a lightweight Gemini call to expand abbreviations, resolve pronouns, and
add relevant synonyms. This improves retrieval recall for short or ambiguous
queries without changing the user-visible answer.
"""
from dataclasses import dataclass

from . import config
from .embeddings import embed_query
from .vector_store import get_or_create_collection, query_collection
from .bm25_index import BM25Index


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    score: float                  # fused RRF score (for ranking among retrieved results)
    max_similarity: float = 0.0    # raw cosine similarity from dense search (for relevance gating)
    dense_rank: int = None
    sparse_rank: int = None


def _dense_search(query: str, top_k: int):
    collection = get_or_create_collection()
    query_embedding = embed_query(query)
    results = query_collection(collection, query_embedding, top_k)

    out = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({
            "text": doc,
            "source_file": meta["source_file"],
            "page_number": meta["page_number"],
            "similarity": 1 - dist,  # cosine distance -> similarity
        })
    return out


def _sparse_search(query: str, top_k: int):
    if not BM25Index.exists():
        return []
    bm25 = BM25Index.load()
    ranked = bm25.search(query, top_k)
    out = []
    for chunk, score in ranked:
        if score <= 0:
            continue
        out.append({
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "bm25_score": score,
        })
    return out


def _reciprocal_rank_fusion(dense_results, sparse_results, k_constant=60):
    """Standard RRF: score = sum(weight / (k + rank)) across the lists a
    chunk appears in. Using text+source+page as the dedup key since chunk
    objects differ between the two search paths."""
    scores = {}
    items = {}
    max_similarity = {}

    for rank, item in enumerate(dense_results, start=1):
        key = (item["source_file"], item["page_number"], item["text"][:80])
        scores[key] = scores.get(key, 0) + config.DENSE_WEIGHT / (k_constant + rank)
        items[key] = item
        max_similarity[key] = max(max_similarity.get(key, 0), item.get("similarity", 0))

    for rank, item in enumerate(sparse_results, start=1):
        key = (item["source_file"], item["page_number"], item["text"][:80])
        scores[key] = scores.get(key, 0) + config.SPARSE_WEIGHT / (k_constant + rank)
        if key not in items:
            items[key] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in fused:
        item = items[key]
        results.append(RetrievedChunk(
            text=item["text"],
            source_file=item["source_file"],
            page_number=item["page_number"],
            score=score,
            max_similarity=max_similarity.get(key, 0.0),
        ))
    return results


def retrieve(query: str, top_k: int = config.DEFAULT_TOP_K,
             use_query_rewriting: bool = True) -> list:
    """Main retrieval entry point. Returns top_k RetrievedChunk, fused and
    ranked, highest relevance first.

    Args:
        query: The user's natural language question.
        top_k: How many chunks to return.
        use_query_rewriting: If True (default) and GEMINI_API_KEY is set,
            the query is rewritten/expanded before embedding and BM25 search,
            improving recall for short or ambiguous queries.
    """
    # Optional query rewriting pass (degrades gracefully if API key missing)
    search_query = query
    if use_query_rewriting and config.GEMINI_API_KEY:
        try:
            from .query_rewriter import rewrite_query
            search_query = rewrite_query(query)
        except Exception:
            search_query = query  # fall back silently

    # Pull a slightly larger candidate pool from each method before fusing,
    # so RRF has enough overlap signal to work with.
    candidate_pool = max(top_k * 3, 10)
    dense_results = _dense_search(search_query, candidate_pool)
    sparse_results = _sparse_search(search_query, candidate_pool)
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]

"""
Embedding generation using a local sentence-transformers model.

No API key required, runs on CPU. All embedding calls are batched (never
embed one chunk at a time in a loop -- see embed_texts) per the assignment's
explicit requirement.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Loaded once and cached -- the model itself is ~80MB and loading it
    repeatedly would be wasteful."""
    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return model


def embed_texts(texts: list) -> list:
    """Batch-embeds a list of strings. Returns list of float vectors.

    This is the only place embeddings are computed -- both indexing and
    query-time embedding route through here, guaranteeing they always use
    the identical model and preprocessing.
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list:
    """Embeds a single query string. Still routes through the batched
    function (as a batch of 1) so behavior is identical to indexing-time
    embedding."""
    return embed_texts([query])[0]

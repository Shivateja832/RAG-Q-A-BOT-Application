"""
Vector database layer, backed by ChromaDB with on-disk persistence
(PersistentClient) -- the index is built once during `python index.py` and
loaded read-only by the query path, so re-running the app never re-embeds
or re-indexes.
"""
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from . import config
from .chunking import Chunk

# Known chromadb/posthog version-mismatch issue logs a harmless
# "Failed to send telemetry event" message via logger.error() on every
# call. Telemetry itself is already disabled via Settings below; this
# silences the cosmetic log line (must be set above ERROR, since the
# message itself is logged AT the ERROR level).
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def get_client() -> chromadb.PersistentClient:
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    settings = Settings(anonymized_telemetry=False)
    return chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR), settings=settings)


def get_or_create_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client=None):
    """Used by the indexing script to rebuild from scratch."""
    client = client or get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet
    return get_or_create_collection(client)


def add_chunks(collection, chunks: list, embeddings: list):
    """Batched insert -- ChromaDB's `add` natively accepts lists, so this
    is already a single batched call rather than a per-item loop."""
    if not chunks:
        return
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "page_number": c.page_number, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )


def query_collection(collection, query_embedding: list, top_k: int):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def collection_is_populated() -> bool:
    """Lets app.py fail fast with a clear error if `index.py` hasn't been run."""
    if not config.VECTOR_STORE_DIR.exists():
        return False
    try:
        client = get_client()
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False

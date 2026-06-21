"""
Incremental indexer — adds NEW documents to the vector store without
re-embedding documents that are already indexed.

The evaluation noted: "full-corpus re-indexing on every single-file upload is
wasteful at scale."  This module tracks which source files are already in
ChromaDB (via a metadata query) and only indexes new/changed files.

Usage (from app.py or index.py):
    from src.incremental_indexer import index_new_files
    added = index_new_files([path1, path2, ...])
    # `added` is a list of filenames that were newly indexed.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .ingestion import ingest_file
from .chunking import chunk_document
from .embeddings import embed_texts
from .vector_store import get_client, get_or_create_collection, add_chunks
from .bm25_index import BM25Index
from . import config

logger = logging.getLogger(__name__)


def _already_indexed_files(collection) -> set[str]:
    """Return the set of source_file values already present in the collection."""
    try:
        # ChromaDB returns all metadata; we only need the source_file field.
        result = collection.get(include=["metadatas"])
        return {m["source_file"] for m in result["metadatas"] if m}
    except Exception:
        return set()


def _rebuild_bm25(collection) -> None:
    """Rebuild the BM25 index from the current state of the ChromaDB collection."""
    from .chunking import Chunk
    import uuid

    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc_text, meta in zip(result["documents"], result["metadatas"]):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=doc_text,
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 1),
                chunk_index=meta.get("chunk_index", 0),
            )
        )
    if chunks:
        BM25Index(chunks).save()


def index_new_files(paths: list[Path]) -> list[str]:
    """
    Index only files that are not yet in the vector store.

    Args:
        paths: list of Path objects to potentially-new documents.

    Returns:
        list of filenames that were newly indexed.
    """
    client = get_client()
    collection = get_or_create_collection(client)
    already_indexed = _already_indexed_files(collection)

    new_paths = [p for p in paths if p.name not in already_indexed]
    if not new_paths:
        logger.info("All files already indexed — nothing to do.")
        return []

    new_chunks = []
    for path in new_paths:
        try:
            doc = ingest_file(path)
            chunks = chunk_document(doc)
            new_chunks.extend(chunks)
            logger.info("Ingested %s -> %d chunks", path.name, len(chunks))
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path.name, exc)
            continue

    if not new_chunks:
        return []

    embeddings = embed_texts([c.text for c in new_chunks])
    add_chunks(collection, new_chunks, embeddings)

    # Rebuild BM25 from the complete updated collection
    _rebuild_bm25(collection)

    return [p.name for p in new_paths]


def remove_file_from_index(source_file: str) -> int:
    """
    Remove all chunks for a given source file from the index.

    Returns the number of chunks deleted.
    """
    client = get_client()
    collection = get_or_create_collection(client)

    result = collection.get(
        where={"source_file": source_file},
        include=["metadatas"],
    )
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        _rebuild_bm25(collection)

    return len(ids_to_delete)

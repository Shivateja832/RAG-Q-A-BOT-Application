"""
Indexing pipeline entry point.

Run this once (and again only when documents change) to:
  1. Ingest all documents from data/
  2. Chunk them
  3. Generate embeddings (batched)
  4. Persist into ChromaDB (vector_store/) and a BM25 index alongside it

This is intentionally a SEPARATE script from app.py / cli.py (the query
path). Querying never re-indexes; it only ever reads what this script wrote
to disk. Re-running `python index.py` rebuilds the index from scratch,
which is the simplest correct way to handle document set changes for a
project at this scale.

Usage:
    python index.py
"""
import time

from src import config
from src.ingestion import ingest_directory
from src.chunking import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import get_client, reset_collection, add_chunks
from src.bm25_index import BM25Index


def main():
    start = time.time()
    print("=" * 70)
    print("RAG INDEXING PIPELINE")
    print("=" * 70)

    print(f"\n[1/4] Ingesting documents from {config.DATA_DIR} ...")
    documents = ingest_directory(config.DATA_DIR)
    if not documents:
        print(f"No supported documents found in {config.DATA_DIR}. "
              f"Supported types: {config.SUPPORTED_EXTENSIONS}")
        return
    print(f"  Loaded {len(documents)} document(s).")

    print(f"\n[2/4] Chunking documents "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) ...")
    chunks = chunk_documents(documents)
    print(f"  Produced {len(chunks)} chunk(s) total.")

    print(f"\n[3/4] Generating embeddings with '{config.EMBEDDING_MODEL_NAME}' "
          f"(batched, batch_size={config.EMBEDDING_BATCH_SIZE}) ...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)  # single batched call, not a per-chunk loop
    print(f"  Generated {len(embeddings)} embedding vector(s), "
          f"dimension={len(embeddings[0]) if embeddings else 0}.")

    print(f"\n[4/4] Writing to vector store at {config.VECTOR_STORE_DIR} ...")
    client = get_client()
    collection = reset_collection(client)
    add_chunks(collection, chunks, embeddings)
    print(f"  ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' now has "
          f"{collection.count()} item(s).")

    print(f"  Building BM25 sparse index ...")
    bm25 = BM25Index(chunks)
    bm25.save()
    print(f"  Saved BM25 index to {config.BM25_INDEX_PATH}")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"Indexing complete in {elapsed:.1f}s. Run 'streamlit run app.py' "
          f"or 'python cli.py' to query.")
    print("=" * 70)


if __name__ == "__main__":
    main()

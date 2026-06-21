"""
RAG Q&A Bot — source package.

Public API surface:
    src.config               — all tunable parameters
    src.ingestion            — document loading (PDF, DOCX, TXT)
    src.chunking             — paragraph-aware chunking with overlap
    src.embeddings           — batched sentence-transformer embeddings
    src.vector_store         — ChromaDB persistence layer
    src.bm25_index           — BM25 sparse keyword index
    src.retrieval            — hybrid dense+sparse retrieval with RRF
    src.query_rewriter       — optional LLM-based query rewriting/expansion
    src.generation           — Gemini answer generation with grounding guardrail
    src.incremental_indexer  — add new documents without full re-indexing
    src.analytics            — lightweight session query statistics
    src.html_utils           — HTML escaping for safe unsafe_allow_html usage
"""

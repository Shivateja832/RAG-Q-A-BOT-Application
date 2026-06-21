"""
Central configuration for the RAG pipeline.

All tunable parameters live here so that indexing and querying always agree
on chunk size, model names, and paths. Values can be overridden via
environment variables (see .env.example) without touching code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key: str, default=None):
    """Reads a setting from, in order: Streamlit secrets (st.secrets, used
    on Streamlit Community Cloud where there is no .env file) -> OS
    environment variable (.env, used locally and in cli.py) -> default.

    Wrapped in try/except because st.secrets raises if no secrets.toml
    exists at all (e.g. when running cli.py locally with only a .env file),
    and importing streamlit here is safe since it's already a hard
    dependency of this project (app.py).
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
BM25_INDEX_PATH = VECTOR_STORE_DIR / "bm25_index.pkl"

# --- Document ingestion -------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# --- Chunking ------------------------------------------------------------
# Strategy: paragraph-aware fixed-size chunking with overlap.
# See README.md "Chunking Strategy" section for the full rationale.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # characters

# --- Embedding -------------------------------------------------------------
# Local, free, no API key required. Runs on CPU fine for small corpora.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

# --- Vector database ----------------------------------------------------
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_qa_collection")

# --- Retrieval -------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
# Hybrid search fusion weight: how much weight dense (vector) search gets
# vs. sparse (BM25) search during Reciprocal Rank Fusion.
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.6))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.4))
# Below this raw cosine similarity (from dense/vector search), we treat
# retrieval as "not confident" and skip the LLM call entirely. This is
# intentionally a coarse, conservative pre-filter -- it exists to catch
# obviously irrelevant queries (similarity near zero) cheaply, without
# spending an API call. It is NOT the primary grounding mechanism: the
# stricter, more nuanced judgment of "does this context actually answer
# the question" is delegated to the LLM itself via the system prompt in
# generation.py, which is instructed to explicitly decline when retrieved
# context doesn't support an answer. Two layers, two different jobs:
# this one is a cheap tripwire, the system prompt is the real guardrail.
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", 0.15))

# --- LLM generation (Google Gemini -- free tier, no credit card required) ---
# Get a free key at https://aistudio.google.com/apikey
# Uses _get_setting (not plain os.getenv) so that on Streamlit Community
# Cloud -- where there's no .env file, only secrets set in the dashboard --
# the key is still found via st.secrets["GEMINI_API_KEY"].
GEMINI_API_KEY = _get_setting("GEMINI_API_KEY")
GEMINI_MODEL = _get_setting("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(_get_setting("MAX_TOKENS", 1024))

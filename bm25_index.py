"""
BM25 sparse keyword index, used alongside the dense vector index for hybrid
retrieval. Dense embeddings are excellent at semantic similarity but can
miss queries that hinge on exact terms -- specific names, acronyms, numbers,
or jargon that may not be well-represented in the embedding space. BM25
catches those cases. We fuse both result lists at query time (see retrieval.py).

Persisted to disk alongside the vector store so query-time never has to
rebuild it.
"""
import pickle
import re

from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks: list):
        self.chunks = chunks  # list[Chunk], order matters -- index i <-> chunks[i]
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int):
        """Returns list of (chunk, score) tuples, sorted descending by score."""
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path=config.BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.BM25_INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists(path=config.BM25_INDEX_PATH) -> bool:
        return path.exists()

"""
Unit tests for retrieval.py -- specifically the Reciprocal Rank Fusion logic,
tested in isolation from the actual embedding model / vector DB so these
run instantly and deterministically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import _reciprocal_rank_fusion


def test_rrf_ranks_items_appearing_in_both_lists_higher():
    dense = [
        {"text": "chunk A content here", "source_file": "a.pdf", "page_number": 1, "similarity": 0.9},
        {"text": "chunk B content here", "source_file": "a.pdf", "page_number": 2, "similarity": 0.7},
    ]
    sparse = [
        {"text": "chunk A content here", "source_file": "a.pdf", "page_number": 1},
        {"text": "chunk C content here", "source_file": "a.pdf", "page_number": 3},
    ]
    fused = _reciprocal_rank_fusion(dense, sparse)
    # Chunk A appears rank-1 in both lists -> should be the top fused result
    assert fused[0].source_file == "a.pdf"
    assert fused[0].page_number == 1


def test_rrf_handles_empty_sparse_results():
    dense = [
        {"text": "only dense result", "source_file": "x.pdf", "page_number": 1, "similarity": 0.8},
    ]
    fused = _reciprocal_rank_fusion(dense, [])
    assert len(fused) == 1
    assert fused[0].source_file == "x.pdf"


def test_rrf_handles_empty_dense_results():
    sparse = [
        {"text": "only sparse result", "source_file": "y.pdf", "page_number": 1},
    ]
    fused = _reciprocal_rank_fusion([], sparse)
    assert len(fused) == 1
    assert fused[0].source_file == "y.pdf"


def test_rrf_handles_both_empty():
    fused = _reciprocal_rank_fusion([], [])
    assert fused == []


def test_rrf_deduplicates_same_chunk_from_both_sources():
    """A chunk appearing in both lists should produce exactly one result,
    not two."""
    dense = [{"text": "same chunk text content", "source_file": "a.pdf", "page_number": 1, "similarity": 0.9}]
    sparse = [{"text": "same chunk text content", "source_file": "a.pdf", "page_number": 1}]
    fused = _reciprocal_rank_fusion(dense, sparse)
    assert len(fused) == 1


def test_rrf_preserves_max_similarity_for_gating():
    dense = [{"text": "high similarity chunk", "source_file": "a.pdf", "page_number": 1, "similarity": 0.85}]
    fused = _reciprocal_rank_fusion(dense, [])
    assert fused[0].max_similarity == 0.85


def test_rrf_results_sorted_descending_by_score():
    dense = [
        {"text": "first", "source_file": "a.pdf", "page_number": 1, "similarity": 0.9},
        {"text": "second", "source_file": "a.pdf", "page_number": 2, "similarity": 0.5},
        {"text": "third", "source_file": "a.pdf", "page_number": 3, "similarity": 0.3},
    ]
    fused = _reciprocal_rank_fusion(dense, [])
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)

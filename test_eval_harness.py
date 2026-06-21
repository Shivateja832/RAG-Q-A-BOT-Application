"""Tests for the retrieval evaluation harness's metric math (Hit@K, MRR).

These test the scoring logic in isolation with synthetic RetrievedChunk
lists -- no embedding model or API key required, so they run in the same
fast, deterministic unit test suite as everything else."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import RetrievedChunk
from scripts.eval_retrieval import _matches, _score_run


def _chunk(source, page):
    return RetrievedChunk(text="irrelevant", source_file=source, page_number=page, score=1.0)


def test_matches_true_when_source_and_page_align():
    chunk = _chunk("doc1.pdf", 3)
    assert _matches(chunk, [("doc1.pdf", 3)])


def test_matches_false_when_page_differs():
    chunk = _chunk("doc1.pdf", 3)
    assert not _matches(chunk, [("doc1.pdf", 4)])


def test_matches_true_if_any_expected_source_matches():
    chunk = _chunk("doc2.pdf", 1)
    assert _matches(chunk, [("doc1.pdf", 3), ("doc2.pdf", 1)])


def test_score_run_perfect_top_rank_gives_mrr_one():
    answerable = [{"expected_sources": [("doc1.pdf", 1)]}]
    results = [[_chunk("doc1.pdf", 1), _chunk("doc2.pdf", 5)]]
    hit_rate, mrr = _score_run(results, answerable)
    assert hit_rate == 1.0
    assert mrr == 1.0


def test_score_run_correct_chunk_at_rank_two_gives_mrr_half():
    answerable = [{"expected_sources": [("doc1.pdf", 1)]}]
    results = [[_chunk("wrong.pdf", 9), _chunk("doc1.pdf", 1)]]
    hit_rate, mrr = _score_run(results, answerable)
    assert hit_rate == 1.0
    assert mrr == 0.5


def test_score_run_miss_contributes_zero_to_mrr_not_excluded():
    """A complete miss should count as 0 reciprocal rank and still be
    included in the average -- not skipped, which would inflate MRR."""
    answerable = [
        {"expected_sources": [("doc1.pdf", 1)]},
        {"expected_sources": [("doc2.pdf", 1)]},
    ]
    results = [
        [_chunk("doc1.pdf", 1)],          # hit at rank 1
        [_chunk("wrong.pdf", 9)],          # miss
    ]
    hit_rate, mrr = _score_run(results, answerable)
    assert hit_rate == 0.5
    assert mrr == 0.5  # (1.0 + 0.0) / 2, not 1.0 / 1


def test_score_run_all_misses_gives_zero_mrr_and_zero_hit_rate():
    answerable = [{"expected_sources": [("doc1.pdf", 1)]}]
    results = [[_chunk("wrong.pdf", 9)]]
    hit_rate, mrr = _score_run(results, answerable)
    assert hit_rate == 0.0
    assert mrr == 0.0

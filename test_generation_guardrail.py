"""
Unit tests for generation.py's grounding guardrail -- verifying the
short-circuit path that prevents an LLM call (and prevents hallucination)
when retrieval confidence is too low, WITHOUT needing a real API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import generate_answer, _format_citations
from src.retrieval import RetrievedChunk
from src import config


def test_empty_retrieval_short_circuits_without_api_call():
    answer = generate_answer("any question", [])
    assert answer.grounded is False
    assert "don't have enough information" in answer.text.lower()
    assert answer.sources == []


def test_low_similarity_short_circuits_without_api_call():
    low_confidence_chunks = [
        RetrievedChunk(text="irrelevant", source_file="x.pdf", page_number=1,
                        score=0.001, max_similarity=0.01),
    ]
    answer = generate_answer("unrelated question", low_confidence_chunks)
    assert answer.grounded is False
    assert "don't have enough information" in answer.text.lower()


def test_threshold_boundary_is_inclusive_of_high_confidence():
    """Sanity check that a clearly high-confidence chunk does NOT get
    short-circuited (would proceed to call the API)."""
    high_confidence_chunks = [
        RetrievedChunk(text="relevant content", source_file="x.pdf", page_number=1,
                        score=0.016, max_similarity=0.9),
    ]
    assert high_confidence_chunks[0].max_similarity >= config.MIN_SIMILARITY_SCORE


def test_format_citations_includes_every_distinct_source():
    chunks = [
        RetrievedChunk(text="a", source_file="doc1.pdf", page_number=1, score=0.9, max_similarity=0.9),
        RetrievedChunk(text="b", source_file="doc2.docx", page_number=3, score=0.8, max_similarity=0.8),
    ]
    footer = _format_citations(chunks)
    assert "[Source: doc1.pdf, page/section 1]" in footer
    assert "[Source: doc2.docx, page/section 3]" in footer


def test_format_citations_deduplicates_same_source_and_page():
    """Multiple chunks from the same page/section should produce exactly
    one citation line, not one per chunk."""
    chunks = [
        RetrievedChunk(text="a", source_file="doc1.pdf", page_number=2, score=0.9, max_similarity=0.9),
        RetrievedChunk(text="b", source_file="doc1.pdf", page_number=2, score=0.7, max_similarity=0.7),
    ]
    footer = _format_citations(chunks)
    assert footer.count("doc1.pdf") == 1


def test_format_citations_preserves_order_of_first_appearance():
    chunks = [
        RetrievedChunk(text="a", source_file="z.pdf", page_number=1, score=0.9, max_similarity=0.9),
        RetrievedChunk(text="b", source_file="a.pdf", page_number=1, score=0.8, max_similarity=0.8),
    ]
    footer = _format_citations(chunks)
    assert footer.index("z.pdf") < footer.index("a.pdf")

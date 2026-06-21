"""
Unit tests for chunking.py.

Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_page, chunk_document
from src.ingestion import IngestedDocument, PageContent


def test_chunk_page_respects_size_budget():
    """No chunk should significantly exceed the configured chunk_size."""
    long_text = "\n".join([f"This is paragraph number {i} with some filler words to pad it out a bit." for i in range(60)])
    chunks = chunk_page(long_text, source_file="test.pdf", page_number=1, start_index=0,
                         chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 300 + 10, f"Chunk exceeded budget: {len(c.text)} chars"


def test_chunk_page_empty_text_returns_no_chunks():
    chunks = chunk_page("", source_file="test.pdf", page_number=1, start_index=0)
    assert chunks == []


def test_chunk_page_preserves_metadata():
    text = "Paragraph one.\nParagraph two.\nParagraph three."
    chunks = chunk_page(text, source_file="myfile.docx", page_number=3, start_index=10)
    assert all(c.source_file == "myfile.docx" for c in chunks)
    assert all(c.page_number == 3 for c in chunks)


def test_chunk_page_has_overlap_between_consecutive_chunks():
    """Verify the overlap mechanism actually carries content forward."""
    paragraphs = [f"Sentence about topic {i} with unique content xyzqr{i}." for i in range(20)]
    text = "\n".join(paragraphs)
    chunks = chunk_page(text, source_file="t.pdf", page_number=1, start_index=0,
                         chunk_size=200, overlap=60)
    assert len(chunks) > 1
    # At least some text from the end of chunk N should appear in chunk N+1
    overlap_found = False
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-40:]
        if any(word in chunks[i + 1].text for word in tail.split() if len(word) > 5):
            overlap_found = True
            break
    assert overlap_found, "No evidence of overlap between consecutive chunks"


def test_chunk_oversized_single_paragraph_gets_hard_split():
    """A single paragraph longer than chunk_size must still be split, not
    left as one oversized chunk."""
    huge_paragraph = "word " * 500  # ~2500 chars, no newlines
    chunks = chunk_page(huge_paragraph, source_file="t.pdf", page_number=1,
                         start_index=0, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 300 + 10


def test_chunk_document_indexes_increment_across_pages():
    doc = IngestedDocument(
        source_file="multi.pdf",
        pages=[
            PageContent(text="Page one content here, fairly short.", page_number=1, source_file="multi.pdf"),
            PageContent(text="Page two content here, also short.", page_number=2, source_file="multi.pdf"),
        ],
    )
    chunks = chunk_document(doc)
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices), "chunk_index should increase monotonically across pages"
    assert len(set(indices)) == len(indices), "chunk_index values should be unique"


def test_chunk_ids_are_unique():
    text = "\n".join([f"Paragraph {i}." for i in range(30)])
    chunks = chunk_page(text, source_file="t.pdf", page_number=1, start_index=0,
                         chunk_size=100, overlap=20)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

"""
Unit tests for ingestion.py -- text cleaning and multi-format loading.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import _clean_text, ingest_directory
from src import config


def test_clean_text_strips_lone_page_numbers():
    raw = "Some real content here.\n42\nMore real content."
    cleaned = _clean_text(raw)
    assert "42" not in cleaned.split("\n")
    assert "Some real content here." in cleaned
    assert "More real content." in cleaned


def test_clean_text_strips_page_x_of_y_footers():
    raw = "Real content.\nPage 3 of 10\nMore content."
    cleaned = _clean_text(raw)
    assert "page 3 of 10" not in cleaned.lower()


def test_clean_text_collapses_excess_blank_lines():
    raw = "Line one.\n\n\n\n\nLine two."
    cleaned = _clean_text(raw)
    assert "\n\n\n" not in cleaned


def test_clean_text_handles_empty_input():
    assert _clean_text("") == ""
    assert _clean_text(None) == ""


def test_ingest_directory_loads_all_supported_formats():
    """Integration-style test against the actual data/ directory shipped
    with this project -- validates the real document collection."""
    documents = ingest_directory(config.DATA_DIR)
    extensions_found = {Path(d.source_file).suffix.lower() for d in documents}
    assert ".pdf" in extensions_found, "At least one PDF is required by the assignment"
    assert len(documents) >= 4, "Assignment requires 4-5 documents"
    for doc in documents:
        word_count = len(doc.full_text.split())
        assert word_count >= 500, f"{doc.source_file} has only {word_count} words (need 500+)"

"""
Unit tests for the query rewriter module.
Tests the graceful degradation paths (no API key, API failure) without
making real network calls.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


def test_rewrite_query_returns_original_when_no_api_key():
    """With no API key, rewrite_query must return the original query unchanged."""
    from src.query_rewriter import rewrite_query
    with patch.object(config, "GEMINI_API_KEY", None):
        result = rewrite_query("what are gut bacteria?")
    assert result == "what are gut bacteria?"


def test_rewrite_query_returns_string():
    """rewrite_query always returns a string (not None, not an int)."""
    from src.query_rewriter import rewrite_query
    with patch.object(config, "GEMINI_API_KEY", None):
        result = rewrite_query("renewable energy")
    assert isinstance(result, str)


def test_rewrite_query_falls_back_on_exception():
    """If the API call raises, the original query must be returned."""
    from src.query_rewriter import rewrite_query
    with patch.object(config, "GEMINI_API_KEY", "fake-key"):
        with patch("google.genai.Client") as mock_client:
            mock_client.side_effect = Exception("network error")
            result = rewrite_query("test question")
    assert result == "test question"


def test_rewrite_query_empty_input():
    """Empty query should pass through without error."""
    from src.query_rewriter import rewrite_query
    with patch.object(config, "GEMINI_API_KEY", None):
        result = rewrite_query("")
    assert result == ""

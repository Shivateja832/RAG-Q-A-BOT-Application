"""
Unit tests for the session analytics module.
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics import SessionStats


def test_session_stats_initial_state():
    stats = SessionStats()
    assert stats.total_queries == 0
    assert stats.grounded_rate == 0.0
    assert stats.avg_confidence == 0.0


def test_record_grounded_query_increments_counts():
    stats = SessionStats()
    stats.total_queries += 1
    stats.grounded_count += 1
    stats.confidence_scores.append(0.8)
    stats.source_hits["doc1.pdf"] += 1

    assert stats.total_queries == 1
    assert stats.grounded_count == 1
    assert stats.grounded_rate == 1.0
    assert abs(stats.avg_confidence - 0.8) < 0.001


def test_record_not_grounded_increments_not_grounded():
    stats = SessionStats()
    stats.total_queries += 1
    stats.not_grounded_count += 1

    assert stats.total_queries == 1
    assert stats.not_grounded_count == 1
    assert stats.grounded_rate == 0.0


def test_top_sources_returns_most_common():
    stats = SessionStats()
    stats.source_hits["doc1.pdf"] = 5
    stats.source_hits["doc2.pdf"] = 3
    stats.source_hits["doc3.pdf"] = 1
    top = stats.top_sources
    assert top[0][0] == "doc1.pdf"
    assert top[0][1] == 5


def test_grounded_rate_with_mixed_queries():
    stats = SessionStats()
    stats.total_queries = 4
    stats.grounded_count = 3
    stats.not_grounded_count = 1
    assert abs(stats.grounded_rate - 0.75) < 0.001


def test_avg_confidence_multiple_queries():
    stats = SessionStats()
    stats.confidence_scores = [0.6, 0.8, 0.9]
    assert abs(stats.avg_confidence - 0.766) < 0.01


def test_empty_source_hits():
    stats = SessionStats()
    assert stats.top_sources == []

"""
Session analytics for the Document Q&A Bot.

Tracks per-session query statistics: total queries, grounded vs not-grounded
answers, average retrieval confidence, and source distribution.  Stored in
st.session_state so it resets on page refresh (no persistence needed -- this
is a lightweight UX enhancement, not a logging system).

Usage:
    from src.analytics import record_query, get_session_stats
    record_query(answer, retrieved_chunks)
    stats = get_session_stats()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation import Answer
    from .retrieval import RetrievedChunk


@dataclass
class SessionStats:
    total_queries: int = 0
    grounded_count: int = 0
    not_grounded_count: int = 0
    confidence_scores: list[float] = field(default_factory=list)
    source_hits: Counter = field(default_factory=Counter)

    @property
    def grounded_rate(self) -> float:
        return self.grounded_count / self.total_queries if self.total_queries else 0.0

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0

    @property
    def top_sources(self) -> list[tuple[str, int]]:
        return self.source_hits.most_common(5)


def _get_or_init_stats() -> SessionStats:
    """Get or initialise the SessionStats object from st.session_state."""
    try:
        import streamlit as st
        if "analytics" not in st.session_state:
            st.session_state.analytics = SessionStats()
        return st.session_state.analytics
    except Exception:
        # CLI context -- return a throw-away object
        return SessionStats()


def record_query(answer: "Answer", retrieved_chunks: list["RetrievedChunk"]) -> None:
    """Record one completed query into the session stats."""
    stats = _get_or_init_stats()
    stats.total_queries += 1

    if answer.grounded:
        stats.grounded_count += 1
        for chunk in retrieved_chunks:
            stats.source_hits[chunk.source_file] += 1
        if retrieved_chunks:
            stats.confidence_scores.append(retrieved_chunks[0].max_similarity)
    else:
        stats.not_grounded_count += 1


def get_session_stats() -> SessionStats:
    return _get_or_init_stats()

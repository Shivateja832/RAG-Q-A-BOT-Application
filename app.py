"""
RAG Q&A Bot — Streamlit UI (v2: security-hardened, incremental indexing,
query rewriting, session analytics)

Security: every user-supplied or document-derived string that appears inside
an unsafe_allow_html block is escaped through html_utils.esc() — prevents
the stored-XSS vector identified in the v1 evaluation where filenames and
chunk text from uploaded documents were rendered raw inside HTML blocks.

Performance: uploading new files uses incremental_indexer.index_new_files()
which skips already-indexed documents instead of re-embedding the full corpus
on every upload.
"""

# ── Torch patch — MUST be first, before any other import ─────────────────────
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

st.set_page_config(
    page_title="RAG Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Stylesheet lives in one constant — never assembled per-value with user data.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-weight: 600; }

.header-wrap { padding: 2rem 0 1.5rem 0; border-bottom: 1px solid #1e2535; margin-bottom: 2rem; }
.header-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e293b; border: 1px solid #334155; border-radius: 999px;
    padding: 4px 12px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; color: #64748b; text-transform: uppercase; margin-bottom: 12px;
}
.badge-dot { width:6px; height:6px; border-radius:50%; background:#3b82f6; display:inline-block; }
.header-title { font-size:2.4rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em; margin:0 0 8px 0; }
.header-sub { font-size:0.95rem; color:#64748b; max-width:560px; line-height:1.6; }

/* Chat bubbles */
.user-row { display:flex; gap:12px; margin:1.2rem 0; align-items:flex-start; }
.bot-row  { display:flex; gap:12px; margin:1.2rem 0; align-items:flex-start; }
.av { width:34px; height:34px; border-radius:8px; display:flex; align-items:center;
      justify-content:center; font-size:15px; flex-shrink:0; }
.av-u { background:#1d4ed8; }
.av-b { background:#1e293b; border:1px solid #334155; }
.bub-u { background:#1e3a5f; border:1px solid #1d4ed8; color:#bfdbfe;
          padding:12px 16px; border-radius:12px; font-size:0.93rem; line-height:1.7; max-width:780px; }

/* Source chips */
.src-wrap { margin-top:10px; padding-top:10px; border-top:1px solid #1e2535; }
.src-label { font-size:10px; font-weight:700; letter-spacing:0.12em;
             text-transform:uppercase; color:#475569; margin-bottom:6px; }
.chip { display:inline-flex; align-items:center; gap:5px; background:#0f172a;
        border:1px solid #1e2535; border-radius:6px; padding:4px 9px;
        font-size:0.76rem; font-family:'JetBrains Mono',monospace; color:#64748b; margin:3px 3px 3px 0; }
.chip-n { color:#3b82f6; font-weight:700; }

/* Confidence badge */
.conf-badge {
    display:inline-flex; align-items:center; gap:4px;
    background:#0f172a; border:1px solid #1e2535; border-radius:6px;
    padding:3px 8px; font-size:0.72rem; font-family:'JetBrains Mono',monospace;
    color:#64748b; margin-left:6px;
}
.conf-high { color:#22c55e; }
.conf-mid  { color:#f59e0b; }
.conf-low  { color:#ef4444; }

/* Chunk card */
.chunk-hdr { font-size:10px; font-weight:700; letter-spacing:0.1em;
             text-transform:uppercase; color:#3b82f6; margin-bottom:4px; }
.chunk-text { color:#94a3b8; font-size:0.85rem; line-height:1.6; margin:4px 0 12px 0; }

/* Not-grounded banner */
.not-grounded {
    background:#1c1008; border:1px solid #92400e; border-left:3px solid #f59e0b;
    border-radius:6px; padding:12px 16px; color:#fbbf24; font-size:0.9rem; margin:4px 0;
}

/* Pipeline tags */
.ptag { display:inline-flex; align-items:center; gap:5px; background:#0f172a;
        border:1px solid #1e2535; border-radius:5px; padding:4px 9px;
        font-size:11px; font-family:'JetBrains Mono',monospace; color:#64748b; margin:3px 0; }
.plbl { font-size:9px; font-weight:700; letter-spacing:0.1em;
        text-transform:uppercase; color:#3b82f6; }

/* Analytics mini-cards */
.stat-row { display:flex; gap:8px; margin:8px 0; }
.stat-card {
    flex:1; background:#0f172a; border:1px solid #1e2535; border-radius:8px;
    padding:10px 12px; text-align:center;
}
.stat-val { font-size:1.3rem; font-weight:700; color:#3b82f6; }
.stat-lbl { font-size:10px; color:#475569; text-transform:uppercase;
            letter-spacing:0.08em; margin-top:2px; }

/* Indexed file list */
.idx-file { font-size:0.78rem; font-family:'JetBrains Mono',monospace;
            color:#64748b; padding:3px 0; }
.idx-dot  { color:#22c55e; margin-right:5px; }

/* Empty state */
.empty { text-align:center; padding:4rem 2rem; }
.empty-icon { font-size:3rem; display:block; margin-bottom:1rem; }
.empty-title { font-size:1.1rem; font-weight:600; color:#475569; margin-bottom:8px; }
.empty-sub { font-size:0.85rem; color:#334155; line-height:1.6; }

/* Inputs */
[data-testid="stTextInput"] input {
    background:#161b27 !important; border:1px solid #1e2535 !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    font-size:0.95rem !important; padding:14px 18px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color:#3b82f6 !important;
    box-shadow:0 0 0 3px rgba(59,130,246,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color:#334155 !important; }

/* Buttons */
.stButton > button {
    background:#2563eb !important; color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:14px 28px !important; width:100% !important;
}
.stButton > button:hover { background:#1d4ed8 !important; }

hr { border-color:#1e2535 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#1e2535; border-radius:3px; }
[data-testid="stFileUploader"] {
    background:#161b27 !important; border:1px dashed #1e2535 !important; border-radius:10px !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.html_utils import esc
from src.ingestion import ingest_directory
from src.chunking import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import (
    get_client, reset_collection, add_chunks,
    get_or_create_collection, collection_is_populated,
)
from src.bm25_index import BM25Index
from src.retrieval import retrieve
from src.generation import generate_answer, Answer, format_citations
from src.incremental_indexer import index_new_files
from src.analytics import record_query, get_session_stats

# ── API key guard ─────────────────────────────────────────────────────────────
if not config.GEMINI_API_KEY:
    st.error(
        "**GEMINI_API_KEY is not set.** Copy `.env.example` to `.env` and add your free key "
        "from [aistudio.google.com/apikey](https://aistudio.google.com/apikey), "
        "then restart the app."
    )
    st.stop()

# ── First-run cold-start indexing ─────────────────────────────────────────────
if not collection_is_populated():
    with st.spinner(
        "First-time setup: building the search index from the bundled document "
        "collection. Downloads the embedding model once, then indexes all documents. "
        "Takes ~60 s and only runs once per deployment…"
    ):
        documents = ingest_directory(config.DATA_DIR)
        chunks    = chunk_documents(documents)
        embeddings = embed_texts([c.text for c in chunks])
        client     = get_client()
        collection = reset_collection(client)
        add_chunks(collection, chunks, embeddings)
        BM25Index(chunks).save()
    st.success("Index built — ready to answer questions.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state:
    st.session_state.messages  = []
if "top_k"     not in st.session_state:
    st.session_state.top_k     = config.DEFAULT_TOP_K
if "rewrite"   not in st.session_state:
    st.session_state.rewrite   = True

# ── Helpers ───────────────────────────────────────────────────────────────────
def _conf_class(sim: float) -> str:
    """Return CSS class for confidence colour coding."""
    if sim >= 0.55:
        return "conf-high"
    if sim >= 0.35:
        return "conf-mid"
    return "conf-low"


def _render_chips(sources: list) -> str:
    """Build source-chip HTML from a deduplicated list of RetrievedChunk.
    All document-derived strings are HTML-escaped."""
    seen: list[tuple] = []
    for c in sources:
        key = (c.source_file, c.page_number)
        if key not in seen:
            seen.append(key)

    chips = ""
    for i, (src, pg) in enumerate(seen, start=1):
        chips += (
            f'<span class="chip">'
            f'<span class="chip-n">[{i}]</span>'
            f'{esc(src)} &sect;{esc(str(pg))}'
            f'</span>'
        )
    return chips


def _confidence_badge(sources: list) -> str:
    """Return an HTML confidence badge based on the top chunk's similarity."""
    if not sources:
        return ""
    sim = getattr(sources[0], "max_similarity", 0.0)
    cls = _conf_class(sim)
    return (
        f'<span class="conf-badge">'
        f'<span class="{cls}">●</span> '
        f'conf {sim:.2f}'
        f'</span>'
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Q&A Bot")
    st.markdown("---")

    # ---- Retrieval controls ----
    st.markdown("#### RETRIEVAL")
    top_k = st.slider(
        "Chunks per query", 1, 10,
        st.session_state.top_k,
        key="top_k_slider",
    )
    st.session_state.top_k = top_k

    rewrite = st.checkbox(
        "Query rewriting",
        value=st.session_state.rewrite,
        key="rewrite_cb",
        help="Expand/clarify the query before retrieval using Gemini (one extra call per question).",
    )
    st.session_state.rewrite = rewrite

    st.markdown("---")

    # ---- Document upload (incremental — skips already-indexed files) ----
    st.markdown("#### DOCUMENTS")

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚡ Index New Documents", use_container_width=True):
            import tempfile
            tmpdir = Path(tempfile.mkdtemp())
            paths  = []
            for f in uploaded:
                dest = tmpdir / f.name
                dest.write_bytes(f.read())
                paths.append(dest)

            with st.spinner("Indexing new files… (existing documents are NOT re-embedded)"):
                try:
                    added = index_new_files(paths)
                    if added:
                        st.success(f"✅ Indexed {len(added)} new file(s): {', '.join(added)}")
                    else:
                        st.info("All uploaded files are already in the index — nothing to do.")
                except Exception as e:
                    st.error(f"Indexing failed: {esc(str(e))}")

    # ---- Currently indexed files ----
    try:
        _meta    = get_or_create_collection().get(include=["metadatas"])["metadatas"]
        _indexed = sorted({m["source_file"] for m in _meta if m})
        if _indexed:
            st.markdown("**Indexed files:**")
            for fname in _indexed:
                st.markdown(
                    f'<div class="idx-file"><span class="idx-dot">✓</span>{esc(fname)}</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    st.markdown("---")

    # ---- Pipeline info ----
    st.markdown("#### PIPELINE")
    for label, value in [
        ("embed",   config.EMBEDDING_MODEL_NAME),
        ("store",   "ChromaDB · persisted"),
        ("search",  "dense + BM25 · RRF"),
        ("rewrite", "Gemini · optional"),
        ("answer",  getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")),
    ]:
        st.markdown(
            f'<div class="ptag"><span class="plbl">{label}</span>{esc(value)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- Session analytics ----
    stats = get_session_stats()
    if stats.total_queries > 0:
        st.markdown("#### SESSION ANALYTICS")
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-card"><div class="stat-val">{stats.total_queries}</div>'
            f'<div class="stat-lbl">Queries</div></div>'
            f'<div class="stat-card"><div class="stat-val">{stats.grounded_rate:.0%}</div>'
            f'<div class="stat-lbl">Grounded</div></div>'
            f'<div class="stat-card"><div class="stat-val">{stats.avg_confidence:.2f}</div>'
            f'<div class="stat-lbl">Avg conf</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if stats.top_sources:
            st.markdown("**Top sources this session:**")
            for src, hits in stats.top_sources:
                st.markdown(
                    f'<div class="idx-file"><span class="idx-dot">◆</span>'
                    f'{esc(src)} ({hits})</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("---")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="header-badge"><span class="badge-dot"></span> RAG · V2</div>
  <div class="header-title">RAG Q&amp;A Bot</div>
  <div class="header-sub">
    Grounded Q&amp;A over your document collection —
    every answer traces back to a numbered source.
    Ask outside the collection and it says so.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty">
          <span class="empty-icon">📄</span>
          <div class="empty-title">No documents queried yet</div>
          <div class="empty-sub">
            Upload documents in the sidebar, click Index, then ask a question below.<br>
            Every answer cites the exact source chunk it came from.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # User query — always escaped before embedding in HTML
                st.markdown(
                    f'<div class="user-row">'
                    f'<div class="av av-u">👤</div>'
                    f'<div class="bub-u">{esc(msg["content"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                answer: Answer = msg["answer"]

                # Strip the auto-appended Sources footer — we render it ourselves
                text = answer.text
                if "Sources:\n" in text:
                    prose, _ = text.rsplit("Sources:\n", 1)
                else:
                    prose = text
                prose = prose.strip()

                # Bot row wrapper
                st.markdown(
                    '<div class="bot-row"><div class="av av-b">🤖</div><div style="flex:1">',
                    unsafe_allow_html=True,
                )

                if not answer.grounded:
                    # Not-grounded — render as styled banner, text escaped
                    st.markdown(
                        f'<div class="not-grounded">{esc(prose)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Grounded prose — use st.write (safe, no unsafe_allow_html)
                    st.write(prose)

                    # Confidence badge + source chips — all doc data escaped
                    if answer.sources:
                        badge = _confidence_badge(answer.sources)
                        chips = _render_chips(answer.sources)
                        st.markdown(
                            f'<div class="src-wrap">'
                            f'<div class="src-label">Cited Sources {badge}</div>'
                            f'{chips}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("</div></div>", unsafe_allow_html=True)

                # Expandable raw chunks — all document content escaped
                if answer.grounded and answer.sources:
                    with st.expander("📋 View source chunks", expanded=False):
                        seen_keys: list[tuple] = []
                        idx = 1
                        for c in answer.sources:
                            key = (c.source_file, c.page_number)
                            if key in seen_keys:
                                continue
                            seen_keys.append(key)
                            st.markdown(
                                f'<div class="chunk-hdr">'
                                f'[{idx}] {esc(c.source_file)} &middot; &sect;{esc(str(c.page_number))}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            excerpt = c.text[:400] + ("…" if len(c.text) > 400 else "")
                            # Chunk text escaped — could contain HTML from the document
                            st.markdown(
                                f'<p class="chunk-text">{esc(excerpt)}</p>',
                                unsafe_allow_html=True,
                            )
                            idx += 1

# ── Input bar ─────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "query",
        placeholder="Ask a question about your documents…",
        label_visibility="collapsed",
        key="query_input",
    )
with col2:
    send = st.button("Send →", use_container_width=True)

# ── Query execution ───────────────────────────────────────────────────────────
if send and query.strip():
    q = query.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("Searching & generating…"):
        try:
            chunks = retrieve(
                q,
                top_k=st.session_state.top_k,
                use_query_rewriting=st.session_state.rewrite,
            )
            answer = generate_answer(q, chunks)
        except Exception as e:
            answer = Answer(text=f"Error: {esc(str(e))}", sources=[], grounded=False)

    # Record into session analytics
    record_query(answer, chunks if "chunks" in dir() else [])

    st.session_state.messages.append({"role": "assistant", "answer": answer})
    st.rerun()

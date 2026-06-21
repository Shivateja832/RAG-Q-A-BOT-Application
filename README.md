# Document Q&A Bot — Hybrid RAG Pipeline

A production-quality Retrieval-Augmented Generation (RAG) system that answers natural language questions over a collection of PDF, DOCX, and TXT documents, with accurate page/section-level source citations and a strict grounding guardrail that prevents the model from answering outside the provided context.

Ask a question → the system optionally **rewrites** it for better retrieval → retrieves the most relevant passages using **hybrid search (dense vector + BM25 keyword, fused via Reciprocal Rank Fusion)** → Google Gemini generates an answer using *only* that retrieved context, with deterministic citations.

---

## What makes this implementation different

Most basic RAG implementations do top-k vector similarity search, eyeball five demo queries, and call it done. This one adds:

- **Query rewriting** — an optional Gemini call before retrieval reformulates short, ambiguous, or pronoun-heavy queries into a form that better matches indexed chunks, improving recall without changing the user's intent. Degrades gracefully (uses original query) if the API call fails.
- **A labeled retrieval evaluation harness** (`scripts/eval_retrieval.py`) — 15 hand-labeled queries with ground-truth (source, page) pairs, scored against Hit@K and Mean Reciprocal Rank, the standard IR metrics for measuring retrieval quality. Includes an ablation mode (`--ablation`) that runs the same query set through dense-only, BM25-only, and the production hybrid pipeline side by side, so the hybrid design decision is empirically justified rather than just asserted in prose. Also reports the grounding guardrail's true-negative rate on deliberately out-of-scope questions.
- **Hybrid retrieval** — dense embeddings (semantic similarity) fused with BM25 (exact keyword matching) via Reciprocal Rank Fusion. Pure vector search misses queries that hinge on exact terms (names, numbers, acronyms); pure keyword search misses paraphrased queries.
- **Three-layer grounding guardrail** — (1) a cheap cosine-similarity pre-filter catches obviously irrelevant queries before spending an API call, (2) a strict system prompt instructs Gemini to explicitly decline when retrieved context doesn't support an answer, and (3) the citation footer is built **deterministically in Python** from the exact chunks sent to the model — never parsed out of the model's free-text reply.
- **Incremental indexing** — uploading a new document in the Streamlit UI only indexes the *new* file; already-indexed documents are not re-embedded. This avoids the full-corpus re-indexing cost on every upload.
- **XSS-safe UI** — all user-supplied data (queries, filenames, chunk text) is HTML-escaped via `html.escape()` before being embedded in `unsafe_allow_html` blocks. Prevents stored-XSS through crafted filenames or document content.
- **Session analytics dashboard** — the sidebar tracks total queries, grounded-answer rate, average retrieval confidence, and most-cited source files for the current session.
- **Page/section-accurate citations** — not just "see `document.pdf`," but the exact page number (PDF) or section (DOCX/TXT) a chunk came from, extracted during ingestion and carried through the entire pipeline.
- **Clean indexing/query separation** — `index.py` is the only thing that embeds and writes to the vector store. `app.py` / `cli.py` only ever read from it.
- **Unit tests** — 40+ tests covering chunking boundary conditions, ingestion text-cleaning, RRF fusion correctness, grounding guardrail, citation-footer determinism, XSS-escape safety, query rewriter degradation, analytics stats, and the eval harness's own metric math.

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Language | Python | 3.11+ |
| PDF parsing | `pdfplumber` | 0.11.4 |
| DOCX parsing | `python-docx` | 1.2.0 |
| Embedding model | `sentence-transformers` (`all-MiniLM-L6-v2`) | 3.3.1 |
| Vector database | `chromadb` (persistent) | 0.5.23 |
| Sparse/keyword search | `rank-bm25` | 0.2.2 |
| LLM — query rewriting + answer generation | `google-genai` SDK → Gemini 2.5 Flash | 2.9.0 |
| Web UI | `streamlit` | 1.40.2 |
| Testing | `pytest`, `pytest-cov` | 8.3.4 / 6.0.0 |

---

## Architecture Overview

```
                         ┌──────────────────────┐
                         │  data/ (PDF/DOCX/TXT) │
                         └───────────┬──────────┘
                                     │
   INDEXING (index.py)              ▼
   ───────────────────       ┌─────────────┐
                              │  Ingestion   │  pdfplumber / python-docx
                              │ (per page/   │  strips headers/footers/
                              │  section)    │  page-number artifacts
                              └──────┬──────┘
                                     ▼
                              ┌─────────────┐
                              │  Chunking    │  paragraph-aware,
                              │ (800 chars,  │  fixed-size + overlap
                              │  150 overlap)│
                              └──────┬──────┘
                                     ▼
                              ┌─────────────┐
                              │  Embedding   │  sentence-transformers
                              │  (batched)   │  all-MiniLM-L6-v2 (local)
                              └──────┬──────┘
                                     ▼
                        ┌────────────┴────────────┐
                        ▼                          ▼
                ┌──────────────┐          ┌──────────────┐
                │   ChromaDB    │          │  BM25 Index   │
                │ (persisted to │          │ (persisted to │
                │ vector_store/)│          │ vector_store/)│
                └──────────────┘          └──────────────┘

   QUERYING (app.py / cli.py)  — never re-indexes, only reads the above
   ───────────────────────────
        User question
              │
              ▼
     ┌─────────────────┐
     │  Query rewriting  │  optional Gemini call — expand/clarify query
     │  (optional)       │  degrades gracefully if API call fails
     └────────┬─────────┘
              ▼
     ┌─────────────────────────────┐
     │  Hybrid retrieval             │
     │  Dense (ChromaDB) + Sparse    │
     │  (BM25) → fused via RRF       │
     └────────┬─────────────────────┘
              ▼
     ┌─────────────────────────────┐
     │  Relevance gate                │  low similarity → skip LLM call,
     │  (coarse pre-filter)           │  return "not found" directly
     └────────┬─────────────────────┘
              ▼
     ┌─────────────────────────────┐
     │  Gemini (generation.py)       │  strict system prompt: answer
     │  with retrieved context       │  ONLY from context, cite sources,
     │                               │  decline if context insufficient
     └────────┬─────────────────────┘
              ▼
     ┌─────────────────────────────┐
     │  HTML-escape all output       │  prevents XSS in UI rendering
     │  + deterministic citation     │  footer built from chunk metadata
     └────────┬─────────────────────┘
              ▼
        Answer + citations + session analytics
```

---

## Chunking Strategy

**Strategy chosen: paragraph-aware fixed-size chunking with character overlap**

Rationale:

1. **Paragraph boundaries respected first** — text is split on natural paragraph breaks (`\n`) before enforcing the character budget. This avoids cutting sentences in half in the common case, preserving readability and semantic coherence within each chunk.
2. **Fixed-size budget (800 chars)** — small enough that each chunk fits comfortably in an LLM context window alongside other chunks, and large enough to contain a complete thought with context. At ~150–200 words per chunk, each chunk is typically one complete paragraph or a small cluster of related sentences.
3. **150-character overlap** — the tail of each chunk is prepended to the next, so a sentence that straddles a chunk boundary is fully present in at least one chunk. This prevents the "split paragraph" retrieval failure mode where neither chunk individually contains enough context to answer the question.
4. **Per-page/section, not per-document** — chunking runs on each page (PDF) or heading section (DOCX) independently, so chunk metadata always carries an accurate location reference.

Alternatives considered:
- **Sentence-based chunking**: Would require a reliable sentence segmenter (spaCy, NLTK), adding a heavy dependency for marginal improvement over paragraph splitting on this corpus.
- **Token-based chunking**: More precise for LLM context limits, but adds tokeniser overhead and couples the chunking strategy to a specific model's tokenisation. Character-based is simpler and predictable.

---

## Embedding Model and Vector Database

**Embedding model: `all-MiniLM-L6-v2` (via `sentence-transformers`)**

- Runs locally with no API key or network call required at index time.
- 384-dimensional vectors, ~80MB model size — fast on CPU for a corpus this size.
- Well-established benchmark performance on semantic similarity tasks; the standard starting point for local embedding.
- Normalised embeddings (L2-normalised at encode time) so cosine similarity reduces to a dot product, which ChromaDB's HNSW index handles efficiently.

**Vector database: ChromaDB (persistent)**

- Zero-infrastructure setup: a single `PersistentClient(path=...)` call; no Docker, no server process.
- Persists to disk (`vector_store/`) so the index survives restarts; `index.py` is the only process that writes to it.
- HNSW cosine-space index for fast approximate nearest-neighbour search on CPU.
- Native batch insert (`collection.add(ids=[...], embeddings=[...], ...)`) satisfies the assignment's "batched embedding" requirement at the storage layer too.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-qa-bot
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

### 3. Build the index (one-time)

```bash
python index.py
```

This ingests `data/`, chunks, embeds, and persists the ChromaDB + BM25 index to `vector_store/`. Takes ~60 seconds on first run (model download + embedding). Subsequent runs are much faster.

### 4. Run the app

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

**Command-line:**
```bash
python cli.py
python cli.py --top-k 8
python cli.py --no-rewrite   # skip query rewriting for faster responses
```

### 5. Run tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google AI Studio API key (free tier) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `MAX_TOKENS` | No | `1024` | Max tokens in generated answer |
| `CHUNK_SIZE` | No | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | No | `150` | Overlap characters between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | No | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_BATCH_SIZE` | No | `32` | Batch size for embedding |
| `CHROMA_COLLECTION_NAME` | No | `document_qa_collection` | ChromaDB collection name |
| `DEFAULT_TOP_K` | No | `5` | Default number of chunks to retrieve |
| `DENSE_WEIGHT` | No | `0.6` | RRF weight for dense (vector) search |
| `SPARSE_WEIGHT` | No | `0.4` | RRF weight for sparse (BM25) search |
| `MIN_SIMILARITY_SCORE` | No | `0.15` | Min cosine similarity to trigger LLM call |

**Never commit actual key values. Always use `.env` locally or Streamlit secrets in deployment.**

---

## Document Collection

Five documents covering diverse topics to enable interesting cross-document queries:

| File | Format | Topic |
|---|---|---|
| `Renewable_Energy_Transition_Report.pdf` | PDF | Solar, wind, storage, LCOE trends |
| `Gut_Microbiome_and_Human_Health.pdf` | PDF | Microbiome science, diet, disease links |
| `Hydrothermal_Vents_Deep_Ocean.docx` | DOCX | Deep-sea ecosystems, chemosynthesis |
| `Printing_Press_Information_Revolution.docx` | DOCX | Gutenberg, information spread, social impact |
| `Startup_Fundraising_VC_Guide.txt` | TXT | Venture capital, term sheets, due diligence |

---

## Example Queries

| Query | Expected answer themes |
|---|---|
| What is the levelized cost of electricity for solar? | LCOE figures, cost trends from the renewable energy report |
| How do gut bacteria influence mental health? | Gut-brain axis, microbiome, depression links |
| What organisms live near hydrothermal vents? | Chemosynthesis, tube worms, extremophiles |
| How did the printing press change society? | Literacy rates, Reformation, standardisation |
| What is a term sheet in venture capital? | VC term definitions, valuation, liquidation preference |
| How does chemosynthesis differ from photosynthesis? | Energy source (chemical vs. light), organisms involved |
| What are probiotic foods? | Fermented foods, lactobacillus, gut health |
| What is a Series A funding round? | Early-stage VC, milestones, typical raise size |
| How much has wind energy capacity grown globally? | Capacity figures, growth percentages from report |
| Who was Johannes Gutenberg? | Inventor context, movable type, Germany |

---

## Known Limitations

- **Retrieval quality bounded by embedding model** — `all-MiniLM-L6-v2` is a general-purpose model not fine-tuned on any of these document topics. Domain-specific queries using very technical jargon may retrieve suboptimal chunks.
- **No multi-turn conversation context** — each query is treated independently. If you say "tell me more about that," the system has no memory of the previous answer.
- **PDF layout artefacts** — complex PDFs with multi-column layouts, tables, or scanned pages (requiring OCR) may produce garbled extracted text. The current ingestion uses `pdfplumber` which is best for single-column text-based PDFs.
- **Chunk boundary fragmentation** — even with paragraph-aware chunking and overlap, a piece of reasoning that spans many paragraphs may be split across chunks, weakening retrieval for questions that require aggregating information across a long passage.
- **Gemini rate limits** — the free tier has per-minute request limits. Heavy use in quick succession may produce API errors (surfaced cleanly, not crashes).
- **BM25 exact-match only** — the sparse search component uses simple tokenisation (`re.findall(r"[a-z0-9]+", ...)`). Stemming and stop-word removal are not applied, so queries with different word forms (e.g., "photosynthesise" vs "photosynthesis") may not match via BM25 (though dense search will still handle them).
- **In-memory BM25 index** — the BM25 index is loaded into memory on each query. For very large corpora (millions of chunks), this would need to move to a disk-backed store.

---

## Retrieval Evaluation

Run the labeled evaluation harness:

```bash
# Standard eval (hybrid pipeline)
python scripts/eval_retrieval.py

# Ablation: compare dense-only vs BM25-only vs hybrid
python scripts/eval_retrieval.py --ablation
```

Metrics reported: **Hit@1**, **Hit@3**, **Hit@5**, **MRR** (Mean Reciprocal Rank).

---

## Deployment (Streamlit Community Cloud)

1. Push your repo to GitHub (ensure `data/` and all source files are committed; `vector_store/` should be in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo + `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy. On first boot, the app auto-indexes the document collection (the spinner will show while this runs).

See `DEPLOYMENT.md` for full details.

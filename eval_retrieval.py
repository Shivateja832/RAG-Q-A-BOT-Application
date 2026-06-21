"""
Retrieval evaluation harness.

Most student RAG submissions demo 5 queries by hand and call it done.
This script instead runs every query in scripts/eval_dataset.py through
the *real* retrieval pipeline (the same retrieve() used by cli.py and
app.py -- not a separate mock) and reports standard information-retrieval
metrics:

  Hit@K       -- for what fraction of queries does at least one chunk from
                 the correct source/page appear anywhere in the top-K
                 retrieved results? The basic "did we find it at all" check.
  MRR         -- Mean Reciprocal Rank: averages 1/rank of the first correct
                 chunk across all queries. Rewards ranking the right answer
                 near the top, not just somewhere in the top-K.
  Guardrail   -- of the genuinely out-of-scope questions in the eval set
  TNR           (expected_sources=None), what fraction does the system
                 correctly decline to answer rather than hallucinate?
                 This is the "true negative rate" -- a RAG system that
                 never declines isn't grounded, it's just always confident.

Methodology note: ground truth in eval_dataset.py was assigned by manually
reading the source documents and noting which page/section actually
contains the answer -- not derived from running this script and copying
its own output, which would make the metric circular and meaningless.

Usage:
    python scripts/eval_retrieval.py            # retrieval metrics only (fast, no API calls)
    python scripts/eval_retrieval.py --full      # also calls Gemini to check the guardrail's
                                                  # true-negative rate on out-of-scope questions
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import retrieve
from src.vector_store import collection_is_populated
from scripts.eval_dataset import EVAL_SET


def _matches(chunk, expected_sources) -> bool:
    return any(
        chunk.source_file == src and chunk.page_number == pg
        for src, pg in expected_sources
    )


def _score_run(results_by_query, answerable):
    """Shared scoring logic for evaluate_retrieval and the ablation runs."""
    hits = 0
    reciprocal_ranks = []
    for ex, retrieved in zip(answerable, results_by_query):
        rank_of_first_hit = None
        for i, chunk in enumerate(retrieved, start=1):
            if _matches(chunk, ex["expected_sources"]):
                rank_of_first_hit = i
                break
        hit = rank_of_first_hit is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank_of_first_hit if hit else 0.0)
    return hits / len(answerable), sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_ablation(top_k: int = 5):
    """Compares dense-only, sparse-only (BM25), and the production hybrid
    (RRF fusion) retrieval on the same labeled query set. This is the
    empirical justification for the hybrid design decision documented in
    the README -- a number proving the choice, not just prose asserting it.
    """
    from src import config as cfg
    from src.retrieval import _dense_search, _sparse_search, _reciprocal_rank_fusion, RetrievedChunk

    answerable = [ex for ex in EVAL_SET if ex["expected_sources"] is not None]

    dense_only_results, sparse_only_results, hybrid_results = [], [], []

    for ex in answerable:
        candidate_pool = max(top_k * 3, 10)
        dense_raw = _dense_search(ex["query"], candidate_pool)
        sparse_raw = _sparse_search(ex["query"], candidate_pool)

        dense_only = [
            RetrievedChunk(text=d["text"], source_file=d["source_file"],
                            page_number=d["page_number"], score=d.get("similarity", 0))
            for d in dense_raw[:top_k]
        ]
        sparse_only = [
            RetrievedChunk(text=d["text"], source_file=d["source_file"],
                            page_number=d["page_number"], score=d.get("bm25_score", 0))
            for d in sparse_raw[:top_k]
        ]
        hybrid = _reciprocal_rank_fusion(dense_raw, sparse_raw)[:top_k]

        dense_only_results.append(dense_only)
        sparse_only_results.append(sparse_only)
        hybrid_results.append(hybrid)

    dense_hit, dense_mrr = _score_run(dense_only_results, answerable)
    sparse_hit, sparse_mrr = _score_run(sparse_only_results, answerable)
    hybrid_hit, hybrid_mrr = _score_run(hybrid_results, answerable)

    print(f"\nAblation: dense-only vs BM25-only vs hybrid (top_k={top_k}, "
          f"{len(answerable)} labeled queries)\n")
    print(f"{'Method':<20} {'Hit@' + str(top_k):<10} {'MRR':<10}")
    print("-" * 40)
    print(f"{'Dense only':<20} {dense_hit:<10.1%} {dense_mrr:<10.3f}")
    print(f"{'BM25 only':<20} {sparse_hit:<10.1%} {sparse_mrr:<10.3f}")
    print(f"{'Hybrid (RRF)':<20} {hybrid_hit:<10.1%} {hybrid_mrr:<10.3f}")
    return {
        "dense": (dense_hit, dense_mrr),
        "sparse": (sparse_hit, sparse_mrr),
        "hybrid": (hybrid_hit, hybrid_mrr),
    }


def evaluate_retrieval(top_k: int = 5):
    answerable = [ex for ex in EVAL_SET if ex["expected_sources"] is not None]
    hits = 0
    reciprocal_ranks = []
    rows = []

    for ex in answerable:
        retrieved = retrieve(ex["query"], top_k=top_k)
        rank_of_first_hit = None
        for i, chunk in enumerate(retrieved, start=1):
            if _matches(chunk, ex["expected_sources"]):
                rank_of_first_hit = i
                break

        hit = rank_of_first_hit is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank_of_first_hit if hit else 0.0)
        rows.append((ex["query"], hit, rank_of_first_hit))

    hit_at_k = hits / len(answerable)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"Retrieval evaluation (top_k={top_k}, {len(answerable)} labeled queries)\n")
    print(f"{'Query':<70} {'Hit':<5} {'Rank':<5}")
    print("-" * 82)
    for query, hit, rank in rows:
        q_display = (query[:67] + "...") if len(query) > 70 else query
        print(f"{q_display:<70} {'✓' if hit else '✗':<5} {rank if rank else '-':<5}")

    print("\n" + "=" * 82)
    print(f"Hit@{top_k}: {hit_at_k:.1%}  ({hits}/{len(answerable)} queries retrieved the correct source)")
    print(f"MRR:    {mrr:.3f}  (1.0 = correct source always ranked #1)")
    print("=" * 82)
    return hit_at_k, mrr


def evaluate_guardrail():
    """Calls the real generation pipeline (requires GEMINI_API_KEY) to
    verify out-of-scope questions are correctly declined, not hallucinated."""
    from src.generation import generate_answer

    unanswerable = [ex for ex in EVAL_SET if ex["expected_sources"] is None]
    correct_declines = 0

    print(f"\nGuardrail evaluation ({len(unanswerable)} out-of-scope queries)\n")
    for ex in unanswerable:
        retrieved = retrieve(ex["query"], top_k=5)
        answer = generate_answer(ex["query"], retrieved)
        declined = not answer.grounded
        correct_declines += int(declined)
        print(f"  [{'✓ declined' if declined else '✗ ANSWERED (bad)'}] {ex['query']}")

    tnr = correct_declines / len(unanswerable)
    print(f"\nGuardrail true-negative rate: {tnr:.1%} ({correct_declines}/{len(unanswerable)})")
    return tnr


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against labeled ground truth.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full", action="store_true",
                         help="Also evaluate the generation guardrail (calls the Gemini API).")
    parser.add_argument("--ablation", action="store_true",
                         help="Compare dense-only vs BM25-only vs hybrid retrieval on the same eval set.")
    args = parser.parse_args()

    if not collection_is_populated():
        print("ERROR: No index found. Run 'python index.py' first.")
        sys.exit(1)

    evaluate_retrieval(top_k=args.top_k)

    if args.ablation:
        evaluate_ablation(top_k=args.top_k)

    if args.full:
        evaluate_guardrail()
    else:
        print("\n(Run with --full to also evaluate the grounding guardrail on out-of-scope "
              "questions -- requires GEMINI_API_KEY and makes live API calls.)")


if __name__ == "__main__":
    main()

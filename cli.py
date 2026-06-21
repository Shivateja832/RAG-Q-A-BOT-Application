"""
Command-line interactive Q&A loop -- the minimum required interface per
the assignment (section 4.7).

Usage:
    python cli.py
    python cli.py --top-k 8
    python cli.py --no-rewrite   # disable query rewriting
"""
import argparse
import sys

from src import config
from src.retrieval import retrieve
from src.generation import generate_answer
from src.vector_store import collection_is_populated


def print_sources(sources):
    if not sources:
        return
    print("\nSources used:")
    for i, s in enumerate(sources, start=1):
        snippet = s.text[:160].replace("\n", " ") + ("..." if len(s.text) > 160 else "")
        confidence = f" | confidence: {s.max_similarity:.3f}" if s.max_similarity else ""
        print(f"  [{i}] {s.source_file}, page/section {s.page_number}{confidence}")
        print(f"      \"{snippet}\"")


def main():
    parser = argparse.ArgumentParser(description="Document Q&A Bot (RAG) -- CLI")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                         help=f"Number of chunks to retrieve (default: {config.DEFAULT_TOP_K})")
    parser.add_argument("--no-rewrite", action="store_true",
                         help="Disable query rewriting/expansion (faster, but lower recall)")
    args = parser.parse_args()

    if not collection_is_populated():
        print("ERROR: No index found. Run 'python index.py' first to build the vector store.")
        sys.exit(1)

    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Copy .env.example to .env and add your free "
              "key from https://aistudio.google.com/apikey, then try again.")
        sys.exit(1)

    use_rewrite = not args.no_rewrite and bool(config.GEMINI_API_KEY)

    print("=" * 70)
    print("Document Q&A Bot -- CLI mode")
    print(f"  top_k = {args.top_k}  |  query rewriting = {'on' if use_rewrite else 'off'}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        retrieved = retrieve(query, top_k=args.top_k, use_query_rewriting=use_rewrite)
        try:
            answer = generate_answer(query, retrieved)
        except Exception as e:
            print(f"\nUnexpected error while generating the answer: {e}")
            print("You can try again, or type 'exit' to quit.")
            continue

        print("\n" + "-" * 70)
        print("ANSWER:")
        print(answer.text)
        print_sources(answer.sources)
        print("-" * 70)


if __name__ == "__main__":
    main()

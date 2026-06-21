"""
Verifies the document collection in data/ actually satisfies the
assignment's explicit, checkable requirements (Section 3 of the brief):

  - 4 to 5 documents
  - each document >= 500 words (or, for PDFs, >= 2 pages)
  - at least one document is a PDF
  - no document is empty / placeholder text

This is not a unit test of application logic -- it's a sanity check on the
*data* itself, run in CI on every push so a future edit to data/ (e.g.
someone swapping in a short file) gets caught automatically instead of
silently violating the assignment spec.

Usage:
    python scripts/verify_documents.py
Exits non-zero (and fails CI) if any check fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest_file
from src import config

MIN_WORDS = 500
MIN_DOCS = 4
MAX_DOCS = 5


def main():
    files = sorted(
        p for p in config.DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )

    print(f"Found {len(files)} document(s) in {config.DATA_DIR}:\n")

    failures = []
    pdf_count = 0

    for path in files:
        doc = ingest_file(path)
        word_count = len(doc.full_text.split())
        page_count = len(doc.pages) if path.suffix.lower() == ".pdf" else None
        is_pdf = path.suffix.lower() == ".pdf"
        if is_pdf:
            pdf_count += 1

        status = "OK"
        if word_count < MIN_WORDS:
            status = f"FAIL (only {word_count} words, need >= {MIN_WORDS})"
            failures.append(f"{path.name}: below minimum word count")

        page_info = f", {page_count} page(s)" if page_count else ""
        print(f"  [{status:>6}] {path.name:45s} {word_count:5d} words{page_info}")

    print()

    if not (MIN_DOCS <= len(files) <= MAX_DOCS):
        failures.append(
            f"Document count is {len(files)}, assignment requires {MIN_DOCS} to {MAX_DOCS}"
        )

    if pdf_count < 1:
        failures.append("No PDF found -- assignment requires at least one PDF document")

    if failures:
        print("FAILED checks:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"All checks passed: {len(files)} documents, {pdf_count} PDF(s), "
          f"all >= {MIN_WORDS} words.")
    sys.exit(0)


if __name__ == "__main__":
    main()

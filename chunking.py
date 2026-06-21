"""
Text chunking.

Strategy: paragraph-aware fixed-size chunking with overlap (documented in
README.md under "Chunking Strategy"). We don't blindly cut at N characters --
we accumulate whole paragraphs up to the target size, which avoids slicing
mid-sentence in the common case, then apply a character-based overlap window
between consecutive chunks so context isn't lost at chunk boundaries.

Each chunk carries metadata: source filename + page/section number, so
retrieval results can always be cited precisely.
"""
from dataclasses import dataclass
import uuid

from . import config
from .ingestion import IngestedDocument


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int  # position within the document, for debugging/ordering


def _split_into_paragraphs(text: str) -> list:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs


def chunk_page(text: str, source_file: str, page_number: int, start_index: int,
                chunk_size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> list:
    """Chunk a single page/section's text. Returns list[Chunk]."""
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current = ""
    idx = start_index

    def flush(buf: str):
        nonlocal idx
        buf = buf.strip()
        if buf:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=buf,
                source_file=source_file,
                page_number=page_number,
                chunk_index=idx,
            ))
            idx += 1

    for para in paragraphs:
        # A single oversized paragraph: hard-split it on its own.
        if len(para) > chunk_size:
            if current:
                flush(current)
                current = ""
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i:i + chunk_size]
                flush(piece)
            continue

        candidate = (current + "\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            overlap_tail = current[-overlap:] if overlap > 0 else ""
            flush(current)
            new_start = (overlap_tail + "\n" + para).strip()
            if len(new_start) <= chunk_size:
                current = new_start
            else:
                # Even overlap_tail + this paragraph alone exceeds chunk_size
                # (e.g. a long paragraph following a full previous chunk).
                # Drop the overlap rather than overflow the size budget.
                current = para if len(para) <= chunk_size else para[:chunk_size]
                if len(para) > chunk_size:
                    # shouldn't happen (oversized paragraphs are handled above),
                    # but guard anyway for safety.
                    flush(current)
                    current = ""

    flush(current)
    return chunks


def chunk_document(document: IngestedDocument) -> list:
    """Chunk every page/section of a document. Returns list[Chunk]."""
    all_chunks = []
    next_index = 0
    for page in document.pages:
        page_chunks = chunk_page(
            text=page.text,
            source_file=page.source_file,
            page_number=page.page_number,
            start_index=next_index,
        )
        all_chunks.extend(page_chunks)
        next_index += len(page_chunks)
    return all_chunks


def chunk_documents(documents: list) -> list:
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        print(f"  {doc.source_file}: {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)
    return all_chunks

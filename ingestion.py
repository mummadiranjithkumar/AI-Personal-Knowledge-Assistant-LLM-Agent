from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from vector_store import VectorStore


DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"


def _ensure_dirs() -> None:
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def compute_doc_id(content_bytes: bytes, filename: str) -> str:
    """
    Compute a stable identifier for a document based on its contents and name.
    """
    h = hashlib.sha256()
    h.update(content_bytes)
    h.update(filename.encode("utf-8"))
    return h.hexdigest()


def _extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_chunks: List[str] = []
    for page in reader.pages:
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_chunks)


def _extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_from_md(path: Path) -> str:
    # For now treat markdown as plain text
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Simple sliding window text splitter.
    """
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap

    return chunks


def ingest_streamlit_files(files, vector_store: VectorStore) -> List[Tuple[str, str]]:
    """
    Ingest a list of Streamlit UploadedFile objects into the vector store.

    Returns a list of (filename, status) for display in the UI.
    """
    _ensure_dirs()

    results: List[Tuple[str, str]] = []

    for f in files:
        try:
            content = f.read()
            filename = f.name
            doc_id = compute_doc_id(content, filename)

            if vector_store.has_document(doc_id):
                results.append((filename, "skipped (already indexed)"))
                continue

            # Persist file to disk for reproducibility / debugging
            dest_path = UPLOADS_DIR / filename
            with dest_path.open("wb") as out:
                out.write(content)

            suffix = dest_path.suffix.lower()
            if suffix == ".pdf":
                text = _extract_text_from_pdf(dest_path)
            elif suffix == ".txt":
                text = _extract_text_from_txt(dest_path)
            elif suffix in {".md", ".markdown"}:
                text = _extract_text_from_md(dest_path)
            else:
                results.append((filename, f"unsupported file type: {suffix}"))
                continue

            chunks = split_text(text)
            if not chunks:
                results.append((filename, "no text content found"))
                continue

            vector_store.add_document(doc_id=doc_id, source=filename, chunk_texts=chunks)
            results.append((filename, f"indexed {len(chunks)} chunks"))

        except Exception as e:
            results.append((getattr(f, "name", "unknown"), f"error: {e}"))

    return results


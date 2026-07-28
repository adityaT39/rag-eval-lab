"""
Splits raw documents into overlapping text chunks for embedding.
Chunk size/overlap are parameters so we can run experiments comparing them.
"""
import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(normalized_text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", normalized_text)


def chunk_document(doc_id: str, title: str, text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Greedily packs sentences into chunks of ~chunk_size characters,
    carrying `overlap` characters of context into the next chunk.

    Each chunk also records its (start_char, end_char) position in the
    normalized document text. This lets us compare "did retrieval find the
    right passage" across DIFFERENT chunk_size configs by checking character
    overlap rather than exact chunk_id equality -- chunk_ids are only
    comparable within a single chunk_size run, but character positions are
    comparable across all of them since they index into the same underlying
    normalized text.
    """
    normalized = normalize(text)
    sentences = split_into_sentences(normalized)
    raw_chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            raw_chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += (" " if current else "") + sentence

    if current.strip():
        raw_chunks.append(current.strip())

    kept = [(i, c) for i, c in enumerate(raw_chunks) if len(c) > 50]  # drop near-empty tail chunks

    result = []
    cursor = 0
    for i, chunk_text in kept:
        start = normalized.find(chunk_text, cursor)
        if start == -1:  # overlap can rewind past cursor; fall back to a full search
            start = normalized.find(chunk_text)
        end = start + len(chunk_text)
        cursor = start + 1  # allow the next chunk's overlap region to be found again

        result.append({
            "chunk_id": f"{doc_id}::{i}",
            "doc_id": doc_id,
            "title": title,
            "text": chunk_text,
            "start_char": start,
            "end_char": end,
        })

    return result


def build_chunks(chunk_size: int = 500, overlap: int = 100, out_name: str = "chunks.json") -> list[dict]:
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    all_chunks = []

    for entry in manifest:
        text = (RAW_DIR / f"{entry['doc_id']}.txt").read_text(encoding="utf-8")
        all_chunks.extend(chunk_document(entry["doc_id"], entry["title"], text, chunk_size, overlap))

    out_path = PROCESSED_DIR / out_name
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Built {len(all_chunks)} chunks from {len(manifest)} documents -> {out_path}")
    return all_chunks


if __name__ == "__main__":
    build_chunks()

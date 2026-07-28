"""
Pure retrieval-scoring logic: no ML dependencies, so it can be unit-tested
and run in CI without installing torch/sentence-transformers/chromadb.

A retrieval "hit" is judged by character-span overlap with the ground-truth
chunk, not exact chunk_id equality -- chunk boundaries (and therefore IDs)
shift when chunk_size changes, so an exact-ID match is only meaningful
within a single chunk_size config. Character positions are stable across
configs because they index into the same underlying normalized document
text (see chunk.py).
"""


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def is_hit(chunk_id: str, offset_map: dict, expected_doc_id: str, expected_start: int, expected_end: int) -> bool:
    entry = offset_map.get(chunk_id)
    if entry is None:
        return False
    doc_id, start, end = entry
    return doc_id == expected_doc_id and spans_overlap(start, end, expected_start, expected_end)


def recall_at_k(retrieved_ids: list[str], offset_map: dict, expected_doc_id: str, expected_start: int, expected_end: int) -> int:
    return 1 if any(is_hit(cid, offset_map, expected_doc_id, expected_start, expected_end) for cid in retrieved_ids) else 0


def reciprocal_rank(retrieved_ids: list[str], offset_map: dict, expected_doc_id: str, expected_start: int, expected_end: int) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if is_hit(cid, offset_map, expected_doc_id, expected_start, expected_end):
            return 1.0 / rank
    return 0.0


def build_offset_map(chunks: list[dict]) -> dict:
    return {c["chunk_id"]: (c["doc_id"], c["start_char"], c["end_char"]) for c in chunks}

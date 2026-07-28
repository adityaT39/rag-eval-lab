import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scoring import (
    build_offset_map,
    is_hit,
    recall_at_k,
    reciprocal_rank,
    spans_overlap,
)


def test_spans_overlap_true_for_overlapping_ranges():
    assert spans_overlap(0, 100, 50, 150) is True
    assert spans_overlap(50, 150, 0, 100) is True


def test_spans_overlap_false_for_disjoint_ranges():
    assert spans_overlap(0, 100, 200, 300) is False


def test_spans_overlap_false_for_merely_touching_ranges():
    # [0, 100) and [100, 200) share the boundary point but no characters
    assert spans_overlap(0, 100, 100, 200) is False


def _sample_offset_map():
    return {
        "doc_a::0": ("doc_a", 0, 100),
        "doc_a::1": ("doc_a", 100, 200),
        "doc_b::0": ("doc_b", 0, 100),
    }


def test_is_hit_requires_same_doc_id_and_overlap():
    offset_map = _sample_offset_map()
    assert is_hit("doc_a::0", offset_map, "doc_a", 50, 150) is True
    assert is_hit("doc_b::0", offset_map, "doc_a", 50, 150) is False  # right span, wrong doc
    assert is_hit("doc_a::1", offset_map, "doc_a", 0, 50) is False    # right doc, no overlap


def test_is_hit_returns_false_for_unknown_chunk_id():
    offset_map = _sample_offset_map()
    assert is_hit("doc_z::99", offset_map, "doc_a", 0, 100) is False


def test_recall_at_k_hit_and_miss():
    offset_map = _sample_offset_map()
    assert recall_at_k(["doc_a::1", "doc_a::0"], offset_map, "doc_a", 0, 50) == 1
    assert recall_at_k(["doc_b::0"], offset_map, "doc_a", 0, 50) == 0


def test_reciprocal_rank_scores_by_position():
    offset_map = _sample_offset_map()
    # hit at rank 1
    assert reciprocal_rank(["doc_a::0", "doc_a::1"], offset_map, "doc_a", 0, 50) == 1.0
    # hit at rank 2
    assert reciprocal_rank(["doc_b::0", "doc_a::0"], offset_map, "doc_a", 0, 50) == 0.5
    # no hit
    assert reciprocal_rank(["doc_b::0"], offset_map, "doc_a", 0, 50) == 0.0


def test_build_offset_map_indexes_by_chunk_id():
    chunks = [
        {"chunk_id": "x::0", "doc_id": "x", "start_char": 0, "end_char": 10},
        {"chunk_id": "x::1", "doc_id": "x", "start_char": 10, "end_char": 20},
    ]
    offset_map = build_offset_map(chunks)
    assert offset_map == {"x::0": ("x", 0, 10), "x::1": ("x", 10, 20)}

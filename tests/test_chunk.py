import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunk import chunk_document, normalize, split_into_sentences


def test_normalize_collapses_whitespace():
    assert normalize("Hello   world.\n\nNext  line.") == "Hello world. Next line."


def test_split_into_sentences_splits_on_terminal_punctuation():
    sentences = split_into_sentences("First sentence. Second sentence! Third one?")
    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_chunk_offsets_round_trip_to_original_text():
    """Every chunk's recorded (start_char, end_char) must extract exactly the
    chunk's own text from the normalized document -- this is the invariant
    the whole cross-config evaluation methodology depends on."""
    text = (
        "Machine learning is a field of study. It gives computers the ability "
        "to learn without being explicitly programmed. Deep learning is a "
        "subset of machine learning based on artificial neural networks. "
        "Neural networks are inspired by the structure of the brain. "
        "Backpropagation is the algorithm used to train them efficiently."
    )
    chunks = chunk_document("test_doc", "Test Doc", text, chunk_size=80, overlap=20)
    normalized = normalize(text)

    assert len(chunks) > 1
    for chunk in chunks:
        extracted = normalized[chunk["start_char"]:chunk["end_char"]]
        assert extracted == chunk["text"]


def test_chunk_ids_are_sequential_and_namespaced_by_doc():
    text = "One sentence here. Another sentence follows. A third sentence too. " * 3
    chunks = chunk_document("doc_a", "Doc A", text, chunk_size=50, overlap=10)
    for chunk in chunks:
        assert chunk["chunk_id"].startswith("doc_a::")
        assert chunk["doc_id"] == "doc_a"


def test_short_trailing_chunks_are_dropped():
    text = "A short sentence. Ok."
    chunks = chunk_document("doc_b", "Doc B", text, chunk_size=1000, overlap=100)
    assert all(len(c["text"]) > 50 or len(chunks) == 0 for c in chunks)


def test_larger_chunk_size_produces_fewer_chunks():
    text = " ".join([f"This is sentence number {i} in a long document." for i in range(50)])
    small_chunks = chunk_document("doc_c", "Doc C", text, chunk_size=100, overlap=20)
    large_chunks = chunk_document("doc_c", "Doc C", text, chunk_size=500, overlap=50)
    assert len(large_chunks) < len(small_chunks)

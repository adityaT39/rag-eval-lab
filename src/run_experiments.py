"""
Runs retrieval quality experiments across configurations (chunk size,
embedding model, with/without reranking) and reports recall@k and MRR
against the reviewed eval set. Results are written to results/ as both
JSON and a Markdown table for the README.

Scoring note: a retrieval "hit" is judged by character-span overlap with
the ground-truth chunk, not exact chunk_id equality. Chunk boundaries (and
therefore chunk_ids) shift when chunk_size changes, so an exact-ID match
would only be meaningful within a single chunk_size config. Character
positions are stable across configs because they index into the same
underlying normalized document text -- see chunk.py.
"""
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from chunk import build_chunks
from embed_store import build_collection, retrieve

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    reranker = get_reranker()
    pairs = [[query, r["text"]] for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked][:top_k]


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


def evaluate_config(collection_name: str, embed_model: str, eval_set: list[dict], offset_map: dict, ground_truth: dict,
                     k: int = 5, use_reranker: bool = False, rerank_pool: int = 20) -> dict:
    from embed_store import get_collection

    collection = get_collection(collection_name)
    recalls, mrrs = [], []

    for item in eval_set:
        gt = ground_truth.get(item["expected_chunk_id"])
        if gt is None:
            continue  # ground-truth chunk_id not found in the reference (500-char) chunking; skip
        expected_doc_id, expected_start, expected_end = gt

        query = item["question"]
        pool_size = rerank_pool if use_reranker else k
        results = retrieve(collection, query, embed_model, k=pool_size)

        if use_reranker:
            results = rerank(query, results, top_k=k)

        retrieved_ids = [r["chunk_id"] for r in results]
        recalls.append(recall_at_k(retrieved_ids, offset_map, expected_doc_id, expected_start, expected_end))
        mrrs.append(reciprocal_rank(retrieved_ids, offset_map, expected_doc_id, expected_start, expected_end))

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "n": len(recalls),
    }


def build_offset_map(chunks: list[dict]) -> dict:
    return {c["chunk_id"]: (c["doc_id"], c["start_char"], c["end_char"]) for c in chunks}


def main():
    eval_set = json.loads((EVAL_DIR / "eval_set_reviewed.json").read_text())
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    results = {}
    offset_maps = {}

    # Ground truth: the eval set's expected_chunk_id was generated using the
    # original 500-char/100-overlap chunking, so that's the reference for
    # "correct" character spans -- regardless of which chunk_size we're
    # currently scoring against.
    reference_chunks = build_chunks(chunk_size=500, overlap=100, out_name="chunks_500.json")
    ground_truth = build_offset_map(reference_chunks)

    # Experiment 1: chunk size comparison (small vs large chunks, fixed embed model)
    for chunk_size, overlap in [(250, 50), (500, 100), (1000, 200)]:
        if chunk_size == 500:
            chunks = reference_chunks
        else:
            chunks = build_chunks(chunk_size=chunk_size, overlap=overlap, out_name=f"chunks_{chunk_size}.json")

        collection_name = f"chunks_{chunk_size}"
        offset_maps[collection_name] = build_offset_map(chunks)
        build_collection(chunks, collection_name, embed_model)

        key = f"chunk_size={chunk_size}"
        results[key] = evaluate_config(collection_name, embed_model, eval_set, offset_maps[collection_name], ground_truth)
        print(f"{key}: {results[key]}")

    # Experiment 2: reranking on/off, using the best chunk size from experiment 1
    best_chunk_size = max(
        [250, 500, 1000],
        key=lambda cs: results[f"chunk_size={cs}"]["recall@5"],
    )
    best_collection = f"chunks_{best_chunk_size}"
    best_offset_map = offset_maps[best_collection]

    results["reranker=off"] = evaluate_config(best_collection, embed_model, eval_set, best_offset_map, ground_truth, use_reranker=False)
    results["reranker=on"] = evaluate_config(best_collection, embed_model, eval_set, best_offset_map, ground_truth, use_reranker=True)
    print(f"reranker=off: {results['reranker=off']}")
    print(f"reranker=on:  {results['reranker=on']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2))

    md_lines = ["| Configuration | Recall@5 | MRR | N |", "|---|---|---|---|"]
    for name, metrics in results.items():
        md_lines.append(f"| {name} | {metrics['recall@5']:.2%} | {metrics['mrr']:.3f} | {metrics['n']} |")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(md_lines))

    print(f"\nResults written to {RESULTS_DIR}/experiment_results.json and results_table.md")


if __name__ == "__main__":
    main()

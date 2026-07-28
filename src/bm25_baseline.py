"""
Keyword-search baseline (BM25) for the same retrieval eval, so results can
answer the obvious question: "is semantic search actually better than just
matching keywords here?" Uses the same 500-char chunking and the same
span-overlap scoring as the embedding-based experiments, so the numbers are
directly comparable.
"""
import json
import re
from chunk import build_chunks
from pathlib import Path

from rank_bm25 import BM25Okapi

from scoring import build_offset_map, recall_at_k, reciprocal_rank

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def run_bm25_eval(chunks: list[dict], eval_set: list[dict], ground_truth: dict, k: int = 5) -> dict:
    offset_map = build_offset_map(chunks)
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    chunk_ids = [c["chunk_id"] for c in chunks]

    recalls, mrrs = [], []
    for item in eval_set:
        gt = ground_truth.get(item["expected_chunk_id"])
        if gt is None:
            continue
        expected_doc_id, expected_start, expected_end = gt

        scores = bm25.get_scores(tokenize(item["question"]))
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        retrieved_ids = [chunk_ids[i] for i in top_k_idx]

        recalls.append(recall_at_k(retrieved_ids, offset_map, expected_doc_id, expected_start, expected_end))
        mrrs.append(reciprocal_rank(retrieved_ids, offset_map, expected_doc_id, expected_start, expected_end))

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "n": len(recalls),
    }


def main():
    eval_set = json.loads((EVAL_DIR / "eval_set_reviewed.json").read_text())
    chunks = build_chunks(chunk_size=500, overlap=100, out_name="chunks_500.json")
    ground_truth = build_offset_map(chunks)

    result = run_bm25_eval(chunks, eval_set, ground_truth)
    print(f"bm25_baseline: {result}")

    RESULTS_DIR.mkdir(exist_ok=True)
    existing = {}
    results_path = RESULTS_DIR / "experiment_results.json"
    if results_path.exists():
        existing = json.loads(results_path.read_text())
    existing["bm25_baseline"] = result
    results_path.write_text(json.dumps(existing, indent=2))

    md_lines = ["| Configuration | Recall@5 | MRR | N |", "|---|---|---|---|"]
    for name, metrics in existing.items():
        md_lines.append(f"| {name} | {metrics['recall@5']:.2%} | {metrics['mrr']:.3f} | {metrics['n']} |")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(md_lines))

    print(f"\nUpdated {results_path} and results_table.md")


if __name__ == "__main__":
    main()

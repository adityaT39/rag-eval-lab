"""
Runs BM25 and the reranked embedding pipeline against both the original
(lexically-aligned) eval set and the paraphrased (naturalistic) eval set,
to test the README's hypothesis: that BM25 only tied embeddings because the
original questions shared vocabulary with their source passages.

Requires chunks_500 to already be embedded (run_experiments.py's chunk_size
loop builds it) -- this script reuses that collection rather than
re-embedding.
"""
import json
from chunk import build_chunks
from pathlib import Path

from bm25_baseline import run_bm25_eval
from run_experiments import evaluate_config
from scoring import build_offset_map

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "chunks_500"


def main():
    original_set = json.loads((EVAL_DIR / "eval_set_reviewed.json").read_text())
    paraphrased_set = json.loads((EVAL_DIR / "eval_set_paraphrased.json").read_text())

    chunks = build_chunks(chunk_size=500, overlap=100, out_name="chunks_500.json")
    ground_truth = build_offset_map(chunks)

    results = {}

    for label, eval_set in [("original", original_set), ("paraphrased", paraphrased_set)]:
        bm25_result = run_bm25_eval(chunks, eval_set, ground_truth)
        embed_result = evaluate_config(
            COLLECTION, EMBED_MODEL, eval_set, ground_truth, ground_truth,
            use_reranker=True,
        )
        results[f"bm25__{label}"] = bm25_result
        results[f"embed_rerank__{label}"] = embed_result
        print(f"bm25__{label}: {bm25_result}")
        print(f"embed_rerank__{label}: {embed_result}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "paraphrase_comparison.json").write_text(json.dumps(results, indent=2))

    md_lines = ["| Method | Eval set | Recall@5 | MRR | N |", "|---|---|---|---|---|"]
    for key, metrics in results.items():
        method, label = key.split("__")
        method_name = "BM25 (keyword)" if method == "bm25" else "Embeddings + reranker"
        md_lines.append(f"| {method_name} | {label} | {metrics['recall@5']:.2%} | {metrics['mrr']:.3f} | {metrics['n']} |")
    (RESULTS_DIR / "paraphrase_comparison.md").write_text("\n".join(md_lines))

    print(f"\nWritten to {RESULTS_DIR}/paraphrase_comparison.json and .md")


if __name__ == "__main__":
    main()

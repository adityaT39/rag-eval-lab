# RAG Eval Lab

[![CI](https://github.com/adityaT39/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/adityaT39/rag-eval-lab/actions/workflows/ci.yml)

A Retrieval-Augmented Generation (RAG) pipeline over a Wikipedia corpus of
machine learning / AI concepts — built to answer one question properly
instead of skipping it: **is the retrieval actually any good?**

Most RAG demos wire an embedding model + an LLM together and call it done.
This project instead treats retrieval as something to be *measured and
improved*: a hand-reviewed evaluation set, quantitative metrics (Recall@k,
MRR), and controlled experiments comparing chunking strategies and reranking.

## Why this exists

I wanted a project that proves I understand what's happening *underneath*
an LLM API call, not just that I can wire one up. This pipeline retrieves
relevant passages from ~50 ML/AI Wikipedia articles and uses Claude to
answer questions grounded in that retrieved context — with every retrieval
config backed by a measured number, not a guess.

## Architecture

```
Wikipedia articles (ingest.py)
        │
        ▼
Chunking (chunk.py) ── configurable chunk size / overlap
        │
        ▼
Embeddings (sentence-transformers) ──► Chroma vector store (embed_store.py)
        │
        ▼
Retrieval (top-k similarity search)
        │
        ▼
[optional] Cross-encoder reranking (run_experiments.py)
        │
        ▼
Claude answer generation, grounded + cited (answer.py)
```

Evaluation loop (separate from the answer path above):

```
Sample chunks → Claude generates a question per chunk (generate_eval_set.py)
        │
        ▼
Human review / spot-check (review_eval_set.py)
        │
        ▼
Retrieval run against reviewed questions → Recall@5, MRR (run_experiments.py)
```

## Results

Ran against a 100-question human-reviewed eval set (all 100 kept on review).
Retrieval quality is measured at the passage level: a "hit" counts if the
retrieved chunk's character span overlaps the source passage the question
was written from — not exact chunk-ID matching, since chunk boundaries (and
therefore IDs) shift across chunk-size configs. See the methodology note
below for why.

| Configuration | Recall@5 | MRR | N |
|---|---|---|---|
| chunk_size=250 | 87.00% | 0.781 | 100 |
| chunk_size=500 | 90.00% | 0.803 | 100 |
| chunk_size=1000 | 87.00% | 0.773 | 100 |
| reranker=off | 90.00% | 0.803 | 100 |
| reranker=on | 97.00% | 0.965 | 100 |
| **BM25 (keyword baseline)** | **97.00%** | 0.942 | 100 |

(Live copy: [`results/results_table.md`](results/results_table.md) /
[`results/experiment_results.json`](results/experiment_results.json))

**What I measured and found:**
1. **Chunk size** (250 / 500 / 1000 chars) — 500 chars performed best
   (90% recall@5). Smaller chunks (250) lose surrounding context around the
   answer; larger chunks (1000) dilute the embedding signal by packing
   unrelated content into one vector. The 500-char middle ground won on both
   metrics.
2. **Cross-encoder reranking** — retrieve a wider candidate pool (20) with
   the embedding model, then re-score with a cross-encoder before taking the
   top-5. This gave the single biggest improvement in the pure-embedding
   pipeline: recall@5 90% → 97%, MRR 0.803 → 0.965. Cross-encoders score the
   query and passage jointly (rather than comparing precomputed independent
   vectors), which is slower per-query but substantially more accurate.
3. **BM25 keyword-search baseline** — the honest finding here: a classic
   keyword-matching retriever (`rank_bm25`, no embeddings, no ML) scores
   essentially the same as the reranked embedding pipeline (97% recall@5,
   0.942 vs 0.965 MRR). This isn't a failure of the embedding approach — it's
   a property of this eval set: the questions were generated directly from
   source passages, so they tend to share literal vocabulary with the answer
   (e.g. asking "what does RU-IRL assume..." when the passage itself uses
   "RU-IRL"). BM25 is unusually strong on this kind of lexically-aligned
   question. On paraphrased, naturalistic user queries — which don't
   reliably reuse the source's exact words — the embedding+reranker approach
   would be expected to pull further ahead. Worth stating plainly rather
   than only reporting the numbers that flatter the fancier system.
4. **Answer-quality (LLM-as-judge), not just retrieval** — retrieval metrics
   only prove the right passage was *found*, not that the final answer is
   *correct*. Sampled 30 questions, ran them through the full pipeline, and
   had Claude grade each generated answer against the known-correct source
   passage:

   | | Strict (exact match) | Lenient (allows minor gaps) |
   |---|---|---|
   | Accuracy | 83.3% | 93.3% |

   (See [`results/answer_quality.json`](results/answer_quality.json) for
   every graded example.) The two "incorrect" cases were themselves useful:
   one was the system correctly *refusing to answer* after a genuine
   retrieval miss (right behavior, not a hallucination), and the other was a
   real hallucination — the model added fabricated statistical detail (a
   Gaussian-noise assumption) that wasn't in the retrieved passage. Being
   able to tell those two failure modes apart is the whole point of grading
   the answer, not just the retrieval.

**A bug worth mentioning, because it's part of the real story:** my first
pass at this experiment used exact `chunk_id` matching to grade correctness,
and got nonsensical results (3-6% recall for the 250/1000-char configs vs.
79% for 500). The cause: the eval set's "correct answer" was tied to
`chunk_id`s from the original 500-char chunking, and those IDs don't mean
the same passage once you re-chunk at a different size. Fixed by tracking
each chunk's character offsets in the source document and scoring hits by
span overlap instead of ID equality — comparable across any chunk_size.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...   # needed for eval-set generation + answers
```

## Running the pipeline end-to-end

```bash
# 1. Download the corpus (~50 Wikipedia articles on ML/AI topics)
python src/ingest.py

# 2. Chunk the corpus (default: 500 chars, 100 overlap)
python src/chunk.py

# 3. Generate a candidate eval set (Claude writes questions per chunk)
python src/generate_eval_set.py

# 4. Human-review the generated questions before trusting them
python src/review_eval_set.py

# 5. Run retrieval experiments (chunk size + reranking) and produce metrics
python src/run_experiments.py

# 6. Run the BM25 keyword-search baseline for comparison
python src/bm25_baseline.py

# 7. Grade actual generated answers (not just retrieval) via LLM-as-judge
python src/evaluate_answers.py

# 8. Ask questions through the full RAG pipeline
python src/answer.py "What is the vanishing gradient problem?"

# 9. Or launch the interactive demo
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v          # unit tests for chunking + scoring logic
ruff check src/ tests/ app.py   # lint
```

Only the pure-logic modules (`chunk.py`, `scoring.py`) are unit tested —
the ML modules (embeddings, vector store, reranking) depend on multi-GB
model downloads that aren't worth the CI minutes for a portfolio project.
Those are covered by the smoke-run commands above instead. Runs
automatically on every push via GitHub Actions (see badge at the top).

## Project structure

```
src/
  ingest.py               # Wikipedia corpus download
  chunk.py                # sentence-aware chunking, size configurable
  scoring.py               # pure retrieval-scoring logic (unit tested, zero ML deps)
  embed_store.py           # embeddings + Chroma vector store + retrieval
  generate_eval_set.py     # LLM-assisted eval question generation
  review_eval_set.py       # human-in-the-loop spot-check CLI
  run_experiments.py       # chunk-size / reranker experiments + metrics
  bm25_baseline.py         # keyword-search baseline for comparison
  evaluate_answers.py      # LLM-as-judge grading of generated answers
  answer.py                 # end-to-end grounded QA with citations
app.py                      # Streamlit demo UI
tests/                      # pytest unit tests for chunk.py / scoring.py
.github/workflows/ci.yml    # lint + test on every push
eval/                       # generated + reviewed eval sets
results/                    # experiment metrics (json + markdown table)
```

## Honest notes on methodology

- The eval set is **LLM-generated, then human-reviewed** — a real and
  commonly used technique (similar to what tools like RAGAS do), not a
  substitute for a fully hand-written eval set. This is disclosed, not hidden,
  because being upfront about eval methodology is part of doing this properly.
- Metrics are computed on ~50-100 examples, small enough to review by hand
  but small enough that results should be read as directional, not
  statistically bulletproof.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no
  API key needed) — chosen so the retrieval half of the pipeline has zero
  external cost or dependency. Only the answer-generation and eval-question
  generation steps call the Claude API.

## Stack

Python · sentence-transformers · ChromaDB · Claude API (Anthropic) ·
cross-encoder reranking · Streamlit

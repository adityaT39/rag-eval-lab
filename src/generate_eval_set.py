"""
Generates a labeled evaluation set: for a sample of chunks, asks Claude to
write a question that chunk answers. This gives us (question -> correct
chunk_id) pairs we can use to score retrieval quality.

This is a known technique (synthetic eval generation, as used by tools like
RAGAS) -- NOT a substitute for human review. Run `review_eval_set.py`
afterwards to spot-check and discard bad pairs before trusting the numbers.
"""
import json
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

PROMPT_TEMPLATE = """You are creating an evaluation dataset for a retrieval system.

Given the passage below, write ONE specific factual question that can ONLY be
answered using information in this passage. The question should not be
answerable from general knowledge alone -- it must require this specific text.

Passage (from the Wikipedia article "{title}"):
\"\"\"
{text}
\"\"\"

Respond with ONLY the question, no preamble, no quotes."""


def generate_eval_set(n_samples: int = 100, seed: int = 42, out_name: str = "eval_set.json"):
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())

    # only use chunks with enough substance to support a specific question
    candidates = [c for c in chunks if len(c["text"]) > 300]
    random.seed(seed)
    sample = random.sample(candidates, min(n_samples, len(candidates)))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    eval_set = []

    for i, chunk in enumerate(sample):
        prompt = PROMPT_TEMPLATE.format(title=chunk["title"], text=chunk["text"])
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        question = response.content[0].text.strip()

        eval_set.append({
            "eval_id": i,
            "question": question,
            "expected_chunk_id": chunk["chunk_id"],
            "expected_doc_id": chunk["doc_id"],
            "source_text": chunk["text"],
            "reviewed": False,  # flip to true (or delete bad rows) in review_eval_set.py
        })
        print(f"[{i+1}/{len(sample)}] {chunk['title']}: {question}")

    out_path = EVAL_DIR / out_name
    out_path.write_text(json.dumps(eval_set, indent=2), encoding="utf-8")
    print(f"\nWrote {len(eval_set)} eval examples -> {out_path}")
    print("Next: run `python src/review_eval_set.py` to spot-check them.")


if __name__ == "__main__":
    generate_eval_set()

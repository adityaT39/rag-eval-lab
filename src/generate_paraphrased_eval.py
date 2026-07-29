"""
Generates a paraphrased version of the reviewed eval set, to test a
hypothesis raised in the README: that BM25 tied the embedding+reranker
pipeline because the original eval questions were generated directly from
the source text and inherited its literal vocabulary. If that's true,
paraphrasing the questions -- same meaning, different words, deliberately
avoiding the source passage's distinctive terms -- should narrow or reverse
that gap. This script only changes question wording; the ground-truth
chunk_id/doc_id mapping is untouched, since paraphrasing doesn't change
which passage answers the question.

A lexical-overlap check (Jaccard similarity between question and source
text, before vs after paraphrasing) is printed as a sanity check that the
paraphrasing actually reduced literal word overlap rather than just
reshuffling the same terms.
"""
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

PARAPHRASE_PROMPT = """Rewrite this question so it asks for the same information but uses DIFFERENT
words than the source passage -- avoid reusing the passage's specific
technical terms, proper nouns, or distinctive phrasing wherever a synonym or
descriptive rephrasing is possible. The rewritten question must still be
answerable using only the source passage.

Original question: {question}

Source passage (for reference, so you know which terms to avoid reusing):
\"\"\"
{source_text}
\"\"\"

Respond with ONLY the rewritten question, no preamble, no quotes."""

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def generate_paraphrased_eval(out_name: str = "eval_set_paraphrased.json"):
    eval_set = json.loads((EVAL_DIR / "eval_set_reviewed.json").read_text())
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    paraphrased = []
    overlaps_before, overlaps_after = [], []

    for i, item in enumerate(eval_set):
        prompt = PARAPHRASE_PROMPT.format(question=item["question"], source_text=item["source_text"])
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        new_question = response.content[0].text.strip()

        source_tokens = tokenize(item["source_text"])
        before = jaccard(tokenize(item["question"]), source_tokens)
        after = jaccard(tokenize(new_question), source_tokens)
        overlaps_before.append(before)
        overlaps_after.append(after)

        paraphrased.append({
            **item,
            "original_question": item["question"],
            "question": new_question,
        })
        print(f"[{i+1}/{len(eval_set)}] overlap {before:.2f} -> {after:.2f} | {new_question[:80]}")

    avg_before = sum(overlaps_before) / len(overlaps_before)
    avg_after = sum(overlaps_after) / len(overlaps_after)

    out_path = EVAL_DIR / out_name
    out_path.write_text(json.dumps(paraphrased, indent=2), encoding="utf-8")
    print(f"\nAvg question/source lexical overlap: {avg_before:.3f} -> {avg_after:.3f}")
    print(f"Wrote {len(paraphrased)} paraphrased questions -> {out_path}")


if __name__ == "__main__":
    generate_paraphrased_eval()

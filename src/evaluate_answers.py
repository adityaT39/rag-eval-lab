"""
Answer-quality evaluation: retrieval metrics (Recall@k, MRR) only tell us
whether the right passage was FOUND -- not whether the final answer Claude
generated from it is actually correct. This script closes that gap with an
LLM-as-judge pass: for a sample of eval questions, generate a real answer
through the full pipeline, then have Claude grade that answer against the
known-correct source passage.

Sampled (not run on the full 100) to control API cost -- each example costs
two Claude calls (one to answer, one to judge).
"""
import json
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from answer import answer_question

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

JUDGE_PROMPT = """You are grading a QA system's answer for factual correctness.

Question: {question}

Source passage (ground truth -- the answer should be consistent with this):
\"\"\"
{source_text}
\"\"\"

System's answer:
\"\"\"
{answer}
\"\"\"

Grade the system's answer as exactly one of:
- CORRECT: matches the source passage, no fabricated details
- PARTIALLY_CORRECT: gets the gist right but has a minor error, omission, or vagueness
- INCORRECT: contradicts the source passage or invents information not in it

Respond in this exact format:
GRADE: <one of the three labels above>
REASON: <one sentence>"""


def parse_grade(judge_response: str) -> tuple[str, str]:
    grade, reason = "UNKNOWN", ""
    for line in judge_response.splitlines():
        if line.startswith("GRADE:"):
            grade = line.removeprefix("GRADE:").strip()
        elif line.startswith("REASON:"):
            reason = line.removeprefix("REASON:").strip()
    return grade, reason


def evaluate_answers(n_samples: int = 30, seed: int = 7, out_name: str = "answer_quality.json"):
    eval_set = json.loads((EVAL_DIR / "eval_set_reviewed.json").read_text())
    random.seed(seed)
    sample = random.sample(eval_set, min(n_samples, len(eval_set)))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    graded = []

    for i, item in enumerate(sample):
        result = answer_question(item["question"])

        judge_prompt = JUDGE_PROMPT.format(
            question=item["question"],
            source_text=item["source_text"],
            answer=result["answer"],
        )
        judge_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        grade, reason = parse_grade(judge_response.content[0].text)

        graded.append({
            "eval_id": item["eval_id"],
            "question": item["question"],
            "answer": result["answer"],
            "grade": grade,
            "reason": reason,
        })
        print(f"[{i+1}/{len(sample)}] {grade}: {item['question'][:70]}")

    counts = {"CORRECT": 0, "PARTIALLY_CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0}
    for g in graded:
        counts[g["grade"]] = counts.get(g["grade"], 0) + 1

    n = len(graded)
    summary = {
        "n": n,
        "counts": counts,
        "accuracy_strict": counts["CORRECT"] / n,
        "accuracy_lenient": (counts["CORRECT"] + counts["PARTIALLY_CORRECT"]) / n,
        "examples": graded,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / out_name).write_text(json.dumps(summary, indent=2))
    print(f"\nStrict accuracy (CORRECT only): {summary['accuracy_strict']:.1%}")
    print(f"Lenient accuracy (CORRECT + PARTIAL): {summary['accuracy_lenient']:.1%}")
    print(f"Written to {RESULTS_DIR / out_name}")


if __name__ == "__main__":
    evaluate_answers()

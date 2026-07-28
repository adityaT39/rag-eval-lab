"""
Interactive CLI to spot-check the LLM-generated eval set before trusting it.
For each question, shows the question + the chunk it's supposed to match,
and lets you keep or discard it. This is the human-in-the-loop step that
makes the eval set defensible.
"""
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def review(in_name: str = "eval_set.json", out_name: str = "eval_set_reviewed.json"):
    eval_set = json.loads((EVAL_DIR / in_name).read_text())
    kept = []

    print(f"Reviewing {len(eval_set)} questions. [Enter]=keep, 'd'=discard, 'q'=quit and save.\n")

    for i, item in enumerate(eval_set):
        print(f"--- {i+1}/{len(eval_set)} ---")
        print(f"Q: {item['question']}")
        print(f"Source ({item['expected_doc_id']}): {item['source_text'][:200]}...")
        choice = input("Keep? [Enter/d/q]: ").strip().lower()

        if choice == "q":
            break
        if choice != "d":
            item["reviewed"] = True
            kept.append(item)
        print()

    out_path = EVAL_DIR / out_name
    out_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    print(f"\nKept {len(kept)}/{len(eval_set)} questions -> {out_path}")


if __name__ == "__main__":
    review()

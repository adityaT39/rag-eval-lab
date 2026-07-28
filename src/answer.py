"""
End-to-end RAG query: retrieve relevant chunks, optionally rerank, then ask
Claude to answer using only the retrieved context (with citations).
"""
import os

import anthropic
from dotenv import load_dotenv

from embed_store import get_collection, retrieve
from run_experiments import rerank

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ANSWER_PROMPT = """Answer the question using ONLY the context passages below.
If the context doesn't contain the answer, say so explicitly -- do not guess.
Cite which passage(s) you used by number.

Question: {question}

Context:
{context}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] ({c['title']}) {c['text']}" for i, c in enumerate(chunks))


def answer_question(question: str, collection_name: str = "chunks_500",
                     k: int = 5, use_reranker: bool = True, rerank_pool: int = 20) -> dict:
    collection = get_collection(collection_name)
    pool_size = rerank_pool if use_reranker else k
    results = retrieve(collection, question, EMBED_MODEL, k=pool_size)

    if use_reranker:
        results = rerank(question, results, top_k=k)
    else:
        results = results[:k]

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = ANSWER_PROMPT.format(question=question, context=format_context(results))
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "question": question,
        "answer": response.content[0].text,
        "sources": [{"title": c["title"], "chunk_id": c["chunk_id"]} for c in results],
    }


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is the difference between supervised and unsupervised learning?"
    result = answer_question(question)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['title']} ({s['chunk_id']})")

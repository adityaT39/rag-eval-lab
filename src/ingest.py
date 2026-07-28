"""
Downloads a corpus of Wikipedia articles on machine learning / AI topics
and saves them as plain-text documents in data/raw/.
"""
import json
import time
from pathlib import Path

import wikipediaapi

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "Machine learning", "Deep learning", "Neural network",
    "Supervised learning", "Unsupervised learning", "Reinforcement learning",
    "Convolutional neural network", "Recurrent neural network",
    "Transformer (deep learning architecture)", "Attention (machine learning)",
    "Large language model", "GPT (language model)", "BERT (language model)",
    "Word embedding", "Natural language processing",
    "Gradient descent", "Backpropagation", "Overfitting",
    "Regularization (mathematics)", "Bias-variance tradeoff",
    "Decision tree learning", "Random forest", "Support vector machine",
    "K-means clustering", "Principal component analysis",
    "Feature engineering", "Cross-validation (statistics)",
    "Confusion matrix", "Precision and recall", "ROC curve",
    "Generative adversarial network", "Autoencoder",
    "Vector database", "Information retrieval",
    "Retrieval-augmented generation", "Prompt engineering",
    "Fine-tuning (machine learning)", "Transfer learning",
    "Hyperparameter optimization", "Ensemble learning",
    "Naive Bayes classifier", "Logistic regression",
    "Linear regression", "Loss function", "Activation function",
    "Batch normalization", "Dropout (neural networks)",
    "Explainable artificial intelligence", "AI alignment",
    "Artificial general intelligence", "Computer vision",
]


def main():
    wiki = wikipediaapi.Wikipedia(user_agent="rag-eval-lab (educational project)", language="en")
    manifest = []

    for title in TOPICS:
        page = wiki.page(title)
        if not page.exists():
            print(f"  [skip] not found: {title}")
            continue

        doc_id = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = RAW_DIR / f"{doc_id}.txt"
        out_path.write_text(page.text, encoding="utf-8")

        manifest.append({
            "doc_id": doc_id,
            "title": page.title,
            "url": page.fullurl,
            "chars": len(page.text),
        })
        print(f"  [ok]   {title} ({len(page.text)} chars)")
        time.sleep(0.2)  # be polite to the API

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} documents. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

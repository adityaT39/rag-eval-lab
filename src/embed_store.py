"""
Embeds text chunks and stores them in a persistent Chroma vector store,
one Chroma collection per experiment configuration so results don't clobber
each other.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
DB_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def build_collection(chunks: list[dict], collection_name: str, embed_model: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))

    existing_names = {c.name for c in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)

    collection = client.create_collection(collection_name, metadata={"embed_model": embed_model})
    embedder = get_embedder(embed_model)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks]

    # Chroma enforces a max batch size per add() call, so insert in slices.
    MAX_BATCH = 5000
    for start in range(0, len(chunks), MAX_BATCH):
        end = start + MAX_BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"Indexed {len(chunks)} chunks into collection '{collection_name}' ({embed_model})")
    return collection


def get_collection(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(collection_name)


def retrieve(collection: chromadb.Collection, query: str, embed_model: str, k: int = 5) -> list[dict]:
    embedder = get_embedder(embed_model)
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)

    return [
        {
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

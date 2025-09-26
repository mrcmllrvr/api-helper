import json
import os
import uuid
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def load_json_list(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file that contains a list of API entries."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level in {path}")
    return data

def make_document(entry: Dict[str, Any], provider: str, index: int, path: str) -> Dict[str, Any]:
    """Turn a JSON entry into one 'chunk' with metadata."""
    metadata = entry.get("metadata", {})

    # Build text for embedding
    parts = []
    if metadata:
        for k in ("operationId", "path", "method", "summary", "tags"):
            if k in metadata and metadata[k]:
                parts.append(f"{k}: {metadata[k]}")
    content = entry.get("content", "")
    if content:
        parts.append(content)

    text = "\n\n".join(parts)

    # Metadata to keep
    md = {
        "provider": provider,
        "source_file": path,
        "source_id": f"{provider}::{index}",
        "operationId": metadata.get("operationId"),
        "path": metadata.get("path"),
        "method": metadata.get("method"),
        "summary": metadata.get("summary"),
    }

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "metadata": md,
    }

def chunk_json_file(path: str) -> List[Dict[str, Any]]:
    """Each JSON entry becomes a single chunk with metadata."""
    provider = os.path.splitext(os.path.basename(path))[0] 
    entries = load_json_list(path)
    return [make_document(entry, provider, i, path) for i, entry in enumerate(entries)]


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, show_progress_bar=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms


def get_chroma_client(persist_dir: str = "./chroma_db"):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    
    return client

def create_or_get_collection(client, name: str = "api_docs"):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)
    

def ingest_files(paths: List[str], persist_dir: str = "./chroma_db"):
    embedder = Embedder()
    client = get_chroma_client(persist_dir)
    collection = create_or_get_collection(client, "api_docs")

    docs = []
    for path in paths:
        docs.extend(chunk_json_file(path))

    print(f"Embedding {len(docs)} documents...")
    embeddings = embedder.embed_texts([d["text"] for d in docs])

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
        embeddings=embeddings.tolist(),
    )

    print("Ingestion complete.")

    return client, collection

if __name__ == "__main__":
    client, coll = ingest_files(
        ["openai.json", "cohere.json"], persist_dir="./chroma_db"
    )

    q = "How do I call the chat API?"
    embedder = Embedder()
    q_emb = embedder.embed_texts([q])[0].tolist()
    res = coll.query(query_embeddings=[q_emb], n_results=5, include=["documents", "metadatas"])
    for doc, md in zip(res["documents"][0], res["metadatas"][0]):
        print("\n--- Result ---")
        print("Provider:", md["provider"])
        print("Path:", md["path"])
        print("Method:", md["method"])
        print("Summary:", md.get("summary"))
        print("Excerpt:", doc[:200], "...")
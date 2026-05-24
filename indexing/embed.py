import os

from sentence_transformers import SentenceTransformer
import numpy as np
import json

from config import BATCH_SIZE, CHUNK_IDS_PATH, EMBEDDING_MODEL, EMBEDDINGS_PATH, PROCESSED_DATA_PATH

def load_model():
    """Load PubMedBERT sentence transformer model from config"""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

def generate_embeddings(chunks: list, model) -> np.ndarray:
    """Generate embeddings in batches, return numpy array of shape (N, 768)"""
    text=[chunk["text"] for chunk in chunks]
    return model.encode(text, batch_size= BATCH_SIZE, show_progress_bar=True)

def save_embeddings(embeddings: np.ndarray, chunks: list) -> None:
    """Save embeddings + chunk IDs to disk for qdrant ingestion"""
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    with open(CHUNK_IDS_PATH, "w") as f:
        json.dump(chunk_ids, f)
    print(f"\nSaved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved chunk IDs to {CHUNK_IDS_PATH}")

if __name__ == "__main__":
    print("Loading processed chunks...")
    with open(PROCESSED_DATA_PATH, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    model = load_model()
    embeddings = generate_embeddings(chunks, model)
    save_embeddings(embeddings, chunks)
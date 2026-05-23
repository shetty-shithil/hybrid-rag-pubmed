from sentence_transformers import SentenceTransformer
import numpy as np

from config import BATCH_SIZE, EMBEDDING_MODEL

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
    pass
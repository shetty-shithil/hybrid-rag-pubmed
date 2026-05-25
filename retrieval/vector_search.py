
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_HOST, QDRANT_PORT, TOP_K_VECTOR


def load_model():
    """Load embedding model — reuse from embed.py concept"""
    model= SentenceTransformer(EMBEDDING_MODEL)
    return model

def get_qdrant_client():
    """Connect to Qdrant"""
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

def vector_search(query: str, model, client, top_k: int = TOP_K_VECTOR) -> list:
    """Embed query and search Qdrant, return top_k results"""
    query_vec = model.encode([query])[0].tolist()  # shape (768,)
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True  # we want the text + metadata back
    )
    return search_result.points  # list of {id, version, score, payload}

if __name__ == "__main__":
    model  = load_model()
    client = get_qdrant_client()
    results = vector_search("does aspirin reduce cardiovascular risk", model, client)
    print(results)
    for r in results:
        print(f"\nScore : {r.score:.4f}")
        print(f"Text  : {r.payload['text'][:200]}")
        print(f"Source: {r.payload['source_id']}")
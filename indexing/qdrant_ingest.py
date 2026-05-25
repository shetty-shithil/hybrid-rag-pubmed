from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import json
import os

from tqdm import tqdm
from config import BATCH_SIZE, CHUNK_IDS_PATH, COLLECTION_NAME, EMBEDDING_DIM, EMBEDDINGS_PATH, PROCESSED_DATA_PATH, QDRANT_HOST, QDRANT_PORT



def get_qdrant_client():
    """Connect to Qdrant running on localhost and return client"""
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

def create_collection(client):
    """Create Qdrant collection with vector config (size=768, cosine distance)"""
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")

def load_data():
    """Load embeddings, chunk_ids and full chunks from disk, return all three"""
    with open(CHUNK_IDS_PATH, "r") as f:
        chunk_ids = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(PROCESSED_DATA_PATH, "r") as f:
        chunks = json.load(f)
    return embeddings, chunk_ids, chunks
    
def ingest_embeddings(client, embeddings, chunks, chunk_ids):
    """Push vectors + payload to Qdrant in batches"""
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Ingesting to Qdrant"):
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        batch_chunks = chunks[i:i+BATCH_SIZE]
        batch_ids = chunk_ids[i:i+BATCH_SIZE]

        points = []
        for emb, chunk, cid in zip(batch_embeddings, batch_chunks, batch_ids):
            point = QdrantClient.models.PointStruct(
                id=int(cid),
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "source_id": chunk["source_id"],
                    "question": chunk["question"]
                }
            )
            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )


if __name__ == "__main__":
    client = get_qdrant_client()
    create_collection(client)
    embeddings, chunk_ids, chunks = load_data()
    ingest_embeddings(client, embeddings, chunks, chunk_ids)
    print("\nIngestion complete!")
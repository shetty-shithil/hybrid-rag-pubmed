import json
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BM25_INDEX_PATH, TOP_K_BM25

def load_bm25_index():
    """Load BM25 object and chunks from pickle file"""
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunks"]

def bm25_search(query: str, bm25, chunks, top_k: int = TOP_K_BM25) -> list:
    """Tokenize query, get BM25 scores, return top_k chunks"""
    tokenized_query = query.lower().split()
    # BM25 returns a score for each chunk in the corpus on the basis of the words in the query and the words in the chunk. Higher score means more relevant.
    scores = bm25.get_scores(tokenized_query) 
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        chunk_info = chunks[idx]
        chunk_info["score"] = scores[idx]
        results.append(chunk_info)
    return results

if __name__ == "__main__":
    bm25, chunks = load_bm25_index()
    query = "does aspirin reduce cardiovascular risk"
    results = bm25_search(query, bm25, chunks)
    # BM25 scores can be any non-negative number, with higher meaning more relevant. The text and source_id are stored in the chunk info.
    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Score: {res['score']}")
        print(f"Text: {res['text']}\n")
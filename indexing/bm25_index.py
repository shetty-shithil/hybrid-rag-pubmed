import json
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from config import PROCESSED_DATA_PATH, BM25_INDEX_PATH


def build_bm25_index():
    print("Loading processed chunks...")
    with open(PROCESSED_DATA_PATH, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # tokenize - BM25 works on token lists, not raw strings
    print("Tokenizing chunks...")
    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in tqdm(chunks, desc="Tokenizing")
    ]

    # build index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # save index + chunks together so we can map results back to text
    payload = {
        "bm25"  : bm25,
        "chunks": chunks       # needed to retrieve text from BM25 results
    }

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    print(f"Vocabulary size: {len(bm25.idf)} unique tokens")
    print(f"Corpus size: {len(tokenized_corpus)} documents")

    return bm25, chunks


if __name__ == "__main__":
    build_bm25_index()
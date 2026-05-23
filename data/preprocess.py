import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from tqdm import tqdm


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'\[\d+\]', '', text)        # remove citation markers like [1]
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Simple word-level chunking with overlap.
    In production you'd use a tokenizer-aware splitter — good talking point for AMA!
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window with overlap

    return chunks


def preprocess():
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, "r") as f:
        samples = json.load(f)

    chunks = []
    chunk_id = 0

    for sample in tqdm(samples, desc="Preprocessing"):
        for passage in sample["passages"]:
            cleaned = clean_text(passage)
            if len(cleaned.split()) < 20:   # skip very short passages
                continue

            for chunk in chunk_text(cleaned):
                chunks.append({
                    "chunk_id"  : str(chunk_id),
                    "source_id" : sample["id"],
                    "question"  : sample["question"],  # useful for eval later
                    "text"      : chunk,
                })
                chunk_id += 1

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    with open(PROCESSED_DATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\n{len(chunks)} chunks saved to {PROCESSED_DATA_PATH}")
    print(f"Avg chunk length: {sum(len(c['text'].split()) for c in chunks) // len(chunks)} words")
    return chunks


if __name__ == "__main__":
    preprocess()
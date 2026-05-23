import json
import os
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, DATASET_CONFIG, MAX_SAMPLES, RAW_DATA_PATH

def download_pubmed():
    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG})...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
    
    samples = []
    split = dataset["train"]
    
    for i, item in enumerate(tqdm(split, desc="Extracting samples")):
        if i >= MAX_SAMPLES:
            break
        
        # each item has a question, context (dict of passages), and long_answer
        context = item.get("context", {})
        passages = context.get("contexts", [])
        question = item.get("question", "")
        answer   = item.get("long_answer", "")
        
        if not passages:
            continue
        
        samples.append({
            "id"      : str(i),
            "question": question,
            "answer"  : answer,
            "passages": passages,   # list of paragraph strings
        })
    
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\nSaved {len(samples)} samples to {RAW_DATA_PATH}")
    return samples


if __name__ == "__main__":
    download_pubmed()
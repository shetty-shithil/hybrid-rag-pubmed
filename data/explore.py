import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from config import DATASET_NAME, DATASET_CONFIG

def explore_raw():
    print("=" * 60)
    print("LOADING DATASET (first 5 samples)...")
    print("=" * 60)
    
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
    split = dataset["train"]
    
    print(f"\nTotal samples in dataset : {len(split)}")
    print(f"Features/columns         : {list(split.features.keys())}")
    
    print("\n" + "=" * 60)
    print("RAW ITEM STRUCTURE (item[0])")
    print("=" * 60)
    item = split[0]
    for key, value in item.items():
        if isinstance(value, dict):
            print(f"\n{key} (dict):")
            for k, v in value.items():
                preview = str(v)[:200] if isinstance(v, str) else str(v)[:200]
                print(f"     {k}: {preview}")
        elif isinstance(value, list):
            print(f"\n{key} (list, len={len(value)}): {str(value[0])[:200]}")
        else:
            print(f"\n{key}: {str(value)[:200]}")

    print("\n" + "=" * 60)
    print("LOOKING AT 5 SAMPLES")
    print("=" * 60)
    for i in range(5):
        item = split[i]
        passages = item.get("context", {}).get("contexts", [])
        question = item.get("question", "")
        answer   = item.get("long_answer", "")

        print(f"\n--- Sample {i} ---")
        print(f"Question     : {question}")
        print(f"Num passages : {len(passages)}")
        print(f"Passage[0]   : {passages[0][:300] if passages else 'N/A'}...")
        print(f"Answer       : {answer[:300] if answer else 'N/A'}...")

    print("\n" + "=" * 60)
    print("PASSAGE LENGTH DISTRIBUTION (first 1000 samples)")
    print("=" * 60)
    lengths = []
    for i, item in enumerate(split):
        if i >= 1000:
            break
        for p in item.get("context", {}).get("contexts", []):
            lengths.append(len(p.split()))

    lengths.sort()
    print(f"  Min words  : {min(lengths)}")
    print(f"  Max words  : {max(lengths)}")
    print(f"  Avg words  : {sum(lengths) // len(lengths)}")
    print(f"  Median     : {lengths[len(lengths) // 2]}")
    print(f"  <50 words  : {sum(1 for l in lengths if l < 50)}")
    print(f"  50-200     : {sum(1 for l in lengths if 50 <= l < 200)}")
    print(f"  200-512    : {sum(1 for l in lengths if 200 <= l < 512)}")
    print(f"  >512 words : {sum(1 for l in lengths if l >= 512)}")
    print(f"\nCHUNK_SIZE=512 covers most passages in one chunk!")


if __name__ == "__main__":
    explore_raw()
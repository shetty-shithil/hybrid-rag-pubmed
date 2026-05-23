import os
from dotenv import load_dotenv

load_dotenv()

# ── Data ──────────────────────────────────────────────────────────────────────
DATASET_NAME        = "pubmed_qa"          # HuggingFace dataset
DATASET_CONFIG      = "pqa_unlabeled"      # unlabeled subset (~200k samples)
MAX_SAMPLES         = 50000                # start with 50k, scale up later
RAW_DATA_PATH       = "data/raw/pubmed.json"
PROCESSED_DATA_PATH = "data/processed/pubmed_chunks.json"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 512                  # tokens per chunk
CHUNK_OVERLAP       = 50                   # overlap between chunks

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "NLP4Science/pubmedbert-base-embeddings"  # domain-tuned
EMBEDDING_DIM       = 768
BATCH_SIZE          = 64

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST         = "localhost"
QDRANT_PORT         = 6333
COLLECTION_NAME     = "pubmed_hybrid"

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH     = "data/processed/bm25_index.pkl"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR        = 20                   # candidates from vector search
TOP_K_BM25          = 20                   # candidates from BM25
TOP_K_FUSION        = 10                   # after RRF merging
TOP_K_RERANK        = 5                    # final chunks sent to LLM
RRF_K               = 60                   # RRF constant (standard is 60)

# ── Generation ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL        = "claude-sonnet-4-20250514"
MAX_TOKENS          = 1024

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL      = "ms-marco-MiniLM-L-12-v2"
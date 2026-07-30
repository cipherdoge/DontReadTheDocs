"""
Central configuration for the library-docs RAG system.
"""

import os

# --- Ollama settings ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Generation model (code writing / answering)
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "qwen3.5:4b")

# Embedding model. NOTE: qwen3.5:4b is a chat/generation model, not an
# embedding model. Ollama needs a model that actually exposes embeddings.
# Pull one with:  ollama pull nomic-embed-text
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")

# --- Storage ---
DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_docs")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

os.makedirs(RAW_DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- Chunking ---
CHUNK_MAX_CHARS = 1500          # soft cap per chunk
CHUNK_OVERLAP_CHARS = 150       # overlap between adjacent chunks (prose only)

# --- Retrieval ---
TOP_K_VECTOR = 12
TOP_K_BM25 = 12
TOP_K_FINAL = 8                 # after hybrid re-ranking, how many go to the LLM

# --- Fetching ---
USER_AGENT = "lib-rag-bot/0.1 (+https://example.com)"
REQUEST_TIMEOUT = 15
MAX_PAGES_PER_LIBRARY = 60

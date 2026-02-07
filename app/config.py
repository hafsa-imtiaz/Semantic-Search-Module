"""
Purpose:
Central configuration file.

What goes here:
    Available embedding model options
    Available vector store options
    Default top-k
    Any constants
"""

# Embedding models with display names
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "Sentence Transformers - All MiniLM L6 v2",
    "bge-small-en": "BGE - Small English",
}

# Vector store options with display names
VECTOR_STORES = {
    "FAISS": "FAISS (In-Memory)",
    "Chroma": "Chroma (Persistent)",
}

# Default parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
MIN_DOCUMENTS = 10
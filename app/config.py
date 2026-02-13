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
    "sentence-transformers/all-mpnet-base-v2": "Recommended - All MPNet Base v2 (Best Overall)",
    "all-MiniLM-L6-v2": "Sentence Transformers - All MiniLM L6 v2",
    "sentence-transformers/all-mpnet-base-v2": "Sentence Transformers - All MPNet Base v2",
    "BAAI/bge-base-en-v1.5": "BGE - Base English",
    "BAAI/bge-large-en-v1.5": "BGE - Large English",
    "intfloat/e5-large-v2": "E5 - Large v2",
    "intfloat/e5-small-v2": "E5 - Small v2",
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
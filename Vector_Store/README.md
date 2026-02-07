# Vector Store Module

## Overview
The `store_manager.py` module provides a factory pattern implementation for vector databases, supporting both FAISS and Chroma backends for semantic similarity search.

## Architecture

### VectorStoreFactory Class

**Purpose**: Creates and returns the appropriate vector store instance based on user selection.

```python
from Vector_Store.store_manager import VectorStoreFactory

# Create a FAISS store
store = VectorStoreFactory.create(
    db_type="FAISS",
    embedding_manager=embedding_manager,
    documents=chunked_documents
)

# Or create a Chroma store
store = VectorStoreFactory.create(
    db_type="Chroma",
    embedding_manager=embedding_manager,
    documents=chunked_documents
)
```

**Parameters**:
- **db_type**: `"FAISS"` or `"Chroma"`
- **embedding_manager**: EmbeddingManager instance (provides `embed_documents()` and `embed_query()`)
- **documents**: List of LangChain Document objects to index

**Returns**: `BaseVectorStore` subclass instance (FaissVectorStore or ChromaVectorStore)

---

## FAISS Vector Store

### **FaissVectorStore Class**

**Use Case**: Fast in-memory semantic search, best for offline analysis and evaluation.

#### Initialization
```python
from Vector_Store.store_manager import FaissVectorStore
from embeddings.embedding_manager import EmbeddingManager

em = EmbeddingManager("all-MiniLM-L6-v2")
store = FaissVectorStore(em, documents)
```

#### How It Works

1. **Embedding Generation**
   - Calls `embedding_manager.embed_documents(texts)` for all chunks
   - Returns embeddings as `numpy.ndarray` of shape (N, embedding_dim)

2. **Index Building**
   - Creates FAISS `IndexFlatL2` (Euclidean distance index)
   - Efficiently indexes embeddings for nearest neighbor search
   - Memory usage: O(N × embedding_dim × 4 bytes)

3. **Search Process**
   - Query embedding generated via `embedding_manager.embed_query(query)`
   - FAISS performs similarity search using L2 distance
   - Returns top-k nearest documents

#### Search Method
```python
def search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Returns:
        [
            {
                "source": "document_name.txt",
                "content": "relevant text snippet...",
                "score": 0.92  # Similarity score (0-1)
            },
            ...
        ]
    """
```

#### Similarity Scoring
- **Raw metric**: L2 Euclidean distance
- **Conversion to score**: `score = 1 / (1 + distance)`
  - Lower distance → Higher score
  - Range: (0, 1]
  - Perfect match: 1.0

#### Advantages
✓ **Fast**: ~0.1–1ms per query (after index built)
✓ **Memory efficient**: Minimal overhead beyond embeddings
✓ **No external dependencies**: Pure FAISS
✓ **Deterministic**: Same results across runs
✓ **Easy debugging**: Direct access to embeddings

#### Disadvantages
✗ **In-memory only**: Data lost on restart
✗ **No persistence**: Cannot load/save index
✗ **No real-time updates**: Must rebuild index for new documents

#### Complexity Analysis
- **Build time**: O(N × embedding_dim) where N = number of documents
- **Query time**: O(N × embedding_dim) for exact search
- **Space**: O(N × embedding_dim × 4 bytes)

---

## Chroma Vector Store

### **ChromaVectorStore Class**

**Use Case**: Production-ready semantic search with persistent storage, better for applications.

#### Initialization
```python
from Vector_Store.store_manager import ChromaVectorStore
from embeddings.embedding_manager import EmbeddingManager

em = EmbeddingManager("all-MiniLM-L6-v2")
store = ChromaVectorStore(em, documents)
```

#### How It Works

1. **Client Setup**
   - Initializes Chroma `EphemeralClient` (in-memory for this session)
   - Could use persistent client by modifying `_initialize_collection()`

2. **Collection Creation**
   - Creates collection with cosine similarity metric
   - Metadata: `{"hnsw:space": "cosine"}`

3. **Document Indexing**
   - Pre-computes embeddings via `embedding_manager.embed_query()`
   - Stores documents + embeddings + metadata in collection
   - Chroma manages the HNSW index internally

4. **Search Process**
   - Query embedding computed via `embedding_manager.embed_query()`
   - Chroma HNSW index performs approximate similarity search
   - Returns top-k results with distances

#### Search Method
```python
def search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Returns:
        [
            {
                "source": "document_name.txt",
                "content": "relevant text snippet...",
                "score": 0.87  # Similarity score (0-1)
            },
            ...
        ]
    """
```

#### Similarity Scoring
- **Raw metric**: Cosine distance (0-2 range)
- **Conversion to score**: `score = 1 - distance`
  - Cosine distance ≈ 0 → Score ≈ 1.0 (identical)
  - Cosine distance ≈ 2 → Score ≈ -1.0 (opposite)
  - Typical range for results: (0, 1]

#### Advantages
✓ **Persistent**: Data can be saved/loaded from disk
✓ **Scalable**: HNSW index handles large datasets
✓ **Fast**: Approximate search ~1–10ms per query
✓ **Flexible**: Supports metadata filtering
✓ **Integration**: Works seamlessly with LangChain

#### Disadvantages
✗ **Approximate search**: May miss some relevant documents
✗ **Complex setup**: Additional dependencies
✗ **Slower build**: Need to create HNSW structure

#### Complexity Analysis
- **Build time**: O(N × log N) for HNSW construction
- **Query time**: O(log N) approximate search
- **Space**: O(N × embedding_dim) for index

---

## Comparison: FAISS vs Chroma

| Feature | FAISS | Chroma |
|---------|-------|--------|
| **Search Speed** | ~0.1–1ms | ~1–10ms |
| **Index Type** | Flat (exact) | HNSW (approximate) |
| **Persistence** | No | Yes |
| **Memory** | Minimal | Moderate |
| **Scalability** | Good for <1M docs | Good for >1M docs |
| **Accuracy** | 100% | 95–99% |
| **Setup** | Simple | Complex |
| **Use Case** | Eval, testing | Production, persistence |

---

## BaseVectorStore (Abstract Base)

Common interface for all vector stores:

```python
class BaseVectorStore:
    def __init__(self, embedding_manager: Any, documents: List[Any]):
        """Initialize with embeddings and documents."""
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search."""
```

All implementations must:
1. Accept embedding_manager and documents in `__init__`
2. Build/initialize index
3. Implement `search(query, top_k)` returning result dicts

---

## Result Format

All vector stores return results in this standardized format:

```python
{
    "source": "document_filename.txt",      # Metadata from document
    "content": "Truncated or full text...", # Document content
    "score": 0.92                           # Similarity score (0-1)
}
```

---

## Integration with GUI

In `app/gui.py`:

```python
from Vector_Store.store_manager import VectorStoreFactory

# Initialize
store = VectorStoreFactory.create(
    db_type=user_selected_db,           # "FAISS" or "Chroma"
    embedding_manager=embedding_manager,
    documents=chunked_documents
)

# Search
results = store.search(query=user_query, top_k=user_top_k)

# Display results
for rank, result in enumerate(results, 1):
    print(f"{rank}. {result['source']} (score: {result['score']:.4f})")
    print(f"   {result['content'][:200]}...")
```

---

## Dependencies

- `faiss-cpu` (for FAISS implementation)
- `chromadb` (for Chroma implementation)
- `numpy` (for array operations)


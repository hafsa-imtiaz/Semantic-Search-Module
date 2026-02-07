# Embeddings Module

## Overview
The `embedding_manager.py` module provides a unified interface for generating semantic embeddings from text using Hugging Face models.

## EmbeddingManager Class

### Purpose
Abstracts the complexity of loading and using embedding models, providing a consistent API for encoding documents and queries into dense vector representations.

### Key Features

#### Dual-Backend Architecture
- **Primary**: LangChain's `HuggingFaceEmbeddings` (recommended)
- **Fallback**: `sentence-transformers.SentenceTransformer` (local, faster)
- Automatically selects available backend with graceful degradation

#### `__init__(model_name: str, device: Optional[str] = None, batch_size: int = 32, **model_kwargs)`

**Parameters**:
- **model_name**: HuggingFace model identifier (e.g., `"all-MiniLM-L6-v2"`, `"bge-small-en"`)
- **device**: `"cuda"` (GPU) or `"cpu"` (default: auto-detect)
- **batch_size**: Documents per batch for memory efficiency (default: 32)
- **model_kwargs**: Additional arguments passed to the model

**Example**:
```python
from embeddings.embedding_manager import EmbeddingManager

# Initialize with a model
em = EmbeddingManager(
    model_name="all-MiniLM-L6-v2",
    device="cuda",
    batch_size=16
)
```

### Core Methods

#### `embed_documents(documents: List[Union[str, Document]]) -> List[List[float]]`

Generates embeddings for multiple texts or LangChain Document objects.

**Features**:
- Accepts both strings and Document objects (extracts `page_content`)
- **Batching**: Processes large lists efficiently without OOM
- **Normalization**: Strips whitespace, handles None values
- **Order preservation**: Returns embeddings in same order as input

**Returns**: `List[List[float]]` - Vectors of shape (N, embedding_dim)

**Example**:
```python
documents = [
    Document(page_content="Machine learning basics..."),
    Document(page_content="Deep learning architectures..."),
]

embeddings = em.embed_documents(documents)
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```

#### `embed_query(query: str) -> List[float]`

Generates embedding for a single query string.

**Features**:
- Uses identical model to `embed_documents()` for consistency
- Returns single vector
- Handles empty/None queries gracefully

**Returns**: `List[float]` - Single vector of shape (embedding_dim,)

**Example**:
```python
query = "What are transformer models?"
query_embedding = em.embed_query(query)
print(f"Query embedding dimension: {len(query_embedding)}")
```

#### `embedding_dim` Property

Returns the dimensionality of embeddings (e.g., 384 for MiniLM).

**Lazy evaluation**: Computes on first access if not cached.

**Example**:
```python
dim = em.embedding_dim
print(f"Each embedding has {dim} dimensions")
```

## Supported Models

### Recommended (from config.py)
| Model ID | Size | Speed | Quality |
|----------|------|-------|---------|
| `all-MiniLM-L6-v2` | 22MB | Fast | Good |
| `bge-small-en` | 33MB | Fast | Excellent |

### Others Supported
Any HuggingFace model with embedding capability (e.g., `bert-base-uncased`, `sentence-t5-base`)

## Implementation Details

### Batching Strategy
```
Input: 1000 documents
Batch size: 32
Processing:
  - Batch 1: docs 0-31
  - Batch 2: docs 32-63
  ...
  - Batch 31: docs 992-999
Memory efficient, no OOM risks
```

### Text Preparation
1. Extract `page_content` from Document objects
2. Convert to string if needed
3. Strip leading/trailing whitespace
4. Replace empty/None with empty string

### Error Handling
- Validates at least one embedding backend available
- Gracefully handles model loading failures
- Returns informative error messages
- No silent failures

## Performance Characteristics

### Speed (Approximate on CPU)
- 32 documents × 500 chars: ~0.5–1.5 seconds (MiniLM)
- Single query: ~10–50 ms

### Memory
- MiniLM model: ~86 MB loaded
- Batch of 32 documents: ~50–100 MB additional

### Quality
- MiniLM: Good for general semantic search
- BGE: Excellent for retrieval-augmented generation

## Integration with Vector Stores

Vector stores receive embeddings via:
```python
embeddings = embedding_manager.embed_documents(chunks)  # List[List[float]]
query_emb = embedding_manager.embed_query(query_str)    # List[float]
```

Both FAISS and Chroma vector stores use these embeddings for similarity search.

## Dependencies
- `langchain` (or `langchain-core`)
- `sentence-transformers`
- `transformers`
- `torch`


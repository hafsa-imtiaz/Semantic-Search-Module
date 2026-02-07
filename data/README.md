# Data Loader Module

## Overview
The `loader.py` module handles document loading and chunking for the semantic search system.

## DocumentLoader Class

### Purpose
Converts raw text files (uploaded or from filesystem) into LangChain `Document` objects, with optional text chunking for large documents.

### Key Features

#### `__init__(chunk_size: int = 500, chunk_overlap: int = 50)`
- **chunk_size**: Maximum characters per chunk (default: 500)
- **chunk_overlap**: Character overlap between chunks (default: 50)
- Overlap ensures context continuity between chunks

#### `load_documents(uploaded_files)`
Loads documents from multiple sources:
- **Streamlit uploaded file objects** (has `.read()` and `.name` attributes)
- **Filesystem paths** (strings or Path-like objects)

**Returns**: `List[Document]` - LangChain Document objects with:
- **page_content**: Full text of the document
- **metadata**: Dictionary with `source` field (filename)

**Example**:
```python
from data.loader import DocumentLoader

loader = DocumentLoader()
documents = loader.load_documents(["path/to/doc1.txt", "path/to/doc2.txt"])
```

#### `chunk_documents(documents: List[Document])`
Splits large documents into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`.

**Returns**: `List[Document]` - Chunked documents preserving original metadata

**Example**:
```python
chunks = loader.chunk_documents(documents)
print(f"Original: {len(documents)} docs → Chunks: {len(chunks)}")
```

## Implementation Details

### Document Format
All documents follow LangChain's standard:
```python
Document(
    page_content="Text content here...",
    metadata={"source": "filename.txt"}
)
```

### Import Compatibility
The module handles different LangChain versions:
- Primary: `langchain.schema.Document`
- Fallback: `langchain_core.documents.base.Document`

### Error Handling
- Automatically detects file type (uploaded file object vs filesystem path)
- Handles UTF-8 decoding with graceful error messages
- Preserves original document metadata

## Usage in System

1. **GUI uploads files** → `load_documents()` converts to Document objects
2. **Documents passed to chunker** → `chunk_documents()` splits for embeddings
3. **Chunks sent to EmbeddingManager** → Embedded into vector space
4. **Chunked documents stored in Vector Store** → Ready for semantic search

## Requirements
- `langchain` or `langchain-core`
- `langchain-text-splitters`
- Python 3.8+


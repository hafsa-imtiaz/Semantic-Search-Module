from typing import List, Dict, Optional, Any
import numpy as np


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create(db_type: str, embedding_manager: Any, documents: List[Any]) -> "BaseVectorStore":
        """Factory method to create a vector store based on the specified type.
        
        Args:
            db_type: Type of vector database ("FAISS" or "Chroma")
            embedding_manager: EmbeddingManager instance for encoding documents and queries
            documents: List of LangChain Document objects to store
            
        Returns:
            An initialized vector store instance
            
        Raises:
            ValueError: If db_type is not supported
        """
        if db_type == "FAISS":
            return FaissVectorStore(embedding_manager, documents)
        elif db_type == "Chroma":
            return ChromaVectorStore(embedding_manager, documents)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


class BaseVectorStore:
    """Base class for vector store implementations."""

    def __init__(self, embedding_manager: Any, documents: List[Any]):
        """Initialize vector store with embedding manager and documents.
        
        Args:
            embedding_manager: EmbeddingManager instance
            documents: List of Document objects to index
        """
        self.embedding_manager = embedding_manager
        self.documents = documents

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries with keys: 'source', 'content', 'score'
        """
        raise NotImplementedError


class FaissVectorStore(BaseVectorStore):
    """FAISS-based vector store for in-memory semantic search."""

    def __init__(self, embedding_manager: Any, documents: List[Any]):
        """Initialize FAISS vector store."""
        super().__init__(embedding_manager, documents)
        
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        self.faiss = faiss
        self.index = None
        self.embeddings = None
        
        # Build the index
        self._build_index()

    def _build_index(self):
        """Generate embeddings for all documents and build FAISS index."""
        # Extract content from documents
        texts = []
        for doc in self.documents:
            if hasattr(doc, "page_content"):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
        
        # Generate embeddings
        self.embeddings = self.embedding_manager.embed_documents(texts)
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        
        # Get embedding dimension
        embedding_dim = embeddings_array.shape[1]
        
        # Create FAISS index
        self.index = self.faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings_array)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search FAISS index for similar documents.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of result dicts containing source, content, and similarity score
        """
        if self.index is None:
            return []
        
        # Embed the query
        query_embedding = self.embedding_manager.embed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS (returns distances, not similarities)
        distances, indices = self.index.search(query_array, min(top_k, len(self.documents)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            idx = int(idx)
            doc = self.documents[idx]
            
            # Convert L2 distance to similarity score (cosine-like normalization)
            # Lower distance = higher similarity
            similarity_score = 1.0 / (1.0 + float(distance))
            
            result = {
                "source": doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown",
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "score": similarity_score
            }
            results.append(result)
        
        return results


class ChromaVectorStore(BaseVectorStore):
    """Chroma-based vector store for persistent semantic search."""

    def __init__(self, embedding_manager: Any, documents: List[Any]):
        """Initialize Chroma vector store."""
        super().__init__(embedding_manager, documents)
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        self.chromadb = chromadb
        self.client = None
        self.collection = None
        
        # Initialize Chroma client and collection
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Chroma collection with documents and embeddings."""
        # Create a client (ephemeral in-memory client for this session)
        try:
            self.client = self.chromadb.EphemeralClient()
        except Exception:
            self.client = self.chromadb.Client()
        
        # Create a collection
        self.collection = self.client.create_collection(
            name="semantic_search_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents to collection
        ids = []
        metadatas = []
        documents_text = []
        embeddings = []
        
        for i, doc in enumerate(self.documents):
            doc_id = f"doc_{i}"
            ids.append(doc_id)
            
            # Extract content
            if hasattr(doc, "page_content"):
                content = doc.page_content
            else:
                content = str(doc)
            documents_text.append(content)
            
            # Extract metadata
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            metadatas.append(metadata)
            
            # Generate embedding
            embedding = self.embedding_manager.embed_query(content)
            embeddings.append(embedding)
        
        # Add to collection (Chroma accepts embeddings directly)
        self.collection.add(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search Chroma collection for similar documents.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of result dicts containing source, content, and similarity score
        """
        if self.collection is None:
            return []
        
        # Query the collection (Chroma handles embedding internally OR we can pass pre-computed)
        # We'll use Chroma's query with the raw query text
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.documents))
        )
        
        # Parse results
        output = []
        if results and results["distances"] and len(results["distances"]) > 0:
            distances = results["distances"][0]
            ids = results["ids"][0] if results["ids"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            documents = results["documents"][0] if results["documents"] else []
            
            for i, (distance, doc_id, metadata, content) in enumerate(
                zip(distances, ids, metadatas, documents)
            ):
                # Chroma returns distances; convert to similarity (cosine distance)
                # For cosine distance: similarity = 1 - distance
                similarity_score = 1.0 - float(distance)
                
                result = {
                    "source": metadata.get("source", "Unknown") if metadata else "Unknown",
                    "content": content,
                    "score": similarity_score
                }
                output.append(result)
        
        return output
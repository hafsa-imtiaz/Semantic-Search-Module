from typing import List, Optional, Union

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    _HAS_LANGCHAIN_EMBED = True
except Exception:
    HuggingFaceEmbeddings = None
    _HAS_LANGCHAIN_EMBED = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False


class EmbeddingManager:
    """Wrapper around an embedding backend.

    Primary backend: LangChain's `HuggingFaceEmbeddings`.
    Fallback: `sentence-transformers.SentenceTransformer`.

    The selected `model_name` (passed from GUI) is forwarded to the backend.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 32, **model_kwargs):
        self.model_name = model_name
        self.batch_size = batch_size or 32
        self.device = device
        self._embedder = None
        self._model = None
        self._embedding_dim: Optional[int] = None

        # Try LangChain HuggingFaceEmbeddings first
        if _HAS_LANGCHAIN_EMBED:
            try:
                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                # allow passing through additional model kwargs
                kwargs.update(model_kwargs)
                self._embedder = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=kwargs)
            except Exception:
                self._embedder = None

        # Fallback to sentence-transformers
        if self._embedder is None and _HAS_SENTENCE_TRANSFORMERS:
            try:
                # SentenceTransformer accepts the model id directly
                self._model = SentenceTransformer(self.model_name)
                # SentenceTransformer exposes embedding dim
                try:
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
                except Exception:
                    self._embedding_dim = None
            except Exception:
                self._model = None

        if self._embedder is None and self._model is None:
            raise RuntimeError("No embedding backend available. Install `langchain` or `sentence-transformers`.")

    @property
    def embedding_dim(self) -> Optional[int]:
        if self._embedding_dim:
            return self._embedding_dim
        # try to infer from embedder by embedding a short text (lazy)
        if self._embedder is not None:
            sample = self.embed_documents(["test"])
            if sample:
                self._embedding_dim = len(sample[0])
                return self._embedding_dim
        if self._model is not None:
            try:
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                return self._embedding_dim
            except Exception:
                return None
        return None

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        prepared = []
        for t in texts:
            if t is None:
                prepared.append("")
            else:
                s = str(t).strip()
                prepared.append(s)
        return prepared

    def embed_documents(self, documents: List[Union[str, object]]) -> List[List[float]]:
        """Embed a list of texts or LangChain `Document`-like objects.

        Args:
            documents: list of strings or objects with `page_content` attribute.

        Returns:
            List of embedding vectors (lists of floats) in the same order.
        """
        if not documents:
            return []

        texts: List[str] = []
        for d in documents:
            if hasattr(d, "page_content"):
                texts.append(getattr(d, "page_content") or "")
            else:
                texts.append(str(d))

        texts = self._prepare_texts(texts)

        embeddings: List[List[float]] = []
        n = len(texts)
        for i in range(0, n, self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self._embedder is not None:
                # LangChain wrapper
                batch_emb = self._embedder.embed_documents(batch)
            else:
                # sentence-transformers returns numpy arrays; convert to lists
                batch_emb = self._model.encode(batch, show_progress_bar=False)
                # ensure list of lists
                batch_emb = [list(map(float, e)) for e in batch_emb]

            embeddings.extend(batch_emb)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string and return its vector."""
        if query is None:
            query = ""
        text = self._prepare_texts([query])[0]

        if self._embedder is not None:
            vec = self._embedder.embed_query(text)
            return list(map(float, vec))

        # sentence-transformers
        vec = self._model.encode(text, show_progress_bar=False)
        return list(map(float, vec))

    def __repr__(self) -> str:
        backend = "HuggingFaceEmbeddings" if self._embedder is not None else "SentenceTransformer"
        return f"<EmbeddingManager model={self.model_name} backend={backend} batch_size={self.batch_size}>"

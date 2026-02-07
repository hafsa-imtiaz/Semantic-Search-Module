"""
This module must:
    Accept uploaded text files
    Read their content
    Convert them into LangChain Document objects
    Optionally chunk large documents
    Return a clean list of documents
"""
import os
from typing import List, Union

# Import Document in a version-compatible way: prefer langchain.schema, fallback to langchain_core
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    try:
        from langchain_core.documents.base import Document  # type: ignore
    except Exception:
        Document = None  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter 


class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, uploaded_files: List[Union[str, "os.PathLike", object]]) -> List[Document]:
        """Load documents from uploaded file-like objects or from file paths.

        Accepts a list where each item is either:
        - a Streamlit uploaded file object (has `.read()` and `.name`)
        - a filesystem path (string or Path-like)

        Returns a list of `Document` instances.
        """
        documents = []

        for item in uploaded_files:
            # Case 1: file-like object (e.g., Streamlit uploaded file)
            if hasattr(item, "read"):
                raw = item.read()
                # raw may be bytes or str
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8")
                else:
                    text = str(raw)
                source = getattr(item, "name", None)

            else:
                # Treat item as a path
                path = str(item)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                # use basename as source
                try:
                    import os

                    source = os.path.basename(path)
                except Exception:
                    source = path

            documents.append(
                Document(page_content=text, metadata={"source": source})
            )

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
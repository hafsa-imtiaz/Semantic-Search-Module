"""
AI Research Assistant - Semantic Search Module (Phase 1)
Streamlit GUI for document upload, embedding initialization, and semantic retrieval.
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import DocumentLoader
from app.config import EMBEDDING_MODELS, VECTOR_STORES


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'vector_db_type' not in st.session_state:
        st.session_state.vector_db_type = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_top_k' not in st.session_state:
        st.session_state.last_top_k = None


def calculate_total_size(files) -> tuple[int, str]:
    """
    Calculate total size of uploaded files.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        Tuple of (size_in_bytes, formatted_string)
    """
    total_bytes = sum(len(file.getvalue()) for file in files)
    
    if total_bytes < 1024:
        return total_bytes, f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        return total_bytes, f"{total_bytes / 1024:.2f} KB"
    else:
        return total_bytes, f"{total_bytes / (1024 * 1024):.2f} MB"


def save_uploaded_files(uploaded_files, temp_dir: Path) -> List[Path]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        temp_dir: Path to temporary directory
        
    Returns:
        List of file paths
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths


def initialize_semantic_memory(
    file_paths: List[Path],
    embedding_model: str,
    vector_db: str
) -> bool:
    """
    Initialize the semantic memory system.
    
    Args:
        file_paths: List of document file paths
        embedding_model: Name of the HuggingFace embedding model
        vector_db: Type of vector database (FAISS or Chroma)
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        from embeddings.embedding_manager import EmbeddingManager
        from Vector_Store.store_manager import VectorStoreFactory
        
        # Load and chunk documents
        loader = DocumentLoader()
        documents = loader.load_documents(file_paths)
        chunks = loader.chunk_documents(documents)
        
        # Initialize embeddings
        embedding_manager = EmbeddingManager(model_name=embedding_model)
        
        # Create vector store
        store_manager = VectorStoreFactory.create(
            db_type=vector_db,
            embedding_manager=embedding_manager,
            documents=chunks
        )
        
        # Store in session state
        st.session_state.vector_store = store_manager
        st.session_state.documents = chunks
        st.session_state.is_initialized = True
        st.session_state.embedding_model = embedding_model
        st.session_state.vector_db_type = vector_db
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing semantic memory: {str(e)}")
        return False


def perform_search(query: str, top_k: int) -> Optional[List[Dict]]:
    """
    Perform semantic search on the vector store.
    
    Args:
        query: User search query
        top_k: Number of results to return
        
    Returns:
        List of result dictionaries or None if error
    """
    try:
        if st.session_state.vector_store is None:
            return None
        
        results = st.session_state.vector_store.search(
            query=query,
            top_k=top_k
        )
        
        return results
        
    except Exception as e:
        st.error(f"Error performing search: {str(e)}")
        return None


def _render_setup_section():
    """Render the setup section (upload and configuration)."""
    
    # ========================================================================
    # Dataset Upload
    # ========================================================================
    st.header("📂 Dataset Upload")
    
    uploaded_files = st.file_uploader(
        "Upload academic documents (.txt format)",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload at least 10-15 documents for optimal semantic search results",
        key="file_uploader"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        total_bytes, size_str = calculate_total_size(uploaded_files)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Uploaded", num_files)
        with col2:
            st.metric("Total Dataset Size", size_str)
        
        if num_files < 10:
            st.warning(f"⚠️ You have uploaded {num_files} files. Consider uploading at least 10-15 documents for better results.")
    else:
        st.info("👆 Please upload your document dataset to begin.")
    
    st.divider()
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    st.header("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda x: EMBEDDING_MODELS[x],
            help="Choose a HuggingFace embedding model for semantic encoding",
            key="embedding_model_select"
        )
    
    with col2:
        vector_db = st.selectbox(
            "Select Vector Database",
            options=list(VECTOR_STORES.keys()),
            format_func=lambda x: VECTOR_STORES[x],
            help="Choose the vector database backend for similarity search",
            key="vector_db_select"
        )
    
    st.markdown("")  # Spacing
    
    # Show initialize or reset button based on state
    if st.session_state.is_initialized:
        # Show reset button when already initialized
        if st.button(
            "🔄 Reset System",
            type="secondary",
            use_container_width=True,
            key="reset_button"
        ):
            st.session_state.is_initialized = False
            st.session_state.vector_store = None
            st.session_state.documents = []
            st.session_state.last_results = None
            st.session_state.last_query = None
            st.session_state.last_top_k = None
            st.rerun()
    else:
        # Show initialize button when not initialized
        init_button_disabled = not uploaded_files
        
        if st.button(
            "🚀 Initialize Semantic Memory",
            disabled=init_button_disabled,
            type="primary",
            use_container_width=True,
            key="init_button"
        ):
            if not uploaded_files:
                st.error("❌ Please upload documents before initializing.")
            else:
                with st.spinner("🔄 Loading documents, generating embeddings, and building vector index..."):
                    # Save uploaded files temporarily
                    temp_dir = Path("/tmp/semantic_search_docs")
                    file_paths = save_uploaded_files(uploaded_files, temp_dir)
                    
                    # Initialize semantic memory
                    success = initialize_semantic_memory(
                        file_paths=file_paths,
                        embedding_model=embedding_model,
                        vector_db=vector_db
                    )
                    
                    if success:
                        st.success(f"✅ Semantic memory initialized successfully!")
                        st.balloons()
                        st.rerun()  # Rerun to update UI and focus on search section
    
    st.divider()


# ============================================================================
# MAIN GUI
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Page title
    st.title("🔍 AI Research Assistant – Semantic Search Module")
    st.markdown("*Phase 1: Document Upload & Semantic Retrieval*")
    st.divider()
    
    # ========================================================================
    # SECTION 1 & 2: Setup (Collapsible after initialization)
    # ========================================================================
    
    # If system is initialized, show setup section in collapsed expander
    if st.session_state.is_initialized:
        with st.expander("⚙️ System Configuration", expanded=False):
            _render_setup_section()
    else:
        # If not initialized, show setup section prominently
        _render_setup_section()
    
    # ========================================================================
    # SECTION 3: Semantic Retrieval (Main focus after initialization)
    # ========================================================================
    
    if st.session_state.is_initialized:
        st.header("🔎 Semantic Retrieval")
        
        # Display system status
        st.success(f"✓ System Ready | Model: {EMBEDDING_MODELS[st.session_state.embedding_model]} | DB: {st.session_state.vector_db_type} | Documents: {len(st.session_state.documents)}")
        st.markdown("")
        
        # Query input - make it prominent
        query = st.text_input(
            "🔍 Enter your research query",
            placeholder="e.g., What are the main applications of transformer architectures?",
            help="Ask questions or describe the information you're looking for",
            key="query_input"
        )
        
        # Search parameters in columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            top_k = st.slider(
                "Number of results (top-k)",
                min_value=1,
                max_value=10,
                value=3,
                help="How many relevant document chunks to retrieve"
            )
        
        with col2:
            st.markdown("")  # Spacing
            search_clicked = st.button(
                "🔍 Search",
                type="primary",
                use_container_width=True
            )
        
        # Handle search
        if search_clicked:
            if not query.strip():
                st.error("❌ Please enter a search query.")
            else:
                with st.spinner("🔍 Searching through semantic space..."):
                    results = perform_search(query, top_k)
                
                # Store results in session state
                st.session_state.last_results = results
                st.session_state.last_query = query
                st.session_state.last_top_k = top_k
        
        # Display results from session state (persists across reruns)
        if st.session_state.last_results is not None:
            results = st.session_state.last_results
            
            if results:
                st.success(f"✨ Found {len(results)} relevant results for: \"{st.session_state.last_query}\"")
                st.markdown("")
                
                # Display results
                for idx, result in enumerate(results, 1):
                    with st.expander(f"**Rank #{idx}** | {result.get('source', 'Unknown Source')}", expanded=(idx == 1)):
                        # Extract result data
                        source = result.get('source', 'Unknown')
                        content = result.get('content', '')
                        score = result.get('score', None)
                        
                        # Display metadata
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**📄 Source:** `{source}`")
                        with col2:
                            if score is not None:
                                st.markdown(f"**Similarity:** {score:.4f}")
                        
                        st.markdown("---")
                        st.markdown("**📝 Content Preview:**")
                        
                        # Preview (first 300 characters)
                        preview = content[:300]
                        if len(content) > 300:
                            preview += "..."
                        
                        st.markdown(f"> {preview}")
                        
                        # Optional: show full content with nested expander
                        if len(content) > 300:
                            st.markdown("")
                            with st.expander("📄 Show Full Content"):
                                st.text_area(
                                    "Full Content",
                                    value=content,
                                    height=200,
                                    key=f"content_{idx}",
                                    label_visibility="collapsed"
                                )
            else:
                st.warning("No results found. Try a different query.")
    
    else:
        # Show placeholder when not initialized
        st.info("👆 Please upload documents and initialize the semantic memory system to start searching.")
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray;'>AI Research Assistant v1.0 | Phase 1: Semantic Search</p>",
        unsafe_allow_html=True
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
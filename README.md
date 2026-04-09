# AI Research Assistant - Semantic Search Module
### CS-4015: Agentic AI | Homework #1 (Phase 1)
**National University of Computer & Emerging Sciences (FAST-NUCES)**

**Student Name:** Hafsa Imtiaz  
**Student ID:** 22i-0959  
**Section:** A  

---

## 🚀 Project Overview
This project serves as the **Memory System** for an AI Research Assistant. It implements a robust Semantic Search Module designed to help university students navigate vast amounts of academic material (lecture notes, research papers, FAQs) by retrieving information based on **contextual meaning** rather than simple keyword matching.

This is Assignment 1 of a multi-part series in the Agentic AI course, focusing on the foundational retrieval-augmented generation (RAG) pipeline.



## 🛠️ System Architecture
The system follows a modular, sequential pipeline architecture using a **Factory Pattern** for vector storage and a **dual-backend** architecture for embeddings.

### 1. Data Module (`loader.py`)
* Handles dynamic file uploads and filesystem loading.
* Implements text chunking with a default size of **500 characters** and **50 character overlap** to ensure semantic continuity across boundaries.

### 2. Embeddings Module (`embedding_manager.py`)
* Supports **6 HuggingFace models** with automatic GPU/CPU detection.
* Models include: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-base-en-v1.5`, `e5-large-v2`, `e5-small-v2`, and `BAAI/bge-large-en-v1.5`.

### 3. Vector Store Module (`store_manager.py`)
* Implements **FAISS** for high-speed in-memory searches.
* Implements **ChromaDB** for persistent document storage.

### 4. GUI Module (`gui.py`)
* Built with **Streamlit** to provide an interactive user experience.
* Allows real-time configuration of models, databases, and `top-k` retrieval parameters.

---

## 📊 Performance Benchmarking
A key component of this project was the automated evaluation of different embedding models using a dataset of 12 academic documents related to AI/ML.

| Model | Load Time (s) | Avg Top Score | Observations |
| :--- | :--- | :--- | :--- |
| **E5 Small v2** | 6.37s | **0.8270** | **Best Performance:** Highest semantic understanding. |
| **BGE Base English** | 6.09s | 0.7377 | **Best Balance:** Fast query speed + strong accuracy. |
| **BGE Large English** | 245.81s | 0.7284 | **Inefficient:** Extremely high initialization time. |
| **All MiniLM L6 v2** | 5.61s | 0.6133 | **Lightweight:** Fast but lower semantic depth. |

> **Note:** During testing, the `all-mpnet-base-v2` model paired with FAISS achieved a maximum similarity score of ~77% on technical research queries, demonstrating high reliability for academic retrieval.

---

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.9+
* HuggingFace API Token (optional, depending on local cache)

### Setup
1. Clone the repository:
   ```bash
   git clone [your-repository-link]
   cd [repository-folder]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app/gui.py
   ```

### How to Use
1. **Upload:** Use the sidebar to upload at least 10-15 `.txt` or `.pdf` documents.
2. **Configure:** Select your preferred Embedding Model and Vector Store (FAISS/Chroma).
3. **Initialize:** Click "Process Documents" to generate the vector index.
4. **Search:** Enter a query in the search bar and adjust the `top-k` slider to view ranked results with relevance scores.

---

## 📁 Repository Structure
```text
├── app/
│   ├── gui.py                 # Streamlit interface for user interaction
│   ├── config.py              # Configuration for models and system paths
│   └── main.py                # Main application entry script
├── data/
│   ├── loader.py              # Document loading and preprocessing logic
│   └── README.md              # Data module documentation
├── embeddings/
│   ├── embedding_manager.py   # Embedding generation and model management
│   └── README.md              # Embeddings module documentation
├── Vector_Store/
│   ├── store_manager.py       # Vector database logic (FAISS/ChromaDB)
│   └── README.md              # Vector store module documentation
├── experiments/
│   ├── test_semantic_search.py # Benchmarking and evaluation scripts
│   ├── report/                # Directory for experiment results
│   └── README.md              # Experiments module documentation
├── HW1_Phase1_AgenticAI.pdf   # Assignment instructions and requirements
└── requirements.txt           # Project dependencies
```

---
**Disclaimer:** This project was developed for academic purposes as part of the CS-4015 Agentic AI course at FAST-NUCES.
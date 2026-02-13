# Experiments Module

Testing and evaluation for the Semantic Search system.

## test_semantic_search.py

### What It Does

This script performs comprehensive testing of embedding models for semantic search:

1. **Tests 6 embedding models** - Evaluates each model's performance and quality
2. **Uses 12 academic documents** - Pre-built test corpus covering AI/ML topics
3. **Runs 3 semantic queries** - Tests retrieval with diverse queries
4. **Returns top-3 results** - Shows ranked documents with similarity scores
5. **Measures performance** - Tracks initialization time and query execution time
6. **Generates reports** - Creates markdown (readable) and JSON (data) reports

### How It Works

**Step 1: Document Loading**
- Loads 12 pre-built academic documents (embedded in script)
- Documents cover: ML, NLP, Deep Learning, CV, RL, Data Science, Transformers, Semantic Search, FAISS, Knowledge Graphs, Attention, QA Systems

**Step 2: Model Testing Loop**
- For each of 6 embedding models:
  - Initializes embedding manager (loads model from HuggingFace)
  - Creates FAISS vector store and indexes all documents
  - Measures initialization time

**Step 3: Query Testing**
- For each query:
  - Embeds the query using the same model
  - Searches FAISS index for top-3 similar documents
  - Records similarity scores and execution time
  - Displays results in console

**Step 4: Report Generation**
- Aggregates results across all models and queries
- Generates markdown report (human-readable)
- Generates JSON report (programmatic access)
- Both saved to `experiments/report/`

### Quick Start

#### Prerequisites
```bash
pip install -r requirements.txt
```

#### Run Tests
```bash
cd c:\Users\Hafsa\Documents\repo\hw1-phase-1-semantic-search-module
venv\Scripts\Activate.ps1
python experiments/test_semantic_search.py
```

**Time**: 5-15 minutes (first run downloads models), 2-5 minutes after (cached)

### Console Output Example

```
================================================================================
SEMANTIC SEARCH MODEL TESTING SUITE
================================================================================

Creating test documents...
✓ Created 12 test documents

================================================================================
Testing Model: all-MiniLM-L6-v2
================================================================================
✓ Initialization time: 2.345s
✓ Documents loaded: 12, Chunks: 24

Query 1: How do neural networks and deep learning work?
[1] Source: document_3.txt
    Similarity: 0.8234
    Preview: Deep learning is a subset of machine learning...
...
```

## Test Details

### Queries Tested

1. "How do neural networks and deep learning work?"
2. "What is the difference between machine learning and artificial intelligence?"
3. "Explain semantic embeddings and how they enable effective search"

### Models Tested

1. **all-MiniLM-L6-v2** - Fast, lightweight, general-purpose
2. **all-mpnet-base-v2** - Larger, better quality
3. **bge-base-en-v1.5** - English-optimized
4. **e5-large-v2** - Maximum quality, larger
5. **e5-small-v2** - Lightweight variant
6. **bge-large-en-v1.5** - Specialized for semantic search

### Evaluation Metrics

- **Initialization Time**: How long to load model and create index
- **Query Time**: How long to search (milliseconds)
- **Similarity Score**: 0-1 score indicating semantic match quality
- **Document Ranking**: Which documents returned for each query

## Reports

### Output Location
```
experiments/report/
├── test_report_20260213_143022.md     ← Markdown report (readable)
└── test_report_20260213_143022.json   ← JSON report (data)
```

### Report Contents

**Markdown Report** includes:
- Summary of results
- Model performance table
- Results for each query
- Similarity scores
- Recommendations

**JSON Report** includes:
- Raw test data
- Aggregated metrics
- All similarity scores
- Timing information

## Troubleshooting

**ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Model download fails**
- Check internet connection
- Try again (models cached after first download)

**Memory issues**
- Edit `test_semantic_search.py`: change `TOP_K = 1` or `TOP_K = 2`
- Or test single models instead of all 6

## Directory Structure

```
experiments/
├── README.md                    ← This file
├── test_semantic_search.py      ← Main testing script
├── report/
│   ├── report_template.md       ← Report template
│   ├── test_report_*.md         ← Generated reports
│   └── test_report_*.json       ← Generated data
└── temp_test_docs/             ← Temporary files (auto-cleaned)
```

See [report_template.md](./report/report_template.md) for report format details.


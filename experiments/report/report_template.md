# Semantic Search Testing Report

**Generated**: {TIMESTAMP}  
**Models Tested**: 6 Embedding Models  
**Vector Store**: FAISS  
**Documents**: 12 | **Queries**: 3 | **Top-K**: 3

---

## Summary

| Metric | Best Model | Score |
|--------|--------|--------|
| Fastest Init | {MODEL} | {TIME}s |
| Fastest Query | {MODEL} | {TIME}ms |
| Highest Quality | {MODEL} | {SCORE} |

---

## Test Queries

1. How do neural networks and deep learning work?
2. What is the difference between machine learning and artificial intelligence?
3. Explain semantic embeddings and how they enable effective search

---

## Model Performance

| Model | Init (s) | Query (ms) | Avg Score |
|-------|:---:|:---:|:---:|
| All MiniLM L6 v2 | {TIME} | {TIME} | {SCORE} |
| All MPNet Base v2 | {TIME} | {TIME} | {SCORE} |
| BGE Base English | {TIME} | {TIME} | {SCORE} |
| E5 Large v2 | {TIME} | {TIME} | {SCORE} |
| E5 Small v2 | {TIME} | {TIME} | {SCORE} |
| AllenAI SPECTER | {TIME} | {TIME} | {SCORE} |

---

## Query 1 Results

"How do neural networks and deep learning work?"

| Rank | Document | Similarity |
|:---:|--|:---:|
| 1 | {DOC_NAME} | {SCORE} |
| 2 | {DOC_NAME} | {SCORE} |
| 3 | {DOC_NAME} | {SCORE} |

---

## Query 2 Results

"What is the difference between machine learning and artificial intelligence?"

| Rank | Document | Similarity |
|:---:|--|:---:|
| 1 | {DOC_NAME} | {SCORE} |
| 2 | {DOC_NAME} | {SCORE} |
| 3 | {DOC_NAME} | {SCORE} |

---

## Query 3 Results

"Explain semantic embeddings and how they enable effective search"

| Rank | Document | Similarity |
|:---:|--|:---:|
| 1 | {DOC_NAME} | {SCORE} |
| 2 | {DOC_NAME} | {SCORE} |
| 3 | {DOC_NAME} | {SCORE} |

---

## Score Summary

**Best Scores Across All Queries:**

- **Highest**: {SCORE} by {MODEL}
- **Lowest**: {SCORE} by {MODEL}
- **Average**: {SCORE}

---

## Recommendations

**For Speed + Quality**: all-MiniLM-L6-v2
- Fast initialization
- Fast queries
- Good quality

**For Maximum Quality**: all-mpnet-base-v2 or e5-large-v2
- Better semantic understanding
- Trade-off: Slower

**For Specialized Content**: bge-base-en-v1.5 or allenai-specter
- Domain-optimized

**For Resource-Constrained**: e5-small-v2 or all-MiniLM-L6-v2
- Minimal memory

---

## Test Environment

- **OS**: Windows
- **Python**: 3.11
- **Vector Store**: FAISS (IndexFlatL2)
- **Vector Dimensions**: Model-dependent
- **Chunks**: 500 chars (50 char overlap)

---

## Conclusion

All models successfully retrieved semantically relevant documents. Performance varies significantly with clear trade-offs between speed and quality.

**Next Steps**:
1. Select model from recommendations
2. Configure in app/config.py
3. Deploy to production
4. Monitor performance

---

**Report Date**: {TIMESTAMP}  
See full logs in: `test_report_YYYYMMDD_HHMMSS.json`



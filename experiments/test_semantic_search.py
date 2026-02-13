"""
Testing Script for Semantic Search Module:
It runs a set of predefined queries and evaluates retrieval quality.

Features:
- Tests all 6 embedding models
- Uses FAISS for consistent, fast retrieval
- Runs 3 semantic queries with k=3 results
- Outputs top result and similarity scores
- Generates a detailed report with comparison metrics
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import DocumentLoader
from embeddings.embedding_manager import EmbeddingManager
from Vector_Store.store_manager import VectorStoreFactory
from app.config import EMBEDDING_MODELS


# ============================================================================
# SAMPLE DOCUMENTS (Academic/Research Content)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "name": "document_1.txt",
        "content": """Machine Learning Overview
Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. It uses algorithms and 
statistical models to analyze patterns and make predictions based on data. 
Common types include supervised learning, unsupervised learning, and reinforcement learning."""
    },
    {
        "name": "document_2.txt",
        "content": """Natural Language Processing
Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
computers and human language. It enables machines to understand, interpret, and generate 
human language in a meaningful way. Applications include sentiment analysis, machine translation, 
question answering, and text summarization."""
    },
    {
        "name": "document_3.txt",
        "content": """Deep Learning and Neural Networks
Deep learning is a subset of machine learning based on neural networks with multiple layers. 
These artificial neural networks mimic the structure and function of biological neurons. 
Deep learning has revolutionized computer vision, natural language processing, and other domains 
by learning hierarchical representations of data."""
    },
    {
        "name": "document_4.txt",
        "content": """Computer Vision Fundamentals
Computer vision is an interdisciplinary field that deals with how computers can gain high-level 
understanding from digital images and videos. It involves image classification, object detection, 
facial recognition, and scene understanding. Convolutional Neural Networks (CNNs) are primary 
models used for computer vision tasks."""
    },
    {
        "name": "document_5.txt",
        "content": """Reinforcement Learning Concepts
Reinforcement Learning (RL) is a paradigm where an agent learns to make decisions by interacting 
with an environment. The agent receives rewards or penalties for its actions and learns to maximize 
cumulative reward. Applications include game playing, robotics, autonomous vehicles, and resource allocation."""
    },
    {
        "name": "document_6.txt",
        "content": """Data Science and Analytics
Data science combines statistics, mathematics, and programming to extract insights from data. 
It involves data collection, cleaning, analysis, visualization, and interpretation. 
Data scientists use tools and techniques to solve business problems and support decision-making 
through evidence-based insights."""
    },
    {
        "name": "document_7.txt",
        "content": """Transformers and BERT Models
Transformers are deep learning architectures based on attention mechanisms that process sequences 
in parallel. BERT (Bidirectional Encoder Representations from Transformers) is a model pre-trained 
on large text corpora to understand language context. Transformers have become the foundation for 
modern NLP applications including language models and question answering systems."""
    },
    {
        "name": "document_8.txt",
        "content": """Semantic Search and Embeddings
Semantic search retrieves documents based on meaning rather than keyword matching. It uses word 
embeddings and sentence embeddings to represent text as dense vectors in high-dimensional space. 
Methods like Word2Vec, GloVe, and sentence-transformers enable effective semantic similarity computation 
for information retrieval and recommendation systems."""
    },
    {
        "name": "document_9.txt",
        "content": """Vector Databases and FAISS
Vector databases store and retrieve dense numerical vectors efficiently. FAISS (Facebook AI Similarity Search) 
is an open-source library for fast similarity search in high-dimensional spaces. FAISS uses approximate nearest 
neighbor search algorithms like IndexFlatL2 and IndexIVFFlat to enable scalable retrieval of similar vectors."""
    },
    {
        "name": "document_10.txt",
        "content": """Knowledge Graphs and Information Retrieval
Knowledge graphs represent structured information about entities and their relationships. 
They support semantic queries and information retrieval tasks. Information retrieval systems 
leverage knowledge graphs to improve search results and provide contextual understanding of entities 
and their connections in large-scale knowledge bases."""
    },
    {
        "name": "document_11.txt",
        "content": """Attention Mechanisms in Deep Learning
Attention mechanisms allow neural networks to focus on relevant parts of input data. 
The self-attention mechanism enables models to weigh the importance of different input positions. 
Multi-head attention processes information from multiple representation subspaces, 
improving model capacity. Attention is fundamental to transformer architectures and modern AI systems."""
    },
    {
        "name": "document_12.txt",
        "content": """Question Answering Systems
Question answering (QA) systems retrieve relevant documents and extract answers to user queries. 
They combine information retrieval with machine comprehension. Extractive QA identifies answer spans 
in documents, while generative QA produces answers from scratch. State-of-the-art QA systems use 
semantic understanding and neural ranking for improved performance."""
    }
]


# ============================================================================
# TEST QUERIES
# ============================================================================

TEST_QUERIES = [
    "How do neural networks and deep learning work?",
    "What is the difference between machine learning and artificial intelligence?",
    "Explain semantic embeddings and how they enable effective search."
]

TOP_K = 3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_documents(temp_dir: Path) -> List[Path]:
    """Create sample test documents in temporary directory."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []
    
    for doc in SAMPLE_DOCUMENTS:
        file_path = temp_dir / doc["name"]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
        file_paths.append(file_path)
    
    return file_paths


def test_model(
    model_name: str,
    file_paths: List[Path],
    queries: List[str],
    top_k: int = 3
) -> Dict:
    """
    Test a single embedding model with all queries.
    
    Args:
        model_name: HuggingFace model identifier
        file_paths: List of document file paths
        queries: List of test queries
        top_k: Number of results to return per query
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*80}")
    
    results = {
        "model": model_name,
        "model_display": EMBEDDING_MODELS.get(model_name, model_name),
        "timestamp": datetime.now().isoformat(),
        "queries": [],
        "performance_metrics": {}
    }
    
    try:
        # Initialize components
        start_time = time.time()
        
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents(file_paths)
        chunks = loader.chunk_documents(documents)
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(model_name=model_name)
        
        # Create FAISS vector store
        vector_store = VectorStoreFactory.create(
            db_type="FAISS",
            embedding_manager=embedding_manager,
            documents=chunks
        )
        
        initialization_time = time.time() - start_time
        results["performance_metrics"]["initialization_time_seconds"] = round(initialization_time, 3)
        
        print(f"✓ Initialization time: {initialization_time:.3f}s")
        print(f"✓ Documents loaded: {len(documents)}, Chunks: {len(chunks)}\n")
        
        # Test each query
        for query_idx, query in enumerate(queries, 1):
            print(f"Query {query_idx}: {query}")
            print("-" * 80)
            
            query_start = time.time()
            search_results = vector_store.search(query=query, top_k=top_k)
            query_time = time.time() - query_start
            
            query_result = {
                "query": query,
                "query_time_seconds": round(query_time, 4),
                "results": []
            }
            
            # Process results
            for rank, result in enumerate(search_results, 1):
                result_info = {
                    "rank": rank,
                    "source": result.get("source", "Unknown"),
                    "similarity_score": round(result.get("score", 0.0), 4),
                    "content_preview": result.get("content", "")[:150] + "..."
                }
                query_result["results"].append(result_info)
                
                # Print result
                print(f"  [{rank}] Source: {result['source']}")
                print(f"      Similarity: {result['score']:.4f}")
                print(f"      Preview: {result['content'][:100]}...")
                print()
            
            results["queries"].append(query_result)
        
        results["status"] = "success"
        return results
        
    except Exception as e:
        print(f"✗ Error testing model {model_name}: {str(e)}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Generate comparison metrics across all models."""
    aggregated = {
        "total_models_tested": len(all_results),
        "successful_models": sum(1 for r in all_results if r.get("status") == "success"),
        "failed_models": sum(1 for r in all_results if r.get("status") == "failed"),
        "model_comparison": [],
        "query_performance": {}
    }
    
    # Model comparison
    for result in all_results:
        if result.get("status") == "success":
            model_info = {
                "model": result["model"],
                "display_name": result["model_display"],
                "initialization_time": result["performance_metrics"].get("initialization_time_seconds", 0),
                "avg_query_time": round(
                    sum(q.get("query_time_seconds", 0) for q in result["queries"]) / len(result["queries"]),
                    4
                ),
                "avg_top_result_score": round(
                    sum(
                        q["results"][0].get("similarity_score", 0)
                        for q in result["queries"]
                        if q["results"]
                    ) / len([q for q in result["queries"] if q["results"]]),
                    4
                )
            }
            aggregated["model_comparison"].append(model_info)
    
    # Query performance across models
    for query_idx in range(len(TEST_QUERIES)):
        query = TEST_QUERIES[query_idx]
        query_scores = []
        
        for result in all_results:
            if result.get("status") == "success" and query_idx < len(result["queries"]):
                query_results = result["queries"][query_idx]
                if query_results["results"]:
                    query_scores.append(query_results["results"][0]["similarity_score"])
        
        aggregated["query_performance"][query] = {
            "max_similarity": max(query_scores) if query_scores else 0,
            "min_similarity": min(query_scores) if query_scores else 0,
            "avg_similarity": round(sum(query_scores) / len(query_scores), 4) if query_scores else 0
        }
    
    return aggregated


def generate_report(all_results: List[Dict], aggregated: Dict, output_file: Path):
    """Generate a detailed markdown report."""
    report = []
    report.append("# Semantic Search Model Testing Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Summary
    report.append("## Summary\n")
    report.append(f"- **Models Tested:** {aggregated['total_models_tested']}\n")
    report.append(f"- **Successful:** {aggregated['successful_models']}\n")
    report.append(f"- **Failed:** {aggregated['failed_models']}\n")
    report.append(f"- **Documents:** {len(SAMPLE_DOCUMENTS)}\n")
    report.append(f"- **Queries:** {len(TEST_QUERIES)}\n")
    report.append(f"- **Top-K:** {TOP_K}\n\n")
    
    # Model Comparison Table
    report.append("## Model Performance Comparison\n\n")
    report.append("| Model | Init Time (s) | Avg Query Time (s) | Avg Top Score |\n")
    report.append("|-------|------|------|------|\n")
    for model in aggregated["model_comparison"]:
        report.append(
            f"| {model['display_name']} | {model['initialization_time']} | "
            f"{model['avg_query_time']} | {model['avg_top_result_score']} |\n"
        )
    report.append("\n")
    
    # Query Performance
    report.append("## Query-Level Performance\n\n")
    for query, metrics in aggregated["query_performance"].items():
        report.append(f"**Query:** _{query}_\n\n")
        report.append(f"- Max Similarity: {metrics['max_similarity']}\n")
        report.append(f"- Min Similarity: {metrics['min_similarity']}\n")
        report.append(f"- Avg Similarity: {metrics['avg_similarity']}\n\n")
    
    # Detailed Results per Model
    report.append("## Detailed Results Per Model\n\n")
    for result in all_results:
        if result.get("status") == "success":
            report.append(f"### {result['model_display']}\n\n")
            
            for query_result in result["queries"]:
                report.append(f"**Query:** {query_result['query']}\n\n")
                report.append(f"Query Execution Time: {query_result['query_time_seconds']}s\n\n")
                
                for res in query_result["results"]:
                    report.append(
                        f"[Rank {res['rank']}] **{res['source']}** "
                        f"(Similarity: {res['similarity_score']})\n\n"
                    )
                    report.append(f"_{res['content_preview']}_\n\n")
        else:
            report.append(f"### {result['model_display']}\n\n")
            report.append(f"❌ **Error:** {result.get('error', 'Unknown error')}\n\n")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✓ Report saved to: {output_file}")


def generate_json_report(all_results: List[Dict], aggregated: Dict, output_file: Path):
    """Generate a JSON report for programmatic access."""
    report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_models": aggregated["total_models_tested"],
            "successful_models": aggregated["successful_models"],
            "failed_models": aggregated["failed_models"],
            "documents_count": len(SAMPLE_DOCUMENTS),
            "queries_count": len(TEST_QUERIES),
            "top_k": TOP_K
        },
        "results": all_results,
        "aggregated_metrics": aggregated
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ JSON report saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main test execution function."""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH MODEL TESTING SUITE")
    print("="*80 + "\n")
    
    # Setup directories
    experiments_dir = Path(__file__).parent
    temp_dir = experiments_dir / "temp_test_docs"
    results_dir = experiments_dir / "report"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test documents
    print("Creating test documents...")
    file_paths = create_test_documents(temp_dir)
    print(f"✓ Created {len(file_paths)} test documents\n")
    
    # Test all models
    all_results = []
    for model_name in EMBEDDING_MODELS.keys():
        result = test_model(
            model_name=model_name,
            file_paths=file_paths,
            queries=TEST_QUERIES,
            top_k=TOP_K
        )
        all_results.append(result)
    
    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80 + "\n")
    
    aggregated = aggregate_results(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_md = results_dir / f"test_report_{timestamp}.md"
    report_json = results_dir / f"test_report_{timestamp}.json"
    
    generate_report(all_results, aggregated, report_md)
    generate_json_report(all_results, aggregated, report_json)
    
    # Final summary
    print("\n" + "="*80)
    print("TEST EXECUTION COMPLETED")
    print("="*80)
    print(f"\n✓ Tested {aggregated['successful_models']} models successfully")
    print(f"✗ {aggregated['failed_models']} models failed")
    print(f"\nReports generated in: {results_dir}")
    print("\nTop Performing Model (by init time):")
    if aggregated["model_comparison"]:
        best = min(aggregated["model_comparison"], key=lambda x: x["initialization_time"])
        print(f"  {best['display_name']}: {best['initialization_time']}s")
    
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n✓ Temporary files cleaned up")


if __name__ == "__main__":
    main()

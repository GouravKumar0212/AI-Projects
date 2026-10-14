#!/usr/bin/env python3
"""
Interactive CLI Demo Runner for 'Ask My Docs' Production RAG.
Demonstrates:
1. Ingestion of domain documents into sliding window chunks
2. BM25 + Dense Vector hybrid retrieval with Reciprocal Rank Fusion (RRF)
3. Cross-Encoder reranking
4. Citation-enforced answer generation with safe fallback
"""
import os
import sys

# Configure UTF-8 stdout encoding for Windows compatibility
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag.pipeline import RAGPipeline

def main():
    print("=" * 70)
    print("[*] Ask My Docs - Domain-Specific RAG Application (CLI Demo)")
    print("=" * 70)

    pipeline = RAGPipeline()

    # 1. Ingestion
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "docs")
    sample_files = [
        os.path.join(docs_dir, "company_security_policy.md"),
        os.path.join(docs_dir, "cloud_architecture_spec.md")
    ]
    
    print("\n[Step 1] Ingesting domain documents...")
    chunk_count = pipeline.ingest_files(sample_files)
    print(f"-> Successfully chunked and indexed {chunk_count} chunks from {len(sample_files)} documents.\n")

    # 2. Test queries
    test_queries = [
        "What encryption standard is required for data at rest and how often are keys rotated?",
        "What is the uptime SLA guarantee for the microservice cluster?",
        "What is the company policy on remote work stipends in Europe?"
    ]

    for i, q in enumerate(test_queries, 1):
        print("-" * 70)
        print(f"Query #{i}: {q}")
        print("-" * 70)
        
        result = pipeline.query(q, top_k_hybrid=4, top_n_rerank=2)
        
        print(f"Generated Answer:\n  {result['answer']}")
        print(f"\nFallback Triggered: {result['fallback_triggered']}")
        
        if result['citations']:
            print("Citations:")
            for c in result['citations']:
                print(f"  - Doc: {c['doc_name']} | Chunk: {c['chunk_id']}")

        print("\nTop Retrieved & Reranked Evidence:")
        for s in result['sources']:
            print(f"  [{s['rerank_rank']}] Rerank Score: {s['rerank_score']} | Hybrid Rank: {s['hybrid_rank']} (BM25: {s['bm25_rank']}, Vec: {s['vector_rank']})")
            print(f"      Source: {s['doc_name']} ({s['chunk_id']})")
            print(f"      Text: {s['text'][:120]}...\n")

    print("=" * 70)
    print("Demo complete! All systems operational.")
    print("=" * 70)

if __name__ == "__main__":
    main()

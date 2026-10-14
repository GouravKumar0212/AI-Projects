import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from rag.chunker import DocumentChunker, Chunk
from rag.hybrid_search import SimpleBM25, HybridSearcher
from rag.reranker import CrossEncoderReranker
from rag.generator import CitationGenerator, FALLBACK_RESPONSE

def test_document_chunker():
    chunker = DocumentChunker(chunk_size_words=20, overlap_words=5)
    sample_text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunker.chunk_text(sample_text, "test_doc.md")
    
    assert len(chunks) >= 3
    assert chunks[0].doc_name == "test_doc.md"
    assert chunks[0].chunk_id == "test_doc.md_c000"
    assert "word0" in chunks[0].text
    assert chunks[0].citation_str() == "[Doc: test_doc.md, Chunk: test_doc.md_c000]"

def test_simple_bm25():
    corpus = [
        ["fido2", "hardware", "token", "authentication"],
        ["postgresql", "database", "replication", "standby"],
        ["kubernetes", "pod", "autoscaler", "cpu"]
    ]
    bm25 = SimpleBM25(corpus)
    scores = bm25.get_scores(["fido2", "authentication"])
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

def test_hybrid_searcher_rrf():
    chunk1 = Chunk(chunk_id="c1", doc_name="doc1", text="AES-256 encryption is mandatory for data at rest.", index=0)
    chunk2 = Chunk(chunk_id="c2", doc_name="doc2", text="Kubernetes autoscaling replicas threshold.", index=1)
    
    searcher = HybridSearcher(use_faiss=False)
    searcher.index([chunk1, chunk2])
    
    results = searcher.hybrid_search("AES-256 encryption", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk"].chunk_id == "c1"
    assert "rrf_score" in results[0]

def test_cross_encoder_reranker():
    chunk1 = Chunk(chunk_id="c1", doc_name="doc1", text="The weather is sunny today.", index=0)
    chunk2 = Chunk(chunk_id="c2", doc_name="doc2", text="TLS 1.3 with mTLS is enforced for all traffic.", index=1)
    
    candidates = [
        {"chunk": chunk1, "rrf_score": 0.05},
        {"chunk": chunk2, "rrf_score": 0.04}
    ]
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("mTLS network traffic", candidates, top_n=2)
    
    assert reranked[0]["chunk"].chunk_id == "c2"
    assert reranked[0]["rerank_rank"] == 1

def test_citation_generator_fallback():
    generator = CitationGenerator(min_relevance_threshold=0.5)
    # Query with no relevance
    low_score_item = [{
        "chunk": Chunk(chunk_id="c1", doc_name="doc1", text="Unrelated text about gardening.", index=0),
        "rerank_score": 0.01
    }]
    answer, citations, fallback = generator.generate("What is quantum computing?", low_score_item)
    assert fallback is True
    assert answer == FALLBACK_RESPONSE
    assert len(citations) == 0

def test_citation_generator_citation_enforcement():
    generator = CitationGenerator(min_relevance_threshold=0.01)
    relevant_item = [{
        "chunk": Chunk(
            chunk_id="policy_c001",
            doc_name="policy.md",
            text="Single-sign-on sessions expire automatically after 8 hours of inactivity.",
            index=0
        ),
        "rerank_score": 2.5
    }]
    answer, citations, fallback = generator.generate("When do sessions expire?", relevant_item)
    assert fallback is False
    assert "[Doc: policy.md, Chunk: policy_c001]" in answer
    assert len(citations) >= 1
    assert citations[0]["doc_name"] == "policy.md"

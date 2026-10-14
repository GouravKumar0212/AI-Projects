import os
from typing import List, Dict, Any, Optional

try:
    from rag.chunker import DocumentChunker, Chunk
    from rag.hybrid_search import HybridSearcher
    from rag.reranker import CrossEncoderReranker
    from rag.generator import CitationGenerator
except ImportError:
    from .chunker import DocumentChunker, Chunk
    from .hybrid_search import HybridSearcher
    from .reranker import CrossEncoderReranker
    from .generator import CitationGenerator

class RAGPipeline:
    """
    End-to-end Domain-Specific RAG Pipeline:
    1. Document Ingestion & Chunking with citation metadata
    2. Hybrid Search (BM25 + Dense Vector Index + RRF Fusion)
    3. Cross-Encoder Reranking
    4. Citation-enforced Generation with safe fallback
    """
    def __init__(
        self,
        chunk_size_words: int = 150,
        overlap_words: int = 30,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_api_key: Optional[str] = None,
        min_relevance_threshold: float = 0.05
    ):
        self.chunker = DocumentChunker(
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words
        )
        self.searcher = HybridSearcher(
            embedding_model_name=embedding_model_name
        )
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name
        )
        self.generator = CitationGenerator(
            api_key=llm_api_key,
            min_relevance_threshold=min_relevance_threshold
        )
        self.all_chunks: List[Chunk] = []

    def ingest_files(self, file_paths: List[str]) -> int:
        """Ingests a list of document file paths and indexes them."""
        new_chunks = []
        for path in file_paths:
            if os.path.exists(path):
                chunks = self.chunker.process_file(path)
                new_chunks.extend(chunks)

        self.all_chunks.extend(new_chunks)
        self.searcher.index(self.all_chunks)
        return len(new_chunks)

    def ingest_texts(self, documents: Dict[str, str]) -> int:
        """Ingests a dictionary of {doc_name: content} and indexes them."""
        new_chunks = self.chunker.process_documents(documents)
        self.all_chunks.extend(new_chunks)
        self.searcher.index(self.all_chunks)
        return len(new_chunks)

    def query(
        self,
        query: str,
        top_k_hybrid: int = 6,
        top_n_rerank: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering:
        Retrieval -> Rerank -> Generation -> Citation verification.
        """
        # 1. Hybrid retrieval (BM25 + Vector + RRF)
        hybrid_candidates = self.searcher.hybrid_search(
            query=query,
            top_k=top_k_hybrid
        )

        # 2. Cross-Encoder reranking
        reranked_results = self.reranker.rerank(
            query=query,
            retrieved_results=hybrid_candidates,
            top_n=top_n_rerank
        )

        # 3. Citation-enforced generation
        answer, citations, fallback_triggered = self.generator.generate(
            query=query,
            reranked_results=reranked_results
        )

        # 4. Format source chunks for transparency & inspectability
        sources = []
        for item in reranked_results:
            c = item["chunk"]
            sources.append({
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "text": c.text,
                "rerank_score": round(item.get("rerank_score", 0.0), 4),
                "rerank_rank": item.get("rerank_rank"),
                "hybrid_rank": item.get("original_hybrid_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "vector_rank": item.get("vector_rank"),
                "rrf_score": round(item.get("rrf_score", 0.0), 5),
            })

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "fallback_triggered": fallback_triggered,
            "sources": sources
        }

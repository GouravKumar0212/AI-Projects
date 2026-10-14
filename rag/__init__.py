"""Ask My Docs - RAG Core Package"""
try:
    from rag.chunker import DocumentChunker, Chunk
    from rag.hybrid_search import HybridSearcher
    from rag.reranker import CrossEncoderReranker
    from rag.generator import CitationGenerator
    from rag.pipeline import RAGPipeline
except ImportError:
    from .chunker import DocumentChunker, Chunk
    from .hybrid_search import HybridSearcher
    from .reranker import CrossEncoderReranker
    from .generator import CitationGenerator
    from .pipeline import RAGPipeline

__all__ = [
    "DocumentChunker",
    "Chunk",
    "HybridSearcher",
    "CrossEncoderReranker",
    "CitationGenerator",
    "RAGPipeline",
]

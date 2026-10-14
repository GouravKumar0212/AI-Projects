import re
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from rag.chunker import Chunk

class SimpleBM25:
    """
    Self-contained, fast BM25Okapi implementation.
    Works independently with zero external C++ dependencies, while
    compatible with standard BM25 conventions.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens = [len(doc) for doc in corpus]
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        df: Dict[str, int] = {}
        for doc in corpus:
            freq: Dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, doc_freq in enumerate(self.doc_freqs):
                if token in doc_freq:
                    tf = doc_freq[token]
                    doc_len = self.doc_lens[idx]
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[idx] += idf_val * (tf * (self.k1 + 1)) / max(1e-6, denom)
        return scores


class HybridSearcher:
    """
    Combines BM25 keyword search and Dense Vector Similarity Search
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
        rrf_k: int = 60
    ):
        self.embedding_model_name = embedding_model_name
        self.use_faiss = use_faiss
        self.rrf_k = rrf_k
        
        self.chunks: List[Chunk] = []
        self.bm25: Optional[SimpleBM25] = None
        self.vector_index = None
        self.embeddings: Optional[np.ndarray] = None
        self.embedder = None
        self._init_embedder()

    def _init_embedder(self):
        """Loads SentenceTransformer if available, else falls back to lightweight embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self.embedder = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.embedder is not None:
            embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        else:
            # Deterministic, lightweight fallback embedding (hashing vectorizer)
            # ensures zero crash if running without internet / gpu / heavy packages
            dim = 384
            vecs = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                tokens = self._tokenize(text)
                for t in tokens:
                    h = abs(hash(t)) % dim
                    vecs[i, h] += 1.0
                norm = np.linalg.norm(vecs[i])
                if norm > 0:
                    vecs[i] /= norm
            return vecs

    def index(self, chunks: List[Chunk]):
        """Indexes chunks for both BM25 and Vector Search."""
        if not chunks:
            self.chunks = []
            self.bm25 = None
            self.embeddings = None
            return

        self.chunks = chunks
        
        # 1. Build BM25 index
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = SimpleBM25(tokenized_corpus)

        # 2. Build Vector index
        texts = [c.text for c in chunks]
        self.embeddings = self._encode_texts(texts)

        # Use FAISS if installed and requested
        if self.use_faiss:
            try:
                import faiss
                dim = self.embeddings.shape[1]
                self.vector_index = faiss.IndexFlatIP(dim)
                self.vector_index.add(self.embeddings)
            except Exception:
                self.vector_index = None
        else:
            self.vector_index = None

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.bm25 or not self.chunks:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q_emb = self._encode_texts([query])
        
        if self.vector_index is not None:
            scores, indices = self.vector_index.search(q_emb, min(top_k, len(self.chunks)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
        else:
            # Cosine similarity via numpy dot product (since embeddings are normalized)
            scores = np.dot(self.embeddings, q_emb[0])
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 and Vector search, then merges rankings using Reciprocal Rank Fusion (RRF):
        RRF_Score(doc) = 1 / (rrf_k + rank_bm25) + 1 / (rrf_k + rank_vector)
        """
        if not self.chunks:
            return []

        bm25_res = self.search_bm25(query, top_k=candidate_pool)
        vec_res = self.search_vector(query, top_k=candidate_pool)

        rrf_scores: Dict[int, float] = {}
        bm25_ranks: Dict[int, int] = {}
        vec_ranks: Dict[int, int] = {}
        bm25_score_map: Dict[int, float] = {}
        vec_score_map: Dict[int, float] = {}

        for rank, (idx, score) in enumerate(bm25_res, start=1):
            bm25_ranks[idx] = rank
            bm25_score_map[idx] = score
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, (idx, score) in enumerate(vec_res, start=1):
            vec_ranks[idx] = rank
            vec_score_map[idx] = score
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "rrf_score": rrf_scores[idx],
                "bm25_rank": bm25_ranks.get(idx, None),
                "bm25_score": bm25_score_map.get(idx, 0.0),
                "vector_rank": vec_ranks.get(idx, None),
                "vector_score": vec_score_map.get(idx, 0.0),
            })
        return results

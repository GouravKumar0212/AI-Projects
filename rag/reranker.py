import re
from typing import List, Dict, Any, Optional

try:
    from rag.chunker import Chunk
except ImportError:
    from .chunker import Chunk

class CrossEncoderReranker:
    """
    Reranks candidate chunks retrieved from hybrid search using a Cross-Encoder
    model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    def _fallback_score(self, query: str, text: str) -> float:
        """
        Heuristic fallback score based on exact query term frequency,
        phrase matching, and coverage for offline/zero-dependency environments.
        """
        q_tokens = re.findall(r"\b\w+\b", query.lower())
        if not q_tokens:
            return 0.0
        t_lower = text.lower()
        score = 0.0
        
        # Exact phrase bonus
        if query.lower().strip() in t_lower:
            score += 2.0
            
        for tok in q_tokens:
            if tok in t_lower:
                score += 1.0 + (t_lower.count(tok) * 0.1)
                
        return score / (len(q_tokens) + 1.0)

    def rerank(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate results from hybrid search and reorders them using the cross-encoder.
        """
        if not retrieved_results:
            return []

        chunks: List[Chunk] = [res["chunk"] for res in retrieved_results]
        
        if self.model is not None:
            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self.model.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(s) for s in scores]
        else:
            scores = [self._fallback_score(query, chunk.text) for chunk in chunks]

        reranked = []
        for i, res in enumerate(retrieved_results):
            entry = dict(res)
            entry["rerank_score"] = float(scores[i])
            entry["original_hybrid_rank"] = i + 1
            reranked.append(entry)

        # Sort by reranker score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, entry in enumerate(reranked, start=1):
            entry["rerank_rank"] = rank

        return reranked[:top_n]

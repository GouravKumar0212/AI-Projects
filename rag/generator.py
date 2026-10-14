import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional

FALLBACK_RESPONSE = "I don't have sufficient evidence in the provided documents to answer this question."

CITATION_PATTERN = re.compile(r"\[Doc:\s*([^,\]]+),\s*Chunk:\s*([^\]]+)\]")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "how", "often", "many",
    "much", "tell", "explain", "describe", "company", "policy", "please"
}

class CitationGenerator:
    """
    Enforces citation in every generated answer by linking statements to
    [Doc: <doc_name>, Chunk: <chunk_id>], with safe fallback when context is insufficient.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        min_relevance_threshold: float = 0.1
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.min_relevance_threshold = min_relevance_threshold

    def build_prompt(self, query: str, chunks_with_scores: List[Dict[str, Any]]) -> str:
        """Formats context with clear document and chunk citation tags."""
        context_blocks = []
        for item in chunks_with_scores:
            c = item["chunk"]
            context_blocks.append(
                f"[Doc: {c.doc_name}, Chunk: {c.chunk_id}]\n{c.text}\n"
            )

        context_str = "\n---\n".join(context_blocks)

        system_prompt = (
            "You are a domain-specific Q&A assistant. Answer user questions strictly using ONLY "
            "the provided context blocks below.\n\n"
            "RULES:\n"
            "1. Answer using ONLY facts directly mentioned in the context.\n"
            "2. Every factual statement or claim MUST end with its exact citation link: [Doc: <doc_name>, Chunk: <chunk_id>].\n"
            "3. If the provided context does not contain sufficient evidence to answer the question, "
            f"you MUST respond EXACTLY with:\n\"{FALLBACK_RESPONSE}\"\n"
            "4. Do NOT speculate, generalize, or use external knowledge.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER (with citations):"
        )
        return system_prompt

    def _call_external_llm(self, prompt: str) -> Optional[str]:
        """Tries calling OpenAI or Gemini if configured."""
        if not self.api_key:
            return None

        # Try OpenAI client if available
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

        return None

    def _extractive_fallback_generate(
        self,
        query: str,
        chunks_with_scores: List[Dict[str, Any]]
    ) -> str:
        """
        High-precision local generator for offline / test / CI environments.
        Identifies the most relevant sentence from top chunks and attributes citations.
        """
        if not chunks_with_scores:
            return FALLBACK_RESPONSE

        top_chunk_item = chunks_with_scores[0]
        chunk = top_chunk_item["chunk"]
        score = top_chunk_item.get("rerank_score", 0.0)

        # If score is below confidence threshold, trigger safe fallback
        if score < self.min_relevance_threshold:
            return FALLBACK_RESPONSE

        q_words = {w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in STOP_WORDS}
        if not q_words:
            q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if s.strip()]
        
        best_sentence = None
        best_overlap = 0

        for sentence in sentences:
            s_words = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        # Require at least 2 key terms or 40% of query key terms to match to prevent hallucination
        min_required_overlap = max(1, min(2, math.ceil(len(q_words) * 0.4)))
        if best_overlap < min_required_overlap or not best_sentence:
            return FALLBACK_RESPONSE

        # Filter out trivial heading lines
        if best_sentence.startswith("#") and len(best_sentence.split()) < 6:
            return FALLBACK_RESPONSE

        citation = f"[Doc: {chunk.doc_name}, Chunk: {chunk.chunk_id}]"
        return f"{best_sentence} {citation}"

    def generate(
        self,
        query: str,
        reranked_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]], bool]:
        """
        Generates an answer with enforced citations.
        Returns: (answer_text, citations_list, fallback_triggered)
        """
        # If no chunks passed or top chunk score is too low, immediately fallback
        if not reranked_results or (reranked_results[0].get("rerank_score", 0.0) < self.min_relevance_threshold):
            return FALLBACK_RESPONSE, [], True

        prompt = self.build_prompt(query, reranked_results)
        
        answer = self._call_external_llm(prompt)
        if not answer:
            answer = self._extractive_fallback_generate(query, reranked_results)

        # Check if fallback was returned
        if FALLBACK_RESPONSE.lower() in answer.lower():
            return FALLBACK_RESPONSE, [], True

        # Extract all citations
        matches = CITATION_PATTERN.findall(answer)
        citations = [{"doc_name": m[0].strip(), "chunk_id": m[1].strip()} for m in matches]

        # Enforce citation: if model answered without any citations, append citation from top evidence
        if not citations and reranked_results:
            top_c = reranked_results[0]["chunk"]
            citation_tag = f"[Doc: {top_c.doc_name}, Chunk: {top_c.chunk_id}]"
            answer = f"{answer} {citation_tag}"
            citations.append({"doc_name": top_c.doc_name, "chunk_id": top_c.chunk_id})

        return answer, citations, False

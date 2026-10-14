import json
import os
import pytest
from rag.pipeline import RAGPipeline
from rag.generator import FALLBACK_RESPONSE

EVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_eval.json")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")

@pytest.fixture(scope="module")
def rag_pipeline():
    pipeline = RAGPipeline(chunk_size_words=100, overlap_words=20)
    doc_files = [
        os.path.join(DOCS_DIR, "company_security_policy.md"),
        os.path.join(DOCS_DIR, "cloud_architecture_spec.md")
    ]
    total_chunks = pipeline.ingest_files(doc_files)
    assert total_chunks > 0, "No chunks were ingested from sample docs!"
    return pipeline

@pytest.fixture(scope="module")
def golden_dataset():
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_evaluation_retrieval_recall_at_3(rag_pipeline, golden_dataset):
    """
    CI Gate 1: Evaluates that Recall@3 on answerable queries exceeds 80%.
    Top-3 retrieved source chunks must contain the expected document.
    """
    answerable = [item for item in golden_dataset if item["should_answer"]]
    hits = 0

    for item in answerable:
        query = item["query"]
        expected_doc = item["expected_doc"]
        
        result = rag_pipeline.query(query, top_k_hybrid=5, top_n_rerank=3)
        retrieved_docs = [s["doc_name"] for s in result["sources"][:3]]
        
        if expected_doc in retrieved_docs:
            hits += 1

    recall = hits / len(answerable)
    print(f"\n[CI Eval Gate 1] Retrieval Recall@3: {recall * 100:.1f}% ({hits}/{len(answerable)})")
    assert recall >= 0.80, f"Recall@3 ({recall:.2f}) fell below minimum required threshold 0.80"

def test_evaluation_safe_fallback_rate(rag_pipeline, golden_dataset):
    """
    CI Gate 2: Evaluates that out-of-domain / unanswerable queries
    safely trigger fallback without hallucinating.
    """
    unanswerable = [item for item in golden_dataset if not item["should_answer"]]
    fallbacks = 0

    for item in unanswerable:
        result = rag_pipeline.query(item["query"])
        if result["fallback_triggered"] or result["answer"] == FALLBACK_RESPONSE:
            fallbacks += 1

    fallback_rate = fallbacks / len(unanswerable)
    print(f"\n[CI Eval Gate 2] Safe Fallback Rate: {fallback_rate * 100:.1f}% ({fallbacks}/{len(unanswerable)})")
    assert fallback_rate == 1.0, f"Safe fallback rate ({fallback_rate:.2f}) must be 100% for out-of-domain queries"

def test_evaluation_citation_compliance(rag_pipeline, golden_dataset):
    """
    CI Gate 3: Evaluates that every answered domain query includes
    at least one verified citation formatted as [Doc: ..., Chunk: ...].
    """
    answerable = [item for item in golden_dataset if item["should_answer"]]
    compliant_count = 0

    for item in answerable:
        result = rag_pipeline.query(item["query"])
        if not result["fallback_triggered"]:
            citations = result.get("citations", [])
            answer_text = result.get("answer", "")
            
            # Must have citation objects and citation string in answer
            if len(citations) > 0 and "[Doc:" in answer_text and "Chunk:" in answer_text:
                compliant_count += 1

    compliance_rate = compliant_count / len(answerable)
    print(f"\n[CI Eval Gate 3] Citation Compliance Rate: {compliance_rate * 100:.1f}%")
    assert compliance_rate >= 0.80, f"Citation compliance ({compliance_rate:.2f}) must be >= 0.80"

def test_evaluation_key_phrase_presence(rag_pipeline, golden_dataset):
    """
    CI Gate 4: Evaluates that answers to domain queries contain
    critical expected technical terms from source documents.
    """
    answerable = [item for item in golden_dataset if item["should_answer"]]
    matched = 0

    for item in answerable:
        result = rag_pipeline.query(item["query"])
        answer_text = result["answer"].lower()
        key_phrases = item.get("key_phrases", [])
        
        # Check if at least one key phrase is captured in the answer or top retrieved source
        source_texts = " ".join([s["text"].lower() for s in result["sources"][:1]])
        phrase_found = any(kp.lower() in answer_text or kp.lower() in source_texts for kp in key_phrases)
        if phrase_found:
            matched += 1

    precision_rate = matched / len(answerable)
    print(f"\n[CI Eval Gate 4] Grounding Keyphrase Precision: {precision_rate * 100:.1f}%")
    assert precision_rate >= 0.80

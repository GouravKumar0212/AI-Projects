import os
import streamlit as st
from rag.pipeline import RAGPipeline
from rag.generator import FALLBACK_RESPONSE

st.set_page_config(
    page_title="Ask My Docs | Domain RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize pipeline in session state
if "pipeline" not in st.session_state:
    pipeline = RAGPipeline()
    # Auto-load sample docs
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    if os.path.exists(sample_dir):
        doc_paths = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.endswith((".md", ".txt"))
        ]
        if doc_paths:
            pipeline.ingest_files(doc_paths)
    st.session_state.pipeline = pipeline

pipeline: RAGPipeline = st.session_state.pipeline

# Custom styling for badges and cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin: 0.2rem;
        border: 1px solid #C7D2FE;
    }
    .score-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 Ask My Docs — Production RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Domain-specific Q&A with BM25 + Dense Vector Hybrid Search, Cross-Encoder Reranking & Enforced Citations.</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status & Config")
    total_chunks = len(pipeline.all_chunks)
    unique_docs = len(set(c.doc_name for c in pipeline.all_chunks))
    
    col1, col2 = st.columns(2)
    col1.metric("Docs", unique_docs)
    col2.metric("Chunks", total_chunks)

    st.subheader("📁 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Index Uploaded Files", use_container_width=True):
            docs_dict = {}
            for uf in uploaded_files:
                content = uf.read().decode("utf-8", errors="ignore")
                docs_dict[uf.name] = content
            new_count = pipeline.ingest_texts(docs_dict)
            st.success(f"Indexed {new_count} chunks from {len(uploaded_files)} files!")
            st.rerun()

    st.divider()
    st.subheader("🔍 Retrieval Parameters")
    top_k_hybrid = st.slider("Hybrid Search Candidates", min_value=2, max_value=15, value=6)
    top_n_rerank = st.slider("Cross-Encoder Rerank Pool", min_value=1, max_value=8, value=3)

# Main query interface
st.subheader("💬 Ask a Domain Question")

# Preset buttons for quick evaluation
st.markdown("**Quick Test Prompts:**")
col_p1, col_p2, col_p3 = st.columns(3)
preset_query = None

if col_p1.button("🔒 Data Encryption & CMEK", use_container_width=True):
    preset_query = "What encryption standard is required for data at rest and how often are keys rotated?"
if col_p2.button("⚡ Uptime SLA Guarantee", use_container_width=True):
    preset_query = "What is the uptime SLA guarantee for the microservice cluster?"
if col_p3.button("❓ Out-of-Domain (Test Fallback)", use_container_width=True):
    preset_query = "What is the company policy on remote work stipends in Europe?"

user_query = st.text_input(
    "Enter your question:",
    value=preset_query or "",
    placeholder="e.g., How quickly must suspected data breaches be reported to the SOC?"
)

if user_query:
    with st.spinner("Executing Hybrid Retrieval -> Cross-Encoder Reranking -> Citation Generation..."):
        result = pipeline.query(
            query=user_query,
            top_k_hybrid=top_k_hybrid,
            top_n_rerank=top_n_rerank
        )

    answer = result["answer"]
    citations = result["citations"]
    fallback = result["fallback_triggered"]
    sources = result["sources"]

    # Display Answer Card
    st.markdown("### 💡 Generated Answer")
    if fallback:
        st.warning(f"⚠️ **Safe Fallback Triggered:** {answer}")
        st.info("The system determined that the retrieved context did not contain sufficient grounding evidence to answer without hallucination.")
    else:
        st.success(answer)

    # Citations
    if citations:
        st.markdown("**Enforced Document Citations:**")
        citation_badges_html = " ".join([
            f'<span class="citation-badge">📄 {c["doc_name"]} (Chunk: {c["chunk_id"]})</span>'
            for c in citations
        ])
        st.markdown(citation_badges_html, unsafe_allow_html=True)

    st.markdown("---")
    
    # Inspection Tabs
    tab1, tab2 = st.tabs(["🔬 Retrieval & Reranker Inspector", "📄 Raw Context Chunks"])
    
    with tab1:
        st.markdown("#### Retrieval Stage Comparison")
        st.caption("Inspect how BM25, Dense Vector, and Cross-Encoder scores interacted for this query.")
        
        table_rows = []
        for s in sources:
            table_rows.append({
                "Rerank Rank": s["rerank_rank"],
                "Rerank Score": s["rerank_score"],
                "Document": s["doc_name"],
                "Chunk ID": s["chunk_id"],
                "Hybrid Rank": s["hybrid_rank"],
                "BM25 Rank": s["bm25_rank"] if s["bm25_rank"] else "-",
                "Vector Rank": s["vector_rank"] if s["vector_rank"] else "-",
                "RRF Fusion Score": s["rrf_score"],
            })
        st.dataframe(table_rows, use_container_width=True)

    with tab2:
        st.markdown("#### Retrieved Evidence Chunks")
        for i, s in enumerate(sources, 1):
            with st.expander(f"Chunk #{i}: {s['doc_name']} [{s['chunk_id']}] (Score: {s['rerank_score']})"):
                st.write(s["text"])
                st.caption(f"Hybrid RRF Score: {s['rrf_score']} | BM25 Rank: {s['bm25_rank']} | Vector Rank: {s['vector_rank']}")

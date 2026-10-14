import os
import sys
import shutil
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline

app = FastAPI(
    title="Ask My Docs - Production RAG API",
    description="Domain-specific Q&A system with hybrid retrieval (BM25 + Dense), Cross-Encoder reranking, and citation enforcement.",
    version="1.0.0"
)

# Initialize pipeline instance
pipeline = RAGPipeline()

# Ingest default sample docs on startup if available
DEFAULT_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
if os.path.exists(DEFAULT_DOCS_DIR):
    sample_files = [
        os.path.join(DEFAULT_DOCS_DIR, f)
        for f in os.listdir(DEFAULT_DOCS_DIR)
        if f.endswith((".md", ".txt"))
    ]
    if sample_files:
        pipeline.ingest_files(sample_files)


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or domain question", example="What encryption is required for data at rest?")
    top_k_hybrid: int = Field(default=6, description="Candidate chunks to retrieve via hybrid search")
    top_n_rerank: int = Field(default=3, description="Top chunks to keep after Cross-Encoder reranking")


class Citation(BaseModel):
    doc_name: str
    chunk_id: str


class SourceChunk(BaseModel):
    chunk_id: str
    doc_name: str
    text: str
    rerank_score: float
    rerank_rank: Optional[int] = None
    hybrid_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rrf_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    fallback_triggered: bool
    sources: List[SourceChunk]


class IngestTextRequest(BaseModel):
    documents: Dict[str, str] = Field(..., description="Mapping of doc_name to text content")


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    total_chunks_indexed: int
    unique_documents: int


@app.get("/health", response_model=HealthResponse)
def health():
    docs = set(c.doc_name for c in pipeline.all_chunks)
    return {
        "status": "healthy",
        "total_chunks_indexed": len(pipeline.all_chunks),
        "unique_documents": len(docs)
    }


@app.post("/query", response_model=QueryResponse)
def query_docs(req: QueryRequest):
    if not pipeline.all_chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please ingest documents before querying."
        )
    result = pipeline.query(
        query=req.query,
        top_k_hybrid=req.top_k_hybrid,
        top_n_rerank=req.top_n_rerank
    )
    return result


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    count = pipeline.ingest_texts(req.documents)
    return {
        "chunks_indexed": count,
        "total_chunks": len(pipeline.all_chunks),
        "message": f"Successfully indexed {count} new chunks."
    }


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: List[UploadFile] = File(...)):
    temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_paths = []
    try:
        for file in files:
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(path)

        count = pipeline.ingest_files(saved_paths)
        return {
            "chunks_indexed": count,
            "total_chunks": len(pipeline.all_chunks),
            "message": f"Successfully indexed {count} chunks from {len(files)} uploaded files."
        }
    finally:
        # Clean up temporary uploads
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

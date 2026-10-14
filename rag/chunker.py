import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation_str(self) -> str:
        return f"[Doc: {self.doc_name}, Chunk: {self.chunk_id}]"


class DocumentChunker:
    """
    Ingests text/markdown/pdf files and splits them into overlapping chunks
    with clean document and chunk citations.
    """
    def __init__(self, chunk_size_words: int = 150, overlap_words: int = 30):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def load_file(self, file_path: str) -> str:
        """Reads text from file path (.txt, .md, .pdf)."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md", ".json", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages_text)
            except ImportError:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def chunk_text(self, text: str, doc_name: str) -> List[Chunk]:
        """Splits text into sliding window word chunks."""
        # Clean text
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split(" ")
        
        if not words or words == [""]:
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size_words - self.overlap_words)
        idx = 0
        
        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + self.chunk_size_words, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words).strip()
            
            if chunk_text:
                chunk_id = f"{doc_name}_c{idx:03d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        text=chunk_text,
                        index=idx,
                        metadata={
                            "word_count": len(chunk_words),
                            "start_word": start_idx,
                            "end_word": end_idx,
                        }
                    )
                )
                idx += 1
                
            if end_idx >= len(words):
                break
                
        return chunks

    def process_file(self, file_path: str) -> List[Chunk]:
        doc_name = os.path.basename(file_path)
        content = self.load_file(file_path)
        return self.chunk_text(content, doc_name)

    def process_documents(self, documents: Dict[str, str]) -> List[Chunk]:
        """Processes dictionary of {doc_name: content}."""
        all_chunks: List[Chunk] = []
        for doc_name, text in documents.items():
            all_chunks.extend(self.chunk_text(text, doc_name))
        return all_chunks

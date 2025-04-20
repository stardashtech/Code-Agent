import os
from config import CHUNK_SIZE

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    """
    Splits large texts into chunks of CHUNK_SIZE.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

class DocAnalysisTool:
    """
    Analyzes documents; reads file content, chunks it, and generates a summary.
    OCR or PDF analysis would be added in a real implementation.
    """
    def analyze(self, doc_path: str) -> str:
        if not os.path.exists(doc_path):
            return f"Document not found: {doc_path}"
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)
            # Simple summary: number of chunks and info about the first chunk
            return f"Document split into {len(chunks)} chunks. First chunk:\n{chunks[0][:200]}"
        except Exception as e:
            return f"Document analysis error: {e}" 
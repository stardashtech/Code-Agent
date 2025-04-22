import os
from config import settings # Import settings object

# Default chunk_size is now defined in settings, function requires it explicitly
def chunk_text(text: str, chunk_size: int):
    """
    Splits large texts into chunks of the specified size.
    """
    # Basic chunking, consider more sophisticated methods (e.g., sentence splitting)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

class DocAnalysisTool:
    """
    Analyzes documents; reads file content, chunks it, and generates a summary.
    Uses chunk size from global settings.
    OCR or PDF analysis would be added in a real implementation.
    """
    def analyze(self, doc_path: str) -> str:
        if not os.path.exists(doc_path):
            return f"Document not found: {doc_path}"
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Use chunk_size from settings here
            chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE)
            if not chunks:
                return "Document is empty or could not be chunked."
            # Simple summary: number of chunks and info about the first chunk
            return f"Document split into {len(chunks)} chunks. First chunk:\n{chunks[0][:200]}..."
        except Exception as e:
            return f"Document analysis error: {e}" 
from tools.vector_search import VectorSearchTool
import logging

logger = logging.getLogger(__name__)

class RAGRetrievalTool:
    """
    RAG (Retrieval-Augmented Generation) tool that uses VectorSearchTool
    to retrieve relevant context for queries from the Qdrant database.
    """
    def __init__(self):
        try:
            # Initialize the VectorSearchTool (which handles Qdrant client setup/check)
            self.vector_search = VectorSearchTool()
            logger.info("RAGRetrievalTool initialized using VectorSearchTool.")
        except Exception as e:
            # Log the error from VectorSearchTool initialization if it failed
            logger.error("RAGRetrievalTool failed to initialize VectorSearchTool: %s", e)
            self.vector_search = None # Ensure it's None if init fails

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieves relevant chunks from the vector store.

        Args:
            query: The query string.
            k: The number of results to retrieve.

        Returns:
            A string containing the retrieved results or an error message.
        """
        if not self.vector_search:
            return "Error: RAGRetrievalTool is not available (failed to initialize VectorSearchTool). Check logs."
            
        try:
            # Use the search method from VectorSearchTool
            results = self.vector_search.search(query=query, k=k)
            if not results:
                return f"RAG retrieval for '{query}' returned no results."
                
            # Join results into a single string context
            context = "\n---\n".join(results)
            return f"Retrieved context:\n{context}"
            
        except Exception as e:
            logger.exception("Error during RAG retrieval using VectorSearchTool: %s", e)
            return f"Error: RAG retrieval failed: {str(e)}" 
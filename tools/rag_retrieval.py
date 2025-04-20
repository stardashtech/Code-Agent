from qdrant_client import QdrantClient
from models.embeddings import get_embedding
from config import QDRANT_HOST, QDRANT_COLLECTION, EMBEDDING_DIMENSION
import logging

logger = logging.getLogger(__name__)

class RAGRetrievalTool:
    """
    RAG (Retrieval-Augmented Generation) tool that uses Qdrant database 
    to retrieve relevant context for queries.
    """
    def __init__(self):
        try:
            self.client = QdrantClient(url=QDRANT_HOST)
            # Check if collection exists, if not provide error message
            self.client.get_collection(collection_name=QDRANT_COLLECTION)
        except Exception as e:
            logger.exception("RAGRetrieval: Failed to get Qdrant collection: %s", e)
            self.client = None

    def retrieve(self, query: str) -> str:
        if not self.client:
            return "Error: Could not connect to Qdrant collection."
        try:
            query_embedding = get_embedding(query).tolist()
            search_result = self.client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=3
            )
            results = [hit.payload.get("chunk", "") for hit in search_result]
            return f"RAG retrieval result: {'; '.join(results)}"
        except Exception as e:
            logger.exception("Error in RAGRetrieval query: %s", e)
            return f"Error: RAGRetrieval query error: {str(e)}" 
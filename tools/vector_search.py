import numpy as np
from qdrant_client import QdrantClient
from models.embeddings import get_embedding
from config import QDRANT_HOST, QDRANT_COLLECTION, EMBEDDING_DIMENSION, CHUNK_SIZE
from tools.doc_analysis import chunk_text

class VectorSearchTool:
    """
    Module that performs vector search using the Qdrant database.
    Large texts are chunked and added to the index for search.
    """
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_HOST)
        # Check if collection exists, create if not
        try:
            self.client.get_collection(collection_name=QDRANT_COLLECTION)
        except Exception:
            self.client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={"size": EMBEDDING_DIMENSION, "distance": "Cosine"}
            )

    def add_document(self, text: str):
        chunks = chunk_text(text)
        points = []
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk).tolist()
            points.append({
                "id": idx,
                "vector": embedding,
                "payload": {"chunk": chunk}
            })
        # Batch insertion
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

    def search(self, query: str, k: int = 3):
        q_embedding = get_embedding(query).tolist()
        search_result = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_embedding,
            limit=k
        )
        results = [hit.payload.get("chunk", "") for hit in search_result]
        return results 
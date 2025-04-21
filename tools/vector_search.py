import numpy as np
import logging
from qdrant_client import QdrantClient, models
from models.embeddings import get_embedding
from config import settings # Import settings object
from tools.doc_analysis import chunk_text

logger = logging.getLogger(__name__)

class VectorSearchTool:
    """
    Module that performs vector search using the Qdrant database.
    Large texts are chunked and added to the index for search.
    """
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST)
        # Check if collection exists, create if not
        try:
            collection_info = self.client.get_collection(collection_name=settings.QDRANT_COLLECTION)
            logger.info(f"Using existing Qdrant collection: {settings.QDRANT_COLLECTION}")
            
            # Check vector params
            vector_params = collection_info.vectors_config.params
            if vector_params.size != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"Collection {settings.QDRANT_COLLECTION} was created with vector size "
                    f"{vector_params.size}, but current config EMBEDDING_DIMENSION "
                    f"is {settings.EMBEDDING_DIMENSION}. Mismatched dimensions may cause errors. "
                    "Consider deleting and recreating the collection."
                )
        except Exception as e:
            logger.info(f"Attempting to create new Qdrant collection: {settings.QDRANT_COLLECTION}")
            try:
                self.client.recreate_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=models.VectorParams( # Use models.VectorParams
                         size=settings.EMBEDDING_DIMENSION, 
                         distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Successfully created collection '{settings.QDRANT_COLLECTION}' with vector size {settings.EMBEDDING_DIMENSION}")
            except Exception as create_e:
                logger.error(f"Failed to create Qdrant collection '{settings.QDRANT_COLLECTION}': {create_e}")
                # Depending on severity, might want to raise here or leave client partially unusable
                raise

    def add_document(self, text: str):
        chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE) # Use settings.CHUNK_SIZE
        points = []
        # Simple sequential IDs for this example
        # Consider using more robust IDs (e.g., hash of chunk, UUID) in a real application
        start_id = np.random.randint(1_000_000) 
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                points.append(models.PointStruct(
                    id=start_id + i,
                    vector=embedding.tolist(),
                    payload={"chunk": chunk}
                ))
            except Exception as embed_e:
                 logger.error(f"Failed to get embedding for chunk {i}: {embed_e}. Skipping chunk.")
                 continue # Skip chunks that fail embedding
                 
        if not points:
             logger.warning("No points generated from document to add to vector store.")
             return

        try:
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
                wait=True # Wait for operation to complete
            )
            logger.debug(f"Successfully added {len(points)} chunks to collection '{settings.QDRANT_COLLECTION}'")
        except Exception as e:
            logger.error(f"Failed to add chunks to collection '{settings.QDRANT_COLLECTION}': {str(e)}")
            # Depending on requirements, might want to raise or just log
            raise

    def search(self, query: str, k: int = 3):
        try:
            q_embedding = get_embedding(query).tolist()
            search_result = self.client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=q_embedding,
                limit=k
            )
            results = [hit.payload.get("chunk", "") for hit in search_result]
            logger.debug(f"Successfully searched '{settings.QDRANT_COLLECTION}', found {len(results)} results for k={k}")
            return results
        except Exception as e:
            logger.error(f"Failed to search collection '{settings.QDRANT_COLLECTION}': {str(e)}")
            # Depending on requirements, might want to raise or just return empty
            raise 
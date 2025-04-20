import numpy as np
import logging
from qdrant_client import QdrantClient
from models.embeddings import get_embedding
from config import QDRANT_HOST, QDRANT_COLLECTION, EMBEDDING_DIMENSION, CHUNK_SIZE
from tools.doc_analysis import chunk_text

logger = logging.getLogger(__name__)

class VectorSearchTool:
    """
    Module that performs vector search using the Qdrant database.
    Large texts are chunked and added to the index for search.
    """
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_HOST)
        # Check if collection exists, create if not
        try:
            collection = self.client.get_collection(collection_name=QDRANT_COLLECTION)
            logger.info(f"Using existing Qdrant collection: {QDRANT_COLLECTION}")
            
            # Check vector size of existing collection
            if collection.config.params.vectors.size != EMBEDDING_DIMENSION:
                logger.warning(
                    f"Collection {QDRANT_COLLECTION} was created with vector size "
                    f"{collection.config.params.vectors.size}, but current EMBEDDING_DIMENSION "
                    f"is {EMBEDDING_DIMENSION}. You must delete and recreate the collection "
                    "for the new vector size to take effect."
                )
        except Exception as e:
            logger.info(f"Creating new Qdrant collection: {QDRANT_COLLECTION}")
            try:
                self.client.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config={"size": EMBEDDING_DIMENSION, "distance": "Cosine"}
                )
                logger.info(f"Successfully created collection with vector size {EMBEDDING_DIMENSION}")
            except Exception as e:
                logger.error(f"Failed to create Qdrant collection: {str(e)}")
                raise

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
        try:
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            logger.debug(f"Successfully added {len(points)} chunks to collection")
        except Exception as e:
            logger.error(f"Failed to add chunks to collection: {str(e)}")
            raise

    def search(self, query: str, k: int = 3):
        try:
            q_embedding = get_embedding(query).tolist()
            search_result = self.client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q_embedding,
                limit=k
            )
            results = [hit.payload.get("chunk", "") for hit in search_result]
            logger.debug(f"Successfully searched collection, found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to search collection: {str(e)}")
            raise 
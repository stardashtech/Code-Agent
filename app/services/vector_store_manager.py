import logging
import uuid
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Callable, Awaitable, Optional
import asyncio # Yeniden deneme için eklendi

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models as rest
from qdrant_client.http import exceptions as qdrant_http_exceptions
# Assuming Config class might be needed or relevant parts passed.
# from app.services.code_agent import Config 

logger = logging.getLogger(__name__)

# Type hint for the async embedding function
AsyncEmbeddingFunc = Callable[[str], Awaitable[List[float]]]

# --- Yeniden Deneme Yapılandırması ---
MAX_RETRIES = 2
RETRY_DELAY = 1 # saniye

class VectorStoreManager:
    """Manages interactions with the Qdrant vector store."""

    def __init__(self, qdrant_host: str, embedding_dimension: int, embedding_func: AsyncEmbeddingFunc, collection_name: Optional[str] = None):
        """Initialize the VectorStoreManager.

        Args:
            qdrant_host: The complete URL for the Qdrant instance (including protocol, host and port).
                         Example: 'http://localhost:6333'
            embedding_dimension: The dimension size of the embeddings.
            embedding_func: An async function that takes text and returns embeddings.
            collection_name: Optional name for the Qdrant collection. Defaults to 'code_embeddings_{dimension}'.
        """
        try:
            self.client = QdrantClient(url=qdrant_host)
            self.client.get_collections() # Test connection early
            logger.info("Qdrant client initialized and connection tested successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Qdrant client at {qdrant_host}: {e}", exc_info=True)
            # Re-raise or handle as appropriate for application startup
            raise RuntimeError(f"Qdrant connection failed: {e}") from e

        self.embedding_dimension = embedding_dimension
        self.embedding_func = embedding_func  # Store the provided embedding function
        self.collection_name = collection_name or f"code_embeddings_{self.embedding_dimension}"
        self._ensure_collection_exists()
        logger.info(f"VectorStoreManager initialized for collection: {self.collection_name}")

    def _get_collection_name(self) -> str:
        """Get the collection name based on the embedding dimension."""
        # Keep consistent with the previous logic in CodeAgent
        return f"code_embeddings_{self.embedding_dimension}"

    def _ensure_collection_exists(self):
        """Ensure the vector store collection exists with correct configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {str(e)}", exc_info=True)
            raise # Re-raise critical error

    def _delete_collection(self):
        """Delete the existing collection if it exists."""
        try:
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {str(e)}", exc_info=True)
            # Don't necessarily raise, might be called during cleanup/reset

    async def search_code(self, query: str, limit: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant code snippets based on the query embedding, with optional filtering.

        Args:
            query: The query string.
            limit: Maximum number of results to return.
            filter_dict: Optional dictionary specifying metadata filters.
                         Example: {"type": "code_master"} to search only master code.
                         Example: {"language": "python", "type": "code_snippet"}

        Returns:
            A list of dictionaries, each representing a found code snippet.
        """
        if not query:
            return []

        # Retry logic wrapper
        for attempt in range(MAX_RETRIES + 1):
            try:
                query_vector = await self.embedding_func(query)
                if not query_vector:
                    logger.warning("Embedding function returned empty vector for query.")
                    return []

                search_params = models.SearchParams(hnsw_ef=128, exact=False)
                
                search_filter = None
                if filter_dict:
                    conditions = []
                    for key, value in filter_dict.items():
                         # Handle simple equality matching for now
                         conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                    if conditions:
                        search_filter = models.Filter(must=conditions)
                        
                logger.debug(f"Performing Qdrant search in '{self.collection_name}' with filter: {search_filter}")

                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    search_params=search_params,
                    limit=limit,
                    with_payload=True # Ensure payload is returned
                )

                results = [
                    {
                        "id": hit.id,
                        "score": hit.score,
                        **hit.payload # Unpack payload directly into result dict
                    }
                    for hit in search_result
                ]
                logger.info(f"Search completed. Found {len(results)} results matching filter: {filter_dict}")
                return results

            except (qdrant_http_exceptions.UnexpectedResponse, ConnectionError) as e: # Sadece UnexpectedResponse ve ConnectionError yakala, diğerleri Exception ile
                logger.warning(f"Qdrant search network/unexpected response (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error("Max retries reached for Qdrant search due to network/response issues.")
                    return []
            except Exception as e: # Diğer tüm Qdrant API veya beklenmedik hataları yakala
                logger.warning(f"Qdrant search failed (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                # Qdrant'a özgü bir hata mı kontrol et (emin olmak zor ama deneyelim)
                is_qdrant_api_error = hasattr(e, 'status_code') # Qdrant hatalarında genelde status_code olur
                if attempt < MAX_RETRIES and is_qdrant_api_error: # Sadece bilinen API hatalarını yeniden dene
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    if not is_qdrant_api_error:
                         logger.error(f"Unexpected non-API error during Qdrant search: {e}", exc_info=True)
                    else: # API hatası ama max deneme sayısına ulaşıldı
                         logger.error(f"Max retries reached for Qdrant search API error: {e}")
                    return [] # Hata durumunda boş liste dön
        return [] # Should not be reached

    async def index_code(self, code: str, metadata: Dict) -> Optional[str]:
        """Index a code snippet with its embedding. Returns the point ID if successful."""
        # --- ISSUE-003 Check ---
        if not code or code.isspace():
            logger.error("Attempted to index empty or whitespace code. Skipping.")
            return None
        # --- End Check ---
        try:
            code_embedding = await self.embedding_func(code)
            
            point_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=code_embedding,
                        payload={
                            "code": code, 
                            "file_path": metadata.get('file_path', ''),
                            "language": metadata.get('language', 'unknown'),
                            "timestamp": datetime.now().isoformat(),
                            "type": "code_snippet", # Add type field
                            **metadata # Include any other metadata passed
                        }
                    )
                ]
            )
            logger.info(f"Indexed code snippet (ID: {point_id}) for file: {metadata.get('file_path', 'N/A')}")
            return point_id
            
        except Exception as e:
            logger.error(f"Error indexing code in Qdrant: {str(e)}", exc_info=True)
            return None # Indicate failure

    async def store_code(self, file_path: str, content: str, language: str, metadata: Optional[Dict] = None) -> str:
        """Store or update code in the vector database. Returns the point ID if successful."""
        if not content or not file_path:
            raise ValueError("File path and content cannot be empty for storing code.")

        # Retry logic wrapper
        for attempt in range(MAX_RETRIES + 1):
            try:
                code_vector = await self.embedding_func(content)
                if not code_vector:
                    raise ValueError(f"Failed to generate embedding for {file_path}")

                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_path)) # Consistent ID based on path
                payload = {
                    "file_path": file_path,
                    "code": content,
                    "language": language,
                    "type": "code_master" # Add a type for easier filtering
                }
                if metadata:
                    payload.update(metadata) # Merge optional metadata

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=code_vector,
                            payload=payload
                        )
                    ],
                    wait=True # Wait for operation to complete
                )

                logger.info(f"Stored/Updated code from {file_path} in Qdrant with ID: {point_id}")
                return point_id

            except (qdrant_http_exceptions.UnexpectedResponse, ConnectionError) as e:
                logger.warning(f"Qdrant upsert network/unexpected response for {file_path} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Max retries reached for Qdrant upsert due to network/response issues: {file_path}")
                    raise ConnectionError(f"Failed to store code in Qdrant after {MAX_RETRIES} retries: {e}") from e
            except ValueError as e: # Catch embedding generation failure specifically
                 logger.error(f"Error storing code for {file_path}: {e}", exc_info=True)
                 raise # Re-raise value errors immediately
            except Exception as e:
                logger.warning(f"Qdrant upsert failed for {file_path} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                is_qdrant_api_error = hasattr(e, 'status_code')
                if attempt < MAX_RETRIES and is_qdrant_api_error:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    if not is_qdrant_api_error:
                         logger.error(f"Unexpected non-API error storing code for {file_path}: {e}", exc_info=True)
                         raise RuntimeError(f"Unexpected error storing code in Qdrant: {e}") from e
                    else:
                         logger.error(f"Max retries reached for Qdrant upsert API error: {file_path} - {e}")
                         raise ConnectionError(f"Failed to store code in Qdrant after {MAX_RETRIES} retries: {e}") from e
        raise RuntimeError("Upsert loop finished unexpectedly")

    async def save_interaction(self, interaction_data: Dict[str, Any]) -> Optional[str]:
        """Save interaction details (query, results, status) to the vector store."""
        # Construct text representation for embedding
        interaction_text = f"""
        Query: {interaction_data.get('query', 'N/A')}
        Status: {interaction_data.get('status', 'unknown')}
        Num Code Snippets: {len(interaction_data.get('code_snippets', []))}
        Fix Status: {interaction_data.get('fix_status', 'unknown')}
        Explanation: {interaction_data.get('explanation', '')}
        """
        try:
            embedding = await self.embedding_func(interaction_text)
            point_id = str(uuid.uuid4())
            
            payload = {
                "query": interaction_data.get('query'),
                "timestamp": datetime.now().isoformat(),
                "type": "interaction",
                "status": interaction_data.get('status'),
                "num_snippets": len(interaction_data.get('code_snippets', [])),
                "has_fix": interaction_data.get('has_fix'),
                "fix_status": interaction_data.get('fix_status'),
                # Optionally store more details, but keep payload manageable
                # "explanation": interaction_data.get('explanation'), 
            }
            # Filter out None values from payload before saving
            payload = {k: v for k, v in payload.items() if v is not None}

            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
            )
            logger.info(f"Saved interaction to vector store with ID: {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Failed to save interaction to Qdrant: {str(e)}", exc_info=True)
            return None # Indicate failure 

    async def delete_points_by_path(self, file_path: str) -> bool:
        """Delete vector points associated with a specific file path."""
        if not file_path:
             logger.warning("Attempted to delete points with empty file_path.")
             return False
             
        logger.info(f"Attempting to delete points for file path: {file_path}")

        # Retry logic wrapper
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Use scroll API to find all points matching the file_path
                # Note: Scrolling can be slow for large collections. Consider if filtering on ID is better if ID generation is stable.
                # Using filter directly in delete might be more efficient if supported and exact match is needed.
                
                # Let's try filtering by payload directly in the delete operation
                delete_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                )
                
                # Perform delete operation
                result = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=delete_filter),
                    wait=True
                )
                
                logger.info(f"Delete operation for {file_path} completed. Result status: {result.status}")
                # Qdrant delete operation might not explicitly return the count of deleted points easily.
                # Status 'completed' indicates the operation finished. We assume success if no error.
                return True

            except (qdrant_http_exceptions.UnexpectedResponse, ConnectionError) as e:
                logger.warning(f"Qdrant delete network/unexpected response for {file_path} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Max retries reached for Qdrant delete due to network/response issues: {file_path}")
                    return False
            except Exception as e:
                logger.warning(f"Qdrant delete failed for {file_path} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {type(e).__name__} - {e}")
                is_qdrant_api_error = hasattr(e, 'status_code')
                if attempt < MAX_RETRIES and is_qdrant_api_error:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    if not is_qdrant_api_error:
                        logger.error(f"Unexpected non-API error deleting points for {file_path}: {e}", exc_info=True)
                    else:
                        logger.error(f"Max retries reached for Qdrant delete API error: {file_path} - {e}")
                    return False
        return False # Should not be reached 
import logging
import uuid
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Callable, Awaitable, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
# Assuming Config class might be needed or relevant parts passed.
# from app.services.code_agent import Config 

logger = logging.getLogger(__name__)

# Type hint for the async embedding function
AsyncEmbeddingFunc = Callable[[str], Awaitable[List[float]]]

class VectorStoreManager:
    """Manages interactions with the Qdrant vector store."""

    def __init__(self, qdrant_host: str, embedding_dimension: int, embedding_func: AsyncEmbeddingFunc):
        """Initialize the VectorStoreManager.

        Args:
            qdrant_host: The URL for the Qdrant instance.
            embedding_dimension: The dimension size of the embeddings.
            embedding_func: An async function that takes text and returns embeddings.
        """
        self.qdrant_client = QdrantClient(url=qdrant_host)
        self.embedding_dimension = embedding_dimension
        self.embedding_func = embedding_func  # Store the provided embedding function
        self.collection_name = self._get_collection_name()
        self._ensure_collection_exists()
        logger.info(f"VectorStoreManager initialized for collection: {self.collection_name}")

    def _get_collection_name(self) -> str:
        """Get the collection name based on the embedding dimension."""
        # Keep consistent with the previous logic in CodeAgent
        return f"code_embeddings_{self.embedding_dimension}"

    def _ensure_collection_exists(self):
        """Ensure the vector store collection exists with correct configuration."""
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
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
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {str(e)}", exc_info=True)
            # Don't necessarily raise, might be called during cleanup/reset

    async def search_code(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant code snippets based on the query embedding."""
        try:
            # Use the passed embedding function
            query_embedding = await self.embedding_func(query)
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            processed_results = []
            for match in results:
                payload = match.payload
                if payload and payload.get('code') is not None:
                    processed_results.append({
                        'code': payload.get('code', ''),
                        'file_path': payload.get('file_path', ''),
                        'language': payload.get('language', 'unknown'),
                        'similarity': match.score,
                        'id': match.id, # Include ID for reference
                        'timestamp': payload.get('timestamp') # Include timestamp
                    })
                else:
                     logger.warning(f"Skipping Qdrant result with missing/empty code payload: ID {match.id}")

            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching code in Qdrant: {str(e)}", exc_info=True)
            return [] # Return empty list on error

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
            self.qdrant_client.upsert(
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

    async def store_code(self, file_path: str, content: str, language: str) -> Optional[str]:
        """Store or update code in the vector database. Returns the point ID if successful."""
        # --- ISSUE-003 Check ---
        if not content or content.isspace():
             logger.error(f"Attempted to store empty or whitespace content for file {file_path}. Skipping.")
             return None
        # --- End Check ---
        try:
            # Get embedding for the code content
            embedding = await self.embedding_func(content)

            # Generate a UUID for the point based on file path and content hash for potential deduplication
            hash_object = hashlib.sha256(f"{file_path}::{content}".encode())
            point_id = str(uuid.UUID(hash_object.hexdigest()[:32]))

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "file_path": file_path,
                            "code": content, 
                            "language": language,
                            "type": "code_master", # Indicate this is a primary source code entry
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                ]
            )

            logger.info(f"Stored/Updated code from {file_path} in Qdrant with ID: {point_id}")
            return point_id

        except Exception as e:
            logger.error(f"Error storing code for {file_path} in Qdrant: {str(e)}", exc_info=True)
            return None # Indicate failure

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

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
            )
            logger.info(f"Saved interaction to vector store with ID: {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Failed to save interaction to Qdrant: {str(e)}", exc_info=True)
            return None # Indicate failure 
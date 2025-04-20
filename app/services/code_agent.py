import asyncio
import json
import logging
import os
import time
import uuid  # UUID için import ekliyorum
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from qdrant_client.http.models import Filter as QdrantFilter
from qdrant_client.http.models import FieldCondition, MatchValue

from app.config import settings
from app.services.mock_openai import MockOpenAI

logger = logging.getLogger(__name__)

class CodeAgent:
    """Code Agent service that handles code search, analysis, and correction using LLMs."""
    
    def __init__(self):
        """Initialize the CodeAgent service with necessary clients and configuration."""
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.use_mock = False
        
        # Initialize OpenAI client
        try:
            api_key = settings.openai_api_key
            logger.info(f"OpenAI API Key is {'present' if api_key else 'missing'}")
            if api_key:
                logger.info(f"API Key starts with: {api_key[:5]}...")
                
            # OpenAI istemcisini sadece api_key ile başlatıyorum, ek parametreler olmadan
            self.client = OpenAI(api_key=api_key)
            logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}", exc_info=True)
            # Initialize with mock client instead
            logger.warning("Using mock OpenAI client for testing purposes")
            self.client = MockOpenAI()
            self.use_mock = True
            
        self.model = settings.openai_model
        logger.info(f"Using model: {self.model}")
        
        # Initialize Qdrant vector store
        try:
            logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or "",
            )
            self.collection_name = settings.qdrant_collection
            self._ensure_collection_exists()
            logger.info("Successfully initialized Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}", exc_info=True)
            self.qdrant_client = None
            self.collection_name = settings.qdrant_collection
        
        logger.info("CodeAgent initialized with model: %s and collection: %s", 
                   self.model, self.collection_name)

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists, create if it doesn't."""
        if self.qdrant_client is None:
            logger.warning("Qdrant client not initialized, cannot ensure collection exists")
            return
            
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info("Creating collection %s in Qdrant", self.collection_name)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info("Collection %s created successfully", self.collection_name)
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            # Continue even if collection creation fails

    async def run(self, query: str) -> Dict[str, Any]:
        """Run the agent on the given query asynchronously."""
        logger.info("Processing query: %s", query)
        
        # Delegate CPU-bound operations to the thread pool
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._run_sync, query
        )

    def _run_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation of the agent run logic."""
        try:
            # Handle case where we're using mock client
            if self.use_mock:
                logger.info("Using mock client for agent run")
                
            # 1. Analyze the query
            analysis = self._analyze_query(query)
            
            # Check if analysis has an error
            if "error_type" in analysis and analysis["error_type"] in ["initialization_error", "api_error"]:
                return {
                    "analysis": analysis,
                    "code_snippets": [],
                    "code_fix": {
                        "explanation": f"Cannot generate code fix: {analysis.get('description', 'Unknown error')}",
                        "fixed_code": "",
                        "file_path": "",
                        "changes": []
                    }
                }
            
            # 2. Search for relevant code
            code_snippets = self._search_code(query, analysis.get("search_terms", []))
            
            # 3. Generate fix with LLM
            if not code_snippets and not self.use_mock:
                logger.warning("No code snippets found for query: %s", query)
                return {
                    "analysis": analysis,
                    "code_snippets": [],
                    "code_fix": {
                        "explanation": "No relevant code found for the query",
                        "fixed_code": "",
                        "file_path": "",
                        "changes": []
                    }
                }
            
            # If using mock and no code snippets, create a dummy snippet for testing
            if self.use_mock and not code_snippets:
                logger.info("Creating mock code snippet for testing")
                code_snippets = [{
                    "file_path": "example.py",
                    "content": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
                    "language": "python",
                    "score": 0.95
                }]
            
            code_fix = self._generate_code_fix(query, code_snippets, analysis)
            
            # 4. Save the interaction
            try:
                self._save_interaction(query, code_snippets, code_fix)
            except Exception as e:
                logger.error(f"Error saving interaction: {str(e)}")
                # Continue even if saving fails
            
            return {
                "analysis": analysis,
                "code_snippets": code_snippets,
                "code_fix": code_fix
            }
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            # Return a structured response even in case of error
            return {
                "analysis": {
                    "error_type": "processing_error",
                    "description": f"Error processing query: {str(e)}"
                },
                "code_snippets": [],
                "code_fix": {
                    "explanation": f"Error during processing: {str(e)}",
                    "fixed_code": "",
                    "file_path": "",
                    "changes": []
                }
            }

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand the problem and extract search terms."""
        if self.client is None:
            logger.error("OpenAI client not initialized")
            return {
                "error_type": "initialization_error",
                "search_terms": [query],
                "file_types": [],
                "language": "unknown",
                "description": "Failed to initialize OpenAI client"
            }
            
        prompt = f"""
        Analyze the following code-related query and provide:
        1. The likely error type or problem category
        2. Key search terms that would help find relevant code
        3. Any specific file types or languages that might be involved
        
        Query: {query}
        
        Provide your analysis in JSON format with keys: 
        error_type, search_terms, file_types, language, description
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                logger.info("Query analysis: %s", analysis)
                return analysis
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return {
                    "error_type": "unknown",
                    "search_terms": [query],
                    "file_types": [],
                    "language": "unknown",
                    "description": "Failed to analyze query"
                }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {
                "error_type": "api_error",
                "search_terms": [query],
                "file_types": [],
                "language": "unknown",
                "description": f"API error: {str(e)}"
            }

    def _search_code(self, query: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Search for relevant code snippets in the vector database."""
        if self.qdrant_client is None:
            logger.warning("Qdrant client not initialized, returning empty search results")
            return []
            
        try:
            # 1. Get vector embedding for the query
            embedding = self._get_embedding(query)
            
            # 2. Construct filter based on search terms if available
            filters = None
            if search_terms:
                conditions = []
                for term in search_terms:
                    # Create a fuzzy match condition for file_path or content fields
                    # This is a simplified example - in real implementation,
                    # you'd use more sophisticated filtering logic
                    conditions.append(
                        FieldCondition(
                            key="content",
                            match=MatchValue(value=term)
                        )
                    )
                
                filters = QdrantFilter(
                    should=conditions  # Like a logical OR
                )
            
            # 3. Perform the search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=5,
                filter=filters
            )
            
            # 4. Format results
            code_snippets = []
            for result in search_results:
                snippet = result.payload
                snippet["score"] = result.score
                code_snippets.append(snippet)
            
            logger.info("Found %d code snippets for query", len(code_snippets))
            return code_snippets
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return []  # Return empty list on error

    def _get_embedding(self, text: str) -> List[float]:
        """Get the vector embedding for a text using OpenAI's embedding API."""
        if self.client is None:
            logger.error("OpenAI client not initialized")
            # Return a zero vector of appropriate size
            return [0.0] * 1536  # Standard size for OpenAI embeddings
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536

    def _generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], 
                          analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code fix using LLM based on the query and found code snippets."""
        if self.client is None:
            logger.error("OpenAI client not initialized")
            return {
                "explanation": "Unable to generate code fix due to initialization error",
                "fixed_code": "",
                "file_path": "",
                "changes": []
            }
            
        # Prepare context from code snippets
        code_context = "\n\n".join([
            f"File: {snippet.get('file_path', 'unknown')}\n```{snippet.get('language', '')}\n{snippet.get('content', '')}\n```"
            for snippet in code_snippets
        ])
        
        prompt = f"""
        You are a coding assistant who helps fix code issues. Based on the following query and code context,
        provide a solution that fixes the problem.
        
        ## Query
        {query}
        
        ## Error Analysis
        Error Type: {analysis.get('error_type', 'Unknown')}
        Description: {analysis.get('description', 'No description available')}
        
        ## Code Context
        {code_context}
        
        Provide your solution in the following JSON format:
        {{
            "explanation": "Clear explanation of the issue and how to fix it",
            "fixed_code": "The corrected code with proper formatting",
            "file_path": "The path to the file that needs to be fixed",
            "changes": [
                {{
                    "line_number": line number where change should be applied,
                    "original": "original line of code",
                    "replacement": "fixed line of code"
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code fixer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            try:
                code_fix = json.loads(response.choices[0].message.content)
                logger.info("Generated code fix for %s", code_fix.get("file_path", "unknown file"))
                return code_fix
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM code fix response as JSON")
                return {
                    "explanation": "Failed to generate structured code fix",
                    "raw_response": response.choices[0].message.content
                }
        except Exception as e:
            logger.error(f"Error generating code fix: {str(e)}")
            return {
                "explanation": f"Error generating code fix: {str(e)}",
                "fixed_code": "",
                "file_path": "",
                "changes": []
            }

    def _save_interaction(self, query: str, code_snippets: List[Dict[str, Any]], 
                         code_fix: Dict[str, Any]) -> None:
        """Save the interaction details to the vector store for future reference."""
        if self.qdrant_client is None:
            logger.warning("Qdrant client not initialized, skipping interaction saving")
            return
            
        # Create a combined text of the interaction for embedding
        interaction_text = f"""
        Query: {query}
        
        Code Snippets:
        {json.dumps(code_snippets, indent=2)}
        
        Solution:
        {json.dumps(code_fix, indent=2)}
        """
        
        # Get embedding for the interaction
        embedding = self._get_embedding(interaction_text)
        
        # Save to Qdrant
        try:
            # Use a timestamp string instead of event loop time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # UUID4 kullanarak geçerli bir ID oluşturuyorum hash yerine
            point_id = str(uuid.uuid4())
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,  # UUID tabanlı ID kullanıyorum
                        vector=embedding,
                        payload={
                            "query": query,
                            "code_snippets": code_snippets,
                            "code_fix": code_fix,
                            "timestamp": timestamp
                        }
                    )
                ]
            )
            logger.info(f"Saved interaction to vector store with ID: {point_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction to vector store: {str(e)}")
            # Continue even if saving fails

    async def store_code(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """Store code in the vector database for future searches."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._store_code_sync, file_path, content, language
        )

    def _store_code_sync(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """Synchronous implementation of code storage."""
        try:
            # Get embedding for the code content
            embedding = self._get_embedding(content)
            
            # Save to Qdrant with UUID
            point_id = str(uuid.uuid4())
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,  # UUID tabanlı ID kullanıyorum hash yerine
                        vector=embedding,
                        payload={
                            "file_path": file_path,
                            "content": content,
                            "language": language,
                            "type": "code_snippet"
                        }
                    )
                ]
            )
            
            logger.info(f"Stored code from {file_path} to Qdrant with ID: {point_id}")
            return {"status": "success", "message": f"Code from {file_path} stored successfully"}
            
        except Exception as e:
            logger.error("Error storing code: %s", str(e), exc_info=True)
            return {"error": f"Error storing code: {str(e)}"}

    async def process_mcp_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Model Context Protocol (MCP) request."""
        try:
            # Check if OpenAI client is initialized
            if self.client is None:
                return {
                    "result": "Error: OpenAI client not initialized",
                    "mcp_context": context
                }
                
            # For now, we'll just wrap the standard agent response in an MCP-friendly format
            result = await self.run(message)
            
            # Add MCP-specific context handling here
            mcp_context = context.copy() if context else {}
            mcp_context["last_query"] = message
            
            return {
                "result": result,
                "mcp_context": mcp_context
            }
        except Exception as e:
            logger.error(f"Error processing MCP request: {str(e)}", exc_info=True)
            return {
                "result": f"Error processing request: {str(e)}",
                "mcp_context": context
            } 
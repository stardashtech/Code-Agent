"""
Mock OpenAI client for testing when API key is not available
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class MockResponse:
    """Mock response object that mimics OpenAI response structure"""
    def __init__(self, choices):
        self.choices = choices

class MockChoices:
    """Mock choices object with message attribute"""
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    """Mock message with content attribute"""
    def __init__(self, content):
        self.content = content

class MockEmbeddingData:
    """Mock embedding data with embedding attribute"""
    def __init__(self, embedding):
        self.embedding = embedding

class MockEmbeddingResponse:
    """Mock embedding response with data attribute"""
    def __init__(self, embedding_data):
        self.data = [MockEmbeddingData(embedding_data)]

class MockOpenAI:
    """Mock OpenAI client for testing"""
    def __init__(self, api_key=None):
        self.api_key = api_key or "mock-api-key"
        logger.info("Initialized MockOpenAI client")
        
        # Set up API endpoints
        self.chat = MockChatCompletion()
        self.embeddings = MockEmbeddings()

class MockChatCompletion:
    """Mock chat completion API"""
    def __init__(self):
        # Add completions to match OpenAI client structure
        self.completions = self
        
    def create(self, model=None, messages=None, response_format=None):
        """Mock create method that returns predefined responses"""
        logger.info(f"MockChatCompletion.create called with model {model}")
        
        # Get the user's query from messages
        query = ""
        for msg in messages:
            if msg["role"] == "user":
                query = msg["content"]
                break
        
        if "analyze" in query.lower():
            # Analysis response
            content = json.dumps({
                "error_type": "syntax_error",
                "search_terms": ["function", "calculate_average", "sum", "len"],
                "file_types": [".py"],
                "language": "python",
                "description": "This is a mock analysis of code"
            })
        else:
            # Code fix response
            content = json.dumps({
                "explanation": "This is a mock code fix explanation",
                "fixed_code": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)",
                "file_path": "example.py",
                "changes": [
                    {
                        "line_number": 1,
                        "original": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
                        "replacement": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"
                    }
                ]
            })
            
        return MockResponse(
            [MockChoices(content)]
        )

class MockEmbeddings:
    """Mock embeddings API"""
    def create(self, input=None, model=None):
        """Mock create method that returns fixed embeddings"""
        logger.info(f"MockEmbeddings.create called with model {model}")
        
        # Return a fixed-size vector of 0.1s 
        embedding = [0.1] * 1536
        
        return MockEmbeddingResponse(embedding) 
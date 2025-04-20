#!/usr/bin/env python3
"""
Debug script to test the CodeAgent class directly
"""
import asyncio
import json
import logging
from app.services.code_agent import CodeAgent
from app.services.mock_openai import MockOpenAI
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

async def test_agent():
    """Test the CodeAgent class directly"""
    logger.info(f"OpenAI API Key: {settings.openai_api_key is not None}")
    if settings.openai_api_key:
        logger.info(f"Key starts with: {settings.openai_api_key[:5]}")
    
    # Initialize agent
    agent = CodeAgent()
    
    # Run a test query
    query = "Check the syntax of this Python function: def calculate_average(numbers): return sum(numbers) / len(numbers)"
    logger.info(f"Running test query: {query}")
    
    # Execute the query
    result = await agent.run(query)
    
    # Print result
    logger.info("Result received:")
    logger.info(f"Analysis: {json.dumps(result.get('analysis', {}), indent=2)}")
    logger.info(f"Code Fix: {json.dumps(result.get('code_fix', {}), indent=2)}")
    logger.info(f"Code Snippets: {len(result.get('code_snippets', []))}")
    
    return result

if __name__ == "__main__":
    logger.info("Starting debug test")
    result = asyncio.run(test_agent())
    logger.info("Debug test completed") 
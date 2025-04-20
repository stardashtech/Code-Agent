import asyncio
import logging
import os
from pprint import pformat

from app.config import settings
from app.services.code_agent import CodeAgent, Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Sample code content (replace with actual file reading if needed)
SAMPLE_CODE = {
    "calculator.py": """
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""",
    "main.py": """
from calculator import add, subtract, multiply, divide

def main():
    # Example calculations
    result1 = add(5, 3)      # Should be 8
    result2 = subtract(10, 4) # Should be 6
    result3 = multiply(2, 6)  # Should be 12
    result4 = divide(15, 3)   # Should be 5.0
    
    print(f"Addition: {result1}")
    print(f"Subtraction: {result2}")
    print(f"Multiplication: {result3}")
    print(f"Division: {result4}")

if __name__ == "__main__":
    main()
"""
}

async def main():
    # Configuration (ensure environment variables like OPENAI_API_KEY are set if needed)
    agent_config = Config(
        openai_api_key=settings.openai_api_key, 
        openai_model=settings.openai_model,
        embedding_dimension=settings.embedding_dimension,
        qdrant_host=settings.qdrant_url,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        cache_ttl=settings.cache_ttl
    )
    agent = CodeAgent(agent_config)
    logger.info("Initialized CodeAgent")

    # 1. Store sample code
    logger.info("\n1. Storing sample code files...")
    try:
        # Clear existing data first for a clean demo
        await agent.store_code("placeholder", "", "python", clear_existing=True)
        for file_path, content in SAMPLE_CODE.items():
            result = await agent.store_code(file_path, content, "python", clear_existing=False)
            logger.info(f"Stored {file_path} in collection {agent.collection_name}: {result}")
    except Exception as e:
         logger.error(f"Error storing code: {e}", exc_info=True)
         return # Stop if storing fails

    # 2. Test different queries
    logger.info("\n2. Testing different queries...")
    queries = [
        "How does the calculator code work?",
        "How can I add a new function to calculate power (exponentiation) in calculator.py?",
        "Is there any potential bug in the divide function?",
        "How can I improve error handling in the calculator functions?"
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        try:
            # Use a basic conversation history for demonstration
            history = [
                {"role": "user", "content": "Tell me about my calculator project."}, 
                {"role": "assistant", "content": "Okay, I see files calculator.py and main.py. What specifically would you like to know or do?"}
            ]
            # Call the agent's run method with history
            result = await agent.run(query, conversation_history=history)

            # Print results based on the new structure
            logger.info(f"\nQuery Assessment: {result.get('clarification_needed', 'N/A')}")
            logger.info(f"Extracted Keywords: {result.get('extracted_keywords', [])}")
            logger.info(f"Decomposed Queries: {result.get('decomposed_queries', [])}")
            logger.info(f"Initial Plan: {result.get('initial_plan', [])}")

            logger.info("\nRelevant Code Snippets:")
            snippets = result.get('code_snippets', [])
            if snippets:
                for snippet in snippets:
                    logger.info(f"\nFile: {snippet.get('file_path', 'N/A')}")
                    logger.info(f"Score: {snippet.get('similarity', 0.0):.2f}")
                    logger.info("Content:")
                    logger.info(snippet.get('code', '[No code content]'))
            else:
                logger.info("No snippets found.")

            logger.info("\nResult Summary:")
            summary = result.get('result_summary', {})
            logger.info(f"Analysis Summary:\n{summary.get('analysis_summary', 'N/A')}")
            fix_details = summary.get('fix_details', {})
            logger.info(f"Fix Details:\n{pformat(fix_details)}") # Use pformat for readable dict output

            logger.info("\nMetadata:")
            metadata = result.get('metadata', {})
            logger.info(pformat(metadata))

            logger.info("\n================================================================================")

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            logger.info("\n================================================================================")


if __name__ == "__main__":
    asyncio.run(main()) 
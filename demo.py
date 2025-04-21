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
    try:
        agent = CodeAgent(agent_config)
        logger.info("Initialized CodeAgent")
        # Check if sandbox runner is available after init
        if not agent.sandbox_runner:
            logger.warning("DockerSandboxRunner failed to initialize (Docker likely not running or accessible). Code validation step will be skipped.")
    except Exception as e:
        logger.critical(f"Failed to initialize CodeAgent: {e}", exc_info=True)
        return # Cannot proceed without agent

    # 1. Store sample code
    logger.info("\n1. Storing sample code files...")
    # Ensure collection is ready (optional: clear if needed for demo reproducibility)
    try:
        # await agent.vector_store_manager._delete_collection() # Uncomment to clear before run
        agent.vector_store_manager._ensure_collection_exists()
    except Exception as e:
        logger.error(f"Failed to prepare vector store collection: {e}", exc_info=True)
        # Decide if we should stop or continue without storing
        # return # Example: Stop if vector store is critical

    try:
        # Store the actual code files
        for file_path, content in SAMPLE_CODE.items():
            result = await agent.store_code(file_path, content, "python") # Removed clear_existing flag
            # Access collection name via vector_store_manager
            collection_name = agent.vector_store_manager.collection_name
            logger.info(f"Store result for {file_path} in {collection_name}: {result}")
    except Exception as e:
         logger.error(f"Error storing code: {e}", exc_info=True)
         # Decide if we should stop or continue

    # 2. Test different queries
    logger.info("\n2. Testing different queries...")
    queries = [
        "How does the calculator code work?",
        "How can I add a new function to calculate power (exponentiation) in calculator.py?",
        "Is there any potential bug in the divide function?",
        "How can I improve error handling in the calculator functions?",
        "Validate the code in calculator.py" # Add a query likely to trigger validation
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
            logger.info(f"\nStatus: {result.get('status', 'N/A')} | Message: {result.get('message', 'N/A')}")
            logger.info(f"Clarification Needed: {result.get('clarification_needed', 'N/A')}")

            # --- Access results via the 'results' key --- #
            agent_results = result.get('results', {})

            logger.info(f"Extracted Keywords: {agent_results.get('extracted_keywords', [])}")
            logger.info(f"Decomposed Queries: {agent_results.get('decomposed_queries', [])}")
            logger.info(f"Initial Plan: {result.get('initial_plan', [])}") # Plan is top-level

            logger.info("\nRelevant Code Snippets:")
            snippets = agent_results.get('code_snippets', [])
            if snippets:
                for snippet in snippets:
                    logger.info(f"\nFile: {snippet.get('file_path', 'N/A')}")
                    logger.info(f"Score: {snippet.get('similarity', 0.0):.2f}")
                    logger.info("Content:")
                    logger.info(snippet.get('code', '[No code content]'))
            else:
                logger.info("No snippets found.")

            # Print search results if they exist
            for search_key in ["web_search_results", "github_search_results", "stackoverflow_search_results"]:
                search_results = agent_results.get(search_key)
                if search_results is not None:
                    logger.info(f"\n{search_key.replace('_', ' ').title()}:")
                    logger.info(pformat(search_results))

            logger.info("\nResult Summary:")
            logger.info(f"Analysis Summary:\n{agent_results.get('analysis_summary', 'N/A')}")
            fix_details = agent_results.get('fix_details', {})
            logger.info(f"Fix Details:\n{pformat(fix_details)}")

            # Print validation status if it exists
            validation_status = agent_results.get('validation_status')
            if validation_status:
                logger.info(f"\nValidation Status:\n{pformat(validation_status)}")
            
            # Print apply fix status if it exists
            apply_fix_status = agent_results.get('apply_fix_status')
            if apply_fix_status:
                logger.info(f"\nApply Fix Status:\n{pformat(apply_fix_status)}")

            logger.info("\nMetadata:")
            metadata = result.get('metadata', {})
            logger.info(pformat(metadata))

            logger.info("\n================================================================================")

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            logger.info("\n================================================================================")


if __name__ == "__main__":
    asyncio.run(main()) 
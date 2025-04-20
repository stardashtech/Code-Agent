import asyncio
from app.services.code_agent import CodeAgent, Config
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Initialize the Config and CodeAgent
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model="gpt-4",
        embedding_dimension=1536,
        qdrant_host="localhost",
        qdrant_port=6333
    )
    agent = CodeAgent(config)
    logger.info("Initialized CodeAgent")
    
    # 2. Store some sample code
    sample_code = {
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
    
    try:
        # Store the sample code files
        logger.info("\n1. Storing sample code files...")
        first_file = True
        for file_path, content in sample_code.items():
            # Clear existing collection only for the first file
            clear_existing = first_file
            result = await agent.store_code(file_path, content, "python", clear_existing=clear_existing)
            logger.info(f"Stored {file_path} in collection {agent.collection_name}: {result}")
            first_file = False
    except Exception as e:
        logger.error(f"Error storing code: {str(e)}")
        return
    
    # 3. Try different types of queries
    queries = [
        "How does the calculator code work?",
        "How can I add a new function to calculate power (exponentiation) in calculator.py?",
        "Is there any potential bug in the divide function?",
        "How can I improve error handling in the calculator functions?"
    ]
    
    logger.info("\n2. Testing different queries...")
    for query in queries:
        try:
            logger.info(f"\nQuery: {query}")
            result = await agent.run(query)
            
            # Print analysis
            logger.info("\nAnalysis:")
            logger.info(result['analysis'])
            
            # Print found code snippets
            logger.info("\nRelevant Code Snippets:")
            for snippet in result['code_snippets']:
                logger.info(f"\nFile: {snippet['file_path']}")
                logger.info(f"Score: {snippet.get('similarity', 0):.2f}")
                logger.info("Content:")
                logger.info(snippet['code'])
            
            # Print suggested fix
            logger.info("\nSuggested Fix:")
            if result['code_fix']['status'] == 'success':
                logger.info(result['code_fix']['explanation'])
            else:
                logger.info(f"Error: {result['code_fix'].get('message', 'Unknown error')}")
            
            logger.info("\n" + "="*80)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main()) 
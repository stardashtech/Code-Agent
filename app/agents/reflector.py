import logging
from typing import TYPE_CHECKING, Tuple, Optional, Dict
import re

if TYPE_CHECKING:
    from app.services.code_agent import LLMProvider

logger = logging.getLogger(__name__)

class Reflector:
    """Agent responsible for query reflection and clarification."""

    def __init__(self, provider: 'LLMProvider'):
        """Initialize the Reflector agent.

        Args:
            provider: An instance of an LLMProvider for making API calls.
        """
        self.provider = provider

    async def assess_query_clarity(self, query: str, conversation_history: Optional[list[Dict[str, str]]] = None) -> str:
        """Assess the clarity of a user query using an LLM, considering history.

        Args:
            query: The user query to assess.
            conversation_history: Optional list of previous messages.

        Returns:
            A string indicating 'CLEAR' if the query is clear, or a
            clarifying question if the query is deemed ambiguous.
        """
        system_prompt = (
            "You are an expert assistant analyzing user queries related to code. "
            "Your goal is to determine if a query is ambiguous or lacks sufficient detail "
            "for a code generation or search task. Consider the conversation history if provided. "
            "If it is ambiguous, formulate a single, specific question to ask the user for clarification. "
            "If the query is clear enough (potentially clarified by history), respond ONLY with the word 'CLEAR'."
        )
        history_context = "\n\nConversation History (for context):\n" + "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in conversation_history[-5:]]) if conversation_history else ""

        user_prompt = (
            f"Analyze the following query: '{query}'.{history_context}\n\n"
            "Is this query clear and specific enough (considering the history) to proceed? "
            "If not, provide the single clarifying question to ask the user. "
            "If it is clear and specific, respond ONLY with the word 'CLEAR'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.provider.generate_completion(messages, temperature=0.2)
            # Simple validation: Check if response is empty or just whitespace
            if not response or response.isspace():
                 logger.warning("LLM returned empty response for query clarity assessment.")
                 return "CLEAR" # Default to clear if assessment fails or is empty
            return response.strip()
        except Exception as e:
            logger.error(f"Error during query clarity assessment: {e}", exc_info=True)
            # In case of error, assume the query is clear to avoid blocking the flow
            return "CLEAR"

    async def extract_keywords(self, query: str, conversation_history: Optional[list[Dict[str, str]]] = None) -> list[str]:
        """Extract key terms or concepts from a user query, considering history.

        Args:
            query: The user query.
            conversation_history: Optional list of previous messages.

        Returns:
            A list of extracted keywords or concepts. Returns empty list on error.
        """
        system_prompt = (
            "You are an expert system analyzing user queries about programming and code. "
            "Your task is to identify and extract the main keywords or technical concepts from the current query, using conversation history for context if provided. "
            "Focus on nouns, technical terms, library names, function names, programming concepts, etc. present in the *current query*. "
            "Return ONLY a comma-separated list of these keywords/concepts. "
            "If no specific keywords are found in the current query, return an empty string."
        )
        history_context = "\n\nConversation History (for context):\n" + "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in conversation_history[-5:]]) if conversation_history else ""

        user_prompt = (
            f"Extract the main keywords/concepts from the *current query* below, using the history for context if needed: '{query}'{history_context}\n\n"
            "Return ONLY a comma-separated list (e.g., keyword1, concept2, library_name) based on the current query."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.provider.generate_completion(messages, temperature=0.1)
            if not response or response.isspace():
                return []
            # Split the comma-separated string and strip whitespace
            keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
            logger.info(f"Extracted keywords for query '{query}': {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Error during keyword extraction: {e}", exc_info=True)
            return []

    async def decompose_query(self, query: str, conversation_history: Optional[list[Dict[str, str]]] = None) -> list[str]:
        """Decompose a potentially complex query into smaller, actionable sub-queries.

        Args:
            query: The user query.
            conversation_history: Optional list of previous messages for context.

        Returns:
            A list of sub-queries. If the query is simple, returns a list
            containing only the original query. Returns [query] on error.
        """
        # Using triple quotes for safer multiline strings
        system_prompt = """
            You are an expert system analyzing user queries about programming and code. 
            Your task is to determine if the given query consists of multiple distinct steps or questions. 
            If the query is simple and represents a single task, respond ONLY with the original query. 
            If the query contains multiple distinct parts, break it down into a numbered list of smaller, 
            self-contained sub-queries. Each sub-query should be actionable on its own.
            Example Input: 'How do I set up a Flask route and connect it to a database using SQLAlchemy?'
            Example Output:
            1. How to set up a Flask route?
            2. How to connect Flask to a database using SQLAlchemy?
            """
        history_context = "\n\nConversation History (for context):\n" + "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in conversation_history[-5:]]) if conversation_history else ""

        user_prompt = f"""
            Analyze the following query: '{query}'{history_context}
            
            Is this query composed of multiple distinct steps or questions? 
            If yes, provide a numbered list of the decomposed sub-queries. 
            If no (it's a single task), respond ONLY with the original query text.
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.provider.generate_completion(messages, temperature=0.1)

            # Check if the response contains a numbered list (heuristic for decomposition)
            if re.match(r"^\s*1\.\s", response):
                # Extract numbered items - revised regex
                sub_queries = [item.strip() for item in re.findall(r"^\s*\d+\.\s*(.*?)\s*$", response, re.MULTILINE)]
                if sub_queries:
                    logger.info(f"Decomposed query '{query}' into: {sub_queries}")
                    return sub_queries

            # If not decomposed or extraction failed, return the original query in a list
            logger.info(f"Query '{query}' was not decomposed.")
            return [query.strip()] # Return original query if not decomposed

        except Exception as e:
            logger.error(f"Error during query decomposition: {e}", exc_info=True)
            return [query.strip()] # Return original query on error 
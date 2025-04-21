from models.llm import call_llm
from agent.logger import get_logger
from config import settings

logger = get_logger(__name__)

class SubgoalManager:
    """
    Breaks down the user query into subgoals using LLM.
    Large texts are chunked.
    """

    def analyze_and_create_subgoals(self, user_query: str):
        prompt = (
            f"{settings.PROMPT_SAFETY_DIRECTIVE}\n"
            "Break down the user's query into logical subgoals. "
            "List each subgoal on a separate line.\n"
            f"User query:\n{user_query}"
        )
        try:
            subgoal_text = call_llm(prompt)
            # Extract subgoals line by line, chunking can be applied to necessary areas using the chunk function
            subgoals = [line.strip("- ").strip() for line in subgoal_text.split("\n") if line.strip()]
            return subgoals
        except Exception as e:
            logger.error("Error creating subgoal: %s", e)
            return [] 
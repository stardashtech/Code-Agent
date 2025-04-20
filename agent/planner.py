from models.llm import call_llm
from agent.logger import get_logger
from config import PROMPT_SAFETY_DIRECTIVE

logger = get_logger(__name__)

class Planner:
    """
    Module that creates the initial plan and performs replanning when needed.
    Security directive is added to the prompt.
    """

    def create_initial_plan(self, user_query: str) -> str:
        prompt = (
            f"{PROMPT_SAFETY_DIRECTIVE}\n"
            "Based on the user's query, specify the general steps. "
            "Please provide clear and actionable steps."
            f"\nUser query: {user_query}"
        )
        try:
            plan = call_llm(prompt)
            return plan
        except Exception as e:
            logger.error("Error creating plan: %s", e)
            return "Plan creation failed."

    def replan_with_context(self, user_query: str, reflection_text: str) -> str:
        prompt = (
            f"{PROMPT_SAFETY_DIRECTIVE}\n"
            "Recreate the plan considering the reflection result. "
            f"\nReflection: {reflection_text}\n"
            f"User query: {user_query}"
        )
        try:
            revised_plan = call_llm(prompt)
            return revised_plan
        except Exception as e:
            logger.error("Error during replan: %s", e)
            return "Replan failed." 
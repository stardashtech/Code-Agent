import re
from models.llm import call_llm
from agent.logger import get_logger
from config import settings

logger = get_logger(__name__)

def _validate_plan_structure(plan_text: str) -> bool:
    """Basic validation to check if the plan seems like a numbered list."""
    lines = [line.strip() for line in plan_text.strip().split('\n') if line.strip()]
    if not lines:
        return False # Empty plan is invalid
    # Check if most lines start with a number/bullet
    numbered_lines = sum(1 for line in lines if re.match(r"^\d+\. |^\* |^- ", line))
    # Allow some flexibility, e.g., >50% lines should be numbered/bulleted
    return numbered_lines >= len(lines) * 0.5

class Planner:
    """
    Module that creates the initial plan and performs replanning when needed.
    Includes basic structural validation for LLM-generated plans.
    Uses global settings for safety directive.
    """

    def create_initial_plan(self, user_query: str) -> str:
        prompt = (
            f"{settings.PROMPT_SAFETY_DIRECTIVE}\n"
            "Based on the user's query, create a concise, step-by-step plan to address it. "
            "Consider the likely intent (e.g., debugging, code generation, explanation, search)."
            "Format the plan as a numbered list (e.g., '1. Step one.\n2. Step two.')."
            f"\nUser query: {user_query}"
        )
        try:
            plan = call_llm(prompt)
            if not _validate_plan_structure(plan):
                logger.warning(f"Initial plan structure validation failed. Raw output: {plan}")
                # Option: Could retry call_llm here, or just return the raw plan
            return plan
        except Exception as e:
            logger.error("Error creating initial plan: %s", e)
            return "Plan creation failed."

    def replan_with_context(self, user_query: str, reflection_text: str, original_subgoals: list[str]) -> str:
        subgoals_str = "\n".join([f"- {sg}" for sg in original_subgoals])
        prompt = (
            f"{settings.PROMPT_SAFETY_DIRECTIVE}\n"
            "Recreate the plan considering the reflection result and the original subgoals. "
            "Format the new plan as a numbered list (e.g., '1. Step one.\n2. Step two.')."
            f"\n--- Context ---"
            f"\nOriginal User Query: {user_query}"
            f"\nOriginal Subgoals:\n{subgoals_str}"
            f"\nReflection Result: {reflection_text}"
            f"\n--- New Plan Request ---"
            "Generate the revised numbered list plan below:"
        )
        try:
            revised_plan = call_llm(prompt)
            if not _validate_plan_structure(revised_plan):
                 logger.warning(f"Revised plan structure validation failed. Raw output: {revised_plan}")
            return revised_plan
        except Exception as e:
            logger.error("Error during replan: %s", e)
            return "Replan failed."
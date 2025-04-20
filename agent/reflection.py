import json
from models.llm import call_llm
from config import PROMPT_SAFETY_DIRECTIVE, ENABLE_CHAIN_OF_THOUGHT_LOGGING
from agent.logger import get_logger

logger = get_logger(__name__)

class Reflection:
    """
    Advanced reflection mechanism: The agent evaluates the result of each subgoal,
    collects chain-of-thought details, and triggers replanning if necessary.
    """

    def reflect(self, user_query: str, subgoal: str, step_result, short_term_mem) -> dict:
        # Hidden chain-of-thought logging
        if ENABLE_CHAIN_OF_THOUGHT_LOGGING:
            debug_text = f"Reflection for subgoal: {subgoal}, result: {step_result}"
            short_term_mem.update_chain_of_thought(debug_text)

        prompt = (
            f"{PROMPT_SAFETY_DIRECTIVE}\n"
            f"User query: {user_query}\n"
            f"Subgoal: {subgoal}\n"
            f"Subgoal output: {step_result}\n"
            "Please evaluate this output and self-criticize. "
            "If improvement is needed for this step, set the 'need_replanning' value to true. "
            "Provide the answer in JSON format: {\"need_replanning\": <bool>, \"reflection_text\": \"...\"}."
        )

        try:
            reflection_output = call_llm(prompt)
            try:
                reflection_data = json.loads(reflection_output)
            except Exception:
                reflection_data = {"need_replanning": False, "reflection_text": reflection_output}
            return reflection_data
        except Exception as e:
            logger.error("Error during reflection: %s", e)
            return {"need_replanning": False, "reflection_text": "Reflection error."} 
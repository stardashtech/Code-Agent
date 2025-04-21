import json
from models.llm import call_llm
from config import settings
from agent.logger import get_logger

logger = get_logger(__name__)

class Reflection:
    """
    Advanced reflection mechanism: The agent evaluates the result of each subgoal,
    assesses query clarity based on context, collects chain-of-thought details,
    and triggers replanning if necessary.
    Uses global settings for safety directive and logging toggle.
    """

    def reflect(self, user_query: str, subgoal: str, step_result, short_term_mem) -> dict:
        # Hidden chain-of-thought logging
        if settings.ENABLE_CHAIN_OF_THOUGHT_LOGGING:
            # Summarize step_result if it's too long to avoid excessive logging/prompt length
            result_summary = str(step_result)
            if len(result_summary) > 500: # Limit summary length
                result_summary = result_summary[:497] + "..."
            debug_text = f"Reflection context - Subgoal: {subgoal}, Result Summary: {result_summary}"
            short_term_mem.update_chain_of_thought(debug_text)

        # Prepare context summary for the prompt
        result_summary_for_prompt = str(step_result)
        if len(result_summary_for_prompt) > 300: # Slightly shorter limit for prompt
            result_summary_for_prompt = result_summary_for_prompt[:297] + "..."

        prompt_header = f"{settings.PROMPT_SAFETY_DIRECTIVE}\n--- Context ---\nUser Query: {user_query}\nExecuted Subgoal: {subgoal}\nSubgoal Output Summary: {result_summary_for_prompt}\n--- Evaluation Task ---\n"
        
        prompt_eval = """1. Assess Clarity: Based **only** on the User Query and the Subgoal Output Summary, is the original User Query fundamentally ambiguous or lacking critical information that prevents progress, even considering the output obtained? Do NOT ask for clarification if the query is reasonably understandable or if the output provides sufficient context to proceed.

2. Self-Critique: Evaluate the Subgoal Output Summary. Was the subgoal executed successfully? Is the output relevant and helpful for the User Query? Are there any errors or obvious issues?

3. Replanning Need: Based on the Clarity and Self-Critique, is replanning necessary? Set 'need_replanning' to true ONLY if: (a) the User Query is **critically ambiguous** after reviewing the output, OR (b) the Subgoal Output indicates a significant failure or requires a completely different approach.
"""

        prompt_footer = """--- Output Format ---
Provide the answer ONLY in JSON format: {{"need_replanning": <bool>, \"reflection_text\": \"Brief explanation for your decisions. **If need_replanning is true due to ambiguity (point 3a), include specific questions or suggestions for clarification in the reflection_text.**\"}}. """

        prompt = prompt_header + prompt_eval + prompt_footer

        try:
            reflection_output_str = call_llm(prompt)
            # Add the raw reflection output to chain-of-thought for debugging
            if settings.ENABLE_CHAIN_OF_THOUGHT_LOGGING:
                 short_term_mem.update_chain_of_thought(f"Raw Reflection LLM Output: {reflection_output_str}")
            try:
                # Attempt to parse the JSON output
                reflection_data = json.loads(reflection_output_str)
                # Basic validation
                if not isinstance(reflection_data.get("need_replanning"), bool):
                    reflection_data["need_replanning"] = False # Default to false if type is wrong
                if not isinstance(reflection_data.get("reflection_text"), str):
                     reflection_data["reflection_text"] = str(reflection_data.get("reflection_text", "")) # Ensure it's a string

            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse reflection JSON: {json_err}. Raw output: {reflection_output_str}")
                # Attempt to salvage the text, assume no replanning needed if format is wrong
                reflection_data = {"need_replanning": False, "reflection_text": f"JSON parsing failed. Raw output: {reflection_output_str}"}
            except Exception as parse_err:
                 logger.warning(f"Unexpected error parsing reflection output: {parse_err}. Raw output: {reflection_output_str}")
                 reflection_data = {"need_replanning": False, "reflection_text": f"Unexpected parsing error. Raw output: {reflection_output_str}"}

            return reflection_data
        except Exception as e:
            logger.error("Error during reflection LLM call: %s", e)
            # Also log this critical failure in chain-of-thought
            if settings.ENABLE_CHAIN_OF_THOUGHT_LOGGING:
                 short_term_mem.update_chain_of_thought(f"CRITICAL ERROR during reflection LLM call: {e}")
            return {"need_replanning": False, "reflection_text": f"Reflection LLM call failed: {e}"} # Indicate error clearly
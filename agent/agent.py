from agent.logger import get_logger
from agent.memory import ShortTermMemory, LongTermMemory
from agent.planner import Planner
from agent.subgoals import SubgoalManager
from agent.reflection import Reflection
from agent.orchestrator import Orchestrator

logger = get_logger(__name__)

class CodeAgent:
    """
    Main agent that brings all components together:
    - Planning (Planner)
    - Subgoals (SubgoalManager)
    - Reflection / Chain-of-thought (Reflection)
    - Orchestrator (Manages the use of Tools)
    - Memory management (Short & Long Term)
    """

    def __init__(self):
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.planner = Planner()
        self.subgoal_manager = SubgoalManager()
        self.reflection = Reflection()
        self.orchestrator = Orchestrator(self.short_term_memory, self.long_term_memory)

    def run(self, user_query: str):
        logger.info("Agent started. User query: %s", user_query)
        self.short_term_memory.add("user_query", user_query)

        # 1. Create initial plan
        initial_plan = self.planner.create_initial_plan(user_query)
        logger.info("Initial plan: %s", initial_plan)

        # 2. Determine subgoals (chunking can be integrated here)
        subgoals = self.subgoal_manager.analyze_and_create_subgoals(user_query)
        logger.info("Determined subgoals: %s", subgoals)

        # 3. Process each subgoal and record hidden chain-of-thought
        execution_trace = []
        final_answer = None
        needs_clarification = False
        clarification_message = ""

        for idx, sg in enumerate(subgoals):
            logger.info("Running subgoal %d: %s", idx + 1, sg)
            step_result = self.orchestrator.execute_subgoal(sg)
            execution_trace.append({"subgoal": sg, "result": step_result})

            # Reflection: Collect chain-of-thought details (hidden)
            reflection_output = self.reflection.reflect(
                user_query=user_query,
                subgoal=sg,
                step_result=step_result,
                short_term_mem=self.short_term_memory
            )
            logger.info("Reflection result: %s", reflection_output)

            if reflection_output.get("need_replanning"):
                logger.warning("Reflection indicated replanning/clarification needed. Stopping execution.")
                # Store the reason/suggestion from reflection
                clarification_message = reflection_output.get("reflection_text", "Replanning or clarification required, but no specific text provided.")
                needs_clarification = True
                # Optional: Log the revised plan if needed for debugging, but don't execute it.
                # revised_plan = self.planner.replan_with_context(
                #     user_query,
                #     clarification_message,
                #     subgoals # Pass original subgoals here
                # )
                # logger.info("Suggested revised plan (not executed): %s", revised_plan)
                break # Stop processing further subgoals

        # 4. Generate final answer ONLY if execution completed without needing clarification
        if not needs_clarification:
            final_answer = self.orchestrator.generate_final_answer(user_query, execution_trace)
            self.long_term_memory.add("last_query", user_query)
            self.long_term_memory.add("last_answer", final_answer)
            result = {
                "status": "completed",
                "plan": initial_plan,
                "subgoals": subgoals,
                "execution_trace": execution_trace,
                "final_answer": final_answer,
                # Optionally include chain_of_thought if needed for completed runs
                # "chain_of_thought": self.short_term_memory.get("chain_of_thought") 
            }
        else:
            # Return a different structure indicating clarification is needed
            result = {
                "status": "clarification_needed",
                "message": clarification_message,
                "plan": initial_plan, # Include context
                "subgoals": subgoals,
                "execution_trace": execution_trace # Include context up to the point of stopping
            }
            
        return result 
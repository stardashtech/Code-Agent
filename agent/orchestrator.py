from tools.code_executor import CodeExecutor
from tools.vector_search import VectorSearchTool
from tools.web_browser import WebBrowserTool
from tools.doc_analysis import DocAnalysisTool
from tools.image_analysis import ImageAnalysisTool
from tools.knowledge_graph import KnowledgeGraphTool
from tools.rag_retrieval import RAGRetrievalTool
from models.llm import call_llm
from agent.logger import get_logger

logger = get_logger(__name__)

class Orchestrator:
    """
    Calls the relevant tools based on subgoals, collects outputs, and generates the final answer.
    Refactored tool dispatch logic into a helper method.
    """

    def __init__(self, short_term_mem, long_term_mem):
        self.short_term_mem = short_term_mem
        self.long_term_mem = long_term_mem
        # Tool Initialization
        self.code_executor = CodeExecutor()
        self.vector_search = VectorSearchTool()
        self.web_browser = WebBrowserTool()
        self.doc_analysis = DocAnalysisTool()
        self.image_analysis = ImageAnalysisTool()
        self.knowledge_graph = KnowledgeGraphTool()
        self.rag_retrieval = RAGRetrievalTool()
        # Store tools in a dictionary for easier access in dispatcher
        self.tools = {
            "code_executor": self.code_executor,
            "vector_search": self.vector_search,
            "web_browser": self.web_browser,
            "doc_analysis": self.doc_analysis,
            "image_analysis": self.image_analysis,
            "knowledge_graph": self.knowledge_graph,
            "rag_retrieval": self.rag_retrieval,
        }

    def _dispatch_tool(self, subgoal: str) -> dict:
        """Determines the appropriate tool based on the subgoal and executes it."""
        subgoal_lower = subgoal.lower()
        tool_used = "unknown"
        tool_input = subgoal # Default to using subgoal as input, refine as needed
        
        # Simple keyword-based routing
        # TODO: Improve routing logic (e.g., LLM-based routing, function calling)
        if "execute code" in subgoal_lower or "run code" in subgoal_lower:
            tool_used = "code_executor"
            # TODO: Extract actual code from subgoal
            tool_input = "print('Hello from Docker Code Execution')" # Placeholder
            execution_result = self.tools[tool_used].execute(tool_input)
            return {"tool_used": tool_used, "result": execution_result}
            
        elif "research" in subgoal_lower or "web search" in subgoal_lower or "browse" in subgoal_lower:
            tool_used = "web_browser"
            # TODO: Extract search query
            tool_input = subgoal
            web_results = self.tools[tool_used].search(tool_input) 
            return {"tool_used": tool_used, "result": web_results}
            
        elif "document analysis" in subgoal_lower or "analyze doc" in subgoal_lower:
            tool_used = "doc_analysis"
            # TODO: Extract doc path
            tool_input = "docs/manual.pdf" # Placeholder
            doc_result = self.tools[tool_used].analyze(tool_input)
            return {"tool_used": tool_used, "result": doc_result}
            
        elif "image analysis" in subgoal_lower or "analyze image" in subgoal_lower:
            tool_used = "image_analysis"
            # TODO: Extract image path
            tool_input = "images/sample.png" # Placeholder
            img_result = self.tools[tool_used].analyze(tool_input)
            return {"tool_used": tool_used, "result": img_result}
            
        elif "knowledge graph" in subgoal_lower or "query kg" in subgoal_lower:
            tool_used = "knowledge_graph"
            # TODO: Extract KG query
            tool_input = subgoal
            kg_result = self.tools[tool_used].query(tool_input)
            return {"tool_used": tool_used, "result": kg_result}
            
        elif "rag" in subgoal_lower or "retrieve context" in subgoal_lower or "search docs" in subgoal_lower:
            tool_used = "rag_retrieval"
            query_for_rag = subgoal
            k_results = 3 # Default
            rag_result = self.tools[tool_used].retrieve(query=query_for_rag, k=k_results)
            return {"tool_used": tool_used, "result": rag_result}
            
        else:
            # Default to LLM call if no specific tool seems appropriate
            tool_used = "llm"
            logger.info(f"No specific tool matched subgoal '{subgoal[:50]}...'. Using LLM.")
            # TODO: Consider a more structured prompt for this LLM fallback
            answer = call_llm(f"{subgoal}\nFulfill this request: {subgoal}") 
            return {"tool_used": tool_used, "result": answer}

    def execute_subgoal(self, subgoal: str):
        tool_used = "unknown"
        try:
            # Call the dispatcher to select and run the tool
            dispatch_result = self._dispatch_tool(subgoal)
            tool_used = dispatch_result.get("tool_used", "unknown")
            logger.info(f"Subgoal '{subgoal[:50]}...' executed using tool: {tool_used}")
            return dispatch_result # Return the dict { "tool_used": ..., "result": ... }
            
        except Exception as e:
             # Catch errors during dispatch or tool execution
             logger.error(f"Error executing subgoal '{subgoal[:50]}...' (tool attempted: '{tool_used}'): {e}", exc_info=True)
             # Return an error structure consistent with successful execution
             return {"tool_used": tool_used, "result": f"Error during execution: {e}"}

    def generate_final_answer(self, user_query: str, execution_trace: list):
        trace_text = ""
        for item in execution_trace:
            # Limit length of result in trace to avoid excessive prompt length
            result_str = str(item.get('result', {}).get('result', '')) # Access nested result
            tool_str = item.get('tool_used', 'unknown')
            if len(result_str) > 200: # Shorter summary for final prompt
                result_str = result_str[:197] + "..."
            trace_text += f"Subgoal: {item['subgoal']}\nTool Used: {tool_str}\nResult: {result_str}\n\n"
            
        prompt = (
            f"--- Execution Trace ---\n{trace_text}\n"
            f"--- User Query ---\n{user_query}\n"
            f"--- Task ---\n"
            "Based on the execution trace and user query, provide a comprehensive final answer. "
            "If a code modification is the best solution, provide the complete, corrected code block within a single markdown code block (e.g., ```python\n# corrected code\n```). "
            "Ensure the code is production-ready. Explain the fix briefly *before* the code block."
            "If no code fix is needed, just provide the explanation or answer."
        )
        
        try:
            final_answer = call_llm(prompt)
            return final_answer
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "Failed to generate final answer due to an error." 
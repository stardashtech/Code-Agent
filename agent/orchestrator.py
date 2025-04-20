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
    """

    def __init__(self, short_term_mem, long_term_mem):
        self.short_term_mem = short_term_mem
        self.long_term_mem = long_term_mem
        self.code_executor = CodeExecutor()
        self.vector_search = VectorSearchTool()
        self.web_browser = WebBrowserTool()
        self.doc_analysis = DocAnalysisTool()
        self.image_analysis = ImageAnalysisTool()
        self.knowledge_graph = KnowledgeGraphTool()
        self.rag_retrieval = RAGRetrievalTool()

    def execute_subgoal(self, subgoal: str):
        subgoal_lower = subgoal.lower()
        if "execute code" in subgoal_lower or "code" in subgoal_lower:
            code_to_run = "print('Hello from Docker Code Execution')"
            execution_result = self.code_executor.execute(code_to_run)
            return {"tool_used": "code_executor", "result": execution_result}
        elif "research" in subgoal_lower or "web" in subgoal_lower:
            results = self.web_browser.search("OpenAI GPT-4")
            return {"tool_used": "web_browser", "result": results}
        elif "document" in subgoal_lower:
            doc_result = self.doc_analysis.analyze("docs/manual.pdf")
            return {"tool_used": "doc_analysis", "result": doc_result}
        elif "image" in subgoal_lower:
            img_result = self.image_analysis.analyze("images/sample.png")
            return {"tool_used": "image_analysis", "result": img_result}
        elif "knowledge graph" in subgoal_lower:
            kg_result = self.knowledge_graph.query("Concepts related to TypeError")
            return {"tool_used": "knowledge_graph", "result": kg_result}
        elif "rag" in subgoal_lower:
            rag_result = self.rag_retrieval.retrieve("Information for error resolution")
            return {"tool_used": "rag_retrieval", "result": rag_result}
        else:
            answer = call_llm(f"{subgoal}\nHow should this subgoal be fulfilled?")
            return {"tool_used": "llm", "result": answer}

    def generate_final_answer(self, user_query: str, execution_trace: list):
        trace_text = ""
        for item in execution_trace:
            trace_text += f"Subgoal: {item['subgoal']}\nResult: {item['result']}\n\n"
        prompt = (
            f"{trace_text}\n"
            f"User query: {user_query}\n"
            "Based on the information above, provide the final answer that best addresses the user's problem."
        )
        final_answer = call_llm(prompt)
        return final_answer 
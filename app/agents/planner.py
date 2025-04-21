import logging
from typing import TYPE_CHECKING, List, Dict, Optional, Any
import json # Added import for JSON parsing
import re # Added import for regular expressions

if TYPE_CHECKING:
    from app.services.code_agent import LLMProvider

logger = logging.getLogger(__name__)

class Planner:
    """Agent responsible for creating execution plans."""

    def __init__(self, provider: Optional['LLMProvider'] = None):
        """Initialize the Planner agent.

        Args:
            provider: Optional instance of an LLMProvider. Needed for advanced planning.
        """
        self.provider = provider # Store provider for future LLM-based planning

    async def create_initial_plan(self, query: str, extracted_keywords: List[str], decomposed_queries: List[str]) -> List[Dict[str, Any]]:
        """Create an execution plan, potentially using the LLM to decide if web search is needed.

        Args:
            query: The original user query.
            extracted_keywords: List of keywords extracted from the query.
            decomposed_queries: List of sub-queries (or original query if not decomposed).

        Returns:
            A list of dictionaries, each representing a plan step.
        """
        default_plan = [
            {"step": "Assess Clarity"},
            {"step": "Extract Keywords"},
            {"step": "Decompose Query"},
            {
                "step": "Search Code (Vector Store)",
                "query": query, # Default to original query
                "use_keywords": True # Flag to indicate using keywords
            },
            {"step": "Analyze Code"},
            {
                "step": "Generate Fix (JSON Output)"
            }
        ]

        if not self.provider:
            logger.warning("LLM Provider not available for dynamic planning. Using default plan.")
            return default_plan

        try:
            planning_prompt = f"""
            Analyze the user query and context to determine the best plan to fulfill the request. 
            The goal is to find relevant code and provide fixes or analysis, potentially applying the fixes.
            Available steps: Assess Clarity, Extract Keywords, Decompose Query, Search Code (Vector Store), Analyze Code, Generate Fix (JSON Output), Web Search, Search GitHub, Search Stack Overflow, Apply Fix, Validate Code (Sandbox).

            Consider the following:
            - If the query asks for general information, latest updates, external documentation, or troubleshooting an error message not likely in the local codebase, include a 'Web Search' step.
            - If the query seems like a common programming question, error, or 'how-to', include 'Search Stack Overflow'. Include a relevant 'language' tag if identifiable.
            - If the query asks for code examples for a specific library/repo or mentions a GitHub repository, include 'Search GitHub'. Include a relevant 'language' tag if identifiable.
            - Include 'Search Code (Vector Store)' to search the local codebase.
            - Multiple search steps (Web, GitHub, Stack Overflow, Vector Store) can be included if relevant.
            - Place search steps generally before 'Analyze Code'.
            - 'Generate Fix (JSON Output)' should come after 'Analyze Code'.
            - Include 'Validate Code (Sandbox)' after 'Generate Fix (JSON Output)' if code was generated and validation seems beneficial before applying.
            - Include 'Apply Fix' only after 'Generate Fix (JSON Output)' and only if a fix involving a specific file path is likely to be generated.
            - The plan should be a sequence of steps.

            User Query: "{query}"
            Keywords: {extracted_keywords}
            Decomposed Queries: {decomposed_queries}

            Output ONLY a valid JSON list of plan steps. Each step should be a dictionary with a 'step' key. 
            If 'Search Code', 'Web Search', 'Search GitHub', or 'Search Stack Overflow' is included, add a 'query' key with the best query string to use. For GitHub and Stack Overflow, also add an optional 'language' key if a language/tag is relevant and identifiable.
            Ensure 'Search Code' includes the 'use_keywords' boolean.
            If 'Apply Fix' or 'Validate Code (Sandbox)' step is included, it does not need additional parameters in the plan itself; it will use the output of previous steps.
            
            Example with Validation and Apply Fix:
            [ {{"step": "Assess Clarity"}}, {{"step": "Extract Keywords"}}, {{"step": "Decompose Query"}}, {{"step": "Web Search", "query": "python requests library SOCKS proxy"}}, {{"step": "Search Stack Overflow", "query": "python requests SOCKS proxy configuration", "language": "python"}}, {{"step": "Search Code (Vector Store)", "query": "proxy configuration requests", "use_keywords": true}}, {{"step": "Analyze Code"}}, {{"step": "Generate Fix (JSON Output)"}}, {{"step": "Validate Code (Sandbox)"}}, {{"step": "Apply Fix"}} ]
            Example for local search only:
            [ {{"step": "Assess Clarity"}}, {{"step": "Extract Keywords"}}, {{"step": "Decompose Query"}}, {{"step": "Search Code (Vector Store)", "query": "{query}", "use_keywords": true}}, {{"step": "Analyze Code"}}, {{"step": "Generate Fix (JSON Output)"}} ]

            Return ONLY the JSON list.
            """

            messages = [
                {"role": "system", "content": "You are a planning assistant. Generate a JSON plan based on the user query and context."}, 
                {"role": "user", "content": planning_prompt}
            ]

            raw_response = await self.provider.generate_completion(messages, temperature=0.1)
            
            # Basic cleaning for potential markdown fences
            cleaned_response = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.MULTILINE)
            cleaned_response = re.sub(r"\s*```$", "", cleaned_response, flags=re.MULTILINE).strip()

            plan = json.loads(cleaned_response)

            # Basic validation
            if isinstance(plan, list) and all(isinstance(item, dict) and 'step' in item for item in plan):
                logger.info(f"Generated dynamic plan using LLM for query: '{query}'")
                # Ensure essential steps are present if LLM missed them (basic safety net)
                required_steps = {"Assess Clarity", "Extract Keywords", "Decompose Query", "Search Code (Vector Store)", "Analyze Code", "Generate Fix (JSON Output)"}
                plan_steps = {item['step'] for item in plan}
                if not required_steps.issubset(plan_steps) and "Web Search" not in plan_steps: # Allow web search to sometimes replace code search
                    logger.warning("LLM plan missing required steps, falling back to default plan.")
                    return default_plan
                    
                # Ensure Search Code step has necessary args if present
                for step in plan:
                    if step['step'] == "Search Code (Vector Store)":
                        step.setdefault('query', query) # Default query if missing
                        step.setdefault('use_keywords', True) # Default to using keywords
                    if step['step'] == "Web Search":
                         step.setdefault('query', query) # Default query if missing
                         
                return plan
            else:
                logger.warning("LLM generated invalid plan structure. Falling back to default plan.")
                return default_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON from LLM: {e}. Response: '{raw_response[:100]}...' Falling back to default plan.")
            return default_plan
        except Exception as e:
            logger.error(f"Error during dynamic planning: {e}. Falling back to default plan.", exc_info=True)
            return default_plan 
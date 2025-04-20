import logging
from typing import TYPE_CHECKING, List, Dict, Optional

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

    async def create_initial_plan(self, query: str, extracted_keywords: List[str], decomposed_queries: List[str]) -> List[Dict[str, str]]:
        """Create an initial plan with potential contingency suggestions (MVP).

        Args:
            query: The original user query.
            extracted_keywords: List of keywords extracted from the query.
            decomposed_queries: List of sub-queries (or original query if not decomposed).

        Returns:
            A list of dictionaries, each representing a plan step with potential suggestions.
        """
        # MVP: Return a fixed plan with simple contingency suggestions.
        plan = [
            {"step": "Assess Clarity"},
            {"step": "Extract Keywords"},
            {"step": "Decompose Query"},
            {
                "step": "Search Code (Vector Store)",
                "contingency_suggestion": "If no relevant results, try searching GitHub/web."
            },
            {"step": "Analyze Code"},
            {
                "step": "Generate Fix (JSON Output)",
                "contingency_suggestion": "If JSON is invalid, try re-generating with simpler instructions."
            }
        ]
        logger.info(f"Generated MVP plan with contingency suggestions for query: '{query}'")
        return plan 
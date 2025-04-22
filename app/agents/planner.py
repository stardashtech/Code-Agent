import logging
from typing import TYPE_CHECKING, List, Dict, Optional, Any, Tuple
import json
import re
from enum import Enum
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.services.code_agent import LLMProvider
    from app.agents.reflector import QueryType

logger = logging.getLogger(__name__)

# Define constants for plan steps
STEP_ASSESS_CLARITY = "Assess Clarity"
STEP_EXTRACT_KEYWORDS = "Extract Keywords"
STEP_DECOMPOSE_QUERY = "Decompose Query"
STEP_ANALYZE_INTENT = "Analyze Intent"
# Search Steps
STEP_SEARCH_CODE_VECTOR = "Search Code (Vector)"
STEP_SEARCH_CODE_TEXT = "Search Code (Text)"
STEP_SEARCH_PKG_MANAGER = "Search Package Manager"
STEP_WEB_SEARCH = "Web Search"
STEP_GITHUB_SEARCH = "GitHub Search"
STEP_STACKOVERFLOW_SEARCH = "Stack Overflow Search"
# Data Fetching/Processing Steps
STEP_FETCH_DOCS_URL = "Fetch Documentation URL"
STEP_SCRAPE_DOCS = "Scrape Documentation"
STEP_EXTRACT_LLM = "Extract Info (LLM)"
STEP_CHECK_VULNS = "Check Vulnerabilities"
# Analysis & Action Steps
STEP_ANALYZE_CODE = "Analyze Code & Context"
STEP_GENERATE_FIX = "Generate Fix"
STEP_APPLY_FIX = "Apply Fix"
STEP_VALIDATE_CODE = "Validate Code"
STEP_UPDATE_DEPENDENCY_FILE = "Update Dependency File"

STEP_USER_CONFIRMATION = "Request User Confirmation"
STEP_FINAL_REPORT = "Generate Final Report"

# Planning steps
STEP_ANALYZE_INTENT = "analyze_intent"
STEP_SEARCH_CODE_TEXT = "search_code_text"
STEP_SEARCH_CODE_SEMANTIC = "search_code_semantic"
STEP_SEARCH_GITHUB = "search_github"
STEP_SEARCH_WEB = "search_web"
STEP_ANALYZE_ERROR = "analyze_error"
STEP_IMPLEMENT_FIX = "implement_fix"
STEP_IMPLEMENT_FEATURE = "implement_feature"
STEP_REFACTOR_CODE = "refactor_code"
STEP_OPTIMIZE_CODE = "optimize_code"
STEP_UPDATE_DEPENDENCIES = "update_dependencies"
STEP_VERIFY_CHANGES = "verify_changes"
STEP_EXPLAIN_CODE = "explain_code"
STEP_GENERATE_TESTS = "generate_tests"

class PlanType(Enum):
    """Different types of plans that can be created."""
    ERROR_FIXING = "error_fixing"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    CODE_REFACTORING = "code_refactoring"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DEPENDENCY_MANAGEMENT = "dependency_management"
    CODE_EXPLANATION = "code_explanation"
    GENERAL = "general"
    
class PlanStep(BaseModel):
    """Model representing a step in an execution plan."""
    step: str
    description: str
    inputs: List[str]
    outputs: List[str]
    estimated_time: Optional[int] = None
    critical: bool = False
    
class ExecutionPlan(BaseModel):
    """Model representing a complete execution plan."""
    plan_type: str
    steps: List[PlanStep]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.steps.append(step)
        
    def insert_step(self, index: int, step: PlanStep) -> None:
        """Insert a step at a specific position."""
        self.steps.insert(index, step)
        
    def remove_step(self, step_name: str) -> None:
        """Remove a step from the plan."""
        self.steps = [s for s in self.steps if s.step != step_name]
        
    def get_step(self, step_name: str) -> Optional[PlanStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.step == step_name:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary."""
        return {
            "plan_type": self.plan_type,
            "steps": [step.dict() for step in self.steps],
            "metadata": self.metadata
        }

class Planner:
    """
    Agent responsible for planning an execution strategy to resolve a user's query.
    
    Features:
    - Adaptive prompt templates based on query intent and type
    - Multiple plan templates for different query types
    - Integration with Reflector's intent analysis
    - Structured execution step generation
    - Dynamic plan modification based on query complexity
    """

    def __init__(self, provider: Optional['LLMProvider'] = None):
        """Initialize the Planner agent.

        Args:
            provider: Optional instance of an LLMProvider. Required to perform planning.
        """
        self.provider = provider
        
        # Base system prompt that will be customized per request
        self._base_system_prompt = """You are an expert planning system for a code assistant agent. 
Your task is to create a detailed execution plan for resolving a user's query about code or development.

The plan should consist of well-defined steps that achieve the user's goal efficiently.
Use your knowledge of software development, debugging, and implementation to create appropriate plans.

For each step, provide:
1. Step identifier (use one of the provided step types)
2. Clear description of what this step accomplishes
3. Required inputs for this step
4. Expected outputs from this step
5. Estimated time to complete (in minutes)
6. Whether this step is critical for success (true/false)

{additional_context}

Available step types:
- analyze_intent: Understand user intent and requirements
- search_code_text: Text-based code search
- search_code_semantic: Semantic code search
- search_github: Search GitHub for examples
- search_web: Search web for information
- analyze_error: Analyze error messages and traces
- implement_fix: Implement code fixes
- implement_feature: Implement new features
- refactor_code: Refactor existing code
- optimize_code: Optimize for performance
- update_dependencies: Update project dependencies
- verify_changes: Verify code changes work correctly
- explain_code: Explain code functionality
- generate_tests: Generate tests for code

Create a focused plan with 3-7 steps that addresses the user's query efficiently.
Return ONLY a JSON array containing the plan steps.
"""
    
    def _get_error_fixing_template(self) -> ExecutionPlan:
        """Return a template for error fixing plans."""
        return ExecutionPlan(
            plan_type=PlanType.ERROR_FIXING.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the error message and context to understand the problem",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["error_type", "affected_files", "probable_causes"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_TEXT,
                    description="Search for the error location in the codebase",
                    inputs=["error_message", "affected_files"],
                    outputs=["error_context", "related_code"],
                    estimated_time=2,
                    critical=True
                ),
                PlanStep(
                    step=STEP_ANALYZE_ERROR,
                    description="Analyze the error context to determine the cause",
                    inputs=["error_context", "related_code", "error_type"],
                    outputs=["root_cause", "fix_approach"],
                    estimated_time=5,
                    critical=True
                ),
                PlanStep(
                    step=STEP_IMPLEMENT_FIX,
                    description="Implement the fix for the identified error",
                    inputs=["root_cause", "fix_approach", "affected_files"],
                    outputs=["code_changes"],
                    estimated_time=8,
                    critical=True
                ),
                PlanStep(
                    step=STEP_VERIFY_CHANGES,
                    description="Verify that the changes resolve the error",
                    inputs=["code_changes", "affected_files"],
                    outputs=["verification_result"],
                    estimated_time=4,
                    critical=True
                )
            ]
        )
    
    def _get_feature_implementation_template(self) -> ExecutionPlan:
        """Return a template for feature implementation plans."""
        return ExecutionPlan(
            plan_type=PlanType.FEATURE_IMPLEMENTATION.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the feature request to understand requirements",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["feature_requirements", "affected_components"],
                    estimated_time=5,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Search for relevant code components to integrate with",
                    inputs=["affected_components", "feature_requirements"],
                    outputs=["integration_points", "existing_patterns"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_GITHUB,
                    description="Search for examples or best practices for implementation",
                    inputs=["feature_requirements", "programming_language"],
                    outputs=["reference_implementations", "best_practices"],
                    estimated_time=4,
                    critical=False
                ),
                PlanStep(
                    step=STEP_IMPLEMENT_FEATURE,
                    description="Implement the requested feature",
                    inputs=["feature_requirements", "integration_points", "existing_patterns", "best_practices"],
                    outputs=["code_changes", "new_files"],
                    estimated_time=15,
                    critical=True
                ),
                PlanStep(
                    step=STEP_GENERATE_TESTS,
                    description="Generate tests for the new feature",
                    inputs=["code_changes", "new_files", "feature_requirements"],
                    outputs=["test_code"],
                    estimated_time=8,
                    critical=False
                ),
                PlanStep(
                    step=STEP_VERIFY_CHANGES,
                    description="Verify that the implementation meets requirements",
                    inputs=["code_changes", "new_files", "test_code", "feature_requirements"],
                    outputs=["verification_result"],
                    estimated_time=5,
                    critical=True
                )
            ]
        )
    
    def _get_code_refactoring_template(self) -> ExecutionPlan:
        """Return a template for code refactoring plans."""
        return ExecutionPlan(
            plan_type=PlanType.CODE_REFACTORING.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the refactoring request to understand goals",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["refactoring_goals", "affected_components"],
                    estimated_time=4,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Search for code components that need refactoring",
                    inputs=["affected_components", "refactoring_goals"],
                    outputs=["refactoring_targets", "code_dependencies"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_GITHUB,
                    description="Search for refactoring patterns and best practices",
                    inputs=["refactoring_goals", "programming_language"],
                    outputs=["refactoring_patterns", "best_practices"],
                    estimated_time=3,
                    critical=False
                ),
                PlanStep(
                    step=STEP_REFACTOR_CODE,
                    description="Refactor the identified code components",
                    inputs=["refactoring_targets", "refactoring_patterns", "code_dependencies"],
                    outputs=["code_changes"],
                    estimated_time=12,
                    critical=True
                ),
                PlanStep(
                    step=STEP_VERIFY_CHANGES,
                    description="Verify that the refactoring maintains functionality",
                    inputs=["code_changes", "refactoring_targets"],
                    outputs=["verification_result"],
                    estimated_time=5,
                    critical=True
                )
            ]
        )
    
    def _get_performance_optimization_template(self) -> ExecutionPlan:
        """Return a template for performance optimization plans."""
        return ExecutionPlan(
            plan_type=PlanType.PERFORMANCE_OPTIMIZATION.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the optimization request to understand performance issues",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["performance_issues", "affected_components"],
                    estimated_time=5,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Search for code components with performance bottlenecks",
                    inputs=["affected_components", "performance_issues"],
                    outputs=["bottleneck_locations", "performance_metrics"],
                    estimated_time=4,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_WEB,
                    description="Search for optimization strategies for specific issues",
                    inputs=["performance_issues", "programming_language"],
                    outputs=["optimization_strategies", "best_practices"],
                    estimated_time=3,
                    critical=False
                ),
                PlanStep(
                    step=STEP_OPTIMIZE_CODE,
                    description="Implement performance optimizations",
                    inputs=["bottleneck_locations", "optimization_strategies"],
                    outputs=["code_changes"],
                    estimated_time=10,
                    critical=True
                ),
                PlanStep(
                    step=STEP_VERIFY_CHANGES,
                    description="Verify performance improvements",
                    inputs=["code_changes", "bottleneck_locations", "performance_metrics"],
                    outputs=["verification_result", "performance_impact"],
                    estimated_time=6,
                    critical=True
                )
            ]
        )
    
    def _get_dependency_management_template(self) -> ExecutionPlan:
        """Return a template for dependency management plans."""
        return ExecutionPlan(
            plan_type=PlanType.DEPENDENCY_MANAGEMENT.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the dependency request to understand requirements",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["dependency_requirements", "affected_components"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_TEXT,
                    description="Search for existing dependency configurations",
                    inputs=["affected_components"],
                    outputs=["dependency_files", "current_dependencies"],
                    estimated_time=2,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_WEB,
                    description="Search for dependency information and compatibility",
                    inputs=["dependency_requirements"],
                    outputs=["dependency_versions", "compatibility_info"],
                    estimated_time=3,
                    critical=False
                ),
                PlanStep(
                    step=STEP_UPDATE_DEPENDENCIES,
                    description="Update dependency configurations",
                    inputs=["dependency_files", "current_dependencies", "dependency_requirements", "dependency_versions"],
                    outputs=["config_changes"],
                    estimated_time=5,
                    critical=True
                ),
                PlanStep(
                    step=STEP_VERIFY_CHANGES,
                    description="Verify dependency changes for compatibility",
                    inputs=["config_changes", "compatibility_info"],
                    outputs=["verification_result"],
                    estimated_time=4,
                    critical=True
                )
            ]
        )
    
    def _get_code_explanation_template(self) -> ExecutionPlan:
        """Return a template for code explanation plans."""
        return ExecutionPlan(
            plan_type=PlanType.CODE_EXPLANATION.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the explanation request to understand focus areas",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["explanation_focus", "code_components"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Search for relevant code to explain",
                    inputs=["code_components", "explanation_focus"],
                    outputs=["relevant_code", "code_relationships"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_EXPLAIN_CODE,
                    description="Generate a clear explanation of the code",
                    inputs=["relevant_code", "code_relationships", "explanation_focus"],
                    outputs=["code_explanation", "visual_diagrams"],
                    estimated_time=10,
                    critical=True
                )
            ]
        )
    
    def _get_general_template(self) -> ExecutionPlan:
        """Return a general purpose plan template."""
        return ExecutionPlan(
            plan_type=PlanType.GENERAL.value,
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the query to understand requirements",
                    inputs=["query", "conversation_history", "code_context"],
                    outputs=["query_type", "key_requirements"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Search for relevant code components",
                    inputs=["key_requirements"],
                    outputs=["relevant_code", "code_context"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_WEB,
                    description="Search for relevant information if needed",
                    inputs=["key_requirements", "relevant_code"],
                    outputs=["reference_information", "best_practices"],
                    estimated_time=3,
                    critical=False
                )
            ]
        )
    
    def _get_tool_usage_template(self) -> ExecutionPlan:
        """Return a template for tool usage demonstration plans."""
        return ExecutionPlan(
            plan_type="tool_usage",
            steps=[
                PlanStep(
                    step=STEP_ANALYZE_INTENT,
                    description="Analyze the query to identify which tool the user wants to use",
                    inputs=["query", "conversation_history"],
                    outputs=["tool_name", "usage_requirements"],
                    estimated_time=2,
                    critical=True
                ),
                PlanStep(
                    step=STEP_SEARCH_CODE_TEXT,
                    description="Search for examples of the tool's usage in the codebase",
                    inputs=["tool_name"],
                    outputs=["usage_examples", "tool_implementation_files"],
                    estimated_time=3,
                    critical=True
                ),
                PlanStep(
                    step="prepare_tool_demo",
                    description="Prepare demonstration of the tool with appropriate examples",
                    inputs=["tool_name", "usage_examples", "usage_requirements"],
                    outputs=["tool_demonstration", "sample_code"],
                    estimated_time=5,
                    critical=True
                ),
                PlanStep(
                    step=STEP_FINAL_REPORT,
                    description="Generate a comprehensive guide on tool usage with examples",
                    inputs=["tool_name", "tool_demonstration", "sample_code"],
                    outputs=["usage_guide"],
                    estimated_time=3,
                    critical=True
                )
            ]
        )
    
    def _generate_plan_template_from_query_type(self, query_type: 'QueryType') -> ExecutionPlan:
        """
        Generate an execution plan template based on query type.
        
        Args:
            query_type: The type of query from the Reflector
            
        Returns:
            An ExecutionPlan template appropriate for the query type
        """
        # Özel bir durumda tool kullanımı şablonunu kontrol et
        query_type_str = str(query_type.value).lower() if query_type else ""
        if query_type_str == "general" and ("tool" in query_type_str or "search" in query_type_str or "github" in query_type_str):
            return self._get_tool_usage_template()
        
        if query_type == QueryType.ERROR_ANALYSIS:
            return self._get_error_fixing_template()
        elif query_type == QueryType.FEATURE_REQUEST:
            return self._get_feature_implementation_template()
        elif query_type == QueryType.CODE_SPECIFIC:
            # For code-specific, decide between refactoring, optimization, and explanation
            # based on the specific nature of the query (would require more context)
            return self._get_code_explanation_template()
        elif query_type == QueryType.PERFORMANCE:
            return self._get_performance_optimization_template()
        elif query_type == QueryType.DEPENDENCY:
            return self._get_dependency_management_template()
        elif query_type == QueryType.SECURITY:
            # For security, we might need a more specialized template
            # For now, use refactoring with security focus
            return self._get_code_refactoring_template()
        elif query_type == QueryType.ARCHITECTURE:
            # For architecture, use refactoring with architecture focus
            return self._get_code_refactoring_template()
        else:
            return self._get_general_template()
    
    def _adapt_template_to_context(self, template: ExecutionPlan, 
                                 intent_analysis: Dict[str, Any],
                                 clarity_assessment: Dict[str, Any]) -> ExecutionPlan:
        """
        Adapt a plan template based on the intent analysis, clarity assessment and context.
        
        Args:
            template: The base template to adapt
            intent_analysis: The intent analysis results
            clarity_assessment: The clarity assessment results
            
        Returns:
            Adapted template with context-specific modifications
        """
        adapted_plan = ExecutionPlan(
            plan_type=template.plan_type,
            steps=template.steps.copy(),
            metadata=template.metadata.copy()
        )
        
        # Store analysis data in metadata
        adapted_plan.metadata["intent_analysis"] = intent_analysis
        adapted_plan.metadata["clarity_assessment"] = clarity_assessment
        
        # If query is unclear, add clarification steps
        if clarity_assessment.get("is_clear", True) is False:
            # Add a clarification step at the beginning
            clarification_step = PlanStep(
                step="request_clarification",
                description="Request clarification from the user on ambiguous aspects",
                inputs=["query", "ambiguities", "missing_context"],
                outputs=["clarified_query", "additional_context"],
                estimated_time=1,
                critical=True
            )
            adapted_plan.insert_step(0, clarification_step)
            adapted_plan.metadata["requires_clarification"] = True
        
        # Add web search step if external resources are needed
        if intent_analysis.get("planning_insights", {}).get("external_resources_needed", False):
            if not any(step.step == STEP_SEARCH_WEB for step in adapted_plan.steps):
                web_search_step = PlanStep(
                    step=STEP_SEARCH_WEB,
                    description="Search for additional information or best practices",
                    inputs=["query", "key_requirements"],
                    outputs=["reference_information", "best_practices"],
                    estimated_time=3,
                    critical=False
                )
                # Insert after the first step (usually analysis)
                adapted_plan.insert_step(1, web_search_step)
        
        # Add code semantic search if deep analysis is required
        if intent_analysis.get("planning_insights", {}).get("analysis_depth_required") == "deep":
            if not any(step.step == STEP_SEARCH_CODE_SEMANTIC for step in adapted_plan.steps):
                code_search_step = PlanStep(
                    step=STEP_SEARCH_CODE_SEMANTIC,
                    description="Perform deep semantic code search for related components",
                    inputs=["query", "affected_components"],
                    outputs=["related_code", "code_dependencies"],
                    estimated_time=4,
                    critical=True
                )
                # Insert early in the plan
                adapted_plan.insert_step(1, code_search_step)
        
        # Add verification step if it's a critical operation
        primary_intent = intent_analysis.get("primary_intent", "other")
        if primary_intent in ["fix_error", "implement_feature", "optimize_performance"] and not any(step.step == STEP_VERIFY_CHANGES for step in adapted_plan.steps):
            verification_step = PlanStep(
                step=STEP_VERIFY_CHANGES,
                description="Verify that the changes function correctly",
                inputs=["code_changes", "requirements"],
                outputs=["verification_result"],
                estimated_time=5,
                critical=True
            )
            adapted_plan.steps.append(verification_step)
            
        # Add test generation for new features if needed
        if primary_intent == "implement_feature" and intent_analysis.get("context_requirements", {}).get("code_examples") in ["needed", "helpful"] and not any(step.step == STEP_GENERATE_TESTS for step in adapted_plan.steps):
            test_step = PlanStep(
                step=STEP_GENERATE_TESTS,
                description="Generate tests for the new functionality",
                inputs=["code_changes", "requirements"],
                outputs=["test_code"],
                estimated_time=8,
                critical=False
            )
            
            # Insert before verification if it exists, otherwise at the end
            verification_index = next((i for i, step in enumerate(adapted_plan.steps) if step.step == STEP_VERIFY_CHANGES), len(adapted_plan.steps))
            adapted_plan.insert_step(verification_index, test_step)
            
        return adapted_plan
    
    def _generate_adaptive_prompt(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None,
                               code_context: Optional[str] = None, 
                               intent_analysis: Optional[Dict[str, Any]] = None,
                               clarity_assessment: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an adaptive system prompt based on query context and analysis.
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            code_context: Optional code context
            intent_analysis: Optional intent analysis
            clarity_assessment: Optional clarity assessment
            
        Returns:
            A customized system prompt
        """
        additional_context = []
        
        # Add context about query type and intent
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "unknown")
            confidence = intent_analysis.get("confidence", 0.0)
            additional_context.append(f"This query involves a {primary_intent} request with confidence {confidence:.2f}.")
            
            # Add specific guidance based on intent
            if primary_intent == "fix_error":
                additional_context.append("Focus on diagnostic steps followed by targeted fixes.")
            elif primary_intent == "implement_feature":
                additional_context.append("Focus on requirements gathering, design, implementation, and testing.")
            elif primary_intent == "refactor_code":
                additional_context.append("Focus on code quality, maintainability, and preserving functionality.")
            elif primary_intent == "optimize_performance":
                additional_context.append("Focus on identifying bottlenecks, benchmarking, and targeted optimizations.")
        
        # Add context about clarity
        if clarity_assessment:
            is_clear = clarity_assessment.get("is_clear", True)
            if not is_clear:
                ambiguities = clarity_assessment.get("ambiguities", [])
                missing = clarity_assessment.get("missing_context", [])
                additional_context.append("The query has clarity issues that should be addressed:")
                if ambiguities:
                    additional_context.append(f"- Ambiguities: {', '.join(ambiguities)}")
                if missing:
                    additional_context.append(f"- Missing context: {', '.join(missing)}")
        
        # Add context about conversation
        if conversation_history and len(conversation_history) > 1:
            additional_context.append(f"This is part of an ongoing conversation with {len(conversation_history)} prior exchanges.")
        
        # Add context about code
        if code_context:
            additional_context.append("Relevant code context is available for this query.")
        
        # Combine all additional context
        additional_context_str = "\n".join(additional_context) if additional_context else "No additional context available."
        
        # Generate the complete system prompt
        prompt = self._base_system_prompt.format(additional_context=additional_context_str)
        return prompt

    async def create_plan(self, query: str, 
                     conversation_history: Optional[List[Dict[str, str]]] = None,
                     code_context: Optional[str] = None,
                     intent_analysis: Optional[Dict[str, Any]] = None,
                     clarity_assessment: Optional[Dict[str, Any]] = None,
                     query_type: Optional['QueryType'] = None) -> ExecutionPlan:
        """
        Create an execution plan for resolving a query.
        
        Args:
            query: The user query to plan for.
            conversation_history: Optional list of previous messages
            code_context: Optional context about the code being discussed
            intent_analysis: Optional pre-computed intent analysis
            clarity_assessment: Optional pre-computed clarity assessment
            query_type: Optional query type from Reflector
            
        Returns:
            An ExecutionPlan object with steps to resolve the query
        """
        if not self.provider:
            logger.error("LLM Provider not available for planning.")
            return self._get_general_template()
        
        # Use available analysis or default to general
        if query_type:
            # Get template based on query type from Reflector
            template = self._generate_plan_template_from_query_type(query_type)
        elif intent_analysis:
            # Select template based on intent
            primary_intent = intent_analysis.get("primary_intent", "other")
            confidence = intent_analysis.get("confidence", 0.0)
            
            logger.info(f"Planning based on intent: {primary_intent} with confidence {confidence}")
            
            # Map intent to template
            template_key = PlanType.GENERAL.value
            if primary_intent == "fix_error" and confidence > 0.6:
                template_key = PlanType.ERROR_FIXING.value
            elif primary_intent == "implement_feature" and confidence > 0.6:
                template_key = PlanType.FEATURE_IMPLEMENTATION.value
            elif primary_intent == "refactor_code" and confidence > 0.6:
                template_key = PlanType.CODE_REFACTORING.value
            elif primary_intent == "optimize_performance" and confidence > 0.6:
                template_key = PlanType.PERFORMANCE_OPTIMIZATION.value
            elif primary_intent == "add_dependency" and confidence > 0.6:
                template_key = PlanType.DEPENDENCY_MANAGEMENT.value
            elif primary_intent == "explain_code" and confidence > 0.6:
                template_key = PlanType.CODE_EXPLANATION.value
                
            # Get the template from the built-in templates
            if template_key == PlanType.ERROR_FIXING.value:
                template = self._get_error_fixing_template()
            elif template_key == PlanType.FEATURE_IMPLEMENTATION.value:
                template = self._get_feature_implementation_template()
            elif template_key == PlanType.CODE_REFACTORING.value:
                template = self._get_code_refactoring_template()
            elif template_key == PlanType.PERFORMANCE_OPTIMIZATION.value:
                template = self._get_performance_optimization_template()
            elif template_key == PlanType.DEPENDENCY_MANAGEMENT.value:
                template = self._get_dependency_management_template()
            elif template_key == PlanType.CODE_EXPLANATION.value:
                template = self._get_code_explanation_template()
            else:
                template = self._get_general_template()
            
            # Adapt template based on available analyses
            template = self._adapt_template_to_context(template, intent_analysis, 
                                                   clarity_assessment or {})
            
            logger.info(f"Selected and adapted the {template_key} template with {len(template.steps)} steps")
            return template
        
        # If no intent analysis or query type, use LLM to generate a custom plan
        formatted_conversation = self._format_conversation(conversation_history)
        formatted_code_context = f"Code Context:\n{code_context}" if code_context else "No code context provided."
        
        # Generate adaptive system prompt
        system_prompt = self._generate_adaptive_prompt(
            query=query,
            conversation_history=conversation_history,
            code_context=code_context,
            intent_analysis=intent_analysis,
            clarity_assessment=clarity_assessment
        )
        
        user_prompt = f"""
        Create an execution plan for the following query:
        
        QUERY: {query}
        
        CONVERSATION HISTORY:
        {formatted_conversation}
        
        {formatted_code_context}
        
        Return ONLY a JSON array containing the plan steps. Each step should include "step", "description", "inputs", "outputs", "estimated_time", and "critical" fields.
        Example:
        [
            {{
                "step": "analyze_intent",
                "description": "Analyze the query to understand requirements",
                "inputs": ["query", "conversation_history"],
                "outputs": ["query_type", "key_requirements"],
                "estimated_time": 3,
                "critical": true
            }},
            ...
        ]
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.provider.generate_completion(messages, temperature=0.2)
            
            # Extract and parse the JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plan_json = json_match.group(0)
                plan_steps = json.loads(plan_json)
                
                if isinstance(plan_steps, list) and len(plan_steps) > 0:
                    logger.info(f"Generated plan with {len(plan_steps)} steps")
                    # Convert to PlanStep objects
                    steps = [
                        PlanStep(
                            step=step["step"],
                            description=step["description"],
                            inputs=step["inputs"],
                            outputs=step["outputs"],
                            estimated_time=step.get("estimated_time", 5),
                            critical=step.get("critical", False)
                        )
                        for step in plan_steps
                    ]
                    
                    # Create and return the complete plan
                    return ExecutionPlan(
                        plan_type=PlanType.GENERAL.value,
                        steps=steps,
                        metadata={
                            "generated_by_llm": True,
                            "input_query": query
                        }
                    )
                else:
                    logger.warning("Generated plan has invalid structure")
            else:
                logger.warning("No valid JSON array found in plan response.")
                
            # Fall back to a default plan if generation fails
            logger.info("Using default plan as fallback")
            return self._get_general_template()
            
        except Exception as e:
            logger.error(f"Error during plan creation: {e}", exc_info=True)
            return self._get_general_template()
    
    def _format_conversation(self, conversation_history: Optional[List[Dict[str, str]]]) -> str:
        """Format conversation history for inclusion in prompts."""
        if not conversation_history:
            return "No previous conversation."
        
        formatted_history = ""
        for i, message in enumerate(conversation_history):
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            formatted_history += f"{role}: {content}\n\n"
            
            # Limit history length to avoid token issues
            if i >= 5:  # Only include the most recent 6 messages
                formatted_history = "...(earlier conversation omitted)...\n\n" + formatted_history
                break
                
        return formatted_history 
import logging
from typing import TYPE_CHECKING, List, Dict, Optional, Any
import json # Added import for JSON parsing
import re # Added import for regular expressions

if TYPE_CHECKING:
    from app.services.code_agent import LLMProvider
    # Add ProactiveIssue import for type hinting
    from analysis.proactive_analyzer import ProactiveIssue 

logger = logging.getLogger(__name__)

# Define constants for plan steps
STEP_ASSESS_CLARITY = "Assess Clarity"
STEP_EXTRACT_KEYWORDS = "Extract Keywords"
STEP_DECOMPOSE_QUERY = "Decompose Query"
# Search Steps
STEP_SEARCH_CODE_VECTOR = "Search Code (Vector)"
STEP_SEARCH_CODE_TEXT = "Search Code (Text)" # Potentially needed for specific keyword searches
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
STEP_UPDATE_DEPENDENCY_FILE = "Update Dependency File" # Re-enabled this step

STEP_USER_CONFIRMATION = "Request User Confirmation" # Potential future step
STEP_FINAL_REPORT = "Generate Final Report"

class Planner:
    """Agent responsible for creating execution plans."""

    def __init__(self, provider: Optional['LLMProvider'] = None):
        """Initialize the Planner agent.

        Args:
            provider: Optional instance of an LLMProvider. Needed for advanced planning.
        """
        self.provider = provider # Store provider for future LLM-based planning

    async def create_plan(self, 
                          query: Optional[str] = None, 
                          extracted_keywords: Optional[List[str]] = None, 
                          decomposed_queries: Optional[List[str]] = None,
                          proactive_issues: Optional[List['ProactiveIssue']] = None
                         ) -> List[Dict[str, Any]]:
        """
        Create an execution plan based on user query OR proactive issues.
        Uses the LLM to decide steps.
        """
        
        # Determine if this is a proactive run or user query run
        is_proactive_run = bool(proactive_issues and not query)
        run_type = "Proactive Issue Resolution" if is_proactive_run else "User Query"
        
        logger.info(f"Creating plan for: {run_type}")

        # Define default plan based on run type (proactive needs different default)
        if is_proactive_run:
             # Default plan for proactive issues might just be analysis initially
             # The LLM should generate the specific steps based on the issue type
             default_plan = [
                 # {"step": STEP_ANALYZE_CODE}, # Maybe start with analysis of the issue?
                 # {"step": STEP_GENERATE_FIX}  # Or directly try to generate a fix?
                 # Let's rely on the LLM prompt for proactive default for now. If it fails, return empty.
                 [] 
             ]
             # Ensure proactive issues are serializable for the prompt
             try:
                 proactive_issues_str = json.dumps(proactive_issues, indent=2)
             except TypeError:
                 logger.error("Could not serialize proactive issues for LLM prompt.")
                 proactive_issues_str = "[Error serializing issues]"
        else:
             # Keep the original default plan for user queries
             default_plan = [
                 {"step": STEP_ASSESS_CLARITY},
                 {"step": STEP_EXTRACT_KEYWORDS},
                 {"step": STEP_DECOMPOSE_QUERY},
                 {
                     "step": STEP_SEARCH_CODE_VECTOR,
                     "query": query or "", 
                     "use_keywords": True
                 },
                 {"step": STEP_ANALYZE_CODE},
                 {"step": STEP_GENERATE_FIX}
             ]
             proactive_issues_str = "N/A" # Not applicable for user query

        if not self.provider:
            logger.warning(f"LLM Provider not available for dynamic planning ({run_type}). Using default plan.")
            return default_plan if not is_proactive_run else [] # Return empty for failed proactive planning

        try:
            # === Construct the Planning Prompt ===
            prompt_header = f"Analyze the context and determine the best plan to fulfill the request ({run_type})."
            prompt_goal = "The goal is to find relevant code/information and provide fixes or analysis, potentially applying the fixes."
            if is_proactive_run:
                 prompt_goal = "The goal is to generate a plan to address the identified proactive issues, potentially generating and applying fixes."
                 
            available_steps = f"""
            Available steps:
            - Core: {STEP_ASSESS_CLARITY}, {STEP_EXTRACT_KEYWORDS}, {STEP_DECOMPOSE_QUERY}, {STEP_SEARCH_CODE_VECTOR}, {STEP_ANALYZE_CODE}, {STEP_GENERATE_FIX}, {STEP_APPLY_FIX}, {STEP_VALIDATE_CODE}.
            - External Search: {STEP_WEB_SEARCH}, {STEP_GITHUB_SEARCH}, {STEP_STACKOVERFLOW_SEARCH}.
            - Package/Docs/Vulns: {STEP_SEARCH_PKG_MANAGER}, {STEP_FETCH_DOCS_URL}, {STEP_SCRAPE_DOCS}, {STEP_EXTRACT_LLM}, {STEP_CHECK_VULNS}.
            - File Update: {STEP_UPDATE_DEPENDENCY_FILE} (Use for applying dependency updates).
            """
            
            guidelines = f"""
            Guidelines:
            {'1. If user query: Always start with: ' + STEP_ASSESS_CLARITY + ', ' + STEP_EXTRACT_KEYWORDS + ', ' + STEP_DECOMPOSE_QUERY + '.' if not is_proactive_run else '1. If proactive issues: Analyze the issues and generate appropriate steps directly.'}
            2.  Include {STEP_SEARCH_CODE_VECTOR} to search the local codebase if understanding local context or usage is needed (e.g., before updating a dependency).
            3.  External Search: 
                - Use {STEP_WEB_SEARCH} for general info, latest updates, external docs, troubleshooting errors.
                - Use {STEP_STACKOVERFLOW_SEARCH} for common programming questions/errors/how-tos (add 'language' tag).
                - Use {STEP_GITHUB_SEARCH} for finding code examples in specific repos or searching GitHub broadly (add 'language' tag).
            4.  Package/Docs/Vulns (Use when query or issue implies specific library/package interaction):
                - Use {STEP_SEARCH_PKG_MANAGER} to find info about a package. Requires 'manager_type' and 'query'.
                - Use {STEP_FETCH_DOCS_URL} *after* finding a package/repo to get its documentation URL. Requires 'source_type' and 'package_name'.
                - Use {STEP_SCRAPE_DOCS} *after* {STEP_FETCH_DOCS_URL}. Requires 'url'.
                - Use {STEP_EXTRACT_LLM} *after* {STEP_SCRAPE_DOCS} or getting other text content. Requires 'prompt_key'.
                - Use {STEP_CHECK_VULNS} to check for vulnerabilities. Requires 'source_type', 'package_name', 'version'.
            5.  Analysis/Fix:
                - Place search/data gathering steps generally *before* {STEP_ANALYZE_CODE}.
                - {STEP_ANALYZE_CODE} should consolidate information gathered about the query or issue.
                - {STEP_GENERATE_FIX} should follow {STEP_ANALYZE_CODE}.
                - If the fix involves updating a dependency file, {STEP_GENERATE_FIX} might output parameters for {STEP_UPDATE_DEPENDENCY_FILE}.
                - {STEP_UPDATE_DEPENDENCY_FILE} updates the dependency file directly. Requires 'file_path', 'package_name', 'new_version'.
                - {STEP_VALIDATE_CODE} can follow {STEP_GENERATE_FIX} or {STEP_UPDATE_DEPENDENCY_FILE} if validation is beneficial.
                - {STEP_APPLY_FIX} applies general code fixes generated by {STEP_GENERATE_FIX}.
            6.  Parameter Requirements: (List specific requirements as before)
                - {STEP_UPDATE_DEPENDENCY_FILE} needs 'file_path', 'package_name', 'new_version'.
                (Add requirements for other steps here)
            """
            
            # Define context_section and output_instructions before they are used
            context_section = f"""
            {'--- User Query Context ---' if not is_proactive_run else '--- Proactive Issues Context ---'}
            {'User Query: "' + (query or 'N/A') + '"' if not is_proactive_run else ''}
            {'Keywords: ' + str(extracted_keywords or []) if not is_proactive_run else ''}
            {'Decomposed Queries: ' + str(decomposed_queries or []) if not is_proactive_run else ''}
            {'Proactive Issues Found:' + proactive_issues_str if is_proactive_run else ''}
            """
            output_instructions = f"""
            Output ONLY a valid JSON list of plan step dictionaries.
            Example (Proactive: Update outdated 'requests' lib in requirements.txt):
            [ 
              {{ # Optionally check vulnerabilities first
                "step": "{STEP_CHECK_VULNS}", "source_type": "pypi", "package_name": "requests", "version": "[from issue context]" 
              }}, 
              {{ # Search local usage before update
                "step": "{STEP_SEARCH_CODE_VECTOR}", "query": "Usage of requests library", "use_keywords": false 
              }},
              {{ # Analyze potential impact
                "step": "{STEP_ANALYZE_CODE}" 
              }},
              {{ # Generate the file update instruction (alternative to generic fix)
                "step": "{STEP_UPDATE_DEPENDENCY_FILE}", "file_path": "[from issue context]", "package_name": "requests", "new_version": "[from issue context]"
              }}
              {{ # Optionally validate after update (e.g., run tests)
                # "step": "{STEP_VALIDATE_CODE}" 
              }} 
            ]

            Return ONLY the JSON list.
            """
            
            planning_prompt = f"{prompt_header}\n{prompt_goal}\n\n{available_steps}\n\n{guidelines}\n\n{context_section}\n\n{output_instructions}"

            messages = [
                {"role": "system", "content": "You are a planning assistant. Generate a JSON plan based on the provided context (user query or proactive issues), following the guidelines and available steps."}, 
                {"role": "user", "content": planning_prompt}
            ]

            raw_response = await self.provider.generate_completion(messages, temperature=0.1)
            
            # Attempt to extract JSON even if there's surrounding text
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                 cleaned_response = json_match.group(0)
            else:
                 # Fallback to previous cleaning attempt if no clear JSON list is found
                 cleaned_response = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.MULTILINE)
                 cleaned_response = re.sub(r"\s*```$", "", cleaned_response, flags=re.MULTILINE).strip()


            plan = json.loads(cleaned_response)

            if isinstance(plan, list) and all(isinstance(item, dict) and 'step' in item for item in plan):
                logger.info(f"Generated dynamic plan using LLM for {run_type}.")
                # Basic safety net for user queries
                if not is_proactive_run:
                     core_steps = {STEP_ASSESS_CLARITY, STEP_EXTRACT_KEYWORDS, STEP_DECOMPOSE_QUERY}
                     plan_steps_set = {item['step'] for item in plan}
                     if not core_steps.issubset(plan_steps_set):
                          logger.warning(f"LLM plan for user query missing core reflection steps, falling back to default.")
                          return default_plan
                
                # Ensure required parameters are defaulted if possible/sensible 
                for step in plan:
                    if step['step'] == STEP_SEARCH_CODE_VECTOR:
                        step.setdefault('query', query or "Analyze code context") # Default query if user query is None
                        step.setdefault('use_keywords', True if extracted_keywords else False)
                    # Add more default logic here if needed for other steps
                         
                return plan
            else:
                logger.warning(f"LLM generated invalid plan structure for {run_type}. Falling back to default plan.")
                return default_plan if not is_proactive_run else []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON from LLM for {run_type}: {e}. Response: '{raw_response[:200]}...' Falling back to default plan.")
            return default_plan if not is_proactive_run else []
        except Exception as e:
            logger.error(f"Error during dynamic planning for {run_type}: {e}. Falling back to default plan.", exc_info=True)
            return default_plan if not is_proactive_run else [] 
import logging
from typing import List, Dict, Optional, Any
import json # For parsing potential JSON output from LLM

# LLM client import
from models.llm import LlmInterface, get_llm_client

logger = logging.getLogger(__name__)

# Define supported languages - useful for informing the LLM
SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "go",
    "csharp", # Use a consistent identifier
}

class CodeDifferError(Exception):
    """Custom exception for code differ errors."""
    pass

class CodeDiffer:
    """
    Compares two versions of code using an LLM to identify semantic differences.
    Relies on an external LLM service for the core comparison logic.
    """

    def __init__(self, llm_client: Optional[LlmInterface] = None):
        """
        Initializes the CodeDiffer.

        Args:
            llm_client: An optional instance of the LLM client.
                        If None, a default client will be instantiated.
        """
        # Initialize or receive LLM client
        self.llm_client = llm_client or get_llm_client()
        
        if self.llm_client is None:
            logger.error("Failed to initialize LLM client for CodeDiffer.")
            raise CodeDifferError("LLM client initialization failed. Cannot perform code comparisons.")

    def _construct_llm_prompt(self, code_before: str, code_after: str, language_name: str) -> str:
        """
        Constructs a prompt for the LLM to perform semantic code comparison.
        """
        # Basic prompt structure - This needs refinement based on LLM capabilities and desired output format
        prompt = f"""
Analyze the semantic differences between the following two code snippets written in {language_name}.
Focus on changes in logic, functionality, dependencies, function signatures, error handling, potential breaking changes, and overall intent. 
Ignore minor stylistic changes unless they affect behavior (e.g., changing variable names might be relevant if it obscures meaning).

Provide the output as a JSON object with the following structure:
{{
  "summary": "A brief overall summary of the changes (e.g., refactored function X, added feature Y, fixed bug Z).",
  "breaking_changes": ["List any potential breaking changes introduced."],
  "semantic_diffs": [
    {{
      "type": "MODIFIED | ADDED | REMOVED", 
      "scope": "e.g., function <n>, class <n>, module level",
      "description": "Detailed description of the semantic change.",
      "risk_assessment": "LOW | MEDIUM | HIGH (optional assessment)"
    }}
    // ... more diffs
  ]
}}

--- Code Before ---
```
{code_before}
```

--- Code After ---
```
{code_after}
```

--- Analysis (JSON format) ---
"""
        return prompt

    def _call_llm_for_diff(self, prompt: str) -> Dict[str, Any]:
        """
        Calls the LLM service to perform the semantic comparison.
        """
        if not self.llm_client:
             logger.error("LLM client is not available. Cannot perform comparison.")
             raise CodeDifferError("LLM client is not available. Cannot perform comparison.")

        logger.info("Sending prompt to LLM for semantic code diff...")
        try:
            # Use the LLM client to generate a response
            response_text = self.llm_client.generate(prompt, max_tokens=1500)
            
            # Attempt to parse the LLM response as JSON
            # Basic cleanup: find the start of the JSON object
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}')
            if json_start_index != -1 and json_end_index != -1:
                json_string = response_text[json_start_index:json_end_index+1]
                
                try:
                    diff_result = json.loads(json_string)
                    logger.info("Successfully parsed LLM response for code diff.")
                    
                    # Validate minimum expected structure is present
                    required_keys = ["summary", "semantic_diffs"]
                    for key in required_keys:
                        if key not in diff_result:
                            logger.warning(f"LLM response is missing required key: {key}. Adding empty default.")
                            if key == "summary":
                                diff_result[key] = "Unable to generate summary."
                            else:  # semantic_diffs
                                diff_result[key] = []
                    
                    # Ensure breaking_changes exists
                    if "breaking_changes" not in diff_result:
                        diff_result["breaking_changes"] = []
                        
                    return diff_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response. Attempting fallback parsing: {e}")
                    # Fallback: try to extract a more limited structure when JSON parsing fails
                    return self._fallback_result_extraction(response_text)
            else:
                logger.error(f"Could not find valid JSON object in LLM response. Attempting fallback parsing.")
                return self._fallback_result_extraction(response_text)
                
        except Exception as e:
            # Catch potential errors from the LLM client itself
            logger.error(f"Error communicating with LLM service: {e}", exc_info=True)
            raise CodeDifferError(f"Error during LLM communication: {e}") from e
    
    def _fallback_result_extraction(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback method to extract meaningful information when JSON parsing fails.
        """
        logger.info("Using fallback parsing for LLM response.")
        
        # Very basic fallback - look for summary-like sections in the text
        lines = response_text.split("\n")
        summary = "Unable to extract structured diff information."
        semantic_diffs = []
        breaking_changes = []
        
        for i, line in enumerate(lines):
            lower_line = line.lower()
            
            # Look for potential summary lines
            if "summary" in lower_line or "overview" in lower_line:
                if i+1 < len(lines) and lines[i+1].strip():
                    summary = lines[i+1].strip()
            
            # Look for breaking changes
            if "breaking" in lower_line:
                if "no breaking changes" in lower_line:
                    continue  # Skip lines saying there are no breaking changes
                clean_line = line.replace("*", "").replace("-", "").strip()
                if clean_line and not clean_line.lower().startswith("breaking"):
                    breaking_changes.append(clean_line)
            
            # Look for semantic differences
            if "added" in lower_line or "modified" in lower_line or "removed" in lower_line:
                clean_line = line.replace("*", "").replace("-", "").strip()
                if clean_line:
                    # Try to determine the type and description
                    if "added" in lower_line:
                        diff_type = "ADDED"
                    elif "removed" in lower_line or "deleted" in lower_line:
                        diff_type = "REMOVED"
                    else:
                        diff_type = "MODIFIED"
                        
                    semantic_diffs.append({
                        "type": diff_type,
                        "scope": "unknown",
                        "description": clean_line,
                        "risk_assessment": "MEDIUM"  # Default when uncertain
                    })
        
        return {
            "summary": summary,
            "breaking_changes": breaking_changes,
            "semantic_diffs": semantic_diffs
        }

    def compare_code(self, code_before: str, code_after: str, language_name: str) -> Dict[str, Any]:
        """
        Compares two strings of code in a specified language using an LLM 
        to identify semantic differences.

        Args:
            code_before: The original code content.
            code_after: The modified code content.
            language_name: The programming language (e.g., 'python', 'javascript').

        Returns:
            A dictionary representing the semantic differences identified by the LLM,
            following the structure requested in the prompt (summary, breaking_changes, semantic_diffs).

        Raises:
            CodeDifferError: If the language is unsupported, LLM communication fails,
                             or the response cannot be parsed.
        """
        language_lower = language_name.lower()
        if language_lower not in SUPPORTED_LANGUAGES:
            raise CodeDifferError(f"Unsupported language for semantic diffing: {language_name}")

        try:
            logger.info(f"Starting LLM-based semantic diff for language: {language_name}")
            prompt = self._construct_llm_prompt(code_before, code_after, language_lower)
            
            diff_result = self._call_llm_for_diff(prompt)
            
            logger.info(f"LLM-based semantic diff completed for {language_name}.")
            return diff_result

        except CodeDifferError as e:
            logger.error(f"Code Differ Error: {e}")
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Unexpected error during LLM code comparison: {e}", exc_info=True)
            raise CodeDifferError(f"Unexpected error during LLM code comparison: {e}") from e


# Example Usage (Updated for LLM approach)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Proper LLM client initialization
        llm_client = get_llm_client()
        differ = CodeDiffer(llm_client=llm_client)
    except Exception as e:
        logger.error(f"Failed to initialize proper LLM client for example: {e}")
        logger.info("Falling back to demo mode with mock responses for example.")
        
        # Create a minimal LLM client for demo purposes (mocks actual LLM calls)
        class MockLlmClient:
            def generate(self, prompt, max_tokens=None):
                # Return a canned response
                return """
                {
                  "summary": "The function was renamed from 'hello' to 'greet', parameter renamed from 'name' to 'person_name', and greeting message changed.",
                  "breaking_changes": ["Function name changed from 'hello' to 'greet'"],
                  "semantic_diffs": [
                    {
                      "type": "MODIFIED", 
                      "scope": "function signature",
                      "description": "Renamed function from 'hello' to 'greet'",
                      "risk_assessment": "HIGH"
                    },
                    {
                      "type": "MODIFIED", 
                      "scope": "parameter name",
                      "description": "Renamed parameter from 'name' to 'person_name'",
                      "risk_assessment": "MEDIUM"
                    },
                    {
                      "type": "MODIFIED", 
                      "scope": "print statement",
                      "description": "Changed greeting message from 'Hello' to 'Greetings'",
                      "risk_assessment": "LOW"
                    },
                    {
                      "type": "ADDED", 
                      "scope": "comments",
                      "description": "Added comment '# Say hi'",
                      "risk_assessment": "LOW"
                    }
                  ]
                }
                """
        
        differ = CodeDiffer(llm_client=MockLlmClient())
    
    # --- Python Example --- 
    print("\n--- Python Diff Example (LLM-based) ---")
    code_py_before = """
def hello(name):
    print(f"Hello, {name}!")
    return name
"""
    code_py_after = """
def greet(person_name):
    # Say hi
    print(f"Greetings, {person_name}!")
    return person_name
"""
    try:
        py_diff = differ.compare_code(code_py_before, code_py_after, "python")
        print("Python Diff Results (LLM Summary):")
        print(json.dumps(py_diff, indent=2))
    except CodeDifferError as e:
        print(f"Error comparing Python code: {e}")
    except Exception as e:
         print(f"Unexpected error: {e}")
         
    # --- JavaScript Example --- 
    print("\n--- JavaScript Diff Example (LLM-based) ---")
    code_js_before = """
function add(a, b) {
    return a + b;
}
console.log(add(1, 2));
"""
    code_js_after = """
function sum(x, y) {
    // Calculate sum
    return x + y;
}
console.log(sum(5, 10)); // Use different values
"""
    try:
        js_diff = differ.compare_code(code_js_before, code_js_after, "javascript")
        print("JavaScript Diff Results (LLM Summary):")
        print(json.dumps(js_diff, indent=2)) 
    except CodeDifferError as e:
        print(f"Error comparing JavaScript code: {e}")
    except Exception as e:
         print(f"Unexpected error: {e}")

    # --- Unsupported Language Example --- 
    print("\n--- Unsupported Language Example ---")
    try:
        differ.compare_code("a=1", "b=2", "lua")
    except CodeDifferError as e:
        print(f"Successfully caught error: {e}")
    except Exception as e:
         print(f"Unexpected error: {e}") 
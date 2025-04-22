import logging
import os
import yaml
import json
from typing import Optional, Dict, Any, List

# Assume an LLM interface exists
from models.llm import LlmInterface # Placeholder import

logger = logging.getLogger(__name__)

class LlmCodeTester:
    \"\"\"
    Uses a Large Language Model (LLM) to evaluate code snippets or examples 
    for correctness, adherence to best practices, or potential issues based on 
    predefined validation prompts.
    \"\"\"
    # Default path for validation prompts
    DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'prompts', 'validation_prompts.yaml')

    def __init__(self, llm_interface: LlmInterface, prompt_file: Optional[str] = None):
        \"\"\"
        Initializes the tester with an LLM interface and loads validation prompts.

        Args:
            llm_interface: An instance of the LLM interface for making calls.
            prompt_file: Optional path to the YAML file containing validation prompts.
                         Defaults to DEFAULT_PROMPT_FILE.
        \"\"\"
        if not isinstance(llm_interface, LlmInterface):
             raise TypeError("llm_interface must be an instance of LlmInterface")
        self.llm = llm_interface
        self.prompts = self._load_prompts(prompt_file or self.DEFAULT_PROMPT_FILE)

    def _load_prompts(self, file_path: str) -> Dict[str, str]:
        \"\"\"Loads validation prompts from a YAML file.\"\"\"
        try:
            # Ensure the directory exists before trying to open the file
            prompt_dir = os.path.dirname(file_path)
            if not os.path.exists(prompt_dir):
                 logger.warning(f"Prompt directory {prompt_dir} does not exist. Creating it.")
                 os.makedirs(prompt_dir)
                 # Create an empty file if it doesn't exist after creating the dir
                 if not os.path.exists(file_path):
                      with open(file_path, 'w', encoding='utf-8') as f:
                           f.write("# Validation Prompts\n")
                      logger.info(f"Created empty prompt file: {file_path}")
                 
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                if not isinstance(prompts, dict):
                    logger.warning(f"Prompt file {file_path} does not contain a dictionary or is empty.")
                    return {}
                return prompts
        except FileNotFoundError:
            # This case should be handled by the creation logic above, but kept as safeguard
            logger.error(f"Prompt file not found at {file_path} despite creation attempt. No prompts loaded.")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing validation prompt file {file_path}: {e}")
            return {}
        except Exception as e:
             logger.error(f"An unexpected error occurred loading validation prompts from {file_path}: {e}", exc_info=True)
             return {}

    def evaluate_code(self, 
                      prompt_key: str, 
                      code_snippet: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Optional[Dict[str, Any]]:
        \"\"\"
        Evaluates a code snippet using a specific validation prompt.

        Args:
            prompt_key: The key identifying the prompt template in the loaded YAML file.
            code_snippet: The code content to evaluate.
            context: Optional dictionary providing additional context (e.g., 
                     {'language': 'python', 'expected_behavior': 'handle errors gracefully'}).

        Returns:
            A dictionary containing the evaluation result (e.g., 
            {'passed': bool, 'feedback': str, 'confidence': float}), or None on failure.
            The exact structure depends on the LLM's response and the prompt design.
        \"\"\"
        if prompt_key not in self.prompts:
            logger.error(f"Error: Validation prompt key '{prompt_key}' not found.")
            return None
        
        prompt_template = self.prompts[prompt_key]
        formatted_prompt = prompt_template # Default if no formatting needed/possible
        
        # Prepare combined context for formatting
        full_context = {
            'code_snippet': code_snippet,
            **(context or {}) # Merge provided context
        }

        try:
            # Attempt to format the prompt using the full context
            formatted_prompt = prompt_template.format(**full_context)
        except KeyError as e:
            logger.warning(f"Missing key '{e}' in context/code_snippet for validation prompt '{prompt_key}'. Using template as is.")
            # Fallback: maybe the prompt only uses {code_snippet}? Or has no placeholders?
            try:
                 formatted_prompt = prompt_template.format(code_snippet=code_snippet)
            except KeyError:
                 logger.warning(f"Prompt '{prompt_key}' does not seem to use '{{code_snippet}}' either. Using raw template.")
                 formatted_prompt = prompt_template # Use raw template
            except Exception as fmt_e:
                 logger.error(f"Error formatting prompt '{prompt_key}' even with fallback: {fmt_e}")
                 return None
        except Exception as e:
            logger.error(f"Error formatting validation prompt '{prompt_key}': {e}")
            return None

        try:
            logger.info(f"Evaluating code using LLM with prompt key: '{prompt_key}'")
            raw_response = self.llm.generate(formatted_prompt)
            logger.debug(f"LLM raw response for '{prompt_key}': {raw_response}")
            
            # --- Response Parsing Logic --- 
            # This part heavily depends on how the prompts are designed. 
            # Aim for prompts that return structured output (e.g., JSON).
            evaluation_result = {
                'passed': None, # True, False, or None if undetermined
                'feedback': str(raw_response), # Default to raw response as feedback
                'confidence': None # Optional confidence score (0.0-1.0)
            }

            if isinstance(raw_response, str):
                raw_response_lower = raw_response.lower().strip()
                # Try simple keyword checks first
                if raw_response_lower.startswith('passed') or "::passed::" in raw_response_lower:
                     evaluation_result['passed'] = True
                elif raw_response_lower.startswith('failed') or "::failed::" in raw_response_lower:
                     evaluation_result['passed'] = False
                
                # Attempt to parse if it looks like JSON
                json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(0))
                        if isinstance(parsed_json, dict):
                            # Update result dict with keys from JSON if they exist
                            evaluation_result['passed'] = parsed_json.get('passed', evaluation_result['passed']) 
                            evaluation_result['feedback'] = parsed_json.get('feedback', evaluation_result['feedback'])
                            evaluation_result['confidence'] = parsed_json.get('confidence', evaluation_result['confidence'])
                            # Add any other fields from the JSON
                            evaluation_result.update({k:v for k,v in parsed_json.items() if k not in ['passed','feedback','confidence']})
                    except json.JSONDecodeError:
                        logger.warning(f"LLM response for '{prompt_key}' looked like JSON but failed to parse.")
                        # Keep raw response as feedback

            elif isinstance(raw_response, dict): # If LLM interface directly returns dict
                 evaluation_result['passed'] = raw_response.get('passed', evaluation_result['passed']) 
                 evaluation_result['feedback'] = raw_response.get('feedback', evaluation_result['feedback'])
                 evaluation_result['confidence'] = raw_response.get('confidence', evaluation_result['confidence'])
                 evaluation_result.update({k:v for k,v in raw_response.items() if k not in ['passed','feedback','confidence']})

            logger.info(f"Code evaluation finished. Passed: {evaluation_result['passed']}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error during LLM code evaluation call for prompt '{prompt_key}': {e}", exc_info=True)
            return None

# Placeholder for LlmInterface if not imported from elsewhere
import abc
import re # Need re for JSON parsing logic
class LlmInterface(abc.ABC):
     @abc.abstractmethod
     def generate(self, prompt: str, **kwargs) -> Any:
         pass

# Example Usage (requires a concrete LlmInterface and prompt file)
if __name__ == '__main__':
    import re # Ensure re is imported for the example section too
    logging.basicConfig(level=logging.INFO)
    print("Running LlmCodeTester Example (requires setup)")

    # 1. Dummy LLM
    class DummyLlmTester(LlmInterface):
        def generate(self, prompt: str, **kwargs) -> Any:
            print("-- Dummy LLM Tester received prompt snippet: --")
            print(prompt[:150] + "...")
            if "check_error_handling" in prompt:
                # Simulate returning JSON
                return '{\"passed\": false, \"feedback\": \"Error handling is missing for potential exceptions.\", \"confidence\": 0.8}'
            elif "check_thread_safety" in prompt:
                return "::passed:: Code appears thread-safe."
            else:
                return "Evaluation inconclusive."

    # 2. Dummy prompt file (validation_prompts.yaml)
    validation_prompt_content = '''
check_error_handling:
  """
  Analyze the following {language} code snippet for proper error handling, especially around potential exceptions or null values. 
  Expected behavior: {expected_behavior}
  
  Code Snippet:
  ```
  {code_snippet}
  ```
  
  Respond with ONLY a JSON object with keys "passed" (boolean), "feedback" (string explanation), and optionally "confidence" (float 0-1).
  """
check_thread_safety:
  """
  Assess the thread safety of the following {language} code snippet, particularly regarding shared resources or state.
  
  Code Snippet:
  ```
  {code_snippet}
  ```
  
  Respond with either '::passed:: [Explanation]' or '::failed:: [Explanation]'.
  """
'''
    validation_prompt_file = LlmCodeTester.DEFAULT_PROMPT_FILE
    os.makedirs(os.path.dirname(validation_prompt_file), exist_ok=True)
    with open(validation_prompt_file, 'w') as f:
        f.write(validation_prompt_content)
    print(f"Dummy validation prompt file created at: {validation_prompt_file}")

    # 3. Initialize and use
    try:
        tester = LlmCodeTester(llm_interface=DummyLlmTester())
        if not tester.prompts:
            print("Error: Validation prompts not loaded.")
        else:
            # Example 1: Check error handling
            print("\n--- Example 1: Check Error Handling ---")
            code1 = "try:\n  x = risky_operation()\nexcept: # Generic except\n  print('An error occurred')" 
            result1 = tester.evaluate_code(
                "check_error_handling", 
                code1, 
                context={'language': 'python', 'expected_behavior': 'Specific exceptions should be caught.'}
            )
            print(f"Evaluation Result 1: {result1}")

            # Example 2: Check thread safety
            print("\n--- Example 2: Check Thread Safety ---")
            code2 = "counter = 0\ndef increment():\n  global counter\n  counter += 1 # Potential race condition" 
            result2 = tester.evaluate_code(
                "check_thread_safety", 
                code2, 
                context={'language': 'python'}
            )
            print(f"Evaluation Result 2: {result2}")
            
    except Exception as e:
        print(f"Error running example: {e}")
    finally:
        # Clean up dummy prompt file
        if os.path.exists(validation_prompt_file):
             try:
                 os.remove(validation_prompt_file)
                 print("Dummy validation prompt file removed.")
             except OSError as e:
                 print(f"Error removing dummy validation prompt file: {e}") 
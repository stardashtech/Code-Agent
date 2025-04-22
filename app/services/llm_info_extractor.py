import logging
import yaml
import os
from typing import Dict, Any, Optional

# Import the call_llm function directly
from app.models.llm import call_llm 

logger = logging.getLogger(__name__)

# Default path for prompts relative to this file or project root
# Adjust if your structure is different
DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'prompts', 'extraction_prompts.yaml')

class LlmInfoExtractor:
    """
    Uses LLM prompts to extract specific information from provided text content.
    """
    def __init__(self, prompt_file_path: Optional[str] = None):
        """
        Initializes the extractor, loading prompts from a YAML file.

        Args:
            prompt_file_path: Path to the YAML file containing prompt templates.
                              Defaults to DEFAULT_PROMPT_FILE.
        """
        self.prompt_file_path = prompt_file_path or DEFAULT_PROMPT_FILE
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Loads prompts from the YAML file."""
        try:
            with open(self.prompt_file_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            if not isinstance(prompts, dict):
                 logger.error(f"Prompt file '{self.prompt_file_path}' does not contain a dictionary.")
                 return {}
            logger.info(f"Successfully loaded {len(prompts)} prompts from {self.prompt_file_path}.")
            return prompts
        except FileNotFoundError:
            logger.error(f"Prompt file not found: '{self.prompt_file_path}'")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing prompt file '{self.prompt_file_path}': {e}", exc_info=True)
            return {}
        except Exception as e:
             logger.error(f"Unexpected error loading prompts from '{self.prompt_file_path}': {e}", exc_info=True)
             return {}

    def _format_prompt(self, prompt_key: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Formats the selected prompt with the given context."""
        prompt_template = self.prompts.get(prompt_key)
        if not prompt_template:
            logger.error(f"Prompt key '{prompt_key}' not found in loaded prompts.")
            return None
            
        try:
            # Use basic .format() for simplicity, requires keys in template like {key}
            # Ensure context has all keys required by the template
            formatted_prompt = prompt_template.format(**(context or {}))
            return formatted_prompt
        except KeyError as e:
             logger.error(f"Missing key '{e}' in context for prompt '{prompt_key}'. Template: {prompt_template}")
             return None # Or return template with placeholders?
        except Exception as e:
             logger.error(f"Error formatting prompt '{prompt_key}': {e}", exc_info=True)
             return None

    def extract(self, prompt_key: str, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Extracts information using the specified prompt key and content.

        Args:
            prompt_key: The key corresponding to the desired prompt in the YAML file.
            content: The text content to analyze and extract information from.
            context: Optional dictionary for formatting the prompt template.

        Returns:
            The extracted information as a string, or None if an error occurs.
        """
        if not self.prompts:
            logger.error("Cannot extract: No prompts loaded.")
            return None
            
        # Prepare the context for formatting, ensuring 'content' is included
        full_context = context or {}
        full_context['content'] = content # Ensure the main content is available for the prompt
        
        formatted_prompt = self._format_prompt(prompt_key, full_context)
        if not formatted_prompt:
            return None # Error logged in _format_prompt
            
        logger.info(f"Sending extraction prompt (key: {prompt_key}) to LLM.")
        # logger.debug(f"Formatted prompt:\n{formatted_prompt}") # Optional: Log full prompt
        
        try:
            # Use the imported call_llm function
            llm_response = call_llm(formatted_prompt)
            logger.info(f"Received LLM response for extraction prompt '{prompt_key}'.")
            # logger.debug(f"LLM Response:\n{llm_response}")
            # Removed TODO: Parsing is context-dependent; caller or specific prompts should handle if JSON needed.
            return llm_response
        except RuntimeError as e:
             logger.error(f"LLM call failed during extraction for prompt '{prompt_key}': {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error during LLM call for prompt '{prompt_key}': {e}", exc_info=True)
            return None

# Example Usage (if run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Ensure config loads correctly and models.llm initializes client
    
    # Assume prompts YAML exists at the default location with a key like 'summarize'
    # Create a dummy prompts file if needed
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'prompts')
    DUMMY_PROMPT_FILE = os.path.join(PROMPT_DIR, 'extraction_prompts.yaml')
    if not os.path.exists(DUMMY_PROMPT_FILE):
         os.makedirs(PROMPT_DIR, exist_ok=True)
         dummy_prompts = {
             'summarize': 'Summarize the following content concisely:\n\n{content}',
             'find_usage': 'Find code examples showing how to use the function mentioned in the context ({function_name}) based on this documentation:\n\n{content}'
         }
         with open(DUMMY_PROMPT_FILE, 'w') as f:
             yaml.dump(dummy_prompts, f)
         logger.info(f"Created dummy prompt file: {DUMMY_PROMPT_FILE}")
         
    extractor = LlmInfoExtractor() # Uses default prompt file
    
    sample_content = """
    This is a long piece of documentation about a Python library called 'requests'.
    It allows you to send HTTP requests easily.
    Example:
    import requests
    response = requests.get('https://api.github.com')
    print(response.status_code)
    """
    
    print("--- Testing Summarization --- ")
    summary = extractor.extract('summarize', sample_content)
    if summary:
        print("Summary Result:")
        print(summary)
    else:
        print("Summarization failed. Check logs.")
        
    print("\n--- Testing Usage Finding --- ")
    usage = extractor.extract('find_usage', sample_content, context={'function_name': 'requests.get'})
    if usage:
         print("Usage Finding Result:")
         print(usage)
    else:
         print("Usage finding failed. Check logs.")

    # Clean up dummy file?
    # os.remove(DUMMY_PROMPT_FILE) # Optional

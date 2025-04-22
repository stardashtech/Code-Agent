import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Adjust import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.code_differ import CodeDiffer, CodeDifferError

# Mock LLM Client for testing
class MockLLMClient:
    def __init__(self, response_mode='success'):
        self.response_mode = response_mode

    def generate(self, prompt, **kwargs): # Simulate a generate method
        if self.response_mode == 'success':
            # Return a valid JSON string representing the diff
            mock_diff = {
                "summary": "Mock summary of changes.",
                "breaking_changes": ["Mock breaking change 1"],
                "semantic_diffs": [
                    {"type": "MODIFIED", "scope": "function mock_func", "description": "Mock description", "risk_assessment": "MEDIUM"}
                ]
            }
            # Simulate the structure expected by the parser in _call_llm_for_diff
            return f"Some text before... {json.dumps(mock_diff)} ...some text after"
        elif self.response_mode == 'invalid_json':
            return "This is not valid JSON { structure: "
        elif self.response_mode == 'no_json':
             return "There is no JSON object in this response."
        elif self.response_mode == 'error':
            raise Exception("Simulated LLM API error")
        else:
            return ""

class TestCodeDifferLLM(unittest.TestCase):

    def test_compare_code_success(self):
        """Test successful code comparison using the mock LLM client."""
        mock_client = MockLLMClient(response_mode='success')
        differ = CodeDiffer(llm_client=mock_client)
        code_before = "def func():\n  pass"
        code_after = "def func():\n  print('hello')"
        
        result = differ.compare_code(code_before, code_after, "python")
        
        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)
        self.assertIn("breaking_changes", result)
        self.assertIn("semantic_diffs", result)
        self.assertEqual(result["summary"], "Mock summary of changes.")
        self.assertEqual(len(result["semantic_diffs"]), 1)
        self.assertEqual(result["semantic_diffs"][0]["scope"], "function mock_func")

    def test_compare_code_unsupported_language(self):
        """Test error handling for unsupported languages."""
        differ = CodeDiffer(llm_client=MockLLMClient()) # Client doesn't matter here
        with self.assertRaisesRegex(CodeDifferError, "Unsupported language"):
            differ.compare_code("a", "b", "lua")

    def test_compare_code_llm_invalid_json_response(self):
        """Test handling when LLM returns non-JSON or invalid JSON."""
        mock_client_invalid = MockLLMClient(response_mode='invalid_json')
        differ_invalid = CodeDiffer(llm_client=mock_client_invalid)
        
        mock_client_no_json = MockLLMClient(response_mode='no_json')
        differ_no_json = CodeDiffer(llm_client=mock_client_no_json)

        with self.assertRaisesRegex(CodeDifferError, "Failed to parse LLM response as JSON"):
            differ_invalid.compare_code("a", "b", "python")
            
        with self.assertRaisesRegex(CodeDifferError, "LLM response did not contain a valid JSON object"):
            differ_no_json.compare_code("a", "b", "python")

    def test_compare_code_llm_api_error(self):
        """Test handling when the LLM client itself raises an error."""
        mock_client = MockLLMClient(response_mode='error')
        differ = CodeDiffer(llm_client=mock_client)
        
        # Wrap the underlying LLM client error
        with self.assertRaisesRegex(CodeDifferError, "Error during LLM communication"): 
            differ.compare_code("a", "b", "python")
            
    def test_compare_code_no_llm_client(self):
        """Test behavior when LLM client is not provided (should error or return error structure)."""
        # Current implementation logs a warning and returns an error structure inside _call_llm_for_diff
        # It doesn't raise an error in __init__ or compare_code directly if client is None.
        differ = CodeDiffer(llm_client=None)
        
        # Test the actual outcome based on current implementation (error structure in result)
        result = differ.compare_code("a", "b", "python") 
        self.assertEqual(result['summary'], "Error: LLM client not configured.")
        self.assertEqual(result['semantic_diffs'], [])
        
        # If we wanted it to raise an error earlier, we would uncomment the raise in __init__
        # and change this test:
        # with self.assertRaisesRegex(CodeDifferError, "LLM client is required"):
        #     differ = CodeDiffer(llm_client=None)

    def test_prompt_construction(self):
        """Verify the prompt structure is as expected."""
        differ = CodeDiffer(llm_client=MockLLMClient())
        code_before = "pass"
        code_after = "return True"
        language = "python"
        prompt = differ._construct_llm_prompt(code_before, code_after, language)
        
        self.assertIn(f"two code snippets written in {language}", prompt)
        self.assertIn("Provide the output as a JSON object", prompt)
        self.assertIn("--- Code Before ---", prompt)
        self.assertIn(code_before, prompt)
        self.assertIn("--- Code After ---", prompt)
        self.assertIn(code_after, prompt)
        self.assertIn("--- Analysis (JSON format) ---", prompt)
        self.assertIn("breaking_changes", prompt)
        self.assertIn("semantic_diffs", prompt)

if __name__ == '__main__':
    unittest.main() 
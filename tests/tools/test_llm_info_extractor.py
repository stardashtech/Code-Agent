import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import yaml
import json

# Adjust import path based on actual structure
from tools.llm_info_extractor import LlmInfoExtractor 
# Import the placeholder ABC for type hinting and dummy implementation
from tools.llm_info_extractor import LlmInterface 

# Dummy LLM Interface for testing
class DummyLlm(LlmInterface):
    def __init__(self, return_value="Default response"):
        self._return_value = return_value
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        if isinstance(self._return_value, Exception):
             raise self._return_value
        return self._return_value

# Sample prompt content
SAMPLE_PROMPTS_YAML = '''
extract_code:
  "Find the {language} code example in:\n{content}"
extract_json:
  "Extract data as JSON from:\n{content}"
no_content_placeholder:
   "This prompt has no content placeholder."
'''

class TestLlmInfoExtractor(unittest.TestCase):

    def test_init_success_and_prompt_loading(self):
        """Test successful initialization and loading prompts from file."""
        mock_llm = DummyLlm()
        # Use mock_open to simulate reading the YAML file
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm, prompt_file="dummy_path.yaml")
        
        self.assertIsInstance(extractor.llm, LlmInterface)
        self.assertEqual(len(extractor.prompts), 3)
        self.assertIn("extract_code", extractor.prompts)
        m_open.assert_called_once_with("dummy_path.yaml", 'r', encoding='utf-8')

    def test_init_prompt_file_not_found(self):
        """Test initialization when prompt file doesn't exist."""
        mock_llm = DummyLlm()
        m_open = mock_open()
        m_open.side_effect = FileNotFoundError("File not found")
        with patch("builtins.open", m_open):
             # Should still initialize but with empty prompts
             extractor = LlmInfoExtractor(llm_interface=mock_llm, prompt_file="bad_path.yaml") 
        self.assertEqual(extractor.prompts, {})

    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML content."""
        mock_llm = DummyLlm()
        m_open = mock_open(read_data="key: value: invalid yaml")
        with patch("builtins.open", m_open):
             extractor = LlmInfoExtractor(llm_interface=mock_llm, prompt_file="invalid.yaml")
        self.assertEqual(extractor.prompts, {})

    def test_extract_success_simple(self):
        """Test successful extraction with simple prompt and content."""
        mock_llm = DummyLlm(return_value="Extracted: Code Example")
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm)

        result = extractor.extract("extract_code", "Some documentation text", context={"language": "Python"})

        self.assertEqual(result, "Extracted: Code Example")
        # Check if the prompt was formatted correctly
        expected_prompt = SAMPLE_PROMPTS_YAML.split('\n')[1].split(':', 1)[1].strip().format(language="Python", content="Some documentation text")
        self.assertEqual(mock_llm.last_prompt, expected_prompt)

    def test_extract_success_json_parsing(self):
        """Test successful extraction where LLM returns JSON string."""
        json_string = '{"key": "value", "items": [1, 2]}'
        mock_llm = DummyLlm(return_value=json_string)
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm)

        result = extractor.extract("extract_json", "Some text to analyze")
        self.assertEqual(result, {"key": "value", "items": [1, 2]}) # Expect parsed dict

    def test_extract_prompt_key_not_found(self):
        """Test extraction attempt with a non-existent prompt key."""
        mock_llm = DummyLlm()
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm)
        result = extractor.extract("invalid_key", "content")
        self.assertIsNone(result)

    def test_extract_llm_failure(self):
        """Test extraction when the LLM call fails."""
        mock_llm = DummyLlm(return_value=Exception("LLM API Error"))
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm)
        result = extractor.extract("extract_code", "content", context={"language":"Go"})
        self.assertIsNone(result)
        
    def test_extract_formatting_error(self):
        """Test extraction when prompt formatting fails (missing context key)."""
        mock_llm = DummyLlm()
        m_open = mock_open(read_data=SAMPLE_PROMPTS_YAML)
        with patch("builtins.open", m_open):
            extractor = LlmInfoExtractor(llm_interface=mock_llm)
        # Missing 'language' in context for 'extract_code' prompt
        result = extractor.extract("extract_code", "content", context={}) 
        # The implementation currently falls back, let's test the fallback
        # It will try formatting with just {content}
        expected_fallback_prompt = SAMPLE_PROMPTS_YAML.split('\n')[1].split(':', 1)[1].strip().format(content="content")
        self.assertEqual(mock_llm.last_prompt, expected_fallback_prompt) 
        # Test case where {content} is missing from prompt
        result_no_content = extractor.extract("no_content_placeholder", "content")
        self.assertIsNone(result_no_content)

if __name__ == '__main__':
    unittest.main() 
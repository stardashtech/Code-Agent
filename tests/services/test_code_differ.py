import unittest
from unittest.mock import patch, MagicMock
import logging

# Adjust import path as needed
from services.code_differ import CodeDiffer, CodeDifferError, SUPPORTED_LANGUAGES

# Disable logging during tests unless specifically needed for debugging
logging.disable(logging.CRITICAL)

# Example code snippets for testing
CODE_PY_BEFORE = """
def hello(name):
    print(f"Hello, {name}!")
    return name
"""
CODE_PY_AFTER_RENAME = """
def greet(person_name):
    # Say hi
    print(f"Greetings, {person_name}!")
    return person_name
"""
CODE_PY_AFTER_ADD_LINE = """
def hello(name):
    print(f"Hello, {name}!")
    print("Another line") # Added line
    return name
"""
CODE_PY_IDENTICAL = CODE_PY_BEFORE

CODE_JS_BEFORE = """
function add(a, b) {
    return a + b;
}
"""
CODE_JS_AFTER = """
function subtract(a, b) {
    return a - b;
}
"""

class TestCodeDiffer(unittest.TestCase):

    def setUp(self):
        \"\"\"Initialize CodeDiffer before each test.\"\"\"
        # Mock the tree_sitter_languages functions to avoid actual grammar loading
        self.mock_language = MagicMock()
        self.mock_parser_instance = MagicMock()
        self.mock_tree = MagicMock()
        self.mock_root_node = MagicMock()
        # Simulate basic tree structure for diffing if needed
        self.mock_root_node.children = [] # Start with empty children
        self.mock_tree.root_node = self.mock_root_node
        self.mock_parser_instance.parse.return_value = self.mock_tree

        # Patch get_language and the Parser class instantiation
        self.patcher_get_language = patch('services.code_differ.get_language', return_value=self.mock_language)
        self.patcher_parser = patch('services.code_differ.Parser')
        
        self.mock_get_language = self.patcher_get_language.start()
        self.mock_parser_class = self.patcher_parser.start()
        # Ensure the Parser() call returns our mock instance
        self.mock_parser_class.return_value = self.mock_parser_instance 
        
        self.differ = CodeDiffer()
        # Reset mocks that might be called during init (though currently none are)
        self.mock_get_language.reset_mock()
        self.mock_parser_class.reset_mock()
        self.mock_parser_instance.reset_mock()
        
    def tearDown(self):
        self.patcher_get_language.stop()
        self.patcher_parser.stop()

    def test_init_parser_cache(self):
        \"\"\"Test that parsers are cached per language.\"\"\"
        # Call _get_parser twice for the same language
        parser1 = self.differ._get_parser("python")
        self.mock_get_language.assert_called_once_with('python')
        self.mock_parser_class.assert_called_once() # Parser class instantiated
        self.mock_parser_instance.set_language.assert_called_once_with(self.mock_language)
        
        # Reset mocks before second call to check caching
        self.mock_get_language.reset_mock()
        self.mock_parser_class.reset_mock()
        self.mock_parser_instance.reset_mock()
        
        parser2 = self.differ._get_parser("python")
        self.assertIs(parser1, parser2) # Should be the same instance
        # Ensure language/parser weren't loaded again
        self.mock_get_language.assert_not_called()
        self.mock_parser_class.assert_not_called()
        self.mock_parser_instance.set_language.assert_not_called()
        
        # Call for a different language
        js_parser = self.differ._get_parser("javascript")
        self.mock_get_language.assert_called_once_with('javascript')
        self.mock_parser_class.assert_called_once() # Parser class instantiated again
        self.mock_parser_instance.set_language.assert_called_with(self.mock_language) # Called with the new language mock
        self.assertIsNot(parser1, js_parser)

    def test_get_parser_unsupported_language(self):
        \"\"\"Test error handling for unsupported languages.\"\"\"
        with self.assertRaisesRegex(CodeDifferError, "Unsupported language"): 
            self.differ._get_parser("lua")

    @patch('services.code_differ.CodeDiffer._find_changed_nodes')
    def test_compare_code_identical(self, mock_find_changes):
        \"\"\"Test comparing identical code results in no changes.\"\"\"
        mock_find_changes.return_value = [] # Simulate no changes found by diff
        
        # Set up mock AST nodes (can be simple mocks)
        mock_node_before = MagicMock(spec=['children', 'type', 'text'])
        mock_node_before.children = []
        mock_node_after = MagicMock(spec=['children', 'type', 'text'])
        mock_node_after.children = []
        
        # Mock _parse_code to return these nodes
        with patch.object(self.differ, '_parse_code', side_effect=[mock_node_before, mock_node_after]) as mock_parse:
            diff = self.differ.compare_code(CODE_PY_IDENTICAL, CODE_PY_IDENTICAL, "python")

        self.assertEqual(diff, [])
        # Ensure parser was obtained and code parsed twice
        self.mock_get_language.assert_called_once_with('python')
        self.assertEqual(mock_parse.call_count, 2)
        # Ensure the diff function was called with the root nodes
        mock_find_changes.assert_called_once_with(mock_node_before, mock_node_after)

    @patch('services.code_differ.CodeDiffer._find_changed_nodes')
    def test_compare_code_with_changes(self, mock_find_changes):
        \"\"\"Test comparing code with differences returns changes from _find_changed_nodes.\"\"\"
        expected_diff = [{'change_type': 'delete', 'node': {'type': 'identifier', 'text': 'hello'}},
                         {'change_type': 'insert', 'node': {'type': 'identifier', 'text': 'greet'}}]
        mock_find_changes.return_value = expected_diff
        
        mock_node_before = MagicMock()
        mock_node_after = MagicMock()
        with patch.object(self.differ, '_parse_code', side_effect=[mock_node_before, mock_node_after]):
            diff = self.differ.compare_code(CODE_PY_BEFORE, CODE_PY_AFTER_RENAME, "python")
            
        self.assertEqual(diff, expected_diff)
        mock_find_changes.assert_called_once_with(mock_node_before, mock_node_after)
        
    # Test the basic _find_changed_nodes logic (might need more complex mocks)
    # This test is limited because mocking the exact node structure is complex
    def test_find_changed_nodes_basic(self):
        \"\"\"Test the basic recursive diff logic (limited test).\"\"\"
        # Create mock nodes with differing simple structures
        # Node 1: A -> B, C
        # Node 2: A -> B, D
        mock_node_b1 = MagicMock(type='b', text=b'b_text', children=[])
        mock_node_c = MagicMock(type='c', text=b'c_text', children=[])
        mock_root1 = MagicMock(type='a', text=b'a_text', children=[mock_node_b1, mock_node_c])
        
        mock_node_b2 = MagicMock(type='b', text=b'b_text', children=[]) # Same as b1
        mock_node_d = MagicMock(type='d', text=b'd_text', children=[])
        mock_root2 = MagicMock(type='a', text=b'a_text', children=[mock_node_b2, mock_node_d])
        
        # Patch _node_to_dict to return predictable dicts for comparison
        def mock_node_to_dict(node): 
             return {'type': node.type, 'text': node.text.decode(), 'start_point': (0,0), 'end_point': (0,1)}
        
        with patch.object(self.differ, '_node_to_dict', side_effect=mock_node_to_dict):
             changes = self.differ._find_changed_nodes(mock_root1, mock_root2)
        
        # Expect C deleted, D inserted (because B matched)
        self.assertEqual(len(changes), 2)
        change_types = [c['change_type'] for c in changes]
        node_types = [c['node']['type'] for c in changes]
        self.assertIn('delete', change_types)
        self.assertIn('insert', change_types)
        self.assertIn('c', node_types)
        self.assertIn('d', node_types)

if __name__ == '__main__':
    unittest.main() 
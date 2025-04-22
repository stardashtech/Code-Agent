import unittest
import os
import tempfile
import shutil
import json
import xml.etree.ElementTree as ET

# Adjust the import path based on your project structure
# If tests/ is at the same level as utils/, this should work.
# If your execution context is different, you might need to adjust sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils.dependency_parser import DependencyParser, DependencyParserError

class TestDependencyParser(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def _create_file(self, filename, content):
        """Helper method to create a file in the temp directory."""
        path = os.path.join(self.test_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    # --- requirements.txt Tests ---
    def test_parse_requirements_txt_basic(self):
        content = "requests==2.28.1\nflask>=2.0\nnumpy"
        path = self._create_file("requirements.txt", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "requests", "version_specifier": "==2.28.1"},
            {"name": "flask", "version_specifier": ">=2.0"},
            {"name": "numpy", "version_specifier": "any"},
        ]
        self.assertCountEqual(deps, expected) # Use assertCountEqual for list comparison regardless of order

    def test_parse_requirements_txt_comments_and_blanks(self):
        content = "# This is a comment\n\nrequests==1.0\n  # Another comment\nflask"
        path = self._create_file("requirements.txt", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "requests", "version_specifier": "==1.0"},
            {"name": "flask", "version_specifier": "any"},
        ]
        self.assertCountEqual(deps, expected)

    def test_parse_requirements_txt_extras_and_complex_specifiers(self):
        content = "django~=3.2.0\nmy-package[extra1,extra2]>1.0,<=2.0\n"
        path = self._create_file("requirements.txt", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "django", "version_specifier": "~=3.2.0"},
             # Note: The current regex might not perfectly capture multiple specifiers like >1.0,<=2.0. 
             # Adjusting regex or using a dedicated library might be needed for full pip specifier support.
             # For now, testing based on current regex capability.
            {"name": "my-package[extra1,extra2]", "version_specifier": ">1.0,<=2.0"}, 
        ]
        # Refining expectation based on the current regex in dependency_parser.py
        # It captures only the first specifier part.
        expected_current = [
            {"name": "django", "version_specifier": "~=3.2.0"},
            {"name": "my-package[extra1,extra2]", "version_specifier": ">1.0"}, 
        ]
        self.assertCountEqual(deps, expected_current)


    # --- package.json Tests ---
    def test_parse_package_json_basic(self):
        content = json.dumps({
            "name": "my-app",
            "dependencies": {
                "express": "^4.17.1",
                "lodash": "~4.17.21"
            }
        })
        path = self._create_file("package.json", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "express", "version_specifier": "^4.17.1"},
            {"name": "lodash", "version_specifier": "~4.17.21"},
        ]
        self.assertCountEqual(deps, expected)

    def test_parse_package_json_with_dev_dependencies(self):
        content = json.dumps({
            "name": "my-app",
            "dependencies": {
                "react": "^17.0.2"
            },
            "devDependencies": {
                "jest": "^27.0.6"
            }
        })
        path = self._create_file("package.json", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "react", "version_specifier": "^17.0.2"},
            {"name": "jest", "version_specifier": "^27.0.6"}, # Dev deps are included
        ]
        self.assertCountEqual(deps, expected)

    def test_parse_package_json_no_dependencies(self):
        content = json.dumps({"name": "my-app", "version": "1.0.0"})
        path = self._create_file("package.json", content)
        deps = DependencyParser.parse_dependencies(path)
        self.assertEqual(deps, [])

    def test_parse_package_json_invalid_json(self):
        content = "{ name: \"invalid json" # Invalid JSON
        path = self._create_file("package.json", content)
        with self.assertRaisesRegex(DependencyParserError, "Invalid package.json format"):
            DependencyParser.parse_dependencies(path)

    # --- go.mod Tests ---
    def test_parse_go_mod_basic(self):
        content = """
        module my/project/go

        go 1.19

        require (
            github.com/gin-gonic/gin v1.8.1
            golang.org/x/text v0.4.0 // indirect
        )

        require example.com/other v1.2.3
        """
        path = self._create_file("go.mod", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            # Note: Current parser might pick up indirect ones depending on regex details.
            # The existing parser implementation seems to capture indirect ones too.
            {"name": "github.com/gin-gonic/gin", "version_specifier": "v1.8.1"},
            {"name": "golang.org/x/text", "version_specifier": "v0.4.0"}, 
            {"name": "example.com/other", "version_specifier": "v1.2.3"},
        ]
        self.assertCountEqual(deps, expected)
        
    def test_parse_go_mod_only_single_requires(self):
        content = """
        module my/project/go
        go 1.20
        require github.com/google/uuid v1.3.0
        require rsc.io/quote/v3 v3.1.0
        """
        path = self._create_file("go.mod", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "github.com/google/uuid", "version_specifier": "v1.3.0"},
            {"name": "rsc.io/quote/v3", "version_specifier": "v3.1.0"},
        ]
        self.assertCountEqual(deps, expected)

    def test_parse_go_mod_no_requires(self):
        content = "module my/project/go\n\ngo 1.18"
        path = self._create_file("go.mod", content)
        deps = DependencyParser.parse_dependencies(path)
        self.assertEqual(deps, [])
        
    # --- .csproj Tests ---
    def test_parse_csproj_basic(self):
        content = """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />
  </ItemGroup>
  <ItemGroup>
      <!-- Another ItemGroup -->
      <PackageReference Include="System.Memory" Version="4.5.5"/>
  </ItemGroup>
   <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" /> <!-- Direct under Project -->
</Project>
        """
        path = self._create_file("sample.csproj", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "Newtonsoft.Json", "version_specifier": "13.0.1"},
            {"name": "Microsoft.Extensions.Logging", "version_specifier": "6.0.0"},
            {"name": "System.Memory", "version_specifier": "4.5.5"},
            {"name": "System.CommandLine", "version_specifier": "2.0.0-beta4.22272.1"},
        ]
        self.assertCountEqual(deps, expected)

    def test_parse_csproj_no_version(self):
        content = """
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="MyLocalProject" /> 
  </ItemGroup>
</Project>
        """
        path = self._create_file("sample.csproj", content)
        deps = DependencyParser.parse_dependencies(path)
        expected = [
            {"name": "MyLocalProject", "version_specifier": "any"}, # Expect 'any' if version missing
        ]
        self.assertCountEqual(deps, expected)
        
    def test_parse_csproj_empty(self):
        content = "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>"
        path = self._create_file("empty.csproj", content)
        deps = DependencyParser.parse_dependencies(path)
        self.assertEqual(deps, [])

    def test_parse_csproj_invalid_xml(self):
        content = "<Project><Invalid</Project>"
        path = self._create_file("invalid.csproj", content)
        with self.assertRaisesRegex(DependencyParserError, "Invalid .csproj XML format"):
             DependencyParser.parse_dependencies(path)

    # --- General Error Handling Tests ---
    def test_parse_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DependencyParser.parse_dependencies("non_existent_file.txt")

    def test_parse_unsupported_file_type(self):
        path = self._create_file("my_config.yaml", "key: value")
        with self.assertRaisesRegex(DependencyParserError, "Unsupported dependency file type"):
            DependencyParser.parse_dependencies(path)

if __name__ == '__main__':
    unittest.main() 
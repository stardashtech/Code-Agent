import unittest
import tempfile
import os
import json
import xml.etree.ElementTree as ET

# Adjust import path based on project structure
from utils.dependency_parser import DependencyParser, DependencyParserError

# Define expected outputs for comparison
EXPECTED_REQS = [
    {'name': 'requests', 'version_specifier': '==2.28.1'},
    {'name': 'flask', 'version_specifier': '>=2.0'},
    {'name': 'django', 'version_specifier': '~=3.2.0'},
    {'name': 'numpy', 'version_specifier': 'any'},
    {'name': 'pandas', 'version_specifier': '<=1.5.0'},
    {'name': 'my-package[extra1,extra2]', 'version_specifier': '>1.0'}
]

EXPECTED_PKG_JSON = [
    {'name': 'express', 'version_specifier': '^4.17.1'},
    {'name': 'lodash', 'version_specifier': '~4.17.21'},
    {'name': 'jest', 'version_specifier': '^27.0.6'} # Includes devDep
]

EXPECTED_GO_MOD = [
    {'name': 'example.com/other', 'version_specifier': 'v1.2.3'}, # Single require first
    {'name': 'github.com/gin-gonic/gin', 'version_specifier': 'v1.8.1'},
    {'name': 'golang.org/x/text', 'version_specifier': 'v0.4.0'} # Found in block
]

EXPECTED_CSPROJ = [
    {'name': 'Newtonsoft.Json', 'version_specifier': '13.0.1'},
    {'name': 'Microsoft.Extensions.Logging', 'version_specifier': '6.0.0'},
    {'name': 'System.CommandLine', 'version_specifier': '2.0.0-beta4.22272.1'} # Direct under Project
]


class TestDependencyParser(unittest.TestCase):

    def setUp(self):
        \"\"\"Create temporary files for testing.\"\"\"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = self.temp_dir.name

        # Create requirements.txt
        self.req_path = os.path.join(self.dir_path, "requirements.txt")
        with open(self.req_path, "w") as f:
            f.write("# This is a comment\n")
            f.write("requests==2.28.1\n")
            f.write("flask>=2.0\n")
            f.write("  django~=3.2.0 # Inline comment\n")
            f.write("numpy\n")
            f.write("pandas<=1.5.0\n")
            f.write("my-package[extra1,extra2]>1.0\n")
            f.write("invalid-line-format\n") # Should be ignored/warned

        # Create package.json
        self.pkg_json_path = os.path.join(self.dir_path, "package.json")
        with open(self.pkg_json_path, "w") as f:
            json.dump({
                "name": "my-app",
                "version": "1.0.0",
                "dependencies": {
                    "express": "^4.17.1",
                    "lodash": "~4.17.21"
                },
                "devDependencies": {
                    "jest": "^27.0.6"
                }
            }, f, indent=2)

        # Create go.mod
        self.go_mod_path = os.path.join(self.dir_path, "go.mod")
        with open(self.go_mod_path, "w") as f:
            f.write("module my/project/go\n\n")
            f.write("go 1.19\n\n")
            f.write("require example.com/other v1.2.3\n") # Single require first
            f.write("require (\n")
            f.write("    github.com/gin-gonic/gin v1.8.1\n")
            f.write("    golang.org/x/text v0.4.0 // indirect\n")
            f.write("    github.com/gin-gonic/gin v1.7.0 // Duplicate, should be ignored\n") # Test deduplication
            f.write(")\n")
            
        # Create sample.csproj
        self.csproj_path = os.path.join(self.dir_path, "sample.csproj")
        with open(self.csproj_path, "w") as f:
            f.write('\ufeff<Project Sdk="Microsoft.NET.Sdk">\n') # Include BOM
            f.write('  <PropertyGroup>\n')
            f.write('    <OutputType>Exe</OutputType>\n')
            f.write('    <TargetFramework>net6.0</TargetFramework>\n')
            f.write('  </PropertyGroup>\n\n')
            f.write('  <ItemGroup>\n')
            f.write('    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />\n')
            f.write('    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />\n')
            f.write('    <PackageReference Include="NoVersionPackage" />\n') # Test no version
            f.write('  </ItemGroup>\n')
            f.write('  <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" />\n') # Direct under Project
            f.write('</Project>\n')

    def tearDown(self):
        \"\"\"Clean up temporary directory.\"\"\"
        self.temp_dir.cleanup()

    def test_parse_requirements_txt(self):
        deps = DependencyParser.parse_dependencies(self.req_path)
        self.assertCountEqual(deps, EXPECTED_REQS) # Use assertCountEqual for list comparison regardless of order
        
    def test_parse_package_json(self):
        deps = DependencyParser.parse_dependencies(self.pkg_json_path)
        self.assertCountEqual(deps, EXPECTED_PKG_JSON)
        
    def test_parse_go_mod(self):
        deps = DependencyParser.parse_dependencies(self.go_mod_path)
        self.assertCountEqual(deps, EXPECTED_GO_MOD)
        
    def test_parse_csproj(self):
        deps = DependencyParser.parse_dependencies(self.csproj_path)
        # Add the package with no version to expected
        expected_csproj_with_no_version = EXPECTED_CSPROJ + [{'name': 'NoVersionPackage', 'version_specifier': 'any'}]
        self.assertCountEqual(deps, expected_csproj_with_no_version)
        
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DependencyParser.parse_dependencies(os.path.join(self.dir_path, "nonexistent.txt"))
            
    def test_unsupported_file_type(self):
        unsupported_path = os.path.join(self.dir_path, "unsupported.yaml")
        with open(unsupported_path, "w") as f: f.write("data: true")
        with self.assertRaisesRegex(DependencyParserError, "Unsupported dependency file type"): 
            DependencyParser.parse_dependencies(unsupported_path)
            
    def test_invalid_package_json(self):
         invalid_json_path = os.path.join(self.dir_path, "invalid.json")
         with open(invalid_json_path, "w") as f: f.write("{\"key\": \"value\",}") # Trailing comma
         with self.assertRaisesRegex(DependencyParserError, "Invalid package.json format"): 
             DependencyParser.parse_dependencies(invalid_json_path)
             
    def test_invalid_csproj(self):
         invalid_csproj_path = os.path.join(self.dir_path, "invalid.csproj")
         with open(invalid_csproj_path, "w") as f: f.write("<Project><MissingTag></Project>") # Malformed XML
         with self.assertRaisesRegex(DependencyParserError, "Invalid .csproj XML format"): 
             DependencyParser.parse_dependencies(invalid_csproj_path)

if __name__ == '__main__':
    unittest.main() 
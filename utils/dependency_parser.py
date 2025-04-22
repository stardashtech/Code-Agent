import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Configure basic logging if not configured by application root
# This prevents 'No handler found' warnings if the script is run standalone
# or if the main app doesn't configure logging early enough.
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DependencyParserError(Exception):
    """Custom exception for dependency parsing errors."""
    pass

class DependencyParser:
    """
    Parses dependency files for various languages to extract package names and versions.
    Supported files:
    - requirements.txt (Python)
    - package.json (Node.js/JS/TS)
    - go.mod (Go)
    - .csproj (C#/.NET)
    """

    @staticmethod
    def _parse_requirements_txt(content: str) -> List[Dict[str, str]]:
        """Parses requirements.txt content."""
        dependencies = []
        # Regex to capture package name and optional version specifier
        # Handles various specifiers (==, >=, <=, ~=, >) and ignores comments/blanks/hashes
        # Allows for extras like [security]
        req_pattern = re.compile(r'^\s*([a-zA-Z0-9_.-]+(?:\[[a-zA-Z0-9_,-]+\])?)\s*([!=<>~]=?\s*[0-9a-zA-Z_.*+-]+(?:\s*,\s*[!=<>~]=?\s*[0-9a-zA-Z_.*+-]+)*)?\s*(?:#.*)?$')
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'): # Ignore comments, blanks, options
                continue
            
            match = req_pattern.match(line)
            if match:
                name = match.group(1).strip()
                version_specifier = match.group(2) if match.group(2) else "any" 
                dependencies.append({"name": name, "version_specifier": version_specifier.strip()})
            else:
                 # Warn about lines that couldn't be parsed but aren't comments/blanks
                 if line: logger.warning(f"Could not parse line in requirements.txt: {line}")
        return dependencies

    @staticmethod
    def _parse_package_json(content: str) -> List[Dict[str, str]]:
        """Parses package.json content."""
        dependencies = []
        try:
            data = json.loads(content)
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {}) # Include dev dependencies as well? Decide based on need.
            
            all_deps = {**deps, **dev_deps} # Merge both dependency types
            
            for name, version_specifier in all_deps.items():
                dependencies.append({"name": name, "version_specifier": version_specifier})
        except json.JSONDecodeError as e:
            raise DependencyParserError(f"Invalid package.json format: {e}")
        return dependencies

    @staticmethod
    def _parse_go_mod(content: str) -> List[Dict[str, str]]:
        """Parses go.mod content (simplified)."""
        dependencies = []
        # Regex to find 'require' blocks and individual require lines
        # Handles single require lines and blocks `require (...)`
        require_block_pattern = re.compile(r'require\s+\((.*?)\)', re.DOTALL)
        single_require_pattern = re.compile(r'^\s*require\s+([\S]+)\s+([\S]+)', re.MULTILINE)
        module_in_block_pattern = re.compile(r'^\s*([^\s]+)\s+([^\s/]+)(?:\s*//\s*indirect)?', re.MULTILINE)

        # Find single require lines first
        for match in single_require_pattern.finditer(content):
             name = match.group(1).strip()
             version = match.group(2).strip()
             # Consider if indirect flag needs to be captured - TBD based on requirements
             dependencies.append({"name": name, "version_specifier": version})

        # Find require blocks
        for block_match in require_block_pattern.finditer(content):
            block_content = block_match.group(1)
            for line_match in module_in_block_pattern.finditer(block_content):
                 name = line_match.group(1).strip()
                 version = line_match.group(2).strip()
                 # Could add 'indirect': bool based on line content if needed
                 dependencies.append({"name": name, "version_specifier": version})
                 
        # Deduplicate based on name, keeping the first encountered (usually non-block ones if they exist)
        # A more robust approach might consider replace directives etc.
        seen_names = set()
        unique_dependencies = []
        for dep in dependencies:
             if dep['name'] not in seen_names:
                  unique_dependencies.append(dep)
                  seen_names.add(dep['name'])
                  
        return unique_dependencies

    @staticmethod
    def _parse_csproj(content: str) -> List[Dict[str, str]]:
        """Parses .csproj content for PackageReference items."""
        dependencies = []
        try:
            # Remove potential byte order mark (BOM) if present
            if content.startswith('\ufeff'):
                content = content[1:]
            root = ET.fromstring(content)
            # Find all ItemGroup elements, then look for PackageReference within them
            # PackageReference might exist directly under Project too, but ItemGroup is common
            for item_group in root.findall('.//ItemGroup'):
                 for pkg_ref in item_group.findall('PackageReference'):
                     name = pkg_ref.get("Include")
                     version = pkg_ref.get("Version") # Version attribute
                     if name: # Name is mandatory
                         dependencies.append({"name": name, "version_specifier": version or "any"})
                         
            # Also check for PackageReference directly under the Project root (less common)
            for pkg_ref in root.findall('PackageReference'):
                 name = pkg_ref.get("Include")
                 version = pkg_ref.get("Version")
                 if name and not any(d['name'] == name for d in dependencies): # Avoid duplicates
                     dependencies.append({"name": name, "version_specifier": version or "any"})
                     
        except ET.ParseError as e:
            raise DependencyParserError(f"Invalid .csproj XML format: {e}")
        except Exception as e: # Catch other potential errors during parsing
             raise DependencyParserError(f"Error parsing .csproj file: {e}")
        return dependencies

    @staticmethod
    def parse_dependencies(file_path: str) -> List[Dict[str, str]]:
        """
        Detects the file type based on the path and parses dependencies.

        Args:
            file_path: Path to the dependency file.

        Returns:
            A list of dictionaries, each containing 'name' and 'version_specifier'.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            DependencyParserError: If the file type is unsupported or parsing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Dependency file not found: {file_path}")
        except Exception as e:
             raise DependencyParserError(f"Error reading file {file_path}: {e}")

        # Use os.path.basename for robust path handling across OS
        filename = os.path.basename(file_path).lower() 

        if filename == "requirements.txt":
            return DependencyParser._parse_requirements_txt(content)
        elif filename == "package.json":
            return DependencyParser._parse_package_json(content)
        elif filename == "go.mod":
            return DependencyParser._parse_go_mod(content)
        elif filename.endswith(".csproj"):
            return DependencyParser._parse_csproj(content)
        else:
            raise DependencyParserError(f"Unsupported dependency file type: {filename}")

# Example Usage (for manual testing)
if __name__ == '__main__':
    # Create dummy files for testing
    # Note: This will create files in the current directory where the script is run from
    # Consider using a temporary directory for real testing frameworks
    
    # requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("# This is a comment\n")
        f.write("requests==2.28.1\n")
        f.write("flask>=2.0\n")
        f.write("  django~=3.2.0 # Inline comment\n")
        f.write("numpy\n")
        f.write("pandas<=1.5.0\n")
        f.write("my-package[extra1,extra2]>1.0\n")

    # package.json
    with open("package.json", "w") as f:
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

    # go.mod
    with open("go.mod", "w") as f:
        f.write("module my/project/go\n\n")
        f.write("go 1.19\n\n")
        f.write("require (\n")
        f.write("    github.com/gin-gonic/gin v1.8.1\n")
        f.write("    golang.org/x/text v0.4.0 // indirect\n")
        f.write(")\n")
        f.write("require example.com/other v1.2.3\n")


    # sample.csproj
    with open("sample.csproj", "w") as f:
        f.write('<Project Sdk="Microsoft.NET.Sdk">\n')
        f.write('  <PropertyGroup>\n')
        f.write('    <OutputType>Exe</OutputType>\n')
        f.write('    <TargetFramework>net6.0</TargetFramework>\n')
        f.write('  </PropertyGroup>\n\n')
        f.write('  <ItemGroup>\n')
        f.write('    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />\n')
        f.write('    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />\n')
        f.write('  </ItemGroup>\n')
        f.write('  <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" />\n') # Direct under Project
        f.write('</Project>\n')

    print("--- Testing requirements.txt ---")
    try:
        deps = DependencyParser.parse_dependencies("requirements.txt")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing package.json ---")
    try:
        deps = DependencyParser.parse_dependencies("package.json")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing go.mod ---")
    try:
        deps = DependencyParser.parse_dependencies("go.mod")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing sample.csproj ---")
    try:
        deps = DependencyParser.parse_dependencies("sample.csproj")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n--- Testing unsupported file ---")
    try:
        with open("unsupported.yaml", "w") as f: f.write("data: true")
        deps = DependencyParser.parse_dependencies("unsupported.yaml")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing non-existent file ---")
    try:
        deps = DependencyParser.parse_dependencies("nonexistent.txt")
        print(json.dumps(deps, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        
    # Clean up dummy files
    import os
    try:
        os.remove("requirements.txt")
        os.remove("package.json")
        os.remove("go.mod")
        os.remove("sample.csproj")
        os.remove("unsupported.yaml")
        print("\nDummy files removed.")
    except OSError as e:
        print(f"Error removing dummy files: {e}")

    # --- Test go.mod ---
    dummy_gomod_content = """
module example.com/myproject

go 1.16

require (
	github.com/gin-gonic/gin v1.7.4
	golang.org/x/sync v0.0.0-20210220032951-036812b2e83c // indirect comment
    example.com/other v1.0.0
    // some comment
)

require github.com/spf13/cobra v1.2.1

replace example.com/other => ../other
"""
    dummy_gomod_path = "dummy_go.mod"
    with open(dummy_gomod_path, "w") as f:
        f.write(dummy_gomod_content)

    print(f"\n--- Parsing {dummy_gomod_path} ---")
    parsed_deps_go = DependencyParser.parse_dependencies(dummy_gomod_path)
    print("Parsed Dependencies (go.mod):")
    for dep in parsed_deps_go:
        print(dep)
    os.remove(dummy_gomod_path)

    print("\n--- Parsing non-existent file ---")
    DependencyParser.parse_dependencies("non_existent_reqs.txt")

    # Consider using a temporary directory for real testing frameworks
    temp_dir = tempfile.mkdtemp() # Create a temp dir for test files
    try:
        req_path = os.path.join(temp_dir, "requirements.txt")
        pkg_path = os.path.join(temp_dir, "package.json")
        go_path = os.path.join(temp_dir, "go.mod")
        csproj_path = os.path.join(temp_dir, "sample.csproj")

        with open(req_path, "w") as f:
            f.write("# This is a comment\n")
            f.write("requests==2.28.1\n")
            f.write("flask>=2.0\n")
            f.write("  django~=3.2.0 # Inline comment\n")
            f.write("numpy\n")
            f.write("pandas<=1.5.0\n")
            f.write("my-package[extra1,extra2]>1.0\n")

        with open(pkg_path, "w") as f:
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

        with open(go_path, "w") as f:
            f.write("module my/project/go\n\n")
            f.write("go 1.19\n\n")
            f.write("require (\n")
            f.write("    github.com/gin-gonic/gin v1.8.1\n")
            f.write("    golang.org/x/text v0.4.0 // indirect\n")
            f.write(")\n")
            f.write("require example.com/other v1.2.3\n")

        with open(csproj_path, "w") as f:
            f.write('<Project Sdk="Microsoft.NET.Sdk">\n')
            f.write('  <PropertyGroup>\n')
            f.write('    <OutputType>Exe</OutputType>\n')
            f.write('    <TargetFramework>net6.0</TargetFramework>\n')
            f.write('  </PropertyGroup>\n\n')
            f.write('  <ItemGroup>\n')
            f.write('    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />\n')
            f.write('    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />\n')
            f.write('  </ItemGroup>\n')
            f.write('  <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" />\n') # Direct under Project
            f.write('</Project>\n')

        print("--- Testing requirements.txt ---")
        try:
            deps = DependencyParser.parse_dependencies(req_path) # Use path in temp dir
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Testing package.json ---")
        try:
            deps = DependencyParser.parse_dependencies(pkg_path) # Use path in temp dir
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Testing go.mod ---")
        try:
            deps = DependencyParser.parse_dependencies(go_path) # Use path in temp dir
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Testing sample.csproj ---")
        try:
            deps = DependencyParser.parse_dependencies(csproj_path) # Use path in temp dir
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            
        # Add tests for error cases
        print("\n--- Testing Non-Existent File ---")
        try:
            deps = DependencyParser.parse_dependencies(os.path.join(temp_dir, "nonexistent.txt"))
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Caught expected error: {e}")

        print("\n--- Testing Unsupported File ---")
        unsupported_path = os.path.join(temp_dir, "myconfig.yaml")
        with open(unsupported_path, "w") as f: f.write("some: yaml")
        try:
            deps = DependencyParser.parse_dependencies(unsupported_path)
            print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Caught expected error: {e}")
            
        # Invalid format tests (optional here, better in unit tests)
        print("\n--- Testing Invalid package.json ---")
        invalid_pkg_path = os.path.join(temp_dir, "invalid_package.json")
        with open(invalid_pkg_path, "w") as f: f.write("{ invalid json")
        try:
             deps = DependencyParser.parse_dependencies(invalid_pkg_path)
             print(json.dumps(deps, indent=2))
        except Exception as e:
            print(f"Caught expected error: {e}")

    finally:
         # Clean up the temporary directory and files
         import shutil
         shutil.rmtree(temp_dir)

# Note: The main block should be removed/replaced by proper unit tests.
# It is modified here to use a temporary directory for slightly cleaner execution. 
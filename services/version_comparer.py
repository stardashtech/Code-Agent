import logging
from typing import List, Dict, Any, Optional, Tuple
from packaging import version as packaging_version
import os

# Assuming these utilities and clients are available and functional
from utils.dependency_parser import DependencyParser, DependencyInfo, Language
from clients.pypi_client import PyPiClient
from clients.npm_client import NpmClient
from clients.go_proxy_client import GoProxyClient
from clients.nuget_client import NuGetClient
from interfaces.api_client import ApiClient  # For type hinting

logger = logging.getLogger(__name__)

class VersionComparisonResult(TypedDict):
    package_name: str
    local_version: Optional[str]
    latest_version: Optional[str]
    is_latest: Optional[bool] # True if local is latest, False if update available, None if comparison failed
    error: Optional[str] # Error message if comparison failed

class VersionComparer:
    """
    Compares local dependency versions parsed from project files against the 
    latest available versions from external package managers.
    """
    def __init__(self, 
                 pypi_client: Optional[PyPiClient] = None, 
                 npm_client: Optional[NpmClient] = None, 
                 go_client: Optional[GoProxyClient] = None, 
                 nuget_client: Optional[NuGetClient] = None):
        """
        Initializes the VersionComparer with API clients for different package managers.
        """
        self.parser = DependencyParser()
        self.clients: Dict[Language, Optional[ApiClient]] = {
            Language.PYTHON: pypi_client or PyPiClient(),
            Language.JAVASCRIPT: npm_client or NpmClient(),
            Language.TYPESCRIPT: npm_client or NpmClient(), # Use npm client for TS too
            Language.GO: go_client or GoProxyClient(),
            Language.CSHARP: nuget_client or NuGetClient(),
        }
        # Ensure all necessary clients are provided or instantiated
        # In a real app, consider dependency injection or a factory
        missing_clients = [lang.name for lang, client in self.clients.items() if client is None]
        if missing_clients:
            # For now, log a warning. Production might require stricter checks.
            logger.warning(f"VersionComparer initialized without clients for: {', '.join(missing_clients)}. Comparison for these languages will fail.")


    def _get_client_for_language(self, language: Language) -> Optional[ApiClient]:
        """Gets the appropriate API client based on the language."""
        client = self.clients.get(language)
        if client is None:
            logger.warning(f"No API client configured for language: {language.name}")
        return client

    def _safe_parse_version(self, version_str: Optional[str]) -> Optional[packaging_version.Version]:
        """Safely parses a version string, returning None on failure."""
        if not version_str:
            return None
        try:
            # Basic cleaning, might need refinement based on observed version formats
            cleaned_version = version_str.lstrip('v=^~<>') 
            return packaging_version.parse(cleaned_version)
        except packaging_version.InvalidVersion:
            logger.debug(f"Could not parse version for comparison: {version_str}")
            return None

    async def compare_project_dependencies(self, project_root: str) -> Dict[Language, List[VersionComparisonResult]]:
        """
        Parses dependencies for all supported languages found in the project root 
        and compares them with the latest versions.

        Args:
            project_root: The root directory of the project to analyze.

        Returns:
            A dictionary mapping each detected language to a list of 
            VersionComparisonResult for its dependencies.
        """
        results: Dict[Language, List[VersionComparisonResult]] = {}
        
        detected_deps = self.parser.parse_dependencies(project_root)

        if not detected_deps:
            logger.info(f"No supported dependency files found in {project_root}")
            return results

        for language, dependencies in detected_deps.items():
            if not dependencies:
                continue
                
            lang_results: List[VersionComparisonResult] = []
            client = self._get_client_for_language(language)

            if client is None:
                # Mark all dependencies for this language as failed
                for dep in dependencies:
                    lang_results.append(VersionComparisonResult(
                        package_name=dep.name,
                        local_version=dep.version_specifier,
                        latest_version=None,
                        is_latest=None,
                        error=f"No API client configured for {language.name}"
                    ))
                results[language] = lang_results
                continue

            # Fetch latest versions concurrently if possible (using asyncio might be better here)
            # For simplicity, doing it sequentially for now.
            for dep in dependencies:
                latest_version_str: Optional[str] = None
                error_msg: Optional[str] = None
                is_latest: Optional[bool] = None
                
                try:
                    # Fetch latest version from the appropriate client
                    # Note: Some clients might return complex objects, adapt as needed
                    # Assuming get_latest_version returns a simple string for now
                    latest_version_info = await client.get_latest_version(dep.name) 
                    
                    if isinstance(latest_version_info, dict): # Handle potential dict response
                        latest_version_str = latest_version_info.get('version')
                    elif isinstance(latest_version_info, str):
                         latest_version_str = latest_version_info
                    else:
                        logger.warning(f"Unexpected type for latest version of {dep.name} ({language.name}): {type(latest_version_info)}")
                        error_msg = "Could not determine latest version format."


                    if latest_version_str:
                        local_version = self._safe_parse_version(dep.version_specifier) # Parse local *specific* version if possible
                        latest_version = self._safe_parse_version(latest_version_str)

                        if local_version and latest_version:
                            # Direct comparison if both are valid versions
                            is_latest = local_version >= latest_version 
                        elif latest_version and not local_version:
                             # If local version is just a specifier (e.g., ^1.0.0), we can't definitively say it *is* latest
                             # but we know an update *might* be available if spec doesn't match latest.
                             # For simplicity, mark as not latest if local parse fails but latest exists.
                             is_latest = False 
                             logger.debug(f"Could not parse local version '{dep.version_specifier}' for {dep.name}, assuming not latest.")
                        elif not latest_version:
                            error_msg = f"Could not parse latest version '{latest_version_str}'."
                        # If only local_version exists, something is wrong upstream - treat as error
                        elif local_version and not latest_version:
                             error_msg = f"Local version '{local_version}' parsed, but failed to get/parse latest version."


                    else:
                       if not error_msg: # Only set error if not already set
                           error_msg = "Failed to fetch latest version from API."

                except Exception as e:
                    logger.error(f"Error comparing version for {dep.name} ({language.name}): {e}", exc_info=True)
                    error_msg = f"Exception during comparison: {e}"

                lang_results.append(VersionComparisonResult(
                    package_name=dep.name,
                    local_version=dep.version_specifier,
                    latest_version=latest_version_str,
                    is_latest=is_latest,
                    error=error_msg
                ))
            
            results[language] = lang_results

        return results

# Example Usage (requires running clients and potentially async context)
async def main():
    logging.basicConfig(level=logging.INFO)
    # Assume clients are configured (e.g., with API keys if needed)
    comparer = VersionComparer() 
    
    # Replace '.' with the actual path to a project you want to analyze
    project_path = '.' 
    if not os.path.exists(project_path):
         logger.error(f"Project path not found: {project_path}")
         return
         
    comparison_results = await comparer.compare_project_dependencies(project_path)

    for lang, lang_results in comparison_results.items():
        print(f"--- {lang.name} Dependencies ---")
        if not lang_results:
            print("  (No dependencies found or client not configured)")
            continue
        for result in lang_results:
            status = "Latest" if result['is_latest'] else ("Update Available" if result['is_latest'] is False else "Comparison Failed")
            error_info = f" (Error: {result['error']})" if result['error'] else ""
            print(f"  - {result['package_name']}: "
                  f"Local='{result['local_version']}', Latest='{result['latest_version']}' "
                  f"-> {status}{error_info}")

if __name__ == "__main__":
    import asyncio
    # Requires clients to be runnable without external dependencies for this simple test
    # In a real app, client instantiation would be more robust.
    try:
        asyncio.run(main())
    except Exception as e:
         logger.error(f"Error running example: {e}")
         print("\\nNOTE: Running this example directly might require specific setup "
               "(like installed clients, API keys, or a sample project structure).") 
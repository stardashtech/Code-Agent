import logging
from typing import List, Dict, Any, Optional, TypedDict

# Assuming these services and clients are available and functional
from services.version_comparer import VersionComparer, VersionComparisonResult, Language
from clients.pypi_client import PyPiClient
from clients.npm_client import NpmClient
from clients.go_proxy_client import GoProxyClient
from clients.nuget_client import NuGetClient
from interfaces.api_client import ApiClient
# Potentially add LLMInfoExtractor later if needed for best practices analysis

logger = logging.getLogger(__name__)

class ProactiveIssue(TypedDict):
    issue_type: str # e.g., 'outdated_dependency', 'vulnerability'
    severity: str # e.g., 'low', 'medium', 'high', 'critical'
    package_name: str
    language: str
    description: str
    details: Optional[Dict[str, Any]] # e.g., {'local_version': '1.0', 'latest_version': '1.1'} or vulnerability info

class ProactiveAnalyzer:
    """
    Analyzes project dependencies proactively to find issues like outdated packages
    and known vulnerabilities.
    """
    def __init__(self, 
                 version_comparer: Optional[VersionComparer] = None,
                 pypi_client: Optional[PyPiClient] = None, 
                 npm_client: Optional[NpmClient] = None, 
                 go_client: Optional[GoProxyClient] = None, 
                 nuget_client: Optional[NuGetClient] = None):
        """
        Initializes the ProactiveAnalyzer.
        """
        self.version_comparer = version_comparer or VersionComparer(
            pypi_client=pypi_client, 
            npm_client=npm_client, 
            go_client=go_client, 
            nuget_client=nuget_client
        )
        # Store clients needed for vulnerability checks
        self.clients: Dict[Language, Optional[ApiClient]] = {
            Language.PYTHON: pypi_client or PyPiClient(),
            Language.JAVASCRIPT: npm_client or NpmClient(),
            Language.TYPESCRIPT: npm_client or NpmClient(),
            Language.GO: go_client or GoProxyClient(),
            Language.CSHARP: nuget_client or NuGetClient(),
        }

    def _get_client_for_language(self, language: Language) -> Optional[ApiClient]:
        """Gets the appropriate API client based on the language."""
        client = self.clients.get(language)
        if client is None:
            logger.warning(f"ProactiveAnalyzer: No API client configured for language: {language.name} for vulnerability checks.")
        return client

    async def analyze_project(self, project_root: str) -> List[ProactiveIssue]:
        """
        Runs the proactive analysis on the specified project root.

        Args:
            project_root: The root directory of the project.

        Returns:
            A list of identified proactive issues.
        """
        issues: List[ProactiveIssue] = []
        logger.info(f"Starting proactive analysis for project: {project_root}")

        try:
            # 1. Compare versions
            comparison_results = await self.version_comparer.compare_project_dependencies(project_root)
            
            if not comparison_results:
                logger.info("No dependencies found or comparison failed.")
                # Continue to potentially check other things if implemented later

            # Process comparison results and check vulnerabilities
            for language, lang_results in comparison_results.items():
                client = self._get_client_for_language(language)
                
                for result in lang_results:
                    # Issue for outdated dependency
                    if result['is_latest'] is False and result['latest_version']:
                        issues.append(ProactiveIssue(
                            issue_type='outdated_dependency',
                            severity='medium', # Default severity, could be refined
                            package_name=result['package_name'],
                            language=language.name,
                            description=f"Package '{result['package_name']}' is outdated.",
                            details={
                                'local_version': result['local_version'],
                                'latest_version': result['latest_version']
                            }
                        ))
                    elif result['error']:
                         logger.warning(f"Skipping vulnerability check for {result['package_name']} ({language.name}) due to version comparison error: {result['error']}")
                         continue # Don't check vulnerabilities if version comparison failed
                    
                    # Issue for vulnerabilities (check even if latest, as latest might have vulns)
                    if client and result['local_version']: # Need client and a version to check
                        try:
                            # Use the local version for vulnerability check, as that's what's running
                            # We might need to parse/clean the local_version here if it's complex
                            # Re-use the safe parser from VersionComparer
                            version_to_check_str: Optional[str] = None
                            parsed_local = self.version_comparer._safe_parse_version(result['local_version'])
                            if parsed_local:
                                version_to_check_str = str(parsed_local)
                            else:
                                # Fallback: try to use the specifier directly if parsing failed?
                                # This might be risky depending on client expectations.
                                # Let's stick to only checking parsed versions for now.
                                logger.warning(f"Could not parse local version '{result['local_version']}' for vulnerability check of {result['package_name']}. Skipping check.")
                                continue # Skip if local version didn't parse

                            # Check if check_vulnerabilities is awaitable
                            # (Assuming clients follow a consistent async/sync pattern)
                            check_vuln_method = getattr(client, 'check_vulnerabilities', None)
                            if asyncio.iscoroutinefunction(check_vuln_method):
                                vulnerabilities = await check_vuln_method(result['package_name'], version_to_check_str)
                            elif callable(check_vuln_method):
                                # Handle potential synchronous client methods
                                vulnerabilities = check_vuln_method(result['package_name'], version_to_check_str)
                            else:
                                logger.warning(f"Client for {language.name} does not have a callable 'check_vulnerabilities' method.")
                                vulnerabilities = []
                                
                            for vuln in vulnerabilities:
                                # Try to determine severity (clients should ideally provide this)
                                sev = str(vuln.get('severity', 'unknown')).lower()
                                if sev not in ['low', 'medium', 'high', 'critical']:
                                    sev = 'medium' # Default if unknown/invalid
                                    
                                issues.append(ProactiveIssue(
                                    issue_type='vulnerability',
                                    severity=sev, 
                                    package_name=result['package_name'],
                                    language=language.name,
                                    description=f"Vulnerability found in '{result['package_name']} {result['local_version']}': {vuln.get('summary', 'No summary provided')}",
                                    details=vuln # Store full vulnerability details
                                ))
                                
                        except Exception as e:
                            logger.error(f"Error checking vulnerabilities for {result['package_name']} ({language.name}): {e}", exc_info=True)
                            # Optionally add an issue indicating vulnerability check failure
                            issues.append(ProactiveIssue(
                                issue_type='analysis_error',
                                severity='low',
                                package_name=result['package_name'],
                                language=language.name,
                                description=f"Failed to check vulnerabilities for package '{result['package_name']}'.",
                                details={'error': str(e)}
                            ))

            # TODO: Integrate other analysis types here (e.g., LLM for best practices)

        except Exception as e:
            logger.error(f"Failed to run proactive analysis on {project_root}: {e}", exc_info=True)
            issues.append(ProactiveIssue(
                issue_type='analysis_error',
                severity='high',
                package_name='project',
                language='unknown',
                description=f"Proactive analysis failed entirely: {e}",
                details={'error': str(e)}
            ))

        logger.info(f"Proactive analysis finished. Found {len(issues)} issues.")
        return issues

# Example Usage (similar setup needed as VersionComparer)
async def main():
    import asyncio
    import os 
    logging.basicConfig(level=logging.INFO)
    analyzer = ProactiveAnalyzer()
    
    project_path = '.' # Replace with actual project path
    if not os.path.exists(project_path):
         logger.error(f"Project path not found: {project_path}")
         return
         
    found_issues = await analyzer.analyze_project(project_path)

    if found_issues:
        print("--- Proactive Analysis Issues Found ---")
        for issue in found_issues:
            print(f"- Type: {issue['issue_type']}")
            print(f"  Severity: {issue['severity']}")
            print(f"  Language: {issue['language']}")
            print(f"  Package: {issue['package_name']}")
            print(f"  Description: {issue['description']}")
            if issue['details']:
                print(f"  Details: {issue['details']}")
            print("---")
    else:
        print("No proactive issues found.")

if __name__ == "__main__":
    import asyncio
    # Requires clients/services to be runnable without external dependencies for this simple test
    try:
        asyncio.run(main())
    except Exception as e:
         logger.error(f"Error running example: {e}", exc_info=True)
         print("\\nNOTE: Running this example directly might require specific setup.") 
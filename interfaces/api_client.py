import abc
from typing import List, Dict, Any, Optional

class ExternalApiClient(abc.ABC):
    """
    Abstract Base Class for clients interacting with external APIs 
    (e.g., code repositories, package managers).

    Defines a common interface for retrieving various types of information.
    """

    @abc.abstractmethod
    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for repositories based on a query and optional language filter.

        Args:
            query: The search query string.
            language: Optional language filter (e.g., 'python', 'javascript').

        Returns:
            A list of dictionaries, each representing a repository found. 
            The structure may vary depending on the specific API source.
        """
        pass

    @abc.abstractmethod
    def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Get the latest stable version string for a given package.

        Args:
            package_name: The name of the package.

        Returns:
            The latest version string (e.g., '1.2.3') or None if not found.
        """
        pass

    @abc.abstractmethod
    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        Fetch the content of a specific file from a repository.

        Args:
            repo_url: The URL or identifier of the repository.
            file_path: The path to the file within the repository.
            revision: Optional specific revision (commit SHA, branch, tag). Defaults to the default branch.

        Returns:
            The content of the file as a string, or None if not found or inaccessible.
        """
        pass

    @abc.abstractmethod
    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempt to find the primary documentation URL for a package.

        Args:
            package_name: The name of the package.
            version: Optional specific version to look for documentation.

        Returns:
            The URL of the documentation or None if not found.
        """
        pass
    
    @abc.abstractmethod
    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for code examples related to a library, function, or class. 
        This might involve searching repository code, issues, or documentation.

        Args:
            library_name: The name of the library/package.
            function_name: Optional specific function name.
            class_name: Optional specific class name.

        Returns:
            A list of dictionaries, each containing information about a found code example 
            (e.g., {'source_url': '...', 'snippet': '...', 'description': '...'}).
        """
        pass

    @abc.abstractmethod
    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Check for known vulnerabilities associated with a specific package version.

        Args:
            package_name: The name of the package.
            version: The specific version string to check.

        Returns:
            A list of dictionaries, each describing a found vulnerability 
            (e.g., {'id': 'CVE-...', 'severity': 'high', 'summary': '...', 'url': '...'}). 
            Returns an empty list if no vulnerabilities are found or the source doesn't provide this info.
        """
        pass 
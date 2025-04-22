import logging
import os
from typing import Optional, Tuple, List

try:
    import git
    from git.exc import InvalidGitRepositoryError, NoSuchPathError, GitCommandError
    GITPYTHON_INSTALLED = True
except ImportError:
    GITPYTHON_INSTALLED = False
    # Define dummy classes or variables if GitPython is not installed 
    # to allow the rest of the application to import this module gracefully.
    class InvalidGitRepositoryError(Exception): pass
    class NoSuchPathError(Exception): pass
    class GitCommandError(Exception): pass
    # Define a dummy Repo class perhaps?
    # class Repo:
    #     def __init__(self, path):
    #         raise NotImplementedError("GitPython not installed")
    # Or just rely on the GITPYTHON_INSTALLED flag.

logger = logging.getLogger(__name__)

def _get_repo(repo_path: str = '.') -> Optional['git.Repo']:
    """Helper to get the GitPython Repo object, handling errors."""
    if not GITPYTHON_INSTALLED:
        logger.warning("GitPython is not installed. Git functionality is disabled.")
        return None
    try:
        # Search parent directories for .git folder
        repo = git.Repo(repo_path, search_parent_directories=True)
        return repo
    except InvalidGitRepositoryError:
        logger.warning(f"Path '{repo_path}' is not part of a valid Git repository.")
        return None
    except NoSuchPathError:
        logger.error(f"Repository path '{repo_path}' does not exist.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing Git repository at '{repo_path}': {e}", exc_info=True)
        return None

def get_current_branch(repo_path: str = '.') -> Optional[str]:
    """Gets the name of the current active branch.

    Returns:
        The branch name, or None if in detached HEAD state or an error occurs.
    """
    repo = _get_repo(repo_path)
    if not repo:
        return None
    try:
        # Check if head is detached
        if repo.head.is_detached:
            logger.info(f"Git HEAD is detached at {repo.head.commit.hexsha[:7]}")
            return None # Or return the commit SHA?
        branch = repo.active_branch
        logger.debug(f"Current active Git branch: {branch.name}")
        return branch.name
    except TypeError as e:
         # Sometimes GitPython throws TypeError on detached head
         if 'detached HEAD' in str(e):
              logger.info(f"Git HEAD is detached (caught TypeError).")
              return None
         else:
              logger.error(f"Error getting current branch: {e}", exc_info=True)
              return None
    except Exception as e:
        logger.error(f"Error getting current branch: {e}", exc_info=True)
        return None

def has_uncommitted_changes(repo_path: str = '.') -> Optional[bool]:
    """
    Checks if there are uncommitted changes in the working directory or index.

    Returns:
        True if dirty, False if clean, None if an error occurs.
    """
    repo = _get_repo(repo_path)
    if not repo:
        return None
    try:
        is_dirty = repo.is_dirty(untracked_files=True)
        logger.debug(f"Repository dirty status (includes untracked): {is_dirty}")
        return is_dirty
    except Exception as e:
        logger.error(f"Error checking repository status: {e}", exc_info=True)
        return None

def get_file_last_commit(file_path: str, repo_path: str = '.') -> Optional[Tuple[str, str, str]]:
    """
    Gets details of the last commit that modified the specified file.

    Args:
        file_path: Path to the file relative to the repository root.
        repo_path: Path to the repository.

    Returns:
        A tuple (commit_hash, author_name, commit_message) or None if error/not found.
    """
    repo = _get_repo(repo_path)
    if not repo:
        return None
        
    # Ensure file_path is relative to repo root for git commands
    try:
         if os.path.isabs(file_path):
              relative_path = os.path.relpath(file_path, repo.working_dir)
         else:
              relative_path = file_path
              
         # Ensure the path uses forward slashes, as Git typically expects
         relative_path = relative_path.replace('\\', '/')
              
    except ValueError as e:
         logger.error(f"Could not determine relative path for {file_path} in repo {repo.working_dir}: {e}")
         return None

    try:
        logger.debug(f"Checking history for file: {relative_path}")
        # Get the log iterator for the specific file, max count 1
        commits = list(repo.iter_commits(paths=relative_path, max_count=1))
        
        if not commits:
            logger.warning(f"No commit history found for file: {relative_path}")
            return None
            
        last_commit = commits[0]
        commit_hash = last_commit.hexsha
        author_name = last_commit.author.name
        commit_message = last_commit.message.strip()
        
        logger.debug(f"Last commit for {relative_path}: {commit_hash[:7]} by {author_name}")
        return commit_hash, author_name, commit_message
        
    except Exception as e:
        logger.error(f"Error getting last commit for file '{relative_path}': {e}", exc_info=True)
        return None

# --- New Git Functions for Branching and Committing ---

def create_branch(branch_name: str, start_point: str = 'HEAD', repo_path: str = '.') -> bool:
    """Creates a new branch."""
    repo = _get_repo(repo_path)
    if not repo:
        return False
    try:
        if branch_name in repo.heads:
             logger.warning(f"Branch '{branch_name}' already exists.")
             # Decide if this should be an error or just a warning. Return True as it exists?
             # Let's return False to indicate it wasn't newly created.
             return False
        new_branch = repo.create_head(branch_name, commit=start_point)
        logger.info(f"Created new branch '{branch_name}' from '{start_point}'.")
        return True
    except Exception as e:
        logger.error(f"Error creating branch '{branch_name}': {e}", exc_info=True)
        return False

def checkout_branch(branch_name: str, repo_path: str = '.') -> bool:
    """Checks out the specified branch."""
    repo = _get_repo(repo_path)
    if not repo:
        return False
    try:
        if branch_name not in repo.heads:
             logger.error(f"Branch '{branch_name}' not found.")
             return False
             
        # Check for dirty state before checkout? Optional safety measure.
        # if repo.is_dirty(untracked_files=True):
        #     logger.error("Cannot checkout branch: repository has uncommitted changes.")
        #     return False
            
        repo.heads[branch_name].checkout()
        logger.info(f"Checked out branch '{branch_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error checking out branch '{branch_name}': {e}", exc_info=True)
        return False

def add_files(file_paths: List[str], repo_path: str = '.') -> bool:
    """Adds specified files to the Git index."""
    repo = _get_repo(repo_path)
    if not repo:
        return False
    try:
        # Convert relative paths to be relative to repo root if needed
        repo_root = repo.working_dir
        abs_file_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in file_paths]
        
        # Filter paths that actually exist
        existing_paths = [p for p in abs_file_paths if os.path.exists(p)]
        missing_paths = [p for p in file_paths if os.path.join(repo_root, p) not in existing_paths and os.path.join(repo_root, p) not in abs_file_paths] # Adjust logic if needed
        
        if missing_paths:
            logger.warning(f"Cannot add missing files: {missing_paths}")
            
        if not existing_paths:
             logger.warning("No existing files provided to add.")
             return False # Or True if adding nothing is success? Let's say False.

        logger.debug(f"Adding files to index: {existing_paths}")
        repo.index.add(existing_paths)
        return True
    except Exception as e:
        logger.error(f"Error adding files {file_paths} to index: {e}", exc_info=True)
        return False

def commit_changes(message: str, repo_path: str = '.') -> bool:
    """Commits the staged changes."""
    repo = _get_repo(repo_path)
    if not repo:
        return False
    try:
        if not repo.index.diff("HEAD"):
             logger.warning("No staged changes detected to commit.")
             return False # Nothing to commit

        repo.index.commit(message)
        logger.info(f"Committed changes with message: '{message}'")
        return True
    except Exception as e:
        logger.error(f"Error committing changes: {e}", exc_info=True)
        return False

def merge_branch(branch_to_merge: str, target_branch: Optional[str] = None, repo_path: str = '.', strategy_option: Optional[str] = None) -> bool:
    """
    Merges the specified branch into the target branch (defaults to current branch).
    Basic implementation: Does not handle merge conflicts automatically.
    """
    repo = _get_repo(repo_path)
    if not repo:
        return False
        
    original_branch = get_current_branch(repo_path)
    if not original_branch and not target_branch:
        logger.error("Cannot merge: Cannot determine current branch (detached HEAD?) and no target specified.")
        return False

    current_branch = target_branch or original_branch
    
    # Ensure we are on the target branch before merging
    if get_current_branch(repo_path) != current_branch:
        if not checkout_branch(current_branch, repo_path):
            logger.error(f"Merge failed: Could not checkout target branch '{current_branch}'.")
            return False
            
    try:
        if branch_to_merge not in repo.heads:
            logger.error(f"Merge failed: Branch to merge '{branch_to_merge}' not found.")
            return False
            
        logger.info(f"Attempting to merge branch '{branch_to_merge}' into '{current_branch}'...")
        
        # Use repo.git.merge for more control, potentially easier conflict handling if needed later
        merge_cmd_args = ['merge']
        if strategy_option:
            merge_cmd_args.extend(['-s', strategy_option])
        merge_cmd_args.append(branch_to_merge)
        
        repo.git.execute(merge_cmd_args) # Using execute directly for simplicity now
        
        # repo.git.merge(branch_to_merge) # Simpler way, might raise on conflict
        
        # Check for conflicts after merge attempt (basic check)
        if repo.is_dirty(): # A merge conflict often leaves the repo dirty
            logger.error(f"Merge resulted in conflicts between '{branch_to_merge}' and '{current_branch}'. Manual resolution needed.")
            # repo.git.merge('--abort') # Option to automatically abort? Requires careful state management.
            return False # Indicate merge failure due to conflicts

        logger.info(f"Successfully merged '{branch_to_merge}' into '{current_branch}'.")
        return True

    except GitCommandError as e:
         # Catch Git command errors, often indicates merge conflicts
         logger.error(f"Error merging branch '{branch_to_merge}' into '{current_branch}': {e}", exc_info=True)
         # Attempt to abort merge if possible? repo.git.merge('--abort')
         try:
              if "merge --abort" in str(e).lower(): # Avoid recursive abort
                  logger.warning("Merge abort command itself failed or was part of the error.")
              else:
                  logger.warning("Attempting to abort merge due to error.")
                  repo.git.merge('--abort')
         except GitCommandError as abort_e:
              logger.error(f"Failed to abort merge after error: {abort_e}")
         return False
    except Exception as e:
        logger.error(f"Unexpected error during merge: {e}", exc_info=True)
        return False
    finally:
        # Optionally switch back to the original branch if target was specified and different
        if target_branch and original_branch and target_branch != original_branch:
             if not checkout_branch(original_branch, repo_path):
                  logger.warning(f"Could not switch back to original branch '{original_branch}' after merge attempt.")


def delete_branch(branch_name: str, force: bool = False, repo_path: str = '.') -> bool:
    """Deletes the specified branch."""
    repo = _get_repo(repo_path)
    if not repo:
        return False
    try:
        if branch_name not in repo.heads:
            logger.warning(f"Branch '{branch_name}' does not exist, cannot delete.")
            return True # Or False? Let's say True as the desired state (no branch) is achieved.

        # Prevent deleting the currently checked out branch unless forced?
        # current = get_current_branch(repo_path)
        # if current == branch_name and not force:
        #     logger.error(f"Cannot delete the currently checked out branch '{branch_name}' without force.")
        #     return False

        logger.info(f"Deleting branch '{branch_name}' (force={force})...")
        repo.delete_head(branch_name, force=force)
        return True
    except Exception as e:
        logger.error(f"Error deleting branch '{branch_name}': {e}", exc_info=True)
        return False


# Example Usage (requires a Git repository)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    if not GITPYTHON_INSTALLED:
         print("GitPython is not installed. Skipping git_utils example.")
    else:
        # Assuming the script is run from within the Code-Agent repository root
        repo_directory = '.' 
        print(f"Checking Git status for repository: {os.path.abspath(repo_directory)}")

        branch = get_current_branch(repo_directory)
        if branch:
            print(f"Current Branch: {branch}")
        else:
            print("Could not determine current branch (possibly detached HEAD or error).")

        is_dirty = has_uncommitted_changes(repo_directory)
        if is_dirty is not None:
            print(f"Repository has uncommitted changes: {is_dirty}")
        else:
            print("Could not determine repository dirty status.")
            
        # Example file check (replace with an actual file path)
        file_to_check = 'README.md' 
        if os.path.exists(file_to_check):
             print(f"\nChecking last commit for: {file_to_check}")
             last_commit_info = get_file_last_commit(file_to_check, repo_directory)
             if last_commit_info:
                  sha, author, msg = last_commit_info
                  print(f"  Hash: {sha}")
                  print(f"  Author: {author}")
                  print(f"  Message: {msg}")
             else:
                  print(f"  Could not retrieve last commit info for {file_to_check}.")
        else:
             print(f"\nSkipping file commit check: '{file_to_check}' not found.") 
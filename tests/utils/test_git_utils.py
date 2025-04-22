import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock the git module BEFORE importing git_utils
# This prevents git_utils from failing if GitPython is not installed in the test environment
mock_git_module = MagicMock()
sys.modules['git'] = mock_git_module
sys.modules['git.exc'] = MagicMock()

# Now import the module under test
from utils import git_utils

# Restore original modules after import (optional, good practice)
# del sys.modules['git'] 
# del sys.modules['git.exc'] 
# ^ Be careful with this, might affect other tests if run in the same session

class TestGitUtils(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        mock_git_module.reset_mock()
        # Mock the Repo object that git.Repo() would return
        self.mock_repo_instance = MagicMock()
        mock_git_module.Repo.return_value = self.mock_repo_instance
        # Common attributes
        self.mock_repo_instance.working_dir = '/fake/repo'
        self.mock_repo_instance.head = MagicMock()
        self.mock_repo_instance.head.is_detached = False
        self.mock_repo_instance.active_branch = MagicMock(name='main')
        self.mock_repo_instance.is_dirty.return_value = False
        
        # Mock git exceptions if needed within tests
        git_utils.InvalidGitRepositoryError = mock_git_module.exc.InvalidGitRepositoryError
        git_utils.NoSuchPathError = mock_git_module.exc.NoSuchPathError
        # Ensure GITPYTHON_INSTALLED is True for most tests
        git_utils.GITPYTHON_INSTALLED = True 
        
    def test_get_repo_success(self):
        \"\"\"Test successfully getting a Repo object.\"\"\"
        repo = git_utils._get_repo('/fake/repo')
        self.assertIs(repo, self.mock_repo_instance)
        mock_git_module.Repo.assert_called_once_with('/fake/repo', search_parent_directories=True)

    def test_get_repo_invalid_repo(self):
        \"\"\"Test handling InvalidGitRepositoryError.\"\"\"
        mock_git_module.Repo.side_effect = git_utils.InvalidGitRepositoryError("Not a repo")
        repo = git_utils._get_repo('/not/a/repo')
        self.assertIsNone(repo)

    def test_get_repo_no_such_path(self):
        \"\"\"Test handling NoSuchPathError.\"\"\"
        mock_git_module.Repo.side_effect = git_utils.NoSuchPathError("Bad path")
        repo = git_utils._get_repo('/bad/path')
        self.assertIsNone(repo)
        
    def test_get_repo_gitpython_not_installed(self):
         \"\"\"Test behavior when GitPython is not installed.\"\"\"
         git_utils.GITPYTHON_INSTALLED = False # Simulate not installed
         repo = git_utils._get_repo()
         self.assertIsNone(repo)
         mock_git_module.Repo.assert_not_called() # Ensure git.Repo wasn't called
         git_utils.GITPYTHON_INSTALLED = True # Restore for other tests

    def test_get_current_branch_success(self):
        \"\"\"Test getting the current branch name successfully.\"\"\"
        self.mock_repo_instance.active_branch.name = 'develop'
        branch = git_utils.get_current_branch()
        self.assertEqual(branch, 'develop')
        mock_git_module.Repo.assert_called_once()
        self.mock_repo_instance.active_branch.name # Accessed

    def test_get_current_branch_detached_head(self):
        \"\"\"Test handling a detached HEAD state.\"\"\"
        self.mock_repo_instance.head.is_detached = True
        self.mock_repo_instance.head.commit.hexsha = 'abcdef123'
        # Simulate TypeError sometimes raised for detached head
        # self.mock_repo_instance.active_branch = PropertyMock(side_effect=TypeError('detached HEAD'))
        # Or rely on the is_detached check:
        del self.mock_repo_instance.active_branch # Ensure accessing it would fail if not checked
        branch = git_utils.get_current_branch()
        self.assertIsNone(branch)
        
    def test_get_current_branch_type_error_detached(self):
         \"\"\"Test handling TypeError indicating detached HEAD.\"\"\"
         # Some GitPython versions might raise TypeError directly
         self.mock_repo_instance.active_branch = property(MagicMock(side_effect=TypeError('Cannot read from detached HEAD'))) 
         branch = git_utils.get_current_branch()
         self.assertIsNone(branch)

    def test_has_uncommitted_changes_clean(self):
        \"\"\"Test checking for changes when the repo is clean.\"\"\"
        self.mock_repo_instance.is_dirty.return_value = False
        is_dirty = git_utils.has_uncommitted_changes()
        self.assertFalse(is_dirty)
        self.mock_repo_instance.is_dirty.assert_called_once_with(untracked_files=True)

    def test_has_uncommitted_changes_dirty(self):
        \"\"\"Test checking for changes when the repo is dirty.\"\"\"
        self.mock_repo_instance.is_dirty.return_value = True
        is_dirty = git_utils.has_uncommitted_changes()
        self.assertTrue(is_dirty)
        self.mock_repo_instance.is_dirty.assert_called_once_with(untracked_files=True)

    def test_get_file_last_commit_success(self):
        \"\"\"Test getting the last commit for a file successfully.\"\"\"
        mock_commit = MagicMock()
        mock_commit.hexsha = 'a1b2c3d4'
        mock_commit.author.name = 'Test Author'
        mock_commit.message = 'Fix: Important bug\n\nMore details.'
        self.mock_repo_instance.iter_commits.return_value = [mock_commit]
        
        # Mock os.path.relpath only if needed (if testing absolute paths)
        with patch('os.path.relpath', return_value='src/file.py') as mock_relpath:
             info = git_utils.get_file_last_commit('/fake/repo/src/file.py')
             mock_relpath.assert_called_once_with('/fake/repo/src/file.py', '/fake/repo')

        self.assertIsNotNone(info)
        self.assertEqual(info, ('a1b2c3d4', 'Test Author', 'Fix: Important bug'))
        # Ensure path uses forward slashes internally
        self.mock_repo_instance.iter_commits.assert_called_once_with(paths='src/file.py', max_count=1)
        
    def test_get_file_last_commit_relative_path(self):
         \"\"\"Test getting the last commit using a relative path.\"\"\"
         mock_commit = MagicMock(hexsha='e5f6g7h8', author=MagicMock(name='Another Author'), message='Feat: New feature')
         self.mock_repo_instance.iter_commits.return_value = [mock_commit]
         
         info = git_utils.get_file_last_commit('data/config.json') # Relative path
         
         self.assertEqual(info, ('e5f6g7h8', 'Another Author', 'Feat: New feature'))
         # Ensure relative path is passed correctly (already forward slashes)
         self.mock_repo_instance.iter_commits.assert_called_once_with(paths='data/config.json', max_count=1)
         
    def test_get_file_last_commit_windows_path(self):
         \"\"\"Test handling of Windows-style paths.\"\"\"
         mock_commit = MagicMock(hexsha='e5f6g7h8', author=MagicMock(name='Author'), message='Fix')
         self.mock_repo_instance.iter_commits.return_value = [mock_commit]
         
         info = git_utils.get_file_last_commit('src\\utils\\helpers.py') # Relative Windows path
         self.assertEqual(info[0], 'e5f6g7h8')
         # Ensure path is converted to forward slashes
         self.mock_repo_instance.iter_commits.assert_called_once_with(paths='src/utils/helpers.py', max_count=1)

    def test_get_file_last_commit_no_history(self):
        \"\"\"Test handling file with no commit history found.\"\"\"
        self.mock_repo_instance.iter_commits.return_value = [] # Simulate no commits returned
        info = git_utils.get_file_last_commit('new_file.txt')
        self.assertIsNone(info)
        self.mock_repo_instance.iter_commits.assert_called_once_with(paths='new_file.txt', max_count=1)
        
    def test_get_file_last_commit_git_error(self):
         \"\"\"Test handling Git errors during commit iteration.\"\"\"
         self.mock_repo_instance.iter_commits.side_effect = Exception("Git command failed")
         info = git_utils.get_file_last_commit('some/file.txt')
         self.assertIsNone(info)

if __name__ == '__main__':
    unittest.main() 
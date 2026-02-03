"""
Tests for the git operations module.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import tempfile

from codecraft_ai.git_operations import (
    GitOperations,
    GitOperationsError,
    NotAGitRepositoryError,
    NoStagedChangesError,
    CommitError,
    FileChange,
    StagedChanges,
    ChangeMetrics,
    CommitResult,
)


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_file_change_creation(self):
        """Test creating a FileChange object."""
        change = FileChange(
            file_path="src/test.py",
            status="M",
            additions=10,
            deletions=5,
            is_binary=False,
        )

        assert change.file_path == "src/test.py"
        assert change.status == "M"
        assert change.additions == 10
        assert change.deletions == 5
        assert change.is_binary is False

    def test_file_change_with_rename(self):
        """Test FileChange with rename."""
        change = FileChange(
            file_path="src/new_name.py",
            status="R",
            additions=0,
            deletions=0,
            old_path="src/old_name.py",
        )

        assert change.status == "R"
        assert change.old_path == "src/old_name.py"


class TestStagedChanges:
    """Tests for StagedChanges dataclass."""

    def test_staged_changes_not_empty(self):
        """Test StagedChanges with files."""
        changes = StagedChanges(
            files=[
                FileChange("test.py", "M", 10, 5, False),
            ],
            total_additions=10,
            total_deletions=5,
            total_files=1,
            diff_content="diff content",
        )

        assert changes.is_empty is False
        assert changes.total_files == 1

    def test_staged_changes_empty(self):
        """Test empty StagedChanges."""
        changes = StagedChanges(
            files=[],
            total_additions=0,
            total_deletions=0,
            total_files=0,
            diff_content="",
        )

        assert changes.is_empty is True


class TestChangeMetrics:
    """Tests for ChangeMetrics dataclass."""

    def test_change_metrics_creation(self):
        """Test creating ChangeMetrics."""
        metrics = ChangeMetrics(
            total_lines_changed=100,
            total_files=5,
            files_added=2,
            files_modified=2,
            files_deleted=1,
            files_renamed=0,
            directories_affected=3,
            file_types={".py": 3, ".js": 2},
            complexity_score=25,
        )

        assert metrics.total_lines_changed == 100
        assert metrics.total_files == 5
        assert metrics.complexity_score == 25


class TestCommitResult:
    """Tests for CommitResult dataclass."""

    def test_successful_commit_result(self):
        """Test successful commit result."""
        result = CommitResult(
            success=True,
            sha="abc123",
            message="Test commit",
        )

        assert result.success is True
        assert result.sha == "abc123"
        assert result.error is None

    def test_failed_commit_result(self):
        """Test failed commit result."""
        result = CommitResult(
            success=False,
            sha=None,
            message="Test commit",
            error="No staged changes",
        )

        assert result.success is False
        assert result.sha is None
        assert result.error == "No staged changes"


class TestGitOperations:
    """Tests for GitOperations class."""

    @patch("codecraft_ai.git_operations.Repo")
    def test_init_valid_repo(self, mock_repo_class):
        """Test initializing with valid git repository."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")

        assert git_ops.repo_path == "/path/to/repo"
        mock_repo_class.assert_called_once()

    @patch("codecraft_ai.git_operations.Repo")
    def test_init_invalid_repo(self, mock_repo_class):
        """Test initializing with invalid git repository."""
        from git import InvalidGitRepositoryError as GitInvalidRepo
        mock_repo_class.side_effect = GitInvalidRepo("Not a git repo")

        with pytest.raises(NotAGitRepositoryError):
            GitOperations("/not/a/repo")

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_current_branch(self, mock_repo_class):
        """Test getting current branch name."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo.active_branch.name = "feature-branch"
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        branch = git_ops.get_current_branch()

        assert branch == "feature-branch"

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_current_branch_detached(self, mock_repo_class):
        """Test getting current branch when HEAD is detached."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        # Mock active_branch to have a 'name' attribute that raises TypeError when accessed
        mock_active_branch = Mock()
        type(mock_active_branch).name = PropertyMock(side_effect=TypeError)
        mock_repo.active_branch = mock_active_branch
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        branch = git_ops.get_current_branch() # This call should catch the TypeError

        assert branch == "HEAD (detached)"

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_remote_url(self, mock_repo_class):
        """Test getting remote URL."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_remote = Mock()
        mock_remote.name = "origin"
        mock_remote.url = "https://github.com/user/repo.git"
        mock_repo.remotes = [mock_remote]
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        url = git_ops.get_remote_url()

        assert url == "https://github.com/user/repo.git"

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_remote_url_no_origin(self, mock_repo_class):
        """Test getting remote URL when no origin exists."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo.remotes = {}
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        url = git_ops.get_remote_url()

        assert url is None

    @patch("codecraft_ai.git_operations.Repo")
    def test_has_uncommitted_changes(self, mock_repo_class):
        """Test checking for uncommitted changes."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo.is_dirty.return_value = True
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        has_changes = git_ops.has_uncommitted_changes()

        assert has_changes is True
        mock_repo.is_dirty.assert_called_once_with(untracked_files=True)

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_untracked_files(self, mock_repo_class):
        """Test getting untracked files."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo.untracked_files = ["new_file.py", "another.txt"]
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        untracked = git_ops.get_untracked_files()

        assert len(untracked) == 2
        assert "new_file.py" in untracked

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_repo_name_from_remote(self, mock_repo_class):
        """Test getting repo name from remote URL."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_remote = Mock()
        mock_remote.name = "origin"
        mock_remote.url = "https://github.com/user/myrepo.git"
        mock_repo.remotes = [mock_remote]
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        name = git_ops.get_repo_name()

        assert name == "user/myrepo"

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_repo_name_ssh_format(self, mock_repo_class):
        """Test getting repo name from SSH remote URL."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_remote = Mock()
        mock_remote = Mock()
        mock_remote.name = "origin"
        mock_remote.url = "git@github.com:user/myrepo.git"
        mock_repo.remotes = [mock_remote]
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        name = git_ops.get_repo_name()

        assert name == "user/myrepo"

    @patch("codecraft_ai.git_operations.Repo")
    def test_get_repo_name_fallback(self, mock_repo_class):
        """Test getting repo name falls back to directory name."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/my-project"
        mock_repo.remotes = {}
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/my-project")
        name = git_ops.get_repo_name()

        assert name == "my-project"

    @patch("codecraft_ai.git_operations.Repo")
    def test_analyze_change_complexity(self, mock_repo_class):
        """Test complexity analysis calculation."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"

        # Mock staged changes
        mock_diff_item = Mock()
        mock_diff_item.new_file = False
        mock_diff_item.deleted_file = False
        mock_diff_item.renamed = False
        mock_diff_item.b_path = "src/test.py"
        mock_diff_item.a_path = "src/test.py"
        mock_diff_item.diff = b"+line1\n+line2\n-old line"
        mock_diff_item.b_blob = None

        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.git.diff.return_value = "diff content"
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        metrics = git_ops.analyze_change_complexity()

        assert isinstance(metrics, ChangeMetrics)
        assert metrics.total_files >= 0
        assert metrics.complexity_score >= 0

    @patch("codecraft_ai.git_operations.Repo")
    def test_create_commit_success(self, mock_repo_class):
        """Test successful commit creation."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"

        # Mock staged changes (non-empty)
        mock_diff_item = Mock()
        mock_diff_item.b_path = "test.py"
        mock_diff_item.a_path = "test.py"
        mock_diff_item.new_file = False
        mock_diff_item.deleted_file = False
        mock_diff_item.renamed = False
        mock_diff_item.diff = b"+new line"
        mock_diff_item.b_blob = None

        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.git.diff.return_value = "diff content"

        mock_commit = Mock()
        mock_commit.hexsha = "abc123"
        mock_repo.index.commit.return_value = mock_commit
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        result = git_ops.create_commit("Test commit message")

        assert result.success is True
        assert result.sha == "abc123"

    @patch("codecraft_ai.git_operations.Repo")
    def test_show_last_commit(self, mock_repo_class):
        """Test showing last commit."""
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/repo"
        mock_repo.git.log.return_value = "commit abc123\nAuthor: Test\nDate: 2024-01-01"
        mock_repo_class.return_value = mock_repo

        git_ops = GitOperations("/path/to/repo")
        log = git_ops.show_last_commit()

        assert "commit" in log.lower() or "abc123" in log
        mock_repo.git.log.assert_called_once_with("-1", "--stat", "--no-color")

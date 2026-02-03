"""
Tests for the GitLab client module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from codecraft_ai.gitlab_client import (
    GitLabClient,
    GitLabClientError,
    AuthenticationError,
    RateLimitError,
    RepositoryNotFoundError,
    RepositoryInfo,
    BranchInfo,
    CommitInfo,
    CommitDiff,
)


class TestGitLabClient:
    """Tests for GitLabClient class."""

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_init_with_token(self, mock_gitlab_class):
        """Test client initialization with explicit token."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token", url="https://gitlab.com")

        assert client.token == "test_token"
        assert client.username == "testuser"
        mock_gitlab_class.assert_called_once_with(
            "https://gitlab.com", private_token="test_token"
        )

    @patch.dict("os.environ", {"GITLAB_TOKEN": "env_token"})
    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_init_with_env_token(self, mock_gitlab_class):
        """Test client initialization with environment token."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "envuser"
        mock_gitlab.user = mock_user
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient()

        assert client.token == "env_token"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_token_raises(self):
        """Test that initialization without token raises error."""
        with pytest.raises(AuthenticationError):
            GitLabClient()

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_list_repositories(self, mock_gitlab_class):
        """Test listing repositories (projects)."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user

        mock_project = Mock()
        mock_project.id = 123
        mock_project.name = "test-project"
        mock_project.path_with_namespace = "testuser/test-project"
        mock_project.description = "A test project"
        mock_project.star_count = 10
        mock_project.forks_count = 2
        mock_project.last_activity_at = "2024-01-01T00:00:00Z"
        mock_project.default_branch = "main"
        mock_project.visibility = "public"
        mock_project.web_url = "https://gitlab.com/testuser/test-project"
        mock_project.languages.return_value = {"Python": 80, "JavaScript": 20}

        mock_gitlab.projects.list.return_value = [mock_project]
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token")
        repos = client.list_repositories()

        assert len(repos) == 1
        assert repos[0].name == "test-project"
        assert repos[0].full_name == "testuser/test-project"
        assert repos[0].language == "Python"
        assert repos[0].stars == 10

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_list_branches(self, mock_gitlab_class):
        """Test listing branches."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user

        mock_project = Mock()
        mock_project.default_branch = "main"

        mock_branch = Mock()
        mock_branch.name = "main"
        mock_branch.protected = False
        mock_branch.commit = {"id": "abc123"}

        mock_project.branches.list.return_value = [mock_branch]
        mock_gitlab.projects.get.return_value = mock_project
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token")
        branches = client.list_branches(123)

        assert len(branches) == 1
        assert branches[0].name == "main"
        assert branches[0].is_default is True

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_list_commits(self, mock_gitlab_class):
        """Test listing commits."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user

        mock_project = Mock()
        mock_project.default_branch = "main"

        mock_commit = Mock()
        mock_commit.id = "abc123def456"
        mock_commit.short_id = "abc123d"
        mock_commit.message = "Test commit message"
        mock_commit.author_name = "Test Author"
        mock_commit.author_email = "test@example.com"
        mock_commit.committed_date = "2024-01-01T00:00:00Z"

        mock_full_commit = Mock()
        mock_full_commit.stats = {"additions": 10, "deletions": 5, "total": 15}

        mock_project.commits.list.return_value = [mock_commit]
        mock_project.commits.get.return_value = mock_full_commit
        mock_gitlab.projects.get.return_value = mock_project
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token")
        commits = client.list_commits(123, "main", limit=10)

        assert len(commits) == 1
        assert commits[0].sha == "abc123def456"
        assert commits[0].short_sha == "abc123d"
        assert commits[0].message == "Test commit message"
        assert commits[0].author_name == "Test Author"

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_get_commit_diff(self, mock_gitlab_class):
        """Test getting commit diff."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user

        mock_project = Mock()

        mock_commit = Mock()
        mock_commit.id = "abc123"
        mock_commit.diff.return_value = [
            {
                "new_path": "test.py",
                "old_path": "test.py",
                "new_file": False,
                "deleted_file": False,
                "renamed_file": False,
                "diff": "@@ -1,3 +1,4 @@\n+new line\n old line",
            }
        ]

        mock_project.commits.get.return_value = mock_commit
        mock_gitlab.projects.get.return_value = mock_project
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token")
        diff = client.get_commit_diff(123, "abc123")

        assert diff.sha == "abc123"
        assert len(diff.files) == 1
        assert diff.files[0]["filename"] == "test.py"

    @patch("codecraft_ai.gitlab_client.gitlab.Gitlab")
    def test_gitlab_url_property(self, mock_gitlab_class):
        """Test gitlab_url property."""
        mock_gitlab = Mock()
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_gitlab.user = mock_user
        mock_gitlab_class.return_value = mock_gitlab

        client = GitLabClient(token="test_token", url="https://gitlab.example.com")

        assert client.gitlab_url == "https://gitlab.example.com"


class TestRepositoryInfo:
    """Tests for RepositoryInfo dataclass."""

    def test_repository_info_creation(self):
        """Test creating a RepositoryInfo object."""
        repo = RepositoryInfo(
            id=123,
            name="test-project",
            full_name="user/test-project",
            description="Test description",
            language="Python",
            stars=100,
            forks=20,
            updated_at=datetime(2024, 1, 1),
            default_branch="main",
            private=False,
            url="https://gitlab.com/user/test-project",
        )

        assert repo.id == 123
        assert repo.name == "test-project"
        assert repo.full_name == "user/test-project"
        assert repo.stars == 100
        assert repo.private is False


class TestBranchInfo:
    """Tests for BranchInfo dataclass."""

    def test_branch_info_creation(self):
        """Test creating a BranchInfo object."""
        branch = BranchInfo(
            name="main",
            is_default=True,
            is_protected=True,
            commit_sha="abc123",
        )

        assert branch.name == "main"
        assert branch.is_default is True
        assert branch.is_protected is True


class TestCommitInfo:
    """Tests for CommitInfo dataclass."""

    def test_commit_info_creation(self):
        """Test creating a CommitInfo object."""
        commit = CommitInfo(
            sha="abc123def456",
            short_sha="abc123d",
            message="Test commit",
            author_name="Test Author",
            author_email="test@example.com",
            date=datetime(2024, 1, 1),
            additions=10,
            deletions=5,
            files_changed=3,
        )

        assert commit.sha == "abc123def456"
        assert commit.short_sha == "abc123d"
        assert commit.additions == 10


class TestCommitDiff:
    """Tests for CommitDiff dataclass."""

    def test_commit_diff_creation(self):
        """Test creating a CommitDiff object."""
        diff = CommitDiff(
            sha="abc123",
            files=[{"filename": "test.py", "status": "modified"}],
            patch="diff content",
            additions=10,
            deletions=5,
        )

        assert diff.sha == "abc123"
        assert len(diff.files) == 1
        assert diff.additions == 10
        assert diff.deletions == 5

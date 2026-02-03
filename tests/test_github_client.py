"""
Tests for the GitHub client module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from codecraft_ai.github_client import (
    GitHubClient,
    GitHubClientError,
    AuthenticationError,
    RateLimitError,
    RepositoryNotFoundError,
    RepositoryInfo,
    BranchInfo,
    CommitInfo,
    CommitDiff,
)


class TestGitHubClient:
    """Tests for GitHubClient class."""

    @patch("codecraft_ai.github_client.Github")
    def test_init_with_token(self, mock_github):
        """Test client initialization with explicit token."""
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github.return_value.get_user.return_value = mock_user

        client = GitHubClient(token="test_token")

        assert client.token == "test_token"
        assert client.username == "testuser"
        mock_github.assert_called_once_with("test_token", per_page=30)

    @patch.dict("os.environ", {"GITHUB_TOKEN": "env_token"})
    @patch("codecraft_ai.github_client.Github")
    def test_init_with_env_token(self, mock_github):
        """Test client initialization with environment token."""
        mock_user = Mock()
        mock_user.login = "envuser"
        mock_github.return_value.get_user.return_value = mock_user

        client = GitHubClient()

        assert client.token == "env_token"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_token_raises(self):
        """Test that initialization without token raises error."""
        with pytest.raises(AuthenticationError):
            GitHubClient()

    @patch("codecraft_ai.github_client.Github")
    def test_list_repositories(self, mock_github):
        """Test listing repositories."""
        # Setup mock
        mock_user = Mock()
        mock_user.login = "testuser"

        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.full_name = "testuser/test-repo"
        mock_repo.description = "A test repository"
        mock_repo.language = "Python"
        mock_repo.stargazers_count = 10
        mock_repo.forks_count = 2
        mock_repo.updated_at = datetime(2024, 1, 1)
        mock_repo.default_branch = "main"
        mock_repo.private = False
        mock_repo.html_url = "https://github.com/testuser/test-repo"

        mock_user.get_repos.return_value = [mock_repo]
        mock_github.return_value.get_user.return_value = mock_user

        client = GitHubClient(token="test_token")
        repos = client.list_repositories()

        assert len(repos) == 1
        assert repos[0].name == "test-repo"
        assert repos[0].full_name == "testuser/test-repo"
        assert repos[0].language == "Python"
        assert repos[0].stars == 10

    @patch("codecraft_ai.github_client.Github")
    def test_list_branches(self, mock_github):
        """Test listing branches."""
        mock_user = Mock()
        mock_user.login = "testuser"

        mock_repo = Mock()
        mock_repo.default_branch = "main"

        mock_branch = Mock()
        mock_branch.name = "main"
        mock_branch.protected = False
        mock_branch.commit.sha = "abc123"

        mock_repo.get_branches.return_value = [mock_branch]
        mock_github.return_value.get_user.return_value = mock_user
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(token="test_token")
        branches = client.list_branches("testuser/test-repo")

        assert len(branches) == 1
        assert branches[0].name == "main"
        assert branches[0].is_default is True

    @patch("codecraft_ai.github_client.Github")
    def test_list_commits(self, mock_github):
        """Test listing commits."""
        mock_user = Mock()
        mock_user.login = "testuser"

        mock_repo = Mock()
        mock_repo.default_branch = "main"

        mock_commit = Mock()
        mock_commit.sha = "abc123def456"
        mock_commit.commit.message = "Test commit message"
        mock_commit.commit.author.name = "Test Author"
        mock_commit.commit.author.email = "test@example.com"
        mock_commit.commit.author.date = datetime(2024, 1, 1)
        mock_commit.stats.additions = 10
        mock_commit.stats.deletions = 5
        mock_commit.files = []

        mock_repo.get_commits.return_value = [mock_commit]
        mock_github.return_value.get_user.return_value = mock_user
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(token="test_token")
        commits = client.list_commits("testuser/test-repo", "main", limit=10)

        assert len(commits) == 1
        assert commits[0].sha == "abc123def456"
        assert commits[0].short_sha == "abc123d"
        assert commits[0].message == "Test commit message"
        assert commits[0].author_name == "Test Author"

    @patch("codecraft_ai.github_client.Github")
    def test_get_commit_diff(self, mock_github):
        """Test getting commit diff."""
        mock_user = Mock()
        mock_user.login = "testuser"

        mock_file = Mock()
        mock_file.filename = "test.py"
        mock_file.status = "modified"
        mock_file.additions = 5
        mock_file.deletions = 2
        mock_file.changes = 7
        mock_file.patch = "@@ -1,3 +1,4 @@\n+new line"

        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.files = [mock_file]
        mock_commit.stats.additions = 5
        mock_commit.stats.deletions = 2

        mock_repo = Mock()
        mock_repo.get_commit.return_value = mock_commit
        mock_github.return_value.get_user.return_value = mock_user
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(token="test_token")
        diff = client.get_commit_diff("testuser/test-repo", "abc123")

        assert diff.sha == "abc123"
        assert len(diff.files) == 1
        assert diff.files[0]["filename"] == "test.py"
        assert diff.additions == 5
        assert diff.deletions == 2

    @patch("codecraft_ai.github_client.Github")
    def test_get_rate_limit_status(self, mock_github):
        """Test getting rate limit status."""
        mock_user = Mock()
        mock_user.login = "testuser"

        mock_rate = Mock()
        mock_rate.core.limit = 5000
        mock_rate.core.remaining = 4999
        mock_rate.core.reset = datetime(2024, 1, 1)

        mock_github.return_value.get_user.return_value = mock_user
        mock_github.return_value.get_rate_limit.return_value = mock_rate

        client = GitHubClient(token="test_token")
        status = client.get_rate_limit_status()

        assert status["limit"] == 5000
        assert status["remaining"] == 4999


class TestRepositoryInfo:
    """Tests for RepositoryInfo dataclass."""

    def test_repository_info_creation(self):
        """Test creating a RepositoryInfo object."""
        repo = RepositoryInfo(
            name="test-repo",
            full_name="user/test-repo",
            description="Test description",
            language="Python",
            stars=100,
            forks=20,
            updated_at=datetime(2024, 1, 1),
            default_branch="main",
            private=False,
            url="https://github.com/user/test-repo",
        )

        assert repo.name == "test-repo"
        assert repo.full_name == "user/test-repo"
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

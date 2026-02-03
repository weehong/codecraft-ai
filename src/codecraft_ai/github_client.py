"""
GitHub API client for repository, branch, and commit operations.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from github import Github, GithubException
from github.Repository import Repository
from github.Branch import Branch
from github.Commit import Commit


@dataclass
class RepositoryInfo:
    """Information about a GitHub repository."""

    name: str
    full_name: str
    description: Optional[str]
    language: Optional[str]
    stars: int
    forks: int
    updated_at: datetime
    default_branch: str
    private: bool
    url: str


@dataclass
class BranchInfo:
    """Information about a repository branch."""

    name: str
    is_default: bool
    is_protected: bool
    commit_sha: str


@dataclass
class CommitInfo:
    """Information about a commit."""

    sha: str
    short_sha: str
    message: str
    author_name: str
    author_email: str
    date: datetime
    additions: int
    deletions: int
    files_changed: int


@dataclass
class CommitDiff:
    """Diff information for a commit."""

    sha: str
    files: List[dict]
    patch: str
    additions: int
    deletions: int


class GitHubClientError(Exception):
    """Base exception for GitHub client errors."""

    pass


class AuthenticationError(GitHubClientError):
    """Authentication failed."""

    pass


class RateLimitError(GitHubClientError):
    """Rate limit exceeded."""

    pass


class RepositoryNotFoundError(GitHubClientError):
    """Repository not found."""

    pass


class GitHubClient:
    """Client for interacting with GitHub API."""

    def __init__(self, token: Optional[str] = None, per_page: int = 30):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token. If not provided, uses GITHUB_TOKEN env var.
            per_page: Number of items to fetch per page (max 100).
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise AuthenticationError(
                "GitHub token not provided. Set GITHUB_TOKEN environment variable "
                "or pass token parameter."
            )

        self.per_page = min(per_page, 100)
        self._github = Github(self.token, per_page=self.per_page)

        # Validate token
        try:
            self._user = self._github.get_user()
            # Force API call to validate token
            _ = self._user.login
        except GithubException as e:
            if e.status == 401:
                raise AuthenticationError("Invalid GitHub token.")
            raise GitHubClientError(f"GitHub API error: {e}")

    def list_repositories(
        self,
        include_private: bool = True,
        sort: str = "updated",
        direction: str = "desc",
    ) -> List[RepositoryInfo]:
        """
        List all accessible repositories.

        Args:
            include_private: Include private repositories.
            sort: Sort field (created, updated, pushed, full_name).
            direction: Sort direction (asc, desc).

        Returns:
            List of RepositoryInfo objects.
        """
        try:
            repos = []
            affiliation = "owner,collaborator,organization_member"

            for repo in self._user.get_repos(
                affiliation=affiliation, sort=sort, direction=direction
            ):
                if not include_private and repo.private:
                    continue

                repos.append(
                    RepositoryInfo(
                        name=repo.name,
                        full_name=repo.full_name,
                        description=repo.description,
                        language=repo.language,
                        stars=repo.stargazers_count,
                        forks=repo.forks_count,
                        updated_at=repo.updated_at,
                        default_branch=repo.default_branch,
                        private=repo.private,
                        url=repo.html_url,
                    )
                )

            return repos

        except GithubException as e:
            if e.status == 403:
                raise RateLimitError("GitHub API rate limit exceeded.")
            raise GitHubClientError(f"Failed to list repositories: {e}")

    def list_branches(self, repo_name: str) -> List[BranchInfo]:
        """
        List branches for a repository.

        Args:
            repo_name: Full repository name (owner/repo).

        Returns:
            List of BranchInfo objects.
        """
        try:
            repo = self._github.get_repo(repo_name)
            branches = []

            for branch in repo.get_branches():
                branches.append(
                    BranchInfo(
                        name=branch.name,
                        is_default=branch.name == repo.default_branch,
                        is_protected=branch.protected,
                        commit_sha=branch.commit.sha,
                    )
                )

            # Sort with default branch first, then alphabetically
            branches.sort(key=lambda b: (not b.is_default, b.name.lower()))
            return branches

        except GithubException as e:
            if e.status == 404:
                raise RepositoryNotFoundError(f"Repository not found: {repo_name}")
            if e.status == 403:
                raise RateLimitError("GitHub API rate limit exceeded.")
            raise GitHubClientError(f"Failed to list branches: {e}")

    def list_commits(
        self,
        repo_name: str,
        branch_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[CommitInfo]:
        """
        List recent commits for a repository branch.

        Args:
            repo_name: Full repository name (owner/repo).
            branch_name: Branch name. If None, uses default branch.
            limit: Maximum number of commits to return.

        Returns:
            List of CommitInfo objects.
        """
        try:
            repo = self._github.get_repo(repo_name)
            sha = branch_name or repo.default_branch

            commits = []
            for commit in repo.get_commits(sha=sha)[:limit]:
                # Get detailed commit info
                git_commit = commit.commit

                commits.append(
                    CommitInfo(
                        sha=commit.sha,
                        short_sha=commit.sha[:7],
                        message=git_commit.message,
                        author_name=git_commit.author.name if git_commit.author else "Unknown",
                        author_email=git_commit.author.email if git_commit.author else "",
                        date=git_commit.author.date if git_commit.author else datetime.now(),
                        additions=commit.stats.additions if commit.stats else 0,
                        deletions=commit.stats.deletions if commit.stats else 0,
                        files_changed=len(commit.files) if commit.files else 0,
                    )
                )

            return commits

        except GithubException as e:
            if e.status == 404:
                raise RepositoryNotFoundError(f"Repository or branch not found: {repo_name}/{branch_name}")
            if e.status == 403:
                raise RateLimitError("GitHub API rate limit exceeded.")
            raise GitHubClientError(f"Failed to list commits: {e}")

    def get_commit_diff(self, repo_name: str, commit_sha: str) -> CommitDiff:
        """
        Get the diff for a specific commit.

        Args:
            repo_name: Full repository name (owner/repo).
            commit_sha: Full or short commit SHA.

        Returns:
            CommitDiff object with file changes and patch.
        """
        try:
            repo = self._github.get_repo(repo_name)
            commit = repo.get_commit(commit_sha)

            files = []
            patches = []

            for file in commit.files:
                file_info = {
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch or "",
                }
                files.append(file_info)

                if file.patch:
                    patches.append(f"--- a/{file.filename}\n+++ b/{file.filename}\n{file.patch}")

            return CommitDiff(
                sha=commit.sha,
                files=files,
                patch="\n\n".join(patches),
                additions=commit.stats.additions if commit.stats else 0,
                deletions=commit.stats.deletions if commit.stats else 0,
            )

        except GithubException as e:
            if e.status == 404:
                raise RepositoryNotFoundError(f"Commit not found: {repo_name}@{commit_sha}")
            if e.status == 403:
                raise RateLimitError("GitHub API rate limit exceeded.")
            raise GitHubClientError(f"Failed to get commit diff: {e}")

    def get_multiple_commit_diffs(
        self, repo_name: str, commit_shas: List[str]
    ) -> List[CommitDiff]:
        """
        Get diffs for multiple commits.

        Args:
            repo_name: Full repository name (owner/repo).
            commit_shas: List of commit SHAs.

        Returns:
            List of CommitDiff objects.
        """
        diffs = []
        for sha in commit_shas:
            diff = self.get_commit_diff(repo_name, sha)
            diffs.append(diff)
        return diffs

    def get_rate_limit_status(self) -> dict:
        """
        Get current rate limit status.

        Returns:
            Dictionary with rate limit information.
        """
        rate = self._github.get_rate_limit()
        return {
            "limit": rate.core.limit,
            "remaining": rate.core.remaining,
            "reset_time": rate.core.reset,
        }

    @property
    def username(self) -> str:
        """Get the authenticated user's username."""
        return self._user.login

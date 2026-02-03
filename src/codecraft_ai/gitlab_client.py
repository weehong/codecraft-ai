"""
GitLab API client for repository, branch, and commit operations.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import gitlab
from gitlab.exceptions import GitlabAuthenticationError, GitlabGetError


@dataclass
class RepositoryInfo:
    """Information about a GitLab repository (project)."""

    id: int
    name: str
    full_name: str  # path_with_namespace
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


class GitLabClientError(Exception):
    """Base exception for GitLab client errors."""

    pass


class AuthenticationError(GitLabClientError):
    """Authentication failed."""

    pass


class RateLimitError(GitLabClientError):
    """Rate limit exceeded."""

    pass


class RepositoryNotFoundError(GitLabClientError):
    """Repository not found."""

    pass


class GitLabClient:
    """Client for interacting with GitLab API."""

    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        per_page: int = 30,
    ):
        """
        Initialize GitLab client.

        Args:
            token: GitLab personal access token. If not provided, uses GITLAB_TOKEN env var.
            url: GitLab instance URL. Defaults to GITLAB_URL env var or https://gitlab.com.
            per_page: Number of items to fetch per page (max 100).
        """
        self.token = token or os.getenv("GITLAB_TOKEN")
        if not self.token:
            raise AuthenticationError(
                "GitLab token not provided. Set GITLAB_TOKEN environment variable "
                "or pass token parameter."
            )

        self.url = url or os.getenv("GITLAB_URL", "https://gitlab.com")
        self.per_page = min(per_page, 100)

        try:
            self._gitlab = gitlab.Gitlab(self.url, private_token=self.token)
            self._gitlab.auth()
            self._user = self._gitlab.user
        except GitlabAuthenticationError:
            raise AuthenticationError("Invalid GitLab token.")
        except Exception as e:
            raise GitLabClientError(f"GitLab API error: {e}")

    def list_repositories(
        self,
        include_private: bool = True,
        sort: str = "updated_at",
        order: str = "desc",
    ) -> List[RepositoryInfo]:
        """
        List all accessible repositories (projects).

        Args:
            include_private: Include private repositories.
            sort: Sort field (created_at, updated_at, name).
            order: Sort order (asc, desc).

        Returns:
            List of RepositoryInfo objects.
        """
        try:
            repos = []
            visibility = None if include_private else "public"

            projects = self._gitlab.projects.list(
                membership=True,
                order_by=sort,
                sort=order,
                per_page=self.per_page,
                visibility=visibility,
                iterator=True,
            )

            for project in projects:
                # Get primary language if available
                language = None
                try:
                    languages = project.languages()
                    if languages:
                        language = max(languages, key=languages.get)
                except Exception:
                    pass

                updated_at = None
                if project.last_activity_at:
                    try:
                        updated_at = datetime.fromisoformat(
                            project.last_activity_at.replace("Z", "+00:00")
                        )
                    except Exception:
                        updated_at = datetime.now()

                repos.append(
                    RepositoryInfo(
                        id=project.id,
                        name=project.name,
                        full_name=project.path_with_namespace,
                        description=project.description,
                        language=language,
                        stars=project.star_count,
                        forks=project.forks_count,
                        updated_at=updated_at,
                        default_branch=project.default_branch or "main",
                        private=project.visibility == "private",
                        url=project.web_url,
                    )
                )

                # Limit to reasonable number
                if len(repos) >= 100:
                    break

            return repos

        except Exception as e:
            raise GitLabClientError(f"Failed to list repositories: {e}")

    def list_branches(self, project_id: int) -> List[BranchInfo]:
        """
        List branches for a repository.

        Args:
            project_id: GitLab project ID.

        Returns:
            List of BranchInfo objects.
        """
        try:
            project = self._gitlab.projects.get(project_id)
            branches = []

            for branch in project.branches.list(per_page=self.per_page, iterator=True):
                branches.append(
                    BranchInfo(
                        name=branch.name,
                        is_default=branch.name == project.default_branch,
                        is_protected=branch.protected,
                        commit_sha=branch.commit["id"],
                    )
                )

                if len(branches) >= 100:
                    break

            # Sort with default branch first, then alphabetically
            branches.sort(key=lambda b: (not b.is_default, b.name.lower()))
            return branches

        except GitlabGetError:
            raise RepositoryNotFoundError(f"Project not found: {project_id}")
        except Exception as e:
            raise GitLabClientError(f"Failed to list branches: {e}")

    def list_commits(
        self,
        project_id: int,
        branch_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[CommitInfo]:
        """
        List recent commits for a repository branch.

        Args:
            project_id: GitLab project ID.
            branch_name: Branch name. If None, uses default branch.
            limit: Maximum number of commits to return.

        Returns:
            List of CommitInfo objects.
        """
        try:
            project = self._gitlab.projects.get(project_id)
            ref = branch_name or project.default_branch

            commits = []
            for commit in project.commits.list(
                ref_name=ref, per_page=min(limit, self.per_page), iterator=True
            ):
                # Parse commit date
                committed_date = None
                if commit.committed_date:
                    try:
                        committed_date = datetime.fromisoformat(
                            commit.committed_date.replace("Z", "+00:00")
                        )
                    except Exception:
                        committed_date = datetime.now()

                # Get stats
                additions = 0
                deletions = 0
                files_changed = 0

                try:
                    # Get full commit details for stats
                    full_commit = project.commits.get(commit.id)
                    if full_commit.stats:
                        additions = full_commit.stats.get("additions", 0)
                        deletions = full_commit.stats.get("deletions", 0)
                        files_changed = full_commit.stats.get("total", 0)
                except Exception:
                    pass

                commits.append(
                    CommitInfo(
                        sha=commit.id,
                        short_sha=commit.short_id,
                        message=commit.message,
                        author_name=commit.author_name or "Unknown",
                        author_email=commit.author_email or "",
                        date=committed_date,
                        additions=additions,
                        deletions=deletions,
                        files_changed=files_changed,
                    )
                )

                if len(commits) >= limit:
                    break

            return commits

        except GitlabGetError:
            raise RepositoryNotFoundError(
                f"Project or branch not found: {project_id}/{branch_name}"
            )
        except Exception as e:
            raise GitLabClientError(f"Failed to list commits: {e}")

    def get_commit_diff(self, project_id: int, commit_sha: str) -> CommitDiff:
        """
        Get the diff for a specific commit.

        Args:
            project_id: GitLab project ID.
            commit_sha: Full or short commit SHA.

        Returns:
            CommitDiff object with file changes and patch.
        """
        try:
            project = self._gitlab.projects.get(project_id)
            commit = project.commits.get(commit_sha)

            files = []
            patches = []
            total_additions = 0
            total_deletions = 0

            for diff in commit.diff():
                file_additions = diff.get("diff", "").count("\n+") - diff.get(
                    "diff", ""
                ).count("\n+++")
                file_deletions = diff.get("diff", "").count("\n-") - diff.get(
                    "diff", ""
                ).count("\n---")

                file_info = {
                    "filename": diff.get("new_path") or diff.get("old_path"),
                    "status": self._get_file_status(diff),
                    "additions": max(0, file_additions),
                    "deletions": max(0, file_deletions),
                    "changes": max(0, file_additions) + max(0, file_deletions),
                    "patch": diff.get("diff", ""),
                }
                files.append(file_info)

                if diff.get("diff"):
                    old_path = diff.get("old_path", "")
                    new_path = diff.get("new_path", "")
                    patches.append(
                        f"--- a/{old_path}\n+++ b/{new_path}\n{diff['diff']}"
                    )

                total_additions += max(0, file_additions)
                total_deletions += max(0, file_deletions)

            return CommitDiff(
                sha=commit.id,
                files=files,
                patch="\n\n".join(patches),
                additions=total_additions,
                deletions=total_deletions,
            )

        except GitlabGetError:
            raise RepositoryNotFoundError(
                f"Commit not found: {project_id}@{commit_sha}"
            )
        except Exception as e:
            raise GitLabClientError(f"Failed to get commit diff: {e}")

    def _get_file_status(self, diff: dict) -> str:
        """Determine the status of a diff item."""
        if diff.get("new_file"):
            return "added"
        elif diff.get("deleted_file"):
            return "deleted"
        elif diff.get("renamed_file"):
            return "renamed"
        else:
            return "modified"

    def get_multiple_commit_diffs(
        self, project_id: int, commit_shas: List[str]
    ) -> List[CommitDiff]:
        """
        Get diffs for multiple commits.

        Args:
            project_id: GitLab project ID.
            commit_shas: List of commit SHAs.

        Returns:
            List of CommitDiff objects.
        """
        diffs = []
        for sha in commit_shas:
            diff = self.get_commit_diff(project_id, sha)
            diffs.append(diff)
        return diffs

    @property
    def username(self) -> str:
        """Get the authenticated user's username."""
        return self._user.username if self._user else "unknown"

    @property
    def gitlab_url(self) -> str:
        """Get the GitLab instance URL."""
        return self.url

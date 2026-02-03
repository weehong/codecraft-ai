"""
Local git operations using GitPython.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from git import Repo, InvalidGitRepositoryError, GitCommandError


@dataclass
class FileChange:
    """Information about a single file change."""

    file_path: str
    status: str  # 'A' (added), 'M' (modified), 'D' (deleted), 'R' (renamed)
    additions: int
    deletions: int
    is_binary: bool = False
    old_path: Optional[str] = None  # For renamed files


@dataclass
class StagedChanges:
    """Collection of staged changes."""

    files: List[FileChange]
    total_additions: int
    total_deletions: int
    total_files: int
    diff_content: str

    @property
    def is_empty(self) -> bool:
        """Check if there are no staged changes."""
        return self.total_files == 0


@dataclass
class ChangeMetrics:
    """Metrics about the changes."""

    total_lines_changed: int
    total_files: int
    files_added: int
    files_modified: int
    files_deleted: int
    files_renamed: int
    directories_affected: int
    file_types: dict = field(default_factory=dict)
    complexity_score: int = 0


@dataclass
class CommitResult:
    """Result of a commit operation."""

    success: bool
    sha: Optional[str]
    message: str
    error: Optional[str] = None


class GitOperationsError(Exception):
    """Base exception for git operations errors."""

    pass


class NotAGitRepositoryError(GitOperationsError):
    """Current directory is not a git repository."""

    pass


class NoStagedChangesError(GitOperationsError):
    """No staged changes to commit."""

    pass


class CommitError(GitOperationsError):
    """Commit operation failed."""

    pass


class GitOperations:
    """Operations for local git repository."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize git operations for a repository.

        Args:
            repo_path: Path to the git repository. Defaults to current directory.
        """
        self.repo_path = repo_path or os.getcwd()

        try:
            self.repo = Repo(self.repo_path, search_parent_directories=True)
            self.repo_path = self.repo.working_dir
        except InvalidGitRepositoryError:
            raise NotAGitRepositoryError(
                f"Not a git repository: {self.repo_path}\n"
                "Please run this command from within a git repository."
            )

    def get_staged_changes(self) -> StagedChanges:
        """
        Get all staged changes (git diff --cached).

        Returns:
            StagedChanges object with file changes and diff content.
        """
        # Get diff between HEAD and staging area
        try:
            staged_diff = self.repo.index.diff("HEAD")
        except GitCommandError:
            # No HEAD (initial commit scenario)
            staged_diff = self.repo.index.diff(None)

        files = []
        total_additions = 0
        total_deletions = 0

        for diff_item in staged_diff:
            status = self._get_change_status(diff_item)
            additions, deletions = self._count_diff_lines(diff_item)

            file_change = FileChange(
                file_path=diff_item.b_path or diff_item.a_path,
                status=status,
                additions=additions,
                deletions=deletions,
                is_binary=diff_item.b_blob and diff_item.b_blob.mime_type.startswith("application/"),
                old_path=diff_item.a_path if diff_item.renamed else None,
            )
            files.append(file_change)
            total_additions += additions
            total_deletions += deletions

        # Also check for new files in index
        try:
            new_files = self.repo.index.diff(None)
            for diff_item in new_files:
                if diff_item.new_file:
                    additions, _ = self._count_diff_lines(diff_item)
                    file_change = FileChange(
                        file_path=diff_item.b_path or diff_item.a_path,
                        status="A",
                        additions=additions,
                        deletions=0,
                        is_binary=False,
                    )
                    files.append(file_change)
                    total_additions += additions
        except GitCommandError:
            pass

        # Get full diff content
        diff_content = self._get_diff_content()

        return StagedChanges(
            files=files,
            total_additions=total_additions,
            total_deletions=total_deletions,
            total_files=len(files),
            diff_content=diff_content,
        )

    def _get_change_status(self, diff_item) -> str:
        """Determine the status of a diff item."""
        if diff_item.new_file:
            return "A"
        elif diff_item.deleted_file:
            return "D"
        elif diff_item.renamed:
            return "R"
        else:
            return "M"

    def _count_diff_lines(self, diff_item) -> Tuple[int, int]:
        """Count additions and deletions in a diff."""
        additions = 0
        deletions = 0

        try:
            diff_text = diff_item.diff
            if diff_text:
                if isinstance(diff_text, bytes):
                    diff_text = diff_text.decode("utf-8", errors="replace")

                for line in diff_text.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        additions += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        deletions += 1
        except Exception:
            pass

        return additions, deletions

    def _get_diff_content(self) -> str:
        """Get the full diff content for staged changes."""
        try:
            return self.repo.git.diff("--cached", "--no-color")
        except GitCommandError:
            # No HEAD (initial commit scenario)
            return self.repo.git.diff("--cached", "--no-color", "--no-index", "/dev/null")

    def get_file_changes(self) -> List[FileChange]:
        """
        Get list of changed files grouped by type.

        Returns:
            List of FileChange objects.
        """
        staged = self.get_staged_changes()
        return staged.files

    def analyze_change_complexity(self) -> ChangeMetrics:
        """
        Analyze the complexity of staged changes.

        Returns:
            ChangeMetrics object with complexity analysis.
        """
        staged = self.get_staged_changes()

        if staged.is_empty:
            return ChangeMetrics(
                total_lines_changed=0,
                total_files=0,
                files_added=0,
                files_modified=0,
                files_deleted=0,
                files_renamed=0,
                directories_affected=0,
                file_types={},
                complexity_score=0,
            )

        files_added = sum(1 for f in staged.files if f.status == "A")
        files_modified = sum(1 for f in staged.files if f.status == "M")
        files_deleted = sum(1 for f in staged.files if f.status == "D")
        files_renamed = sum(1 for f in staged.files if f.status == "R")

        # Collect unique directories
        directories = set()
        for f in staged.files:
            dir_path = os.path.dirname(f.file_path)
            if dir_path:
                directories.add(dir_path)

        # Collect file types
        file_types = {}
        for f in staged.files:
            ext = os.path.splitext(f.file_path)[1].lower() or "no_extension"
            file_types[ext] = file_types.get(ext, 0) + 1

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            staged.total_additions + staged.total_deletions,
            len(staged.files),
            len(directories),
            len(file_types),
        )

        return ChangeMetrics(
            total_lines_changed=staged.total_additions + staged.total_deletions,
            total_files=len(staged.files),
            files_added=files_added,
            files_modified=files_modified,
            files_deleted=files_deleted,
            files_renamed=files_renamed,
            directories_affected=len(directories),
            file_types=file_types,
            complexity_score=complexity_score,
        )

    def _calculate_complexity_score(
        self, lines: int, files: int, dirs: int, types: int
    ) -> int:
        """
        Calculate a complexity score for the changes.

        Higher scores indicate more complex changes that might need splitting.
        """
        score = 0

        # Lines changed contribution
        if lines > 500:
            score += 50
        elif lines > 200:
            score += 30
        elif lines > 100:
            score += 15
        elif lines > 50:
            score += 5

        # Files changed contribution
        if files > 20:
            score += 30
        elif files > 10:
            score += 20
        elif files > 5:
            score += 10
        elif files > 2:
            score += 5

        # Directories affected contribution
        if dirs > 10:
            score += 20
        elif dirs > 5:
            score += 10
        elif dirs > 2:
            score += 5

        # File types contribution (different types = potentially different concerns)
        if types > 5:
            score += 15
        elif types > 3:
            score += 10
        elif types > 1:
            score += 5

        return score

    def validate_staged_changes(self) -> bool:
        """
        Validate that there are staged changes to commit.

        Returns:
            True if there are staged changes.

        Raises:
            NoStagedChangesError: If no changes are staged.
        """
        staged = self.get_staged_changes()
        if staged.is_empty:
            raise NoStagedChangesError(
                "No changes staged for commit.\n"
                "Use 'git add <file>' to stage changes."
            )
        return True

    def create_commit(self, message: str) -> CommitResult:
        """
        Create a commit with the given message.

        Args:
            message: Commit message.

        Returns:
            CommitResult with success status and commit SHA.
        """
        try:
            self.validate_staged_changes()

            # Create the commit
            commit = self.repo.index.commit(message)

            return CommitResult(
                success=True,
                sha=commit.hexsha,
                message=message,
            )

        except NoStagedChangesError as e:
            return CommitResult(
                success=False,
                sha=None,
                message=message,
                error=str(e),
            )
        except GitCommandError as e:
            return CommitResult(
                success=False,
                sha=None,
                message=message,
                error=f"Git commit failed: {e}",
            )

    def show_last_commit(self) -> str:
        """
        Get the last commit log with stats (git log -1 --stat).

        Returns:
            Formatted commit log string.
        """
        try:
            return self.repo.git.log("-1", "--stat", "--no-color")
        except GitCommandError as e:
            return f"Error getting commit log: {e}"

    def get_current_branch(self) -> str:
        """
        Get the name of the current branch.

        Returns:
            Branch name or 'HEAD' if detached.
        """
        try:
            return self.repo.active_branch.name
        except TypeError:
            return "HEAD (detached)"

    def get_remote_url(self) -> Optional[str]:
        """
        Get the URL of the origin remote.

        Returns:
            Remote URL or None if not configured.
        """
        try:
            if "origin" in self.repo.remotes:
                return self.repo.remotes.origin.url
            return None
        except Exception:
            return None

    def has_uncommitted_changes(self) -> bool:
        """
        Check if there are any uncommitted changes (staged or unstaged).

        Returns:
            True if there are uncommitted changes.
        """
        return self.repo.is_dirty(untracked_files=True)

    def get_unstaged_changes(self) -> List[str]:
        """
        Get list of files with unstaged changes.

        Returns:
            List of file paths with unstaged changes.
        """
        try:
            unstaged = self.repo.index.diff(None)
            return [d.a_path or d.b_path for d in unstaged]
        except GitCommandError:
            return []

    def get_untracked_files(self) -> List[str]:
        """
        Get list of untracked files.

        Returns:
            List of untracked file paths.
        """
        return self.repo.untracked_files

    def stage_files(self, files: List[str]) -> None:
        """
        Stage specific files.

        Args:
            files: List of file paths to stage.
        """
        self.repo.index.add(files)

    def unstage_files(self, files: List[str]) -> None:
        """
        Unstage specific files.

        Args:
            files: List of file paths to unstage.
        """
        self.repo.index.reset(paths=files)

    def get_repo_name(self) -> str:
        """
        Get the repository name from remote URL or directory name.

        Returns:
            Repository name.
        """
        remote_url = self.get_remote_url()
        if remote_url:
            # Extract repo name from URL
            # Handle SSH format: git@github.com:user/repo.git
            # Handle HTTPS format: https://github.com/user/repo.git
            match = re.search(r"[/:]([^/]+/[^/]+?)(?:\.git)?$", remote_url)
            if match:
                return match.group(1)

        # Fallback to directory name
        return os.path.basename(self.repo_path)

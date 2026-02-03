"""
Intelligent commit splitting logic.

Analyzes staged changes and suggests splitting large commits
into smaller, logical commits.
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from .git_operations import FileChange, StagedChanges, ChangeMetrics
from .conventional_commit import CommitType, CommitTypeDetector


class FileCategory(Enum):
    """Categories of files based on their purpose."""

    SOURCE = "source"
    TEST = "test"
    DOCS = "docs"
    CONFIG = "config"
    BUILD = "build"
    STYLE = "style"
    OTHER = "other"


@dataclass
class SplitGroup:
    """A group of related files that should be committed together."""

    name: str
    description: str
    files: List[FileChange]
    category: FileCategory
    suggested_type: CommitType
    total_additions: int = 0
    total_deletions: int = 0
    rationale: str = ""

    @property
    def total_lines(self) -> int:
        """Total lines changed in this group."""
        return self.total_additions + self.total_deletions

    @property
    def file_count(self) -> int:
        """Number of files in this group."""
        return len(self.files)


@dataclass
class SplitProposal:
    """Proposal for splitting a commit into multiple commits."""

    should_split: bool
    groups: List[SplitGroup]
    rationale: str
    original_metrics: ChangeMetrics

    @property
    def total_commits(self) -> int:
        """Total number of proposed commits."""
        return len(self.groups) if self.should_split else 1


class FileCategorizer:
    """Categorizes files based on their paths and types."""

    # Patterns for categorizing files
    CATEGORY_PATTERNS = {
        FileCategory.TEST: [
            r"test[s]?/",
            r"__tests__/",
            r"_test\.",
            r"\.test\.",
            r"\.spec\.",
            r"test_",
        ],
        FileCategory.DOCS: [
            r"\.md$",
            r"\.rst$",
            r"\.txt$",
            r"^docs?/",
            r"README",
            r"CHANGELOG",
            r"LICENSE",
            r"CONTRIBUTING",
        ],
        FileCategory.CONFIG: [
            r"\.json$",
            r"\.ya?ml$",
            r"\.toml$",
            r"\.ini$",
            r"\.cfg$",
            r"\.conf$",
            r"\.env",
            r"\.gitignore$",
            r"\.editorconfig$",
            r"\.prettierrc",
            r"\.eslintrc",
            r"tsconfig",
            r"jest\.config",
            r"webpack\.config",
            r"babel\.config",
        ],
        FileCategory.BUILD: [
            r"Dockerfile",
            r"docker-compose",
            r"Makefile$",
r"^\\.github/",
            r"\.gitlab-ci",
            r"\.travis",
            r"\.circleci/",
            r"azure-pipelines",
            r"Jenkinsfile",
            r"package\.json$",
            r"requirements\.txt$",
            r"setup\.py$",
            r"pyproject\.toml$",
            r"go\.mod$",
            r"Cargo\.toml$",
            r"pom\.xml$",
            r"build\.gradle",
        ],
        FileCategory.STYLE: [
            r"\.css$",
            r"\.scss$",
            r"\.sass$",
            r"\.less$",
            r"\.styled\.",
        ],
    }

    # File extensions that indicate source code
    SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".clj", ".ex", ".exs", ".erl", ".hs",
    }

    @classmethod
    def categorize(cls, file_path: str) -> FileCategory:
        """
        Categorize a file based on its path.

        Args:
            file_path: Path to the file.

        Returns:
            FileCategory for the file.
        """
        # Check patterns in order of specificity
        for category, patterns in cls.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return category

        # Check file extension for source files
        ext = os.path.splitext(file_path)[1].lower()
        if ext in cls.SOURCE_EXTENSIONS:
            return FileCategory.SOURCE

        return FileCategory.OTHER


class ComponentDetector:
    """Detects logical components from file paths."""

    @classmethod
    def detect_component(cls, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group files by their logical component.

        Args:
            file_paths: List of file paths.

        Returns:
            Dictionary mapping component names to file paths.
        """
        components = {}

        for path in file_paths:
            component = cls._extract_component(path)
            if component not in components:
                components[component] = []
            components[component].append(path)

        return components

    @classmethod
    def _extract_component(cls, path: str) -> str:
        """Extract component name from a file path."""
        parts = path.split("/")

        # Skip common root directories
        skip_dirs = {"src", "lib", "pkg", "app", "internal", "cmd"}

        # Find the meaningful directory
        for i, part in enumerate(parts):
            if part.lower() in skip_dirs:
                if i + 1 < len(parts) - 1:  # Not the last directory
                    return parts[i + 1]
            elif part not in (".", "..") and i < len(parts) - 1:
                return part

        # Fallback to filename without extension
        if parts:
            return os.path.splitext(parts[-1])[0]

        return "root"


class CommitSplitter:
    """Analyzes changes and suggests commit splits."""

    def __init__(
        self,
        max_commit_size: int = 200,
        complexity_threshold: int = 50,
    ):
        """
        Initialize the commit splitter.

        Args:
            max_commit_size: Maximum lines per commit before suggesting split.
            complexity_threshold: Complexity score threshold for suggesting split.
        """
        self.max_commit_size = max_commit_size
        self.complexity_threshold = complexity_threshold

    def analyze(
        self, staged_changes: StagedChanges, metrics: ChangeMetrics
    ) -> SplitProposal:
        """
        Analyze staged changes and propose splits if needed.

        Args:
            staged_changes: Staged changes from git.
            metrics: Change metrics analysis.

        Returns:
            SplitProposal with recommendations.
        """
        # Check if split is needed
        should_split = self._should_split(staged_changes, metrics)

        if not should_split:
            return SplitProposal(
                should_split=False,
                groups=[],
                rationale="Changes are small enough for a single commit.",
                original_metrics=metrics,
            )

        # Generate split groups
        groups = self._generate_groups(staged_changes)

        # Filter out empty groups and single-file trivial groups
        groups = [g for g in groups if g.file_count > 0]

        # If we only have one meaningful group, don't split
        if len(groups) <= 1:
            return SplitProposal(
                should_split=False,
                groups=[],
                rationale="All changes belong to a single logical group.",
                original_metrics=metrics,
            )

        # Generate rationale
        rationale = self._generate_rationale(metrics, groups)

        return SplitProposal(
            should_split=True,
            groups=groups,
            rationale=rationale,
            original_metrics=metrics,
        )

    def _should_split(
        self, staged_changes: StagedChanges, metrics: ChangeMetrics
    ) -> bool:
        """Determine if changes should be split."""
        # Check total lines changed
        total_lines = staged_changes.total_additions + staged_changes.total_deletions
        if total_lines > self.max_commit_size:
            return True

        # Check complexity score
        if metrics.complexity_score > self.complexity_threshold:
            return True

        # Check for mixed change types (e.g., source + tests + docs)
        categories = set()
        for f in staged_changes.files:
            category = FileCategorizer.categorize(f.file_path)
            categories.add(category)

        # If we have multiple unrelated categories, suggest split
        unrelated_categories = {
            FileCategory.SOURCE,
            FileCategory.TEST,
            FileCategory.DOCS,
        }
        if len(categories.intersection(unrelated_categories)) >= 2:
            return True

        return False

    def _generate_groups(self, staged_changes: StagedChanges) -> List[SplitGroup]:
        """Generate logical groups for splitting."""
        groups = []

        # Group by category first
        category_files: Dict[FileCategory, List[FileChange]] = {}
        for f in staged_changes.files:
            category = FileCategorizer.categorize(f.file_path)
            if category not in category_files:
                category_files[category] = []
            category_files[category].append(f)

        # Create groups for each category
        for category, files in category_files.items():
            if not files:
                continue

            # Further split large source groups by component
            if category == FileCategory.SOURCE and len(files) > 5:
                component_groups = self._split_by_component(files, category)
                groups.extend(component_groups)
            else:
                group = self._create_group(files, category)
                groups.append(group)

        # Sort groups by suggested commit order
        groups = self._sort_groups(groups)

        return groups

    def _split_by_component(
        self, files: List[FileChange], category: FileCategory
    ) -> List[SplitGroup]:
        """Split files by their logical component."""
        groups = []
        file_paths = [f.file_path for f in files]
        file_map = {f.file_path: f for f in files}

        components = ComponentDetector.detect_component(file_paths)

        for component, paths in components.items():
            component_files = [file_map[p] for p in paths]
            group = self._create_group(
                component_files, category, component_name=component
            )
            groups.append(group)

        return groups

    def _create_group(
        self,
        files: List[FileChange],
        category: FileCategory,
        component_name: Optional[str] = None,
    ) -> SplitGroup:
        """Create a SplitGroup from files."""
        total_additions = sum(f.additions for f in files)
        total_deletions = sum(f.deletions for f in files)

        # Determine suggested commit type
        file_paths = [f.file_path for f in files]
        suggested_type = CommitTypeDetector.detect_type(file_paths)

        # Generate group name and description
        if component_name:
            name = f"{category.value}: {component_name}"
            description = f"Changes to {component_name} ({category.value})"
        else:
            name = category.value
            description = f"{category.value.title()} changes"

        # Generate rationale
        if category == FileCategory.TEST:
            rationale = "Test files should be committed separately to clearly identify test changes."
        elif category == FileCategory.DOCS:
            rationale = "Documentation changes should be in their own commit for clear history."
        elif category == FileCategory.CONFIG:
            rationale = "Configuration changes may need separate review and rollback capability."
        elif category == FileCategory.BUILD:
            rationale = "Build/CI changes should be isolated for easier debugging of build issues."
        else:
            rationale = f"Group of related {category.value} changes."

        return SplitGroup(
            name=name,
            description=description,
            files=files,
            category=category,
            suggested_type=suggested_type,
            total_additions=total_additions,
            total_deletions=total_deletions,
            rationale=rationale,
        )

    def _sort_groups(self, groups: List[SplitGroup]) -> List[SplitGroup]:
        """Sort groups by recommended commit order."""
        # Priority order: build/config first, then source, tests, docs last
        priority = {
            FileCategory.BUILD: 0,
            FileCategory.CONFIG: 1,
            FileCategory.SOURCE: 2,
            FileCategory.STYLE: 3,
            FileCategory.TEST: 4,
            FileCategory.DOCS: 5,
            FileCategory.OTHER: 6,
        }

        return sorted(groups, key=lambda g: priority.get(g.category, 99))

    def _generate_rationale(
        self, metrics: ChangeMetrics, groups: List[SplitGroup]
    ) -> str:
        """Generate rationale for the split proposal."""
        reasons = []

        total_lines = metrics.total_lines_changed
        if total_lines > self.max_commit_size:
            reasons.append(
                f"Total changes ({total_lines} lines) exceed recommended "
                f"maximum ({self.max_commit_size} lines)"
            )

        if metrics.complexity_score > self.complexity_threshold:
            reasons.append(
                f"Complexity score ({metrics.complexity_score}) exceeds "
                f"threshold ({self.complexity_threshold})"
            )

        if len(groups) > 2:
            categories = set(g.category.value for g in groups)
            reasons.append(
                f"Changes span multiple categories: {', '.join(categories)}"
            )

        if metrics.directories_affected > 5:
            reasons.append(
                f"Changes affect {metrics.directories_affected} directories"
            )

        rationale = "Suggested split because:\n"
        rationale += "\n".join(f"  - {r}" for r in reasons)

        return rationale


def suggest_commit_split(
    staged_changes: StagedChanges,
    metrics: ChangeMetrics,
    max_commit_size: int = 200,
    complexity_threshold: int = 50,
) -> SplitProposal:
    """
    Convenience function to analyze and suggest commit splits.

    Args:
        staged_changes: Staged changes from git.
        metrics: Change metrics analysis.
        max_commit_size: Maximum lines per commit.
        complexity_threshold: Complexity threshold.

    Returns:
        SplitProposal with recommendations.
    """
    splitter = CommitSplitter(
        max_commit_size=max_commit_size,
        complexity_threshold=complexity_threshold,
    )
    return splitter.analyze(staged_changes, metrics)

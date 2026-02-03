"""
Tests for the commit splitter module.
"""

import pytest

from codecraft_ai.commit_splitter import (
    CommitSplitter,
    FileCategorizer,
    ComponentDetector,
    FileCategory,
    SplitGroup,
    SplitProposal,
    suggest_commit_split,
)
from codecraft_ai.git_operations import FileChange, StagedChanges, ChangeMetrics
from codecraft_ai.conventional_commit import CommitType


class TestFileCategory:
    """Tests for FileCategory enum."""

    def test_file_category_values(self):
        """Test FileCategory enum values."""
        assert FileCategory.SOURCE.value == "source"
        assert FileCategory.TEST.value == "test"
        assert FileCategory.DOCS.value == "docs"
        assert FileCategory.CONFIG.value == "config"
        assert FileCategory.BUILD.value == "build"
        assert FileCategory.STYLE.value == "style"
        assert FileCategory.OTHER.value == "other"


class TestFileCategorizer:
    """Tests for FileCategorizer class."""

    def test_categorize_test_files(self):
        """Test categorizing test files."""
        test_paths = [
            "tests/test_api.py",
            "test_utils.py",
            "src/__tests__/Button.test.tsx",
            "api.spec.js",
        ]

        for path in test_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.TEST, f"Failed for {path}"

    def test_categorize_docs_files(self):
        """Test categorizing documentation files."""
        doc_paths = [
            "README.md",
            "docs/guide.md",
            "CHANGELOG.rst",
            "LICENSE",
        ]

        for path in doc_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.DOCS, f"Failed for {path}"

    def test_categorize_config_files(self):
        """Test categorizing configuration files."""
        config_paths = [
            "config.json",
            "settings.yaml",
            ".eslintrc",
            ".prettierrc",
            "tsconfig.json",
        ]

        for path in config_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.CONFIG, f"Failed for {path}"

    def test_categorize_build_files(self):
        """Test categorizing build files."""
        build_paths = [
            "Dockerfile",
            ".github/workflows/ci.yml",
            "Makefile",
            "package.json",
            "requirements.txt",
        ]

        for path in build_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.BUILD, f"Failed for {path}"

    def test_categorize_source_files(self):
        """Test categorizing source files."""
        source_paths = [
            "src/app.py",
            "lib/utils.js",
            "main.go",
            "App.tsx",
        ]

        for path in source_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.SOURCE, f"Failed for {path}"

    def test_categorize_style_files(self):
        """Test categorizing style files."""
        style_paths = [
            "styles.css",
            "app.scss",
            "theme.less",
        ]

        for path in style_paths:
            category = FileCategorizer.categorize(path)
            assert category == FileCategory.STYLE, f"Failed for {path}"


class TestComponentDetector:
    """Tests for ComponentDetector class."""

    def test_detect_component_single_file(self):
        """Test detecting component from single file."""
        components = ComponentDetector.detect_component(["src/api/users.py"])
        assert "api" in components

    def test_detect_component_multiple_files(self):
        """Test detecting components from multiple files."""
        paths = [
            "src/api/users.py",
            "src/api/auth.py",
            "src/models/user.py",
        ]
        components = ComponentDetector.detect_component(paths)

        assert "api" in components
        assert "models" in components
        assert len(components["api"]) == 2
        assert len(components["models"]) == 1

    def test_detect_component_empty_list(self):
        """Test detecting components from empty list."""
        components = ComponentDetector.detect_component([])
        assert len(components) == 0


class TestSplitGroup:
    """Tests for SplitGroup dataclass."""

    def test_split_group_properties(self):
        """Test SplitGroup properties."""
        files = [
            FileChange("test.py", "M", 10, 5, False),
            FileChange("utils.py", "A", 20, 0, False),
        ]

        group = SplitGroup(
            name="source",
            description="Source code changes",
            files=files,
            category=FileCategory.SOURCE,
            suggested_type=CommitType.FEAT,
            total_additions=30,
            total_deletions=5,
        )

        assert group.total_lines == 35
        assert group.file_count == 2


class TestSplitProposal:
    """Tests for SplitProposal dataclass."""

    def test_split_proposal_with_groups(self):
        """Test SplitProposal with groups."""
        groups = [
            SplitGroup(
                name="source",
                description="Source changes",
                files=[],
                category=FileCategory.SOURCE,
                suggested_type=CommitType.FEAT,
            ),
            SplitGroup(
                name="tests",
                description="Test changes",
                files=[],
                category=FileCategory.TEST,
                suggested_type=CommitType.TEST,
            ),
        ]

        metrics = ChangeMetrics(
            total_lines_changed=200,
            total_files=10,
            files_added=5,
            files_modified=5,
            files_deleted=0,
            files_renamed=0,
            directories_affected=5,
            complexity_score=60,
        )

        proposal = SplitProposal(
            should_split=True,
            groups=groups,
            rationale="Large change",
            original_metrics=metrics,
        )

        assert proposal.total_commits == 2

    def test_split_proposal_no_split(self):
        """Test SplitProposal when not splitting."""
        metrics = ChangeMetrics(
            total_lines_changed=50,
            total_files=2,
            files_added=1,
            files_modified=1,
            files_deleted=0,
            files_renamed=0,
            directories_affected=1,
            complexity_score=10,
        )

        proposal = SplitProposal(
            should_split=False,
            groups=[],
            rationale="Small change",
            original_metrics=metrics,
        )

        assert proposal.total_commits == 1


class TestCommitSplitter:
    """Tests for CommitSplitter class."""

    def test_should_not_split_small_changes(self):
        """Test that small changes are not split."""
        splitter = CommitSplitter(max_commit_size=200, complexity_threshold=50)

        files = [FileChange("test.py", "M", 10, 5, False)]
        staged = StagedChanges(
            files=files,
            total_additions=10,
            total_deletions=5,
            total_files=1,
            diff_content="",
        )
        metrics = ChangeMetrics(
            total_lines_changed=15,
            total_files=1,
            files_added=0,
            files_modified=1,
            files_deleted=0,
            files_renamed=0,
            directories_affected=1,
            complexity_score=5,
        )

        proposal = splitter.analyze(staged, metrics)

        assert proposal.should_split is False

    def test_should_split_large_changes(self):
        """Test that large changes are split."""
        splitter = CommitSplitter(max_commit_size=100, complexity_threshold=50)

        files = [
            FileChange("src/api.py", "M", 100, 50, False),
            FileChange("tests/test_api.py", "M", 50, 20, False),
            FileChange("README.md", "M", 10, 0, False),
        ]
        staged = StagedChanges(
            files=files,
            total_additions=160,
            total_deletions=70,
            total_files=3,
            diff_content="",
        )
        metrics = ChangeMetrics(
            total_lines_changed=230,
            total_files=3,
            files_added=0,
            files_modified=3,
            files_deleted=0,
            files_renamed=0,
            directories_affected=3,
            complexity_score=60,
        )

        proposal = splitter.analyze(staged, metrics)

        assert proposal.should_split is True
        assert len(proposal.groups) >= 2

    def test_should_split_mixed_categories(self):
        """Test that mixed categories suggest split."""
        splitter = CommitSplitter(max_commit_size=500, complexity_threshold=100)

        files = [
            FileChange("src/app.py", "M", 20, 10, False),
            FileChange("tests/test_app.py", "M", 20, 10, False),
            FileChange("docs/README.md", "M", 5, 0, False),
        ]
        staged = StagedChanges(
            files=files,
            total_additions=45,
            total_deletions=20,
            total_files=3,
            diff_content="",
        )
        metrics = ChangeMetrics(
            total_lines_changed=65,
            total_files=3,
            files_added=0,
            files_modified=3,
            files_deleted=0,
            files_renamed=0,
            directories_affected=3,
            complexity_score=30,
        )

        proposal = splitter.analyze(staged, metrics)

        # Should suggest split due to mixed categories (source + test + docs)
        assert proposal.should_split is True


class TestSuggestCommitSplit:
    """Tests for suggest_commit_split convenience function."""

    def test_suggest_commit_split_small_change(self):
        """Test convenience function for small changes."""
        files = [FileChange("test.py", "M", 5, 2, False)]
        staged = StagedChanges(
            files=files,
            total_additions=5,
            total_deletions=2,
            total_files=1,
            diff_content="",
        )
        metrics = ChangeMetrics(
            total_lines_changed=7,
            total_files=1,
            files_added=0,
            files_modified=1,
            files_deleted=0,
            files_renamed=0,
            directories_affected=1,
            complexity_score=5,
        )

        proposal = suggest_commit_split(staged, metrics)

        assert proposal.should_split is False

    def test_suggest_commit_split_with_custom_threshold(self):
        """Test convenience function with custom thresholds."""
        files = [
            FileChange("src/feature.py", "M", 30, 10, False),
            FileChange("tests/test_feature.py", "A", 20, 0, False),
            FileChange("docs/feature.md", "M", 5, 0, False),
        ]
        staged = StagedChanges(
            files=files,
            total_additions=55,
            total_deletions=10,
            total_files=3,
            diff_content="",
        )
        metrics = ChangeMetrics(
            total_lines_changed=65,
            total_files=3,
            files_added=1,
            files_modified=2,
            files_deleted=0,
            files_renamed=0,
            directories_affected=3,
            complexity_score=20,
        )

        # With low threshold and mixed categories, should suggest split
        proposal = suggest_commit_split(
            staged, metrics, max_commit_size=50, complexity_threshold=10
        )

        assert proposal.should_split is True
        assert len(proposal.groups) >= 2

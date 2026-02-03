"""
Tests for the conventional commit module.
"""

import pytest

from codecraft_ai.conventional_commit import (
    CommitType,
    ConventionalCommit,
    ConventionalCommitParser,
    CommitMessageFormatter,
    ScopeExtractor,
    CommitTypeDetector,
)


class TestCommitType:
    """Tests for CommitType enum."""

    def test_commit_type_values(self):
        """Test that all expected commit types exist."""
        expected_types = [
            "feat", "fix", "docs", "style", "refactor",
            "test", "chore", "perf", "ci", "build", "revert"
        ]

        for type_name in expected_types:
            commit_type = CommitType.from_string(type_name)
            assert commit_type is not None
            assert commit_type.type_name == type_name

    def test_commit_type_from_string_invalid(self):
        """Test that invalid types return None."""
        assert CommitType.from_string("invalid") is None
        assert CommitType.from_string("") is None

    def test_commit_type_all_types(self):
        """Test getting all type names."""
        all_types = CommitType.all_types()
        assert "feat" in all_types
        assert "fix" in all_types
        assert len(all_types) == 11

    def test_commit_type_properties(self):
        """Test CommitType properties."""
        feat = CommitType.FEAT
        assert feat.type_name == "feat"
        assert feat.description == "A new feature"
        assert feat.color == "green"


class TestConventionalCommit:
    """Tests for ConventionalCommit dataclass."""

    def test_format_basic(self):
        """Test basic commit message formatting."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope=None,
            subject="add new feature",
        )

        formatted = commit.format()
        assert formatted == "feat: add new feature"

    def test_format_with_scope(self):
        """Test commit message with scope."""
        commit = ConventionalCommit(
            type=CommitType.FIX,
            scope="api",
            subject="fix authentication bug",
        )

        formatted = commit.format()
        assert formatted == "fix(api): fix authentication bug"

    def test_format_with_body(self):
        """Test commit message with body."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope="auth",
            subject="add login endpoint",
            body="Implement JWT-based authentication.",
        )

        formatted = commit.format()
        assert "feat(auth): add login endpoint" in formatted
        assert "Implement JWT-based authentication." in formatted

    def test_format_breaking_change(self):
        """Test commit message with breaking change."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope="api",
            subject="change response format",
            breaking=True,
            breaking_description="Response format changed from XML to JSON",
        )

        formatted = commit.format()
        assert "feat(api)!: change response format" in formatted
        assert "BREAKING CHANGE:" in formatted
        assert "XML to JSON" in formatted

    def test_validate_valid_commit(self):
        """Test validation of valid commit."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope="api",
            subject="add new endpoint",
        )

        is_valid, errors = commit.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_subject_too_long(self):
        """Test validation catches long subject."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope=None,
            subject="this is a very long subject line that exceeds the fifty character limit",
        )

        is_valid, errors = commit.validate()
        assert is_valid is False
        assert any("too long" in e.lower() for e in errors)

    def test_validate_subject_uppercase(self):
        """Test validation catches uppercase subject."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope=None,
            subject="Add new feature",  # Starts with uppercase
        )

        is_valid, errors = commit.validate()
        assert is_valid is False
        assert any("lowercase" in e.lower() for e in errors)

    def test_validate_subject_trailing_period(self):
        """Test validation catches trailing period."""
        commit = ConventionalCommit(
            type=CommitType.FEAT,
            scope=None,
            subject="add new feature.",  # Has trailing period
        )

        is_valid, errors = commit.validate()
        assert is_valid is False
        assert any("period" in e.lower() for e in errors)


class TestConventionalCommitParser:
    """Tests for ConventionalCommitParser."""

    def test_parse_basic(self):
        """Test parsing basic commit message."""
        commit = ConventionalCommitParser.parse("feat: add new feature")

        assert commit is not None
        assert commit.type == CommitType.FEAT
        assert commit.scope is None
        assert commit.subject == "add new feature"

    def test_parse_with_scope(self):
        """Test parsing commit with scope."""
        commit = ConventionalCommitParser.parse("fix(api): handle null values")

        assert commit is not None
        assert commit.type == CommitType.FIX
        assert commit.scope == "api"
        assert commit.subject == "handle null values"

    def test_parse_breaking_change(self):
        """Test parsing breaking change."""
        commit = ConventionalCommitParser.parse("feat(api)!: change response format")

        assert commit is not None
        assert commit.breaking is True

    def test_parse_with_body(self):
        """Test parsing commit with body."""
        message = """feat(auth): add login endpoint

This implements JWT-based authentication.
Includes login, logout, and refresh endpoints."""

        commit = ConventionalCommitParser.parse(message)

        assert commit is not None
        assert commit.subject == "add login endpoint"
        assert commit.body is not None
        assert "JWT-based" in commit.body

    def test_parse_invalid_type(self):
        """Test parsing invalid commit type."""
        commit = ConventionalCommitParser.parse("invalid: some message")
        assert commit is None

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        commit = ConventionalCommitParser.parse("no colon here")
        assert commit is None


class TestCommitMessageFormatter:
    """Tests for CommitMessageFormatter."""

    def test_format_subject_lowercase(self):
        """Test subject formatting to lowercase."""
        formatted = CommitMessageFormatter.format_subject("Add new feature")
        assert formatted == "add new feature"

    def test_format_subject_remove_period(self):
        """Test subject formatting removes period."""
        formatted = CommitMessageFormatter.format_subject("add new feature.")
        assert formatted == "add new feature"

    def test_format_subject_truncate(self):
        """Test subject truncation."""
        long_subject = "this is a very long subject line that needs to be truncated because it exceeds the limit"
        formatted = CommitMessageFormatter.format_subject(long_subject)
        assert len(formatted) <= 50
        assert formatted.endswith("...")

    def test_format_body_preserves_bullets(self):
        """Test body formatting preserves bullet points."""
        body = "- First item\n- Second item\n- Third item"
        formatted = CommitMessageFormatter.format_body(body)
        assert "- First item" in formatted
        assert "- Second item" in formatted

    def test_format_bullet_list(self):
        """Test formatting bullet list."""
        items = ["First item", "Second item", "Third item"]
        formatted = CommitMessageFormatter.format_bullet_list(items)
        assert "- First item" in formatted
        assert "- Second item" in formatted
        assert "- Third item" in formatted

    def test_create_commit_message(self):
        """Test creating full commit message."""
        message = CommitMessageFormatter.create_commit_message(
            commit_type=CommitType.FEAT,
            subject="Add login feature",
            scope="auth",
            body="Implement user authentication.",
        )

        assert "feat(auth):" in message
        assert "add login feature" in message
        assert "Implement user authentication." in message


class TestScopeExtractor:
    """Tests for ScopeExtractor."""

    def test_extract_scope_single_file(self):
        """Test extracting scope from single file."""
        scope = ScopeExtractor.extract_scope(["src/components/Button.tsx"])
        assert scope == "components"

    def test_extract_scope_common_pattern(self):
        """Test extracting scope from common patterns."""
        test_cases = [
            (["tests/test_api.py"], "tests"),
            (["docs/README.md"], "docs"),
            ([".github/workflows/ci.yml"], "ci"),
        ]

        for paths, expected in test_cases:
            scope = ScopeExtractor.extract_scope(paths)
            assert scope == expected, f"Expected {expected} for {paths}, got {scope}"

    def test_extract_scope_multiple_files_same(self):
        """Test extracting scope from multiple files in same directory."""
        paths = [
            "src/api/users.py",
            "src/api/auth.py",
            "src/api/routes.py",
        ]
        scope = ScopeExtractor.extract_scope(paths)
        assert scope == "api"

    def test_extract_scope_empty_list(self):
        """Test extracting scope from empty list."""
        scope = ScopeExtractor.extract_scope([])
        assert scope is None


class TestCommitTypeDetector:
    """Tests for CommitTypeDetector."""

    def test_detect_type_docs(self):
        """Test detecting docs commit type."""
        paths = ["README.md", "docs/guide.md"]
        commit_type = CommitTypeDetector.detect_type(paths)
        assert commit_type == CommitType.DOCS

    def test_detect_type_tests(self):
        """Test detecting test commit type."""
        paths = ["tests/test_api.py", "tests/test_models.py"]
        commit_type = CommitTypeDetector.detect_type(paths)
        assert commit_type == CommitType.TEST

    def test_detect_type_ci(self):
        """Test detecting CI commit type."""
        paths = [".github/workflows/ci.yml"]
        commit_type = CommitTypeDetector.detect_type(paths)
        assert commit_type == CommitType.CI

    def test_detect_type_build(self):
        """Test detecting build commit type."""
        paths = ["package.json", "requirements.txt"]
        commit_type = CommitTypeDetector.detect_type(paths)
        assert commit_type == CommitType.BUILD

    def test_detect_type_from_diff_content(self):
        """Test detecting type from diff content."""
        paths = ["src/app.py"]
        diff_content = "fix bug in authentication"
        commit_type = CommitTypeDetector.detect_type(paths, diff_content)
        assert commit_type == CommitType.FIX

    def test_detect_type_empty_paths(self):
        """Test detecting type with empty paths."""
        commit_type = CommitTypeDetector.detect_type([])
        assert commit_type == CommitType.CHORE

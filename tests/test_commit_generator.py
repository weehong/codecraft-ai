"""
Tests for the commit generator module.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from codecraft_ai.commit_generator import (
    CommitGenerator,
    GeneratedCommit,
    CommitGeneratorError,
    APIError,
    RateLimitError,
    InvalidResponseError,
    generate_commit_message,
    validate_conventional_commit,
)
from codecraft_ai.conventional_commit import CommitType
from codecraft_ai.commit_config import CommitConfig


class TestGeneratedCommit:
    """Tests for GeneratedCommit dataclass."""

    def test_from_dict_basic(self):
        """Test creating GeneratedCommit from dictionary."""
        data = {
            "type": "feat",
            "scope": "api",
            "subject": "add new endpoint",
            "body": None,
            "breaking": False,
            "breaking_description": None,
        }

        commit = GeneratedCommit.from_dict(data)

        assert commit.type == CommitType.FEAT
        assert commit.scope == "api"
        assert commit.subject == "add new endpoint"
        assert commit.breaking is False

    def test_from_dict_with_body(self):
        """Test creating GeneratedCommit with body."""
        data = {
            "type": "fix",
            "scope": "auth",
            "subject": "fix login bug",
            "body": "Handle null values properly.",
            "breaking": False,
            "breaking_description": None,
        }

        commit = GeneratedCommit.from_dict(data)

        assert commit.type == CommitType.FIX
        assert commit.body == "Handle null values properly."
        assert "fix(auth):" in commit.formatted_message

    def test_from_dict_breaking_change(self):
        """Test creating GeneratedCommit with breaking change."""
        data = {
            "type": "feat",
            "scope": None,
            "subject": "change api format",
            "body": None,
            "breaking": True,
            "breaking_description": "Response format changed",
        }

        commit = GeneratedCommit.from_dict(data)

        assert commit.breaking is True
        assert commit.breaking_description == "Response format changed"
        assert "!" in commit.formatted_message

    def test_from_dict_invalid_type_fallback(self):
        """Test fallback for invalid commit type."""
        data = {
            "type": "invalid",
            "scope": None,
            "subject": "some change",
            "body": None,
            "breaking": False,
            "breaking_description": None,
        }

        commit = GeneratedCommit.from_dict(data)

        # Should fallback to CHORE
        assert commit.type == CommitType.CHORE

    def test_from_dict_missing_fields(self):
        """Test handling missing fields."""
        data = {
            "type": "feat",
        }

        commit = GeneratedCommit.from_dict(data)

        assert commit.type == CommitType.FEAT
        assert commit.subject == "update code"  # Default


class TestCommitGenerator:
    """Tests for CommitGenerator class."""

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_init_with_config(self, mock_openai):
        """Test initialization with config."""
        config = CommitConfig(
            openai_api_key="test_key",
            openai_model="gpt-4",
        )

        generator = CommitGenerator(config)

        assert generator.config == config
        mock_openai.assert_called_once_with(api_key="test_key")

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        config = CommitConfig(openai_api_key=None)

        with pytest.raises(CommitGeneratorError):
            CommitGenerator(config)

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_generate_commit_message(self, mock_openai):
        """Test generating commit message."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "type": "feat",
                "scope": "api",
                "subject": "add new endpoint",
                "body": None,
                "breaking": False,
                "breaking_description": None,
            })))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        config = CommitConfig(
            openai_api_key="test_key",
            openai_model="gpt-4",
        )

        generator = CommitGenerator(config)
        commit = generator.generate_commit_message(
            diff_content="+new line\n-old line",
            file_paths=["src/api.py"],
        )

        assert commit.type == CommitType.FEAT
        assert commit.scope == "api"
        assert commit.subject == "add new endpoint"

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_generate_commit_message_with_context(self, mock_openai):
        """Test generating commit message with additional context."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "type": "fix",
                "scope": "auth",
                "subject": "handle edge case",
                "body": None,
                "breaking": False,
                "breaking_description": None,
            })))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        config = CommitConfig(
            openai_api_key="test_key",
            openai_model="gpt-4",
        )

        generator = CommitGenerator(config)
        commit = generator.generate_commit_message(
            diff_content="+fix\n-bug",
            file_paths=["src/auth.py"],
            context={
                "project_type": "web",
                "language": "Python",
                "existing_messages": ["feat: add login", "fix: handle errors"],
            },
        )

        assert commit is not None
        assert commit.type == CommitType.FIX

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_generate_commit_message_truncates_long_diff(self, mock_openai):
        """Test that long diffs are truncated."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "type": "feat",
                "scope": None,
                "subject": "update code",
                "body": None,
                "breaking": False,
                "breaking_description": None,
            })))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        config = CommitConfig(openai_api_key="test_key")
        generator = CommitGenerator(config)

        # Create a very long diff
        long_diff = "+" + "x" * 10000 + "\n" + "-" + "y" * 10000

        commit = generator.generate_commit_message(
            diff_content=long_diff,
            file_paths=["test.py"],
        )

        # Should not raise and should truncate
        assert commit is not None

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_generate_commit_message_empty_response(self, mock_openai):
        """Test handling empty API response."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        config = CommitConfig(openai_api_key="test_key")
        generator = CommitGenerator(config)

        with pytest.raises(InvalidResponseError):
            generator.generate_commit_message(
                diff_content="+test",
                file_paths=["test.py"],
            )

    @patch("codecraft_ai.commit_generator.OpenAI")
    def test_generate_commit_message_invalid_json(self, mock_openai):
        """Test handling invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="not valid json"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        config = CommitConfig(openai_api_key="test_key")
        generator = CommitGenerator(config)

        with pytest.raises(InvalidResponseError):
            generator.generate_commit_message(
                diff_content="+test",
                file_paths=["test.py"],
            )


class TestValidateConventionalCommit:
    """Tests for validate_conventional_commit function."""

    def test_validate_valid_commit(self):
        """Test validating a valid commit message."""
        is_valid, errors = validate_conventional_commit("feat(api): add new endpoint")
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_format(self):
        """Test validating invalid format."""
        is_valid, errors = validate_conventional_commit("no format here")
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_with_body(self):
        """Test validating commit with body."""
        message = """feat(auth): add login

Implement JWT authentication."""
        is_valid, errors = validate_conventional_commit(message)
        assert is_valid is True

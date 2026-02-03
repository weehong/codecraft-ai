"""
OpenAI-powered commit message generation.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI, OpenAIError

from .commit_config import CommitConfig, get_openai_prompt_config
from .conventional_commit import (
    CommitType,
    ConventionalCommit,
    CommitMessageFormatter,
    ScopeExtractor,
    CommitTypeDetector,
)
from .commit_splitter import SplitGroup


@dataclass
class GeneratedCommit:
    """A generated commit message."""

    type: CommitType
    scope: Optional[str]
    subject: str
    body: Optional[str]
    breaking: bool
    breaking_description: Optional[str]
    formatted_message: str
    confidence: float = 0.9

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedCommit":
        """Create from dictionary (OpenAI response)."""
        commit_type = CommitType.from_string(data.get("type", "chore"))
        if not commit_type:
            commit_type = CommitType.CHORE

        scope = data.get("scope")
        subject = data.get("subject", "update code")
        body = data.get("body")
        breaking = data.get("breaking", False)
        breaking_description = data.get("breaking_description")

        # Format the message
        formatted = CommitMessageFormatter.create_commit_message(
            commit_type=commit_type,
            subject=subject,
            scope=scope,
            body=body,
            breaking=breaking,
            breaking_description=breaking_description,
        )

        return cls(
            type=commit_type,
            scope=scope,
            subject=subject,
            body=body,
            breaking=breaking,
            breaking_description=breaking_description,
            formatted_message=formatted,
        )


class CommitGeneratorError(Exception):
    """Base exception for commit generator errors."""

    pass


class APIError(CommitGeneratorError):
    """OpenAI API error."""

    pass


class RateLimitError(CommitGeneratorError):
    """API rate limit exceeded."""

    pass


class InvalidResponseError(CommitGeneratorError):
    """Invalid response from API."""

    pass


class CommitGenerator:
    """Generates commit messages using OpenAI."""

    def __init__(self, config: Optional[CommitConfig] = None):
        """
        Initialize the commit generator.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or CommitConfig.from_env()

        if not self.config.openai_api_key:
            raise CommitGeneratorError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.prompt_config = get_openai_prompt_config()

    def generate_commit_message(
        self,
        diff_content: str,
        file_paths: List[str],
        context: Optional[Dict] = None,
    ) -> GeneratedCommit:
        """
        Generate a commit message for the given diff.

        Args:
            diff_content: Git diff content.
            file_paths: List of changed file paths.
            context: Optional additional context.

        Returns:
            GeneratedCommit with the generated message.
        """
        # Truncate diff if too large
        max_diff_length = 8000
        if len(diff_content) > max_diff_length:
            diff_content = diff_content[:max_diff_length] + "\n... (truncated)"

        # Build the prompt
        messages = self._build_messages(diff_content, file_paths, context)

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                temperature=self.config.openai_temperature,
                max_tokens=self.config.openai_max_tokens,
                response_format={"type": "json_object"},
            )

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise InvalidResponseError("Empty response from API")

            data = json.loads(content)
            return GeneratedCommit.from_dict(data)

        except json.JSONDecodeError as e:
            raise InvalidResponseError(f"Failed to parse API response as JSON: {e}")
        except OpenAIError as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError("OpenAI API rate limit exceeded. Please try again later.")
            raise APIError(f"OpenAI API error: {e}")

    def _build_messages(
        self,
        diff_content: str,
        file_paths: List[str],
        context: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages for the OpenAI API call."""
        # System message
        system_message = (
            f"{self.prompt_config['system_role']}\n\n"
            f"{self.prompt_config['format_instructions']}\n\n"
            f"{self.prompt_config['output_format']}"
        )

        # Add examples
        examples_text = "\nExamples:\n"
        for example in self.prompt_config["examples"][:2]:
            examples_text += f"\nDiff description: {example['diff']}\n"
            examples_text += f"Response: {json.dumps(example['response'], indent=2)}\n"

        system_message += examples_text

        # User message
        user_content = f"Generate a commit message for the following changes:\n\n"

        # Add file list
        user_content += "Changed files:\n"
        for path in file_paths[:20]:  # Limit to first 20 files
            user_content += f"  - {path}\n"
        if len(file_paths) > 20:
            user_content += f"  ... and {len(file_paths) - 20} more files\n"

        # Add context if provided
        if context:
            if context.get("project_type"):
                user_content += f"\nProject type: {context['project_type']}"
            if context.get("language"):
                user_content += f"\nPrimary language: {context['language']}"
            if context.get("existing_messages"):
                user_content += "\nRecent commit messages for style reference:\n"
                for msg in context["existing_messages"][:3]:
                    user_content += f"  - {msg[:100]}\n"

        # Add diff
        user_content += f"\nDiff content:\n```\n{diff_content}\n```"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def generate_split_commits(
        self, groups: List[SplitGroup], context: Optional[Dict] = None
    ) -> List[GeneratedCommit]:
        """
        Generate commit messages for split commit groups.

        Args:
            groups: List of split groups.
            context: Optional additional context.

        Returns:
            List of GeneratedCommit objects.
        """
        commits = []

        for group in groups:
            file_paths = [f.file_path for f in group.files]

            # Build a mini-diff representation for the group
            diff_summary = self._build_group_diff_summary(group)

            try:
                commit = self.generate_commit_message(
                    diff_content=diff_summary,
                    file_paths=file_paths,
                    context=context,
                )
                commits.append(commit)
            except CommitGeneratorError:
                # Fallback to a basic commit message
                fallback = self._create_fallback_commit(group)
                commits.append(fallback)

        return commits

    def _build_group_diff_summary(self, group: SplitGroup) -> str:
        """Build a diff summary for a group of files."""
        summary = f"Category: {group.category.value}\n"
        summary += f"Description: {group.description}\n"
        summary += f"Total lines: +{group.total_additions} -{group.total_deletions}\n"
        summary += "\nFiles:\n"

        for f in group.files:
            status_map = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
            status = status_map.get(f.status, f.status)
            summary += f"  - {f.file_path} ({status}, +{f.additions} -{f.deletions})\n"

        return summary

    def _create_fallback_commit(self, group: SplitGroup) -> GeneratedCommit:
        """Create a fallback commit message when generation fails."""
        file_paths = [f.file_path for f in group.files]

        # Try to extract scope
        scope = ScopeExtractor.extract_scope(file_paths)

        # Determine subject based on category
        category_subjects = {
            "source": "update source code",
            "test": "update tests",
            "docs": "update documentation",
            "config": "update configuration",
            "build": "update build configuration",
            "style": "update styles",
            "other": "update files",
        }
        subject = category_subjects.get(group.category.value, "update files")

        formatted = CommitMessageFormatter.create_commit_message(
            commit_type=group.suggested_type,
            subject=subject,
            scope=scope,
        )

        return GeneratedCommit(
            type=group.suggested_type,
            scope=scope,
            subject=subject,
            body=None,
            breaking=False,
            breaking_description=None,
            formatted_message=formatted,
            confidence=0.5,
        )

    def regenerate_with_feedback(
        self,
        previous_message: str,
        feedback: str,
        diff_content: str,
        file_paths: List[str],
    ) -> GeneratedCommit:
        """
        Regenerate a commit message with user feedback.

        Args:
            previous_message: The previously generated message.
            feedback: User feedback on what to change.
            diff_content: Original diff content.
            file_paths: List of changed files.

        Returns:
            New GeneratedCommit.
        """
        # Build messages with feedback context
        messages = self._build_messages(diff_content, file_paths)

        # Add previous message and feedback
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "type": "chore",
                "scope": None,
                "subject": previous_message.split("\n")[0],
                "body": None,
                "breaking": False,
                "breaking_description": None,
            }),
        })
        messages.append({
            "role": "user",
            "content": f"Please improve the commit message based on this feedback: {feedback}",
        })

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                temperature=self.config.openai_temperature,
                max_tokens=self.config.openai_max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise InvalidResponseError("Empty response from API")

            data = json.loads(content)
            return GeneratedCommit.from_dict(data)

        except Exception as e:
            raise APIError(f"Failed to regenerate commit message: {e}")


def generate_commit_message(
    diff_content: str,
    file_paths: List[str],
    config: Optional[CommitConfig] = None,
    context: Optional[Dict] = None,
) -> GeneratedCommit:
    """
    Convenience function to generate a commit message.

    Args:
        diff_content: Git diff content.
        file_paths: List of changed file paths.
        config: Optional configuration.
        context: Optional additional context.

    Returns:
        GeneratedCommit with the generated message.
    """
    generator = CommitGenerator(config)
    return generator.generate_commit_message(diff_content, file_paths, context)


def validate_conventional_commit(message: str) -> tuple[bool, List[str]]:
    """
    Validate that a commit message follows Conventional Commits.

    Args:
        message: Commit message to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    from .conventional_commit import ConventionalCommitParser

    parsed = ConventionalCommitParser.parse(message)
    if not parsed:
        return False, ["Message does not follow Conventional Commits format."]

    return parsed.validate()

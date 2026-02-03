"""
Configuration management for the AI-powered commit message generator.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv


@dataclass
class CommitConfig:
    """Configuration for the commit message generator."""

    # GitHub settings
    github_token: Optional[str] = None
    github_per_page: int = 30

    # GitLab settings
    gitlab_token: Optional[str] = None
    gitlab_url: str = "https://gitlab.com"
    gitlab_per_page: int = 30

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1024

    # Commit splitting settings
    max_commit_size: int = 200
    complexity_threshold: int = 50

    # Commit types customization
    custom_types: Dict[str, str] = field(default_factory=dict)

    # Scope patterns based on project structure
    scope_patterns: Dict[str, str] = field(default_factory=dict)

    # Files to exclude from analysis
    exclude_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "CommitConfig":
        """
        Load configuration from environment variables.

        Args:
            env_path: Optional path to .env file.

        Returns:
            CommitConfig instance.
        """
        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        return cls(
            # GitHub settings
            github_token=os.getenv("GITHUB_TOKEN"),
            github_per_page=int(os.getenv("GITHUB_PER_PAGE", "30")),
            # GitLab settings
            gitlab_token=os.getenv("GITLAB_TOKEN"),
            gitlab_url=os.getenv("GITLAB_URL", "https://gitlab.com"),
            gitlab_per_page=int(os.getenv("GITLAB_PER_PAGE", "30")),
            # OpenAI settings
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
            # Commit splitting
            max_commit_size=int(os.getenv("MAX_COMMIT_SIZE", "200")),
            complexity_threshold=int(os.getenv("COMPLEXITY_THRESHOLD", "50")),
            # Default exclude patterns
            exclude_patterns=[
                "*.lock",
                "*.log",
                "node_modules/",
                "__pycache__/",
                ".git/",
                "*.pyc",
                ".DS_Store",
            ],
        )

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the configuration.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check required settings for specific features
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for AI-powered commit generation.")

        # Validate OpenAI model (models that support JSON response format)
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo-0125"]
        if self.openai_model not in valid_models:
            errors.append(
                f"Invalid OPENAI_MODEL: {self.openai_model}. "
                f"Valid options: {', '.join(valid_models)}"
            )

        # Validate temperature
        if not 0.0 <= self.openai_temperature <= 2.0:
            errors.append("OPENAI_TEMPERATURE must be between 0.0 and 2.0.")

        # Validate max_commit_size
        if self.max_commit_size < 10:
            errors.append("MAX_COMMIT_SIZE must be at least 10.")

        return len(errors) == 0, errors

    def validate_github(self) -> tuple[bool, List[str]]:
        """
        Validate GitHub-specific configuration.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        if not self.github_token:
            errors.append("GITHUB_TOKEN is required for GitHub repository access.")

        if self.github_per_page < 1 or self.github_per_page > 100:
            errors.append("GITHUB_PER_PAGE must be between 1 and 100.")

        return len(errors) == 0, errors

    def validate_gitlab(self) -> tuple[bool, List[str]]:
        """
        Validate GitLab-specific configuration.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        if not self.gitlab_token:
            errors.append("GITLAB_TOKEN is required for GitLab repository access.")

        if self.gitlab_per_page < 1 or self.gitlab_per_page > 100:
            errors.append("GITLAB_PER_PAGE must be between 1 and 100.")

        if not self.gitlab_url:
            errors.append("GITLAB_URL is required.")

        return len(errors) == 0, errors

    def validate_openai(self) -> tuple[bool, List[str]]:
        """
        Validate OpenAI-specific configuration.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required.")

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "github_per_page": self.github_per_page,
            "gitlab_url": self.gitlab_url,
            "gitlab_per_page": self.gitlab_per_page,
            "openai_model": self.openai_model,
            "openai_temperature": self.openai_temperature,
            "openai_max_tokens": self.openai_max_tokens,
            "max_commit_size": self.max_commit_size,
            "complexity_threshold": self.complexity_threshold,
            "has_github_token": bool(self.github_token),
            "has_gitlab_token": bool(self.gitlab_token),
            "has_openai_key": bool(self.openai_api_key),
        }


class ConfigurationError(Exception):
    """Configuration error."""

    pass


def get_config(env_path: Optional[str] = None) -> CommitConfig:
    """
    Get the configuration, validating required settings.

    Args:
        env_path: Optional path to .env file.

    Returns:
        CommitConfig instance.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    config = CommitConfig.from_env(env_path)

    is_valid, errors = config.validate()
    if not is_valid:
        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return config


def get_openai_prompt_config() -> dict:
    """
    Get configuration for OpenAI prompt engineering.

    Returns:
        Dictionary with prompt configuration.
    """
    return {
        "system_role": "You are an expert at writing concise, descriptive git commit messages following the Conventional Commits v1.0.0 specification.",
        "format_instructions": """
Follow these rules for commit messages:
1. Use one of these types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert
2. Optionally include a scope in parentheses after the type: feat(api):
3. Use imperative mood in the subject line (e.g., "add" not "added")
4. Keep the subject line under 50 characters
5. Start subject with lowercase letter
6. Do not end subject with a period
7. If a body is needed, separate it from the subject with a blank line
8. Wrap body text at 72 characters
9. Use the body to explain what and why, not how
10. Start each sentence in the body with a capital letter
11. Each sentence in the body should start with an action verb (Introduce, Add, Update, Include, Fix, etc.)
12. Body sentences should be complete and descriptive
""",
        "output_format": """
Return your response as a JSON object with these fields:
{
    "type": "feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert",
    "scope": "optional scope string or null",
    "subject": "imperative mood description under 50 chars",
    "body": "optional longer description or null",
    "breaking": false,
    "breaking_description": "description if breaking is true, else null"
}
""",
        "examples": [
            {
                "diff": "Added Dockerfile and docker-compose.yml for containerization",
                "response": {
                    "type": "feat",
                    "scope": "deploy",
                    "subject": "add Dockerfile for app containerization",
                    "body": "Introduce Dockerfile and related configurations for containerizing the\napplication.\nIncludes Docker Compose setup and environment variable examples for both\nDocker and Compose environments.",
                    "breaking": False,
                    "breaking_description": None,
                },
            },
            {
                "diff": "Added new user authentication endpoint",
                "response": {
                    "type": "feat",
                    "scope": "auth",
                    "subject": "add user authentication endpoint",
                    "body": "Implement JWT-based authentication with refresh tokens.\nInclude login, logout, and token refresh endpoints.",
                    "breaking": False,
                    "breaking_description": None,
                },
            },
            {
                "diff": "Fixed null pointer exception in data parser",
                "response": {
                    "type": "fix",
                    "scope": "parser",
                    "subject": "handle null values in data parser",
                    "body": "Add null checks before processing data fields.\nPrevent crashes when API returns incomplete responses.",
                    "breaking": False,
                    "breaking_description": None,
                },
            },
            {
                "diff": "Updated README with installation instructions",
                "response": {
                    "type": "docs",
                    "scope": None,
                    "subject": "add installation instructions to README",
                    "body": None,
                    "breaking": False,
                    "breaking_description": None,
                },
            },
        ],
    }

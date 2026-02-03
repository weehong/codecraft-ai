"""
Conventional Commit formatting and validation.

Implements the Conventional Commits v1.0.0 specification.
https://www.conventionalcommits.org/en/v1.0.0/
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class CommitType(Enum):
    """Valid commit types according to Conventional Commits."""

    FEAT = ("feat", "A new feature", "green")
    FIX = ("fix", "A bug fix", "red")
    DOCS = ("docs", "Documentation only changes", "blue")
    STYLE = ("style", "Changes that do not affect the meaning of the code", "magenta")
    REFACTOR = ("refactor", "A code change that neither fixes a bug nor adds a feature", "yellow")
    TEST = ("test", "Adding missing tests or correcting existing tests", "cyan")
    CHORE = ("chore", "Other changes that don't modify src or test files", "dim")
    PERF = ("perf", "A code change that improves performance", "green")
    CI = ("ci", "Changes to CI configuration files and scripts", "blue")
    BUILD = ("build", "Changes that affect the build system or external dependencies", "yellow")
    REVERT = ("revert", "Reverts a previous commit", "red")

    def __init__(self, type_name: str, description: str, color: str):
        self.type_name = type_name
        self.description = description
        self.color = color

    @classmethod
    def from_string(cls, type_str: str) -> Optional["CommitType"]:
        """Get CommitType from string name."""
        type_str = type_str.lower()
        for commit_type in cls:
            if commit_type.type_name == type_str:
                return commit_type
        return None

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all valid type names."""
        return [ct.type_name for ct in cls]


@dataclass
class ConventionalCommit:
    """Represents a conventional commit message."""

    type: CommitType
    scope: Optional[str]
    subject: str
    body: Optional[str] = None
    footer: Optional[str] = None
    breaking: bool = False
    breaking_description: Optional[str] = None

    def format(self) -> str:
        """Format the commit message according to conventional commit spec."""
        # Build the header line
        header = self.type.type_name
        if self.scope:
            header += f"({self.scope})"
        if self.breaking:
            header += "!"
        header += f": {self.subject}"

        # Build the full message
        parts = [header]

        if self.body:
            parts.append("")  # Blank line
            parts.append(self.body)

        if self.breaking and self.breaking_description:
            parts.append("")
            parts.append(f"BREAKING CHANGE: {self.breaking_description}")

        if self.footer:
            parts.append("")
            parts.append(self.footer)

        return "\n".join(parts)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the commit message.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Validate subject length
        if len(self.subject) > 50:
            errors.append(f"Subject line too long ({len(self.subject)} chars). Maximum is 50 characters.")

        # Validate subject format
        if self.subject and self.subject[0].isupper():
            errors.append("Subject should start with lowercase letter.")

        if self.subject and self.subject.endswith("."):
            errors.append("Subject should not end with a period.")

        # Validate body line lengths
        if self.body:
            for i, line in enumerate(self.body.split("\n"), 1):
                if len(line) > 72:
                    errors.append(f"Body line {i} too long ({len(line)} chars). Maximum is 72 characters.")

        # Validate scope format
        if self.scope:
            if not re.match(r"^[a-z][a-z0-9-]*$", self.scope):
                errors.append("Scope should be lowercase alphanumeric with hyphens.")

        return len(errors) == 0, errors


class ConventionalCommitParser:
    """Parser for conventional commit messages."""

    # Regex pattern for parsing conventional commits
    PATTERN = re.compile(
        r"^(?P<type>\w+)"  # Type
        r"(?:\((?P<scope>[^)]+)\))?"  # Optional scope
        r"(?P<breaking>!)?"  # Optional breaking change indicator
        r":\s*"  # Colon separator
        r"(?P<subject>.+)$",  # Subject line
        re.MULTILINE,
    )

    @classmethod
    def parse(cls, message: str) -> Optional[ConventionalCommit]:
        """
        Parse a commit message into a ConventionalCommit.

        Args:
            message: Full commit message string.

        Returns:
            ConventionalCommit object or None if parsing fails.
        """
        lines = message.strip().split("\n")
        if not lines:
            return None

        # Parse header
        header = lines[0]
        match = cls.PATTERN.match(header)
        if not match:
            return None

        type_str = match.group("type")
        commit_type = CommitType.from_string(type_str)
        if not commit_type:
            return None

        scope = match.group("scope")
        breaking = bool(match.group("breaking"))
        subject = match.group("subject").strip()

        # Parse body and footer
        body = None
        footer = None
        breaking_description = None

        if len(lines) > 2:
            # Skip blank line after header
            remaining = "\n".join(lines[2:]) if lines[1] == "" else "\n".join(lines[1:])

            # Check for BREAKING CHANGE footer
            if "BREAKING CHANGE:" in remaining:
                parts = remaining.split("BREAKING CHANGE:", 1)
                body = parts[0].strip() or None
                footer_parts = parts[1].strip().split("\n", 1)
                breaking_description = footer_parts[0].strip()
                if len(footer_parts) > 1:
                    footer = footer_parts[1].strip()
                breaking = True
            else:
                # Check for other footers (issue references, etc.)
                body_lines = []
                footer_lines = []
                in_footer = False

                for line in remaining.split("\n"):
                    # Check if this looks like a footer line
                    if re.match(r"^[\w-]+(-by)?:\s", line, re.IGNORECASE) or re.match(
                        r"^(Fixes|Closes|Resolves)\s+#\d+", line, re.IGNORECASE
                    ):
                        in_footer = True
                        footer_lines.append(line)
                    elif in_footer:
                        footer_lines.append(line)
                    else:
                        body_lines.append(line)

                body = "\n".join(body_lines).strip() or None
                footer = "\n".join(footer_lines).strip() or None

        return ConventionalCommit(
            type=commit_type,
            scope=scope,
            subject=subject,
            body=body,
            footer=footer,
            breaking=breaking,
            breaking_description=breaking_description,
        )


class CommitMessageFormatter:
    """Formats commit messages according to Conventional Commits."""

    MAX_SUBJECT_LENGTH = 50
    MAX_BODY_LINE_LENGTH = 72

    @classmethod
    def format_subject(cls, subject: str) -> str:
        """
        Format the subject line.

        - Lowercase first letter
        - Remove trailing period
        - Truncate if too long
        """
        if not subject:
            return subject

        # Lowercase first letter
        subject = subject[0].lower() + subject[1:] if subject else subject

        # Remove trailing period
        subject = subject.rstrip(".")

        # Truncate if necessary
        if len(subject) > cls.MAX_SUBJECT_LENGTH:
            subject = subject[: cls.MAX_SUBJECT_LENGTH - 3] + "..."

        return subject

    @classmethod
    def format_body(cls, body: str) -> str:
        """
        Format the body text.

        - Wrap lines to 72 characters
        - Preserve bullet points and code blocks
        """
        if not body:
            return body

        lines = body.split("\n")
        formatted_lines = []

        for line in lines:
            # Preserve code blocks and bullet points
            if line.startswith("```") or line.startswith("  ") or line.startswith("- ") or line.startswith("* "):
                formatted_lines.append(line)
            elif len(line) > cls.MAX_BODY_LINE_LENGTH:
                # Wrap long lines
                formatted_lines.extend(cls._wrap_line(line))
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    @classmethod
    def _wrap_line(cls, line: str) -> List[str]:
        """Wrap a single line to MAX_BODY_LINE_LENGTH."""
        words = line.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + (1 if current_line else 0) <= cls.MAX_BODY_LINE_LENGTH:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    @classmethod
    def format_bullet_list(cls, items: List[str]) -> str:
        """Format a list of items as bullet points."""
        return "\n".join(f"- {item}" for item in items)

    @classmethod
    def create_commit_message(
        cls,
        commit_type: CommitType,
        subject: str,
        scope: Optional[str] = None,
        body: Optional[str] = None,
        footer: Optional[str] = None,
        breaking: bool = False,
        breaking_description: Optional[str] = None,
    ) -> str:
        """
        Create a formatted conventional commit message.

        Args:
            commit_type: Type of commit.
            subject: Subject line (will be formatted).
            scope: Optional scope.
            body: Optional body text.
            footer: Optional footer (issue references, etc.).
            breaking: Whether this is a breaking change.
            breaking_description: Description of breaking change.

        Returns:
            Formatted commit message string.
        """
        commit = ConventionalCommit(
            type=commit_type,
            scope=scope,
            subject=cls.format_subject(subject),
            body=cls.format_body(body) if body else None,
            footer=footer,
            breaking=breaking,
            breaking_description=breaking_description,
        )

        return commit.format()


class ScopeExtractor:
    """Extracts scope from file paths."""

    # Common scope mappings based on directory patterns
    SCOPE_PATTERNS = {
        r"^src/components/": "components",
        r"^src/api/": "api",
        r"^src/services/": "services",
        r"^src/utils/": "utils",
        r"^src/hooks/": "hooks",
        r"^src/store/": "store",
        r"^src/models/": "models",
        r"^src/views/": "views",
        r"^src/pages/": "pages",
        r"^src/lib/": "lib",
        r"^tests?/": "tests",
        r"^docs?/": "docs",
        r"^config/": "config",
        r"^scripts/": "scripts",
        r"\.github/": "ci",
        r"^\.": "config",
    }

    @classmethod
    def extract_scope(cls, file_paths: List[str]) -> Optional[str]:
        """
        Extract a common scope from file paths.

        Args:
            file_paths: List of changed file paths.

        Returns:
            Scope string or None if no common scope found.
        """
        if not file_paths:
            return None

        if len(file_paths) == 1:
            return cls._scope_from_path(file_paths[0])

        # Find common scope for multiple files
        scopes = [cls._scope_from_path(path) for path in file_paths]
        scopes = [s for s in scopes if s]  # Remove None values

        if not scopes:
            return None

        # Return the most common scope
        scope_counts = {}
        for scope in scopes:
            scope_counts[scope] = scope_counts.get(scope, 0) + 1

        most_common = max(scope_counts, key=scope_counts.get)

        # Only return if it's reasonably common
        if scope_counts[most_common] >= len(file_paths) / 2:
            return most_common

        return None

    @classmethod
    def _scope_from_path(cls, path: str) -> Optional[str]:
        """Extract scope from a single file path."""
        # Check against known patterns
        for pattern, scope in cls.SCOPE_PATTERNS.items():
            if re.match(pattern, path):
                return scope

        # Try to extract from first directory
        parts = path.split("/")
        if len(parts) > 1:
            first_dir = parts[0].lower()
            if first_dir in ("src", "lib", "pkg"):
                if len(parts) > 2:
                    return parts[1].lower()
            elif first_dir not in (".", ".."):
                return first_dir

        return None


class CommitTypeDetector:
    """Detects the appropriate commit type from changes."""

    # File patterns that suggest specific commit types
    TYPE_PATTERNS = {
        CommitType.DOCS: [
            r"\.md$",
            r"\.rst$",
            r"\.txt$",
            r"^docs?/",
            r"README",
            r"LICENSE",
            r"CHANGELOG",
        ],
        CommitType.TEST: [
            r"test[s_]?/",
            r"_test\.",
            r"\.test\.",
            r"\.spec\.",
            r"__tests__/",
        ],
        CommitType.CI: [
            r"\.github/",
            r"\.gitlab-ci",
            r"Jenkinsfile",
            r"\.travis",
            r"\.circleci/",
            r"azure-pipelines",
        ],
        CommitType.BUILD: [
            r"package\.json$",
            r"package-lock\.json$",
            r"yarn\.lock$",
            r"requirements\.txt$",
            r"setup\.py$",
            r"pyproject\.toml$",
            r"Makefile$",
            r"Dockerfile",
            r"docker-compose",
            r"\.gradle",
            r"pom\.xml$",
        ],
        CommitType.STYLE: [
            r"\.css$",
            r"\.scss$",
            r"\.less$",
            r"\.styled\.",
        ],
        CommitType.CHORE: [
            r"\.gitignore$",
            r"\.editorconfig$",
            r"\.prettierrc",
            r"\.eslintrc",
            r"tsconfig\.json$",
        ],
    }

    @classmethod
    def detect_type(
        cls, file_paths: List[str], diff_content: Optional[str] = None
    ) -> CommitType:
        """
        Detect the most appropriate commit type.

        Args:
            file_paths: List of changed file paths.
            diff_content: Optional diff content for analysis.

        Returns:
            Detected CommitType.
        """
        if not file_paths:
            return CommitType.CHORE

        # Check each file against patterns
        type_matches = {}
        for commit_type, patterns in cls.TYPE_PATTERNS.items():
            for path in file_paths:
                for pattern in patterns:
                    if re.search(pattern, path, re.IGNORECASE):
                        type_matches[commit_type] = type_matches.get(commit_type, 0) + 1
                        break

        # If we found matches, return the most common
        if type_matches:
            return max(type_matches, key=type_matches.get)

        # Analyze diff content for hints
        if diff_content:
            content_lower = diff_content.lower()

            # Check for bug fix indicators
            if any(word in content_lower for word in ["fix", "bug", "issue", "error", "crash"]):
                return CommitType.FIX

            # Check for feature indicators
            if any(word in content_lower for word in ["add", "new", "feature", "implement"]):
                return CommitType.FEAT

            # Check for refactor indicators
            if any(word in content_lower for word in ["refactor", "rename", "move", "restructure"]):
                return CommitType.REFACTOR

            # Check for performance indicators
            if any(word in content_lower for word in ["performance", "optimize", "speed", "cache"]):
                return CommitType.PERF

        # Default to feat for new files, refactor for modifications
        new_files = sum(1 for p in file_paths if "new" in p.lower() or not any(
            re.search(pattern, p) for patterns in cls.TYPE_PATTERNS.values() for pattern in patterns
        ))

        if new_files > len(file_paths) / 2:
            return CommitType.FEAT

        return CommitType.REFACTOR

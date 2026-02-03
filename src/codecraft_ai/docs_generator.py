"""
Documentation Generator - Docs-as-Code Tool

Analyzes git diffs and generates documentation updates using AI providers.
Supports OpenAI (default) and DeepSeek.
"""

import logging
import os
import re
import subprocess
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION MAPPING ---
PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "base_url": None,  # Defaults to OpenAI's standard URL
        "model": "gpt-4-turbo",
        "name": "OpenAI GPT-4 Turbo"
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "name": "DeepSeek V3"
    }
}

# --- THE MASTER PROMPT ---
SYSTEM_PROMPT = """
You are the Lead Documentation Architect.
Your goal: Analyze code changes and generate Markdown documentation updates.

RULES:
1. Analyze the GIT DIFF provided.
2. Structure output for a standard Docs-as-Code hierarchy:
   - architecture/ (decisions, context)
   - design/ (data-model, security)
   - technical/ (api, setup)
   - operations/ (config, deploy)
3. Output format:
   - Start with "### FILE: [Relative Path]" (e.g., "04-operations/configuration.md")
   - Provide the Markdown content to append or replace.
   - Use Mermaid.js for diagrams.
4. If trivial (typo, formatting), output "NO_UPDATES".
5. If the input is a FULL COMMIT HISTORY (multiple commits from root to HEAD),
   generate comprehensive project documentation covering the entire codebase:
   - Provide a high-level architecture overview.
   - Document all major components, modules, and their interactions.
   - Include setup/installation instructions based on observed configuration.
   - Summarize the project's evolution from the commit history.
   - Treat this as initial documentation generation, not an incremental update.
"""

SEGMENT_SYSTEM_PROMPT = """
You are the Lead Documentation Architect.
You are processing SEGMENT {segment_num} of {total_segments} from a full commit history.

Your goal: Document the components, features, and architectural decisions introduced
in THIS segment's commits. A later step will combine all segment outputs into one
coherent documentation set.

RULES:
1. Analyze the commits provided in this segment.
2. Structure output for a standard Docs-as-Code hierarchy:
   - architecture/ (decisions, context)
   - design/ (data-model, security)
   - technical/ (api, setup)
   - operations/ (config, deploy)
3. Output format:
   - Start with "### FILE: [Relative Path]" (e.g., "04-operations/configuration.md")
   - Provide the Markdown content to append or replace.
   - Use Mermaid.js for diagrams.
4. Focus on what is NEW or CHANGED in this segment's commits.
5. If this segment contains only trivial changes (typos, formatting), output "NO_UPDATES".
"""

COMBINER_SYSTEM_PROMPT = """
You are the Lead Documentation Architect.
You are merging {num_segments} segment outputs into one coherent documentation set.

RULES:
1. Combine all segment outputs into a single, unified documentation set.
2. Deduplicate overlapping content — keep the most complete version.
3. Resolve contradictions: later segments override earlier ones.
4. Maintain the Docs-as-Code hierarchy:
   - architecture/ (decisions, context)
   - design/ (data-model, security)
   - technical/ (api, setup)
   - operations/ (config, deploy)
5. Output format:
   - Start with "### FILE: [Relative Path]" (e.g., "04-operations/configuration.md")
   - Provide the Markdown content to append or replace.
   - Use Mermaid.js for diagrams.
6. Produce comprehensive project documentation that reads as a complete set,
   not as a collection of incremental updates.
"""

logger = logging.getLogger(__name__)


def get_git_diff(repo_path: str) -> str:
    """Get the git diff between the last two commits.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        SystemExit: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_staged_diff(repo_path: str) -> str:
    """Get the git diff of staged changes.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_unstaged_diff(repo_path: str) -> str:
    """Get the git diff of unstaged changes.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_commit_diff(repo_path: str, commit_id: str = "HEAD") -> str:
    """Get the git diff for a specific commit.

    Args:
        repo_path: Path to the git repository.
        commit_id: Commit SHA or reference (default: HEAD).

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "show", "--format=", commit_id],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error getting commit {commit_id}: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_full_history_diff(repo_path: str) -> str:
    """Get the full patch history from the first commit to HEAD.

    Runs git log with patches in chronological order (--reverse) to produce
    a complete history of all changes from the repository root to the current HEAD.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The full patch history as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        # Find the root commit (first commit in history)
        root_result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        root_commit = root_result.stdout.strip().splitlines()[0]

        result = subprocess.run(
            [
                "git", "log", "-p", "--reverse",
                "--format=commit %H%nAuthor: %an%nDate: %ad%nSubject: %s%n",
                f"{root_commit}..HEAD"
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        # If there's only one commit (root), the range root..HEAD is empty.
        # In that case, show the root commit itself.
        output = result.stdout
        if not output.strip():
            result = subprocess.run(
                [
                    "git", "log", "-p", "--reverse",
                    "--format=commit %H%nAuthor: %an%nDate: %ad%nSubject: %s%n",
                    root_commit
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_full_history_commits(repo_path: str) -> list[dict]:
    """Parse the full commit history into individual commit dicts.

    Reuses get_full_history_diff() to get the raw output, then splits it
    into individual commits by matching 'commit <hash>' boundaries.

    Args:
        repo_path: Path to the git repository.

    Returns:
        A list of dicts with keys: hash, author, date, subject, patch.

    Raises:
        RuntimeError: If git command fails.
    """
    raw = get_full_history_diff(repo_path)
    if not raw.strip():
        return []

    # Split on commit boundaries: lines that start with "commit " followed by a 40-char hex hash
    parts = re.split(r'^(commit [0-9a-f]{40})$', raw, flags=re.MULTILINE)

    commits = []
    # parts[0] is any text before the first "commit ..." line (usually empty)
    # After that, parts alternate: [marker, body, marker, body, ...]
    i = 1
    while i < len(parts) - 1:
        marker = parts[i]       # "commit <hash>"
        body = parts[i + 1]     # everything until next commit marker
        i += 2

        commit_hash = marker.split(" ", 1)[1].strip()
        lines = body.lstrip("\n").split("\n")

        author = ""
        date = ""
        subject = ""
        patch_start = 0

        for idx, line in enumerate(lines):
            if line.startswith("Author: "):
                author = line[len("Author: "):].strip()
            elif line.startswith("Date: "):
                date = line[len("Date: "):].strip()
            elif line.startswith("Subject: "):
                subject = line[len("Subject: "):].strip()
                patch_start = idx + 1
                break
            elif line.startswith("diff ") or line.startswith("---"):
                # No subject line found, patch starts here
                patch_start = idx
                break
        else:
            patch_start = len(lines)

        patch = "\n".join(lines[patch_start:]).strip()

        commits.append({
            "hash": commit_hash,
            "author": author,
            "date": date,
            "subject": subject,
            "patch": patch,
        })

    return commits


def format_commit_for_prompt(commit: dict) -> str:
    """Format a commit dict back into text for an AI prompt.

    Args:
        commit: Dict with keys: hash, author, date, subject, patch.

    Returns:
        Formatted string with commit header and patch.
    """
    header = f"commit {commit['hash']}\n"
    if commit.get("author"):
        header += f"Author: {commit['author']}\n"
    if commit.get("date"):
        header += f"Date: {commit['date']}\n"
    if commit.get("subject"):
        header += f"Subject: {commit['subject']}\n"
    header += "\n"
    if commit.get("patch"):
        header += commit["patch"]
    return header


def chunk_commits_by_char_limit(commits: list[dict], char_limit: int) -> list[list[dict]]:
    """Group commits into segments that fit within a character limit.

    Never splits mid-commit. If a single commit exceeds the limit,
    it gets its own segment.

    Args:
        commits: List of commit dicts.
        char_limit: Maximum characters per segment.

    Returns:
        List of segments, where each segment is a list of commit dicts.
    """
    segments = []
    current_segment = []
    current_size = 0

    for commit in commits:
        commit_text = format_commit_for_prompt(commit)
        commit_size = len(commit_text)

        if current_segment and (current_size + commit_size) > char_limit:
            # Current segment is full, start a new one
            segments.append(current_segment)
            current_segment = []
            current_size = 0

        current_segment.append(commit)
        current_size += commit_size

    if current_segment:
        segments.append(current_segment)

    return segments


def generate_docs_chunked(
    commits: list[dict],
    provider_key: str,
    char_limit: int,
    progress_callback: callable = None,
) -> str:
    """Generate documentation from commits using chunked multi-call flow.

    Orchestrates:
    1. Chunk commits into segments
    2. For each segment: build prompt, call AI, invoke progress callback
    3. Combine outputs via combiner AI call or direct concatenation
    4. Return combined documentation string

    Args:
        commits: List of commit dicts from get_full_history_commits().
        provider_key: AI provider key ('openai' or 'deepseek').
        char_limit: Character limit per segment (provider-dependent).
        progress_callback: Optional callable(segment_num, total_segments, chars).

    Returns:
        Combined documentation string.

    Raises:
        RuntimeError: If all segment API calls fail.
    """
    # Reserve space for the system prompt in each segment
    prompt_overhead = 500  # approximate overhead for system prompt + user message wrapper
    effective_limit = max(char_limit - prompt_overhead, char_limit // 2)

    segments = chunk_commits_by_char_limit(commits, effective_limit)
    total_segments = len(segments)

    if total_segments == 0:
        return "No commits to process."

    config = PROVIDERS[provider_key]
    api_key = os.getenv(config["env_var"])
    if not api_key:
        raise ValueError(f"{config['env_var']} environment variable not found.")

    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    max_retries = 3
    backoff_delays = [1, 2, 4]

    segment_outputs = []
    failed_segments = 0

    for seg_idx, segment in enumerate(segments, 1):
        segment_text = "\n\n".join(format_commit_for_prompt(c) for c in segment)
        system_prompt = SEGMENT_SYSTEM_PROMPT.format(
            segment_num=seg_idx,
            total_segments=total_segments,
        )
        user_content = f"Here are the commits for segment {seg_idx}/{total_segments}:\n\n{segment_text}"

        result = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    stream=False,
                )
                result = response.choices[0].message.content
                break
            except Exception as e:
                logger.warning(
                    "Segment %d/%d attempt %d failed: %s",
                    seg_idx, total_segments, attempt + 1, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff_delays[attempt])

        if result and "NO_UPDATES" not in result:
            segment_outputs.append(result)
            if progress_callback:
                progress_callback(seg_idx, total_segments, len(result))
        else:
            if result is None:
                failed_segments += 1
                logger.error("Segment %d/%d failed after %d retries", seg_idx, total_segments, max_retries)
            if progress_callback:
                progress_callback(seg_idx, total_segments, 0)

    if not segment_outputs:
        if failed_segments == total_segments:
            raise RuntimeError("All segment API calls failed.")
        return "No documentation updates required."

    # Single segment — no combining needed
    if len(segment_outputs) == 1:
        return segment_outputs[0]

    # Try to combine via AI combiner call
    combined_text = "\n\n---\n\n".join(
        f"## Segment {i+1} Output\n\n{output}"
        for i, output in enumerate(segment_outputs)
    )

    if len(combined_text) <= char_limit:
        combiner_system = COMBINER_SYSTEM_PROMPT.format(num_segments=len(segment_outputs))
        combiner_user = f"Here are the segment outputs to combine:\n\n{combined_text}"

        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": combiner_system},
                    {"role": "user", "content": combiner_user},
                ],
                temperature=0.2,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Combiner call failed, falling back to concatenation: %s", e)

    # Fallback: direct concatenation with section separators
    return "\n\n---\n\n".join(
        f"<!-- Segment {i+1}/{len(segment_outputs)} -->\n\n{output}"
        for i, output in enumerate(segment_outputs)
    )


def generate_docs(diff_content: str, provider_key: str) -> str:
    """Generate documentation using the specified AI provider.

    Args:
        diff_content: The git diff to analyze.
        provider_key: The AI provider to use ('openai' or 'deepseek').

    Returns:
        The generated documentation as a string.

    Raises:
        ValueError: If API key is not configured.
        RuntimeError: If API call fails.
    """
    # Load configuration based on the chosen provider
    config = PROVIDERS[provider_key]
    api_key = os.getenv(config["env_var"])

    if not api_key:
        raise ValueError(f"{config['env_var']} environment variable not found.")

    # Initialize Client
    # Note: If base_url is None, the library defaults to OpenAI
    client = OpenAI(api_key=api_key, base_url=config["base_url"])

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the Git Diff:\n\n{diff_content}"}
            ],
            temperature=0.2,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error calling API: {e}")


def run_docs_generator(
    repo_path: str = ".",
    output_dir: str = None,
    provider: str = "openai",
    diff_source: str = "commit",
    commit_id: str = None,
    progress_callback: callable = None,
) -> tuple[bool, str]:
    """Run the documentation generator.

    Args:
        repo_path: Path to the git repository.
        output_dir: Directory to save the output. Defaults to <repo>/docs.
        provider: AI provider to use ('openai' or 'deepseek').
        diff_source: Source of diff ('commit', 'staged', 'unstaged', 'all',
                     'current_commit', 'specific_commit').
        commit_id: Specific commit ID when diff_source is 'specific_commit'.

    Returns:
        Tuple of (success: bool, message: str).
    """
    # Resolve Paths
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.join(repo_path, "docs")

    # Get Changes based on diff_source
    try:
        if diff_source == "commit":
            diff = get_git_diff(repo_path)
        elif diff_source == "staged":
            diff = get_staged_diff(repo_path)
        elif diff_source == "unstaged":
            diff = get_unstaged_diff(repo_path)
        elif diff_source == "all":
            # Combine staged and unstaged
            staged = get_staged_diff(repo_path)
            unstaged = get_unstaged_diff(repo_path)
            diff = staged + "\n" + unstaged
        elif diff_source == "current_commit":
            diff = get_commit_diff(repo_path, "HEAD")
        elif diff_source == "specific_commit":
            if not commit_id:
                return False, "No commit ID provided for specific_commit source."
            diff = get_commit_diff(repo_path, commit_id)
        elif diff_source == "full_history":
            # Use chunked processing for full history
            commits = get_full_history_commits(repo_path)
            if not commits:
                return False, "No commits found in repository."

            char_limit = 30000 if provider == "deepseek" else 15000
            try:
                docs_update = generate_docs_chunked(
                    commits, provider, char_limit, progress_callback
                )
            except (ValueError, RuntimeError) as e:
                return False, str(e)

            if "NO_UPDATES" in docs_update:
                return True, "No documentation updates required."

            # Resolve output dir and save
            repo_path_abs = os.path.abspath(os.path.expanduser(repo_path))
            if output_dir:
                out_dir = os.path.abspath(os.path.expanduser(output_dir))
            else:
                out_dir = os.path.join(repo_path_abs, "docs")
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, "docs_suggestion.md")
            with open(output_file, "w") as f:
                f.write(docs_update)

            return True, f"Documentation saved to: {output_file}"
        else:
            return False, f"Invalid diff_source: {diff_source}"
    except RuntimeError as e:
        return False, str(e)

    if not diff.strip():
        return False, "No changes found."

    # Safety Truncate (DeepSeek handles larger contexts better)
    limit = 30000 if provider == "deepseek" else 15000
    truncated = False
    if len(diff) > limit:
        diff = diff[:limit]
        truncated = True

    # Generate Content
    try:
        docs_update = generate_docs(diff, provider)
    except (ValueError, RuntimeError) as e:
        return False, str(e)

    # Save Output
    if "NO_UPDATES" in docs_update:
        return True, "No documentation updates required."

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "docs_suggestion.md")

    with open(output_file, "w") as f:
        f.write(docs_update)

    message = f"Documentation saved to: {output_file}"
    if truncated:
        message = f"Warning: Diff was truncated to {limit} chars.\n{message}"

    return True, message


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Docs-as-Code updates from git changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use OpenAI (default) on current repo
  %(prog)s --provider deepseek      # Use DeepSeek instead
  %(prog)s -p deepseek -r /my/repo  # DeepSeek on specific repo
  %(prog)s -o ./my-docs             # Custom output directory
        """
    )

    # Path Arguments
    parser.add_argument(
        "--repo", "-r",
        default=".",
        help="Path to the project root (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Directory to save the output (default: <repo>/docs)"
    )

    # Provider Argument
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        default="openai",
        help="Choose the AI provider (default: openai)"
    )

    # Diff source argument
    parser.add_argument(
        "--diff-source", "-s",
        choices=["commit", "staged", "unstaged", "all", "full_history"],
        default="commit",
        help="Source of changes to analyze (default: commit)"
    )

    args = parser.parse_args()

    print(f"Project Path:    {os.path.abspath(args.repo)}")
    print(f"Using Provider:  {PROVIDERS[args.provider]['name']}")
    print(f"Diff Source:     {args.diff_source}")
    print()
    print("AI is analyzing changes...")

    success, message = run_docs_generator(
        repo_path=args.repo,
        output_dir=args.output,
        provider=args.provider,
        diff_source=args.diff_source
    )

    if success:
        print()
        print("=" * 50)
        print(message)
        print("=" * 50)
    else:
        print(f"Error: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()

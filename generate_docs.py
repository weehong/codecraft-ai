#!/usr/bin/env python3
"""
Documentation Generator - Docs-as-Code Tool

Analyzes git diffs and generates documentation updates using AI providers.
Supports OpenAI (default) and DeepSeek.
"""

import os
import subprocess
import sys
import argparse
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
"""


def get_git_diff(repo_path):
    """Get the git diff between the last two commits."""
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
        print(f"Error running git in {repo_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Git is not installed.")
        sys.exit(1)


def generate_docs(diff_content, provider_key):
    """Generate documentation using the specified AI provider."""
    # Load configuration based on the chosen provider
    config = PROVIDERS[provider_key]
    api_key = os.getenv(config["env_var"])

    if not api_key:
        print(f"Error: {config['env_var']} environment variable not found.")
        sys.exit(1)

    print(f"Using Provider:  {config['name']}")

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
        print(f"Error calling API: {e}")
        sys.exit(1)


def main():
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

    args = parser.parse_args()

    # Resolve Paths
    repo_path = os.path.abspath(os.path.expanduser(args.repo))
    if args.output:
        output_dir = os.path.abspath(os.path.expanduser(args.output))
    else:
        output_dir = os.path.join(repo_path, "docs")

    print(f"Project Path:    {repo_path}")

    # Get Changes
    diff = get_git_diff(repo_path)
    if not diff.strip():
        print("No changes found in the last commit.")
        return

    # Safety Truncate (DeepSeek handles larger contexts better)
    limit = 30000 if args.provider == "deepseek" else 15000
    if len(diff) > limit:
        print(f"Warning: Diff is large. Truncating to {limit} chars...")
        diff = diff[:limit]

    # Generate Content
    print("AI is analyzing changes...")
    docs_update = generate_docs(diff, args.provider)

    # Save Output
    if "NO_UPDATES" in docs_update:
        print("No documentation updates required.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "docs_suggestion.md")

        with open(output_file, "w") as f:
            f.write(docs_update)

        print("\n" + "=" * 50)
        print(f"DONE! Review suggestions in:\n   {output_file}")
        print("=" * 50)


if __name__ == "__main__":
    main()

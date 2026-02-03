# Sonar JaCoCo Analyzer

A Python tool for analyzing SonarCloud/SonarQube code quality issues, JaCoCo coverage reports, and AI-powered commit message generation.

## Features

- **Code Smell Analysis**: Analyzes code smell data from SonarCloud/SonarQube
- **Rule Mapping**: Maps issues to their corresponding quality rules
- **Technical Debt Calculation**: Tracks effort required and technical debt metrics
- **Severity Analysis**: Breaks down issues by severity levels
- **JaCoCo Coverage Analysis**: Analyzes JaCoCo HTML coverage reports
- **Structured Output**: Generates clean JSON output for integration with other tools
- **AI-Powered Commit Messages**: Generate conventional commit messages using OpenAI
- **GitHub Integration**: Analyze commits from GitHub repositories
- **GitLab Integration**: Analyze commits from GitLab repositories (including self-hosted)
- **Intelligent Commit Splitting**: Automatically suggest splitting large changes

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/username/codecraft-ai.git
cd codecraft-ai

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

After installation, you can use the `codecraft-ai` command:

```bash
# Interactive mode (prompts for options)
codecraft-ai

# Fetch data from SonarCloud API
codecraft-ai --api

# Use local input.json file
codecraft-ai --file

# Analyze JaCoCo coverage report
codecraft-ai --jacoco path/to/report.zip

# AI-powered commit message generator (interactive)
codecraft-ai --commit

# Quick commit: auto-detect repo, generate and commit in one step
codecraft-ai --quick-commit
# or shorthand
codecraft-ai -q

# Reset saved selections
codecraft-ai --reset

# Show help
codecraft-ai --help
```

### Standalone Commit CLI

The commit message generator is also available as a standalone command:

```bash
# Run the AI-powered commit message generator (interactive)
git-commit-ai

# Quick mode: auto-detect repo, generate and commit in one step
git-commit-ai --quick
# or shorthand
git-commit-ai -q
```

### Running without Installation

```bash
python -m codecraft_ai.cli
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Required variables for API mode:
- `SONAR_TOKEN`: Your SonarCloud user token
- `SONAR_ORGANIZATION`: Your organization key

Optional:
- `SONAR_COOKIES`: Browser session cookies (alternative auth)
- `SONAR_XSRF_TOKEN`: XSRF token (required with cookies)

### AI Commit Generator Configuration

Required for the commit message generator:
- `OPENAI_API_KEY`: Your OpenAI API key

Optional (for remote repository modes):
- `GITHUB_TOKEN`: GitHub personal access token (for GitHub repository mode)
- `GITLAB_TOKEN`: GitLab personal access token (for GitLab repository mode)
- `GITLAB_URL`: GitLab instance URL (default: `https://gitlab.com`)

Optional (AI settings):
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o`)
- `OPENAI_TEMPERATURE`: Temperature for generation (default: `0.3`)
- `MAX_COMMIT_SIZE`: Lines threshold for commit splitting (default: `200`)

## Project Structure

```
codecraft-ai/
├── src/
│   └── codecraft_ai/
│       ├── __init__.py           # Package initialization
│       ├── api.py                # SonarCloud API client
│       ├── cli.py                # Command-line interface
│       ├── jacoco.py             # JaCoCo report analyzer
│       ├── commit_cli.py         # Commit generator CLI
│       ├── commit_config.py      # Commit generator configuration
│       ├── commit_generator.py   # OpenAI-powered message generation
│       ├── commit_splitter.py    # Intelligent commit splitting
│       ├── conventional_commit.py # Conventional commit formatting
│       ├── git_operations.py     # Local git operations
│       ├── github_client.py      # GitHub API client
│       └── gitlab_client.py      # GitLab API client
├── tests/                        # Test files
├── docs/                         # Documentation
├── .env.example                  # Example environment configuration
├── .gitignore
├── pyproject.toml                # Modern Python packaging configuration
├── requirements.txt              # Runtime dependencies
└── README.md
```

## Output Format

The analyzer generates JSON output with:

```json
{
  "metadata": {
    "analysis_timestamp": "2025-09-20T12:00:00Z",
    "total_issues_analyzed": 18,
    "unique_rules_violated": 7,
    "total_technical_debt_minutes": 120
  },
  "issue_rule_mappings": [...],
  "rules_summary": [...],
  "severity_distribution": {...},
  "type_distribution": {...}
}
```

## AI Commit Message Generator

The commit message generator helps create conventional commit messages using AI:

### Features

- **Conventional Commits**: Generates messages following the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) specification
- **Local Repository Support**: Analyze staged changes in your local git repository
- **GitHub Integration**: Analyze commits from GitHub repositories
- **GitLab Integration**: Analyze commits from GitLab repositories (including self-hosted instances)
- **Intelligent Splitting**: Automatically suggests splitting large changes into smaller, logical commits
- **User Approval**: Always requires user approval before creating commits

### Workflow

1. **Quick Commit Mode** (Recommended for fast workflows):
   - Stage your changes with `git add`
   - Run `git-commit-ai -q` or `codecraft-ai -q`
   - Review the AI-generated commit message
   - Confirm with `Y` to create the commit

2. **Local Repository Mode** (Interactive):
   - Stage your changes with `git add`
   - Run `git-commit-ai` or `codecraft-ai --commit`
   - Review the AI-generated commit message
   - Approve, edit, regenerate, or cancel

3. **GitHub Repository Mode**:
   - Select a repository from your GitHub account
   - Choose a branch and select commits to analyze
   - Get AI-generated commit message suggestions

4. **GitLab Repository Mode**:
   - Select a project from your GitLab account
   - Choose a branch and select commits to analyze
   - Get AI-generated commit message suggestions
   - Supports both gitlab.com and self-hosted instances

### Commit Types

The generator supports all standard conventional commit types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes
- `revert`: Reverting changes

See [docs/commit-generator.md](docs/commit-generator.md) for detailed documentation.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

This project is open source. Please ensure compliance with your organization's security policies when handling SonarCloud data.

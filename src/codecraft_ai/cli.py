"""
Command-line interface for Sonar JaCoCo Analyzer.
"""

import atexit
import json
import os
import readline
import sys
from collections import defaultdict
from datetime import datetime, timezone

# History file for input persistence
HISTORY_FILE = os.path.expanduser("~/.sonar_jacoco_history")
PATH_HISTORY_FILE = os.path.expanduser("~/.sonar_jacoco_path_history")
HISTORY_MAX_LENGTH = 500
_history_initialized = False

# Default output directory
OUTPUT_DIR = "output"

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm

from .api import (
    SonarCloudAPI,
    load_env_file,
    select_project_interactive,
    load_config,
    save_config,
    reset_config,
)
from .jacoco import (
    analyze_jacoco_report,
    format_analysis_result,
    format_analysis_result_chunked,
    get_available_packages,
    format_analysis_for_package,
    JaCoCoAnalysisResult,
    find_7zip_executables,
    set_7zip_path,
)
from .commit_cli import (
    main as commit_main,
    run_quick_commit,
    setup_path_completion,
    expand_path,
    fetch_openai_credits,
    fetch_anthropic_credits,
    fetch_deepseek_credits,
)
from .docs_generator import (
    run_docs_generator,
    get_full_history_commits,
    PROVIDERS as DOCS_PROVIDERS,
)

console = Console()


def setup_input_history():
    """
    Initialize readline input history.

    Loads history from file and configures readline for persistent history.
    This allows users to use up/down arrows to navigate through previous inputs.
    """
    global _history_initialized
    if _history_initialized:
        return
    _history_initialized = True

    # Configure history settings
    readline.set_history_length(HISTORY_MAX_LENGTH)

    # Load existing history file if it exists
    try:
        if os.path.exists(HISTORY_FILE):
            readline.read_history_file(HISTORY_FILE)
    except (IOError, OSError, PermissionError):
        # Silently ignore history load errors
        pass

    # Register save function to run at exit
    atexit.register(save_input_history)


def save_input_history():
    """
    Save readline input history to file.

    Called automatically at exit via atexit, but can also be called manually.
    Filters out menu selections and other non-meaningful inputs.
    """
    try:
        # Ensure parent directory exists
        history_dir = os.path.dirname(HISTORY_FILE)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)

        # Get current history length
        history_length = readline.get_current_history_length()

        # Collect meaningful entries
        meaningful_entries = []
        for i in range(1, history_length + 1):
            entry = readline.get_history_item(i)
            if entry and is_meaningful_history_entry(entry):
                meaningful_entries.append(entry)

        # Clear current history and add back only meaningful entries
        readline.clear_history()
        for entry in meaningful_entries:
            readline.add_history(entry)

        readline.write_history_file(HISTORY_FILE)
    except (IOError, OSError, PermissionError):
        # Silently ignore history save errors
        pass


def is_meaningful_history_entry(entry: str) -> bool:
    """
    Check if a history entry is meaningful and should be saved.

    Filters out menu selections (single digits, single letters) and other
    non-meaningful inputs that clutter the history.

    Args:
        entry: The history entry to check.

    Returns:
        True if the entry should be saved, False otherwise.
    """
    if not entry or not entry.strip():
        return False

    entry = entry.strip()

    # Filter out single characters (menu selections like "1", "a", "e")
    if len(entry) == 1:
        return False

    # Filter out pure numeric entries (menu selections like "1", "2", "12")
    if entry.isdigit():
        return False

    # Filter out common menu-style inputs
    menu_patterns = {
        "all", "q", "yes", "no", "y", "n",
        "approve", "edit", "regenerate", "cancel",
    }
    if entry.lower() in menu_patterns:
        return False

    # Filter out range selections like "1-5", "2-10"
    if "-" in entry and all(part.isdigit() for part in entry.split("-") if part):
        return False

    # Filter out space-separated number lists like "1 2 3"
    parts = entry.split()
    if all(part.isdigit() or (part.count("-") == 1 and all(p.isdigit() for p in part.split("-") if p)) for part in parts):
        return False

    return True


def remove_last_history_if_not_meaningful():
    """
    Remove the last readline history entry if it's not meaningful.

    This should be called after menu selections to prevent them from
    appearing in history when using up/down arrows.
    """
    history_length = readline.get_current_history_length()
    if history_length > 0:
        last_entry = readline.get_history_item(history_length)
        if last_entry and not is_meaningful_history_entry(last_entry):
            # Remove the last entry by setting history length to one less
            readline.remove_history_item(history_length - 1)


def add_to_history(text: str):
    """
    Add a text entry to the input history.

    Args:
        text: The text to add to history.
    """
    if text and text.strip():
        readline.add_history(text.strip())


def menu_ask(prompt_text: str, choices: list = None, default: str = None) -> str:
    """
    Ask for menu input and clean up history if the input is not meaningful.

    This wraps Prompt.ask and removes menu selections (like "1", "2", "y")
    from readline history immediately after input.

    Args:
        prompt_text: The prompt to display.
        choices: Optional list of valid choices.
        default: Optional default value.

    Returns:
        The user's input.
    """
    if choices:
        result = Prompt.ask(prompt_text, choices=choices, default=default)
    elif default is not None:
        result = Prompt.ask(prompt_text, default=default)
    else:
        result = Prompt.ask(prompt_text)
    remove_last_history_if_not_meaningful()
    return result


def clear_history():
    """
    Clear the readline input history.

    Removes all entries from the current session and deletes the history file.
    """
    # Clear readline's in-memory history
    readline.clear_history()

    # Delete the history file if it exists
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            return True
        except (IOError, OSError, PermissionError):
            return False
    return True


def clear_output():
    """
    Clear the output directory.

    Removes all files and subdirectories in the output directory.

    Returns:
        Tuple of (success: bool, files_removed: int)
    """
    import shutil

    if not os.path.exists(OUTPUT_DIR):
        return True, 0

    files_removed = 0
    try:
        # Count files before removal
        for root, dirs, files in os.walk(OUTPUT_DIR):
            files_removed += len(files)

        # Remove the directory and all contents
        shutil.rmtree(OUTPUT_DIR)
        return True, files_removed
    except (IOError, OSError, PermissionError):
        return False, 0


def analyze_codesmell_data(json_data):
    """
    Analyze the code smell data and generate comprehensive output.

    Args:
        json_data: Parsed JSON data from SonarCloud

    Returns:
        Dictionary containing analysis results
    """
    issues = json_data.get("issues", [])
    rules = json_data.get("rules", [])

    # Create rule lookup dictionary
    rule_lookup = {rule["key"]: rule for rule in rules}

    # Analyze data
    issue_rule_mappings = []
    rule_violation_counts = defaultdict(int)
    severity_counts = defaultdict(int)
    type_counts = defaultdict(int)

    total_effort = 0
    total_debt = 0

    for issue in issues:
        rule_key = issue.get("rule")
        rule_violation_counts[rule_key] += 1

        # Count severities and types
        severity_counts[issue.get("severity", "UNKNOWN")] += 1
        type_counts[issue.get("type", "UNKNOWN")] += 1

        # Extract effort and debt (convert from minutes to minutes)
        effort = (
            int(issue.get("effort", "0").replace("min", ""))
            if issue.get("effort")
            else 0
        )
        debt = (
            int(issue.get("debt", "0").replace("min", "")) if issue.get("debt") else 0
        )

        total_effort += effort
        total_debt += debt

        # Get rule information
        rule_info = rule_lookup.get(rule_key, {})

        # Create issue-rule mapping
        mapping = {
            "issue_key": issue.get("key"),
            "rule_key": rule_key,
            "rule_name": rule_info.get("name", "Unknown Rule"),
            "rule_description": rule_info.get("htmlDesc", ""),
            "component": issue.get("component", ""),
            "line": issue.get("line"),
            "message": issue.get("message", ""),
            "severity": issue.get("severity"),
            "type": issue.get("type"),
            "effort": effort,
            "debt": debt,
            "creation_date": issue.get("creationDate"),
            "update_date": issue.get("updateDate"),
        }

        issue_rule_mappings.append(mapping)

    # Generate rules summary
    rules_summary = []
    for rule_key, count in rule_violation_counts.items():
        rule_info = rule_lookup.get(rule_key, {})
        rules_summary.append(
            {
                "rule_key": rule_key,
                "rule_name": rule_info.get("name", "Unknown Rule"),
                "violation_count": count,
                "rule_type": rule_info.get("type", "UNKNOWN"),
                "rule_severity": rule_info.get("severity", "UNKNOWN"),
                "rule_description": rule_info.get("htmlDesc", ""),
            }
        )

    # Sort by violation count (descending)
    rules_summary.sort(key=lambda x: x["violation_count"], reverse=True)

    return {
        "metadata": {
            "analysis_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "total_issues_analyzed": len(issues),
            "unique_rules_violated": len(rule_violation_counts),
            "total_technical_debt_minutes": total_debt,
            "total_effort_minutes": total_effort,
        },
        "issue_rule_mappings": issue_rule_mappings,
        "rules_summary": rules_summary,
        "severity_distribution": dict(severity_counts),
        "type_distribution": dict(type_counts),
        "rule_violation_counts": dict(rule_violation_counts),
    }


def generate_output_json(analysis_result, output_file="output/output.json"):
    """
    Generate the output JSON file with analysis results.

    Args:
        analysis_result: Analysis results dictionary
        output_file: Output filename
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    console.print(
        f"[green]Analysis complete![/green] Output saved to: [bold]{output_file}[/bold]"
    )


def format_time(minutes):
    """Format minutes into a human-readable string."""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}h {mins}m" if mins else f"{hours}h"
    days = hours // 24
    hours = hours % 24
    return f"{days}d {hours}h" if hours else f"{days}d"


def get_severity_style(severity):
    """Get rich style for severity level."""
    styles = {
        "BLOCKER": "bold red",
        "CRITICAL": "red",
        "MAJOR": "yellow",
        "MINOR": "cyan",
        "INFO": "dim",
    }
    return styles.get(severity, "")


def print_analysis_report(analysis_result):
    """
    Print a formatted analysis report to console.

    Args:
        analysis_result: Analysis results dictionary
    """
    metadata = analysis_result["metadata"]
    rules_summary = analysis_result["rules_summary"]
    severity_dist = analysis_result["severity_distribution"]
    type_dist = analysis_result["type_distribution"]

    console.print()

    # Header
    console.print(
        Panel.fit("[bold]CODE SMELL ANALYSIS REPORT[/bold]", border_style="cyan")
    )
    console.print()

    # Summary Table
    summary_table = Table(title="Summary", show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="blue")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Total Issues", str(metadata["total_issues_analyzed"]))
    summary_table.add_row("Unique Rules", str(metadata["unique_rules_violated"]))
    summary_table.add_row(
        "Effort Required", format_time(metadata["total_effort_minutes"])
    )
    summary_table.add_row(
        "Technical Debt", format_time(metadata["total_technical_debt_minutes"])
    )
    console.print(summary_table)
    console.print()

    # Severity Distribution
    if severity_dist:
        severity_table = Table(
            title="Severity Distribution", box=None, padding=(0, 1)
        )
        severity_table.add_column("Severity", width=10)
        severity_table.add_column("Bar", width=20)
        severity_table.add_column("Count", justify="right", width=6)
        severity_table.add_column("%", justify="right", width=7)

        total = sum(severity_dist.values())
        severity_order = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]

        for severity in severity_order:
            if severity in severity_dist:
                count = severity_dist[severity]
                style = get_severity_style(severity)
                pct = (count / total) * 100 if total > 0 else 0
                bar_width = int((count / total) * 15) if total > 0 else 0
                bar = Text(
                    "█" * bar_width + "░" * (15 - bar_width), style=style
                )
                severity_table.add_row(
                    Text(severity, style=style), bar, str(count), f"{pct:.1f}%"
                )

        # Handle any other severities not in the order
        for severity, count in severity_dist.items():
            if severity not in severity_order:
                style = get_severity_style(severity)
                pct = (count / total) * 100 if total > 0 else 0
                bar_width = int((count / total) * 15) if total > 0 else 0
                bar = Text(
                    "█" * bar_width + "░" * (15 - bar_width), style=style
                )
                severity_table.add_row(
                    Text(severity, style=style), bar, str(count), f"{pct:.1f}%"
                )

        console.print(severity_table)
        console.print()

    # Type Distribution
    if type_dist:
        type_table = Table(title="Type Distribution", box=None, padding=(0, 1))
        type_table.add_column("Type", width=16)
        type_table.add_column("Bar", width=20)
        type_table.add_column("Count", justify="right", width=6)
        type_table.add_column("%", justify="right", width=7)

        total = sum(type_dist.values())
        type_styles = {"CODE_SMELL": "yellow", "BUG": "red", "VULNERABILITY": "red"}

        for issue_type, count in sorted(
            type_dist.items(), key=lambda x: x[1], reverse=True
        ):
            style = type_styles.get(issue_type, "")
            pct = (count / total) * 100 if total > 0 else 0
            bar_width = int((count / total) * 15) if total > 0 else 0
            bar = Text("█" * bar_width + "░" * (15 - bar_width), style=style)
            type_table.add_row(
                Text(issue_type, style=style), bar, str(count), f"{pct:.1f}%"
            )

        console.print(type_table)
        console.print()

    # Top Violated Rules
    if rules_summary:
        rules_table = Table(title="Top Violated Rules", box=None, padding=(0, 1))
        rules_table.add_column("#", justify="right", width=3, style="dim")
        rules_table.add_column("Bar", width=12)
        rules_table.add_column("Count", justify="right", width=5)
        rules_table.add_column("Rule", width=40)

        max_count = rules_summary[0]["violation_count"] if rules_summary else 1

        for i, rule in enumerate(rules_summary[:10], 1):
            count = rule["violation_count"]
            style = get_severity_style(rule.get("rule_severity", ""))
            bar_width = int((count / max_count) * 10) if max_count > 0 else 0
            bar = Text("█" * bar_width + "░" * (10 - bar_width), style=style)
            rule_name = (
                rule["rule_name"][:38] + ".."
                if len(rule["rule_name"]) > 40
                else rule["rule_name"]
            )
            rules_table.add_row(str(i), bar, f"{count}x", Text(rule_name, style=style))

        console.print(rules_table)
        console.print()

    # Footer
    console.print(f"[dim]Analysis completed at: {metadata['analysis_timestamp']}[/dim]")
    console.rule(style="cyan")
    console.print()


def run_with_api():
    """Run the analyzer by fetching data from SonarCloud API."""
    # Load environment variables
    load_env_file(".env")

    # Initialize API client
    api = SonarCloudAPI.from_env()

    if not api.organization and not api.token:
        console.print("[red]No configuration found.[/red]")
        console.print(
            "[dim]Please set up your .env file with SONAR_ORGANIZATION and SONAR_TOKEN[/dim]"
        )
        console.print("[dim]See .env.example for configuration options.[/dim]")
        return False

    try:
        # List projects
        with console.status("[cyan]Fetching projects from SonarCloud...[/cyan]"):
            projects = api.list_projects()

        if not projects:
            console.print("[yellow]No projects found in the organization.[/yellow]")
            return False

        console.print(f"[green]Found {len(projects)} project(s).[/green]")
        console.print()

        # Let user select a project
        selected = select_project_interactive(projects, console)

        if not selected:
            console.print("[dim]No project selected. Exiting.[/dim]")
            return False

        project_key = selected["key"]
        project_name = selected["name"]

        console.print()
        console.print(f"[bold]Selected:[/bold] [green]{project_name}[/green]")

        # Fetch issues from API
        with console.status(
            f"[cyan]Fetching issues for project: {project_key}...[/cyan]"
        ):
            json_data = api.get_issues(project_key)

        console.print(f"[green]Retrieved {json_data['total']} issues.[/green]")

        # Analyze the data
        with console.status("[cyan]Analyzing code smell data...[/cyan]"):
            analysis_result = analyze_codesmell_data(json_data)

        # Add project info to metadata
        analysis_result["metadata"]["project_key"] = project_key
        analysis_result["metadata"]["project_name"] = project_name

        # Generate output
        generate_output_json(analysis_result)

        # Print report
        print_analysis_report(analysis_result)

        return True

    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        return False
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return False


def run_with_file():
    """Run the analyzer using a local input.json file."""
    with console.status("[cyan]Reading data from input.json...[/cyan]"):
        with open("input.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)

    # Analyze the data
    with console.status("[cyan]Analyzing code smell data...[/cyan]"):
        analysis_result = analyze_codesmell_data(json_data)

    # Generate output
    generate_output_json(analysis_result)

    # Print report
    print_analysis_report(analysis_result)


def print_jacoco_report(result: JaCoCoAnalysisResult):
    """
    Print a formatted JaCoCo analysis report to console.

    Args:
        result: JaCoCoAnalysisResult object
    """
    console.print()

    # Header
    console.print(
        Panel.fit("[bold]JACOCO COVERAGE ANALYSIS REPORT[/bold]", border_style="cyan")
    )
    console.print()

    # Summary Table
    summary_table = Table(title="Summary", show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="blue")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Files Analyzed", str(result.total_files_analyzed))
    summary_table.add_row(
        "Missed Branches", f"[yellow]{len(result.missed_branches)}[/yellow]"
    )
    summary_table.add_row(
        "Uncovered Lines", f"[red]{len(result.uncovered_lines)}[/red]"
    )
    console.print(summary_table)
    console.print()

    # Missed Branches by File
    if result.missed_branches:
        console.print("[bold yellow]MISSED BRANCHES[/bold yellow]")
        console.print()

        # Group by file
        by_file = {}
        for mb in result.missed_branches:
            if mb.file_path not in by_file:
                by_file[mb.file_path] = []
            by_file[mb.file_path].append(mb)

        for file_path, branches in sorted(by_file.items()):
            console.print(f"  [bold]{file_path}[/bold]")
            for mb in sorted(branches, key=lambda x: x.line_number):
                source_preview = (
                    mb.source_line[:60] + "..."
                    if len(mb.source_line) > 60
                    else mb.source_line
                )
                console.print(
                    f"    [dim]L{mb.line_number}:[/dim] [yellow]{mb.branch_info}[/yellow]"
                )
                console.print(f"        [dim]{source_preview}[/dim]")
            console.print()

    # Uncovered Lines by File
    if result.uncovered_lines:
        console.print("[bold red]UNCOVERED LINES[/bold red]")
        console.print()

        # Group by file
        by_file = {}
        for ul in result.uncovered_lines:
            if ul.file_path not in by_file:
                by_file[ul.file_path] = []
            by_file[ul.file_path].append(ul)

        for file_path, lines in sorted(by_file.items()):
            console.print(
                f"  [bold]{file_path}[/bold] [dim]({len(lines)} uncovered lines)[/dim]"
            )

            # Show first 10 lines, then summarize
            sorted_lines = sorted(lines, key=lambda x: x.line_number)
            display_lines = sorted_lines[:10]

            for ul in display_lines:
                source_preview = (
                    ul.source_line[:70] + "..."
                    if len(ul.source_line) > 70
                    else ul.source_line
                )
                console.print(
                    f"    [red]L{ul.line_number}:[/red] [dim]{source_preview}[/dim]"
                )

            if len(sorted_lines) > 10:
                remaining = len(sorted_lines) - 10
                console.print(
                    f"    [dim]... and {remaining} more uncovered lines[/dim]"
                )
            console.print()

    # Footer
    console.rule(style="cyan")
    console.print()


def generate_jacoco_ai_prompt(formatted_result: dict, json_str: str) -> str:
    """
    Generate an AI prompt for writing unit tests to cover missed coverage.

    Args:
        formatted_result: The formatted analysis result dictionary.
        json_str: The JSON string representation of the result.

    Returns:
        A well-structured AI prompt for generating unit tests.
    """
    summary = formatted_result.get("summary", {})
    total_files = summary.get("total_files_analyzed", 0)
    total_missed = summary.get("total_missed_branches", 0)
    total_uncovered = summary.get("total_uncovered_lines", 0)

    ai_prompt = f'''You are a **Senior Test Engineer** specializing in Java/Kotlin unit testing with expertise in JUnit 5, Mockito, and test-driven development. Your task is to write comprehensive unit tests to cover the missed coverage identified in the JaCoCo report.

---

## Context

You have been provided with JaCoCo code coverage analysis results showing:
- **Files Analyzed**: {total_files}
- **Missed Branches**: {total_missed} (conditional logic not fully tested)
- **Uncovered Lines**: {total_uncovered} (lines with zero test coverage)

---

## Your Task

Write unit tests to achieve full coverage for the identified gaps. Follow these instructions:

### 1. Analyze Coverage Gaps
Review the JSON data and identify:
- Which classes/methods have uncovered lines
- Which conditional branches are not tested
- The source code context for each gap

### 2. Write Unit Tests

For each uncovered area, generate complete, runnable unit test code that:

**Test Structure:**
- Use JUnit 5 (`@Test`, `@BeforeEach`, `@DisplayName`, etc.)
- Use Mockito for mocking dependencies (`@Mock`, `@InjectMocks`, `@ExtendWith(MockitoExtension.class)`)
- Follow the Arrange-Act-Assert (AAA) pattern
- Use descriptive test method names: `methodName_stateUnderTest_expectedBehavior()`

**For Uncovered Lines:**
- Write tests that execute the uncovered code paths
- Verify expected behavior with appropriate assertions
- Handle edge cases and boundary conditions

**For Missed Branches:**
- Write separate tests for each branch condition (true/false paths)
- Test null checks, empty collections, boundary values
- Cover exception handling paths

### 3. Test Code Requirements

Each test class should include:
```java
@ExtendWith(MockitoExtension.class)
@DisplayName("Tests for [ClassName]")
class [ClassName]Test {{

    @Mock
    private [DependencyType] dependency;

    @InjectMocks
    private [ClassName] underTest;

    @BeforeEach
    void setUp() {{
        // Any additional setup
    }}

    @Test
    @DisplayName("Should [expected behavior] when [condition]")
    void methodName_condition_expectedResult() {{
        // Arrange

        // Act

        // Assert
    }}
}}
```

### 4. Output Format

For each file with coverage gaps, provide:

1. **File**: `[ClassName].java`
2. **Coverage Issues**: Brief summary of what's not covered
3. **Test Code**: Complete test class with all necessary imports
4. **Explanation**: Brief note on what each test covers

---

## Constraints

- Generate syntactically correct, compilable Java code
- Use standard testing libraries (JUnit 5, Mockito, AssertJ)
- Do not modify production code; only write test code
- Prioritize tests by coverage impact (most uncovered lines first)

---

## JaCoCo Coverage Data (JSON)

```json
{json_str}
```

---

Please generate the unit tests now. Start with the file that has the most coverage gaps.'''

    return ai_prompt


def generate_jacoco_ai_prompt_chunked(chunk: dict, json_str: str) -> str:
    """
    Generate an AI prompt for a chunked JaCoCo analysis result.

    Args:
        chunk: A chunk dictionary from format_analysis_result_chunked.
        json_str: The JSON string representation of the chunk.

    Returns:
        A well-structured AI prompt for generating unit tests.
    """
    chunk_info = chunk.get("chunk_info", {})
    overall = chunk.get("overall_summary", {})
    chunk_num = chunk_info.get("chunk_number", 1)
    total_chunks = chunk_info.get("total_chunks", 1)
    branches_in_chunk = chunk_info.get("missed_branches_in_chunk", 0)
    lines_in_chunk = chunk_info.get("uncovered_lines_in_chunk", 0)
    split_by = chunk_info.get("split_by", "package")

    # Determine the element count label based on split_by mode
    split_labels = {"package": "Packages", "class": "Classes", "package_class": "Packages (with classes)"}
    element_label = split_labels.get(split_by, "Packages")
    element_count = (
        chunk_info.get("package_in_chunk", 0)
        or chunk_info.get("class_in_chunk", 0)
        or chunk_info.get("package_class_in_chunk", 0)
        or chunk_info.get("packages_in_chunk", 0)
    )

    ai_prompt = f'''You are a **Senior Test Engineer** specializing in Java/Kotlin unit testing with expertise in JUnit 5, Mockito, and test-driven development. Your task is to write comprehensive unit tests to cover the missed coverage identified in this JaCoCo report chunk.

---

## Context

This is **Chunk {chunk_num} of {total_chunks}** from the JaCoCo coverage analysis (split by **{split_by.replace("_", " ")}**).

**This Chunk Contains:**
- **{element_label}**: {element_count}
- **Missed Branches**: {branches_in_chunk}
- **Uncovered Lines**: {lines_in_chunk}

**Overall Project Totals:**
- **Total Files Analyzed**: {overall.get("total_files_analyzed", 0)}
- **Total Missed Branches**: {overall.get("total_missed_branches", 0)}
- **Total Uncovered Lines**: {overall.get("total_uncovered_lines", 0)}

---

## Your Task

Write unit tests to achieve full coverage for the identified gaps in THIS chunk. Follow these instructions:

### 1. Analyze Coverage Gaps
Review the JSON data and identify:
- Which classes/methods have uncovered lines
- Which conditional branches are not tested
- The source code context for each gap

### 2. Write Unit Tests

For each uncovered area, generate complete, runnable unit test code that:

**Test Structure:**
- Use JUnit 5 (`@Test`, `@BeforeEach`, `@DisplayName`, etc.)
- Use Mockito for mocking dependencies (`@Mock`, `@InjectMocks`, `@ExtendWith(MockitoExtension.class)`)
- Follow the Arrange-Act-Assert (AAA) pattern
- Use descriptive test method names: `methodName_stateUnderTest_expectedBehavior()`

**For Uncovered Lines:**
- Write tests that execute the uncovered code paths
- Verify expected behavior with appropriate assertions
- Handle edge cases and boundary conditions

**For Missed Branches:**
- Write separate tests for each branch condition (true/false paths)
- Test null checks, empty collections, boundary values
- Cover exception handling paths

### 3. Test Code Requirements

Each test class should include:
```java
@ExtendWith(MockitoExtension.class)
@DisplayName("Tests for [ClassName]")
class [ClassName]Test {{

    @Mock
    private [DependencyType] dependency;

    @InjectMocks
    private [ClassName] underTest;

    @BeforeEach
    void setUp() {{
        // Any additional setup
    }}

    @Test
    @DisplayName("Should [expected behavior] when [condition]")
    void methodName_condition_expectedResult() {{
        // Arrange

        // Act

        // Assert
    }}
}}
```

---

## JaCoCo Coverage Data - Chunk {chunk_num}/{total_chunks} (JSON)

```json
{json_str}
```

---

Please generate the unit tests for this chunk now. Start with the package that has the most coverage gaps.'''

    return ai_prompt


def generate_jacoco_ai_prompt_package(package_result: dict, json_str: str) -> str:
    """
    Generate an AI prompt for a package-specific JaCoCo analysis result.

    Args:
        package_result: A package dictionary from format_analysis_for_package.
        json_str: The JSON string representation of the package result.

    Returns:
        A well-structured AI prompt for generating unit tests.
    """
    package_name = package_result.get("package", "Unknown")
    summary = package_result.get("summary", {})
    missed_branches = summary.get("missed_branches", 0)
    uncovered_lines = summary.get("uncovered_lines", 0)

    ai_prompt = f'''You are a **Senior Test Engineer** specializing in Java/Kotlin unit testing with expertise in JUnit 5, Mockito, and test-driven development. Your task is to write comprehensive unit tests to cover the missed coverage in the specified package.

---

## Context

**Package:** `{package_name}`

**Coverage Issues:**
- **Missed Branches**: {missed_branches}
- **Uncovered Lines**: {uncovered_lines}

---

## Your Task

Write unit tests to achieve full coverage for the identified gaps in this package. Follow these instructions:

### 1. Analyze Coverage Gaps
Review the JSON data and identify:
- Which classes/methods have uncovered lines
- Which conditional branches are not tested
- The source code context for each gap

### 2. Write Unit Tests

For each uncovered area, generate complete, runnable unit test code that:

**Test Structure:**
- Use JUnit 5 (`@Test`, `@BeforeEach`, `@DisplayName`, etc.)
- Use Mockito for mocking dependencies (`@Mock`, `@InjectMocks`, `@ExtendWith(MockitoExtension.class)`)
- Follow the Arrange-Act-Assert (AAA) pattern
- Use descriptive test method names: `methodName_stateUnderTest_expectedBehavior()`

**For Uncovered Lines:**
- Write tests that execute the uncovered code paths
- Verify expected behavior with appropriate assertions
- Handle edge cases and boundary conditions

**For Missed Branches:**
- Write separate tests for each branch condition (true/false paths)
- Test null checks, empty collections, boundary values
- Cover exception handling paths

### 3. Test Code Requirements

Each test class should include:
```java
@ExtendWith(MockitoExtension.class)
@DisplayName("Tests for [ClassName]")
class [ClassName]Test {{

    @Mock
    private [DependencyType] dependency;

    @InjectMocks
    private [ClassName] underTest;

    @BeforeEach
    void setUp() {{
        // Any additional setup
    }}

    @Test
    @DisplayName("Should [expected behavior] when [condition]")
    void methodName_condition_expectedResult() {{
        // Arrange

        // Act

        // Assert
    }}
}}
```

---

## JaCoCo Coverage Data - Package: {package_name} (JSON)

```json
{json_str}
```

---

Please generate the unit tests for this package now.'''

    return ai_prompt


def _offer_ai_prompt_actions(ai_prompt: str, output_dir: str) -> None:
    """
    Offer actions for the AI prompt (copy to clipboard or save to file).

    Args:
        ai_prompt: The AI prompt string.
        output_dir: The directory for saving files.
    """
    console.print()
    console.print("[bold cyan]AI Prompt for Unit Test Generation[/bold cyan]")
    console.print("[dim]An AI prompt has been generated to help write unit tests for missed coverage.[/dim]")
    console.print()

    # Ask if user wants to copy to clipboard
    if Confirm.ask("Copy AI prompt to clipboard?", default=True):
        try:
            import subprocess
            # Try xclip (Linux)
            process = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
            )
            process.communicate(ai_prompt.encode())
            console.print("[green]AI prompt copied to clipboard![/green]")
            console.print("[dim]Paste into your AI assistant to generate unit tests.[/dim]")
        except (FileNotFoundError, Exception):
            console.print("[yellow]Clipboard not available (xclip not installed).[/yellow]")
            # Fall back to saving to file
            _save_ai_prompt_to_file(ai_prompt, output_dir)
    else:
        # Save to file instead
        _save_ai_prompt_to_file(ai_prompt, output_dir if output_dir else "output")


def _save_ai_prompt_to_file(ai_prompt: str, output_dir: str) -> None:
    """
    Save the AI prompt to a file.

    Args:
        ai_prompt: The AI prompt string to save.
        output_dir: The directory to save the file in.
    """
    # Ensure output directory exists
    if not output_dir:
        output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = os.path.join(output_dir, f"ai_prompt_unit_tests_{timestamp}.md")

    try:
        prompt_file = Prompt.ask("Output filename for AI prompt", default=prompt_file)

        # Ensure directory exists for custom path
        prompt_dir = os.path.dirname(prompt_file)
        if prompt_dir and not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir, exist_ok=True)

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(ai_prompt)

        console.print(f"[green]AI prompt saved to:[/green] [bold]{prompt_file}[/bold]")
        console.print("[dim]Copy the contents and paste into your AI assistant to generate unit tests.[/dim]")
    except (IOError, OSError) as e:
        console.print(f"[red]Error saving AI prompt: {e}[/red]")


def select_7zip_executable():
    """
    Find and let user select a 7-Zip executable.

    Returns:
        True if a 7zip executable was selected or py7zr is available, False otherwise
    """
    # First check if py7zr is available
    try:
        import py7zr  # noqa: F401

        console.print("[dim]Using py7zr library for 7z extraction[/dim]")
        return True
    except ImportError:
        pass

    # Find available 7zip executables
    executables = find_7zip_executables()

    if not executables:
        console.print("[yellow]No 7-Zip executable found on your system.[/yellow]")
        console.print()
        console.print("[dim]Options to fix this:[/dim]")
        console.print("    1. Install py7zr: [green]pip install py7zr[/green]")
        console.print("    2. Install 7-Zip and add it to your PATH")
        console.print()

        # Allow manual entry
        try:
            manual_path = Prompt.ask(
                "Enter path to 7z executable manually (or press Enter to cancel)",
                default="",
            )
            if manual_path:
                manual_path = manual_path.strip().strip('"').strip("'")
                if os.path.isfile(manual_path) and os.access(manual_path, os.X_OK):
                    set_7zip_path(manual_path)
                    console.print(f"[green]Using:[/green] {manual_path}")
                    return True
                else:
                    console.print(f"[red]Invalid executable path:[/red] {manual_path}")
                    return False
            return False
        except (EOFError, KeyboardInterrupt):
            return False

    if len(executables) == 1:
        # Only one option, use it automatically
        set_7zip_path(executables[0])
        console.print(f"[dim]Using 7-Zip:[/dim] {executables[0]}")
        return True

    # Multiple options, let user choose
    console.print()
    console.print("[bold]Select 7-Zip executable:[/bold]")
    console.print()

    for i, exe_path in enumerate(executables, 1):
        console.print(f"    [green][{i}][/green] {exe_path}")

    console.print(f"    [green][{len(executables) + 1}][/green] Enter path manually")
    console.print()

    valid_choices = [str(i) for i in range(1, len(executables) + 2)]

    try:
        choice = menu_ask(
            "[bold]Enter choice[/bold]", choices=valid_choices
        )
        choice_idx = int(choice) - 1

        if choice_idx < len(executables):
            selected_path = executables[choice_idx]
            set_7zip_path(selected_path)
            console.print(f"[green]Selected:[/green] {selected_path}")
            return True
        else:
            # Manual entry
            manual_path = Prompt.ask("Enter path to 7z executable")
            manual_path = manual_path.strip().strip('"').strip("'")
            if os.path.isfile(manual_path) and os.access(manual_path, os.X_OK):
                set_7zip_path(manual_path)
                console.print(f"[green]Using:[/green] {manual_path}")
                return True
            else:
                console.print(f"[red]Invalid executable path:[/red] {manual_path}")
                return False

    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False


def find_jacoco_files():
    """
    Find JaCoCo archive files and potential report directories in current directory.

    Returns:
        Tuple of (archive_files, report_dirs) where each is a list of paths
    """
    archive_files = []
    report_dirs = []

    # Look for archive files
    for filename in os.listdir("."):
        lower_name = filename.lower()
        if lower_name.endswith((".zip", ".7z", ".7zip")):
            # Check if name suggests JaCoCo
            if "jacoco" in lower_name or "coverage" in lower_name:
                archive_files.append(filename)

    # Look for directories that might contain JaCoCo reports
    for dirname in os.listdir("."):
        if os.path.isdir(dirname):
            lower_name = dirname.lower()
            # Check if name suggests JaCoCo or if it contains index.html
            if "jacoco" in lower_name or "coverage" in lower_name:
                report_dirs.append(dirname)
            elif os.path.exists(os.path.join(dirname, "index.html")):
                # Could be an extracted JaCoCo report
                report_dirs.append(dirname)

    return sorted(archive_files), sorted(report_dirs)


def _save_current_history_to_list() -> list:
    """Save current readline history to a list."""
    history = []
    for i in range(1, readline.get_current_history_length() + 1):
        item = readline.get_history_item(i)
        if item:
            history.append(item)
    return history


def _restore_history_from_list(history: list):
    """Restore readline history from a list."""
    readline.clear_history()
    for item in history:
        readline.add_history(item)


def _load_path_history():
    """Load path-specific history from file."""
    readline.clear_history()
    try:
        if os.path.exists(PATH_HISTORY_FILE):
            readline.read_history_file(PATH_HISTORY_FILE)
    except (IOError, OSError, PermissionError):
        pass


def _save_path_history():
    """Save path-specific history to file."""
    try:
        readline.write_history_file(PATH_HISTORY_FILE)
    except (IOError, OSError, PermissionError):
        pass


def prompt_for_jacoco_path(prompt_text: str, path_type: str = "file") -> str:
    """
    Prompt user for a path with tab completion support.

    Uses a separate history file for paths, so arrow keys only show previous paths,
    not menu selections or other inputs.

    Args:
        prompt_text: Text to display in the prompt.
        path_type: Either "file" or "directory" to indicate expected path type.

    Returns:
        Expanded path string, or empty string if cancelled.
    """
    # Setup tab completion
    setup_path_completion()

    # Save main history and switch to path history
    main_history = _save_current_history_to_list()
    _load_path_history()

    console.print()
    console.print(f"[bold]{prompt_text}[/bold]")
    console.print("[dim]Supports: Tab completion, ~, $HOME, $USER, etc.[/dim]")
    console.print("[dim]Use ↑/↓ arrow keys to navigate previous paths.[/dim]")
    console.print()

    try:
        # Use raw input to support readline
        user_input = input("Path: ").strip()

        if not user_input:
            return ""

        # Strip quotes if present
        user_input = user_input.strip('"').strip("'")

        # Expand the path
        expanded = expand_path(user_input)

        return expanded

    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""
    finally:
        # Save path history and restore main history
        _save_path_history()
        _restore_history_from_list(main_history)
        # Reset completer to avoid affecting other prompts
        readline.set_completer(None)


def run_jacoco_analysis(path: str = None, output_format: str = None,
                        split_by: str = "package", max_tokens: int = 18000,
                        packages: str = "all", output_dir: str = "output"):
    """Run JaCoCo coverage report analysis.

    Args:
        path: Optional path to JaCoCo archive (zip/7z) or report directory.
              If not provided, user will be prompted.
        output_format: If set ('full', 'chunked', 'package'), skip interactive
                       prompts and save directly in the requested format.
        split_by: Split strategy for chunked output ('package', 'class', 'package_class').
        max_tokens: Max tokens per chunk for chunked output.
        packages: Package filter for package output ('all' or comma-separated names).
        output_dir: Output directory for saved files.
    """
    console.print("[bold]JaCoCo Coverage Report Analyzer[/bold]")
    console.print()

    # If path provided via command line, use it directly
    if path:
        path = path.strip().strip('"').strip("'")
        # Expand environment variables and user home directory
        path = os.path.expandvars(os.path.expanduser(path))

        if not os.path.exists(path):
            console.print(f"[red]Error:[/red] Path not found: {path}")
            return

        console.print(f"[dim]Analyzing:[/dim] {path}")
        console.print()

        # Determine if it's a file or directory
        if os.path.isfile(path):
            # Check if it's a 7z file and we need to select executable
            if path.endswith(".7z") or path.endswith(".7zip"):
                if not select_7zip_executable():
                    return

            with console.status(
                "[cyan]Extracting and analyzing JaCoCo report...[/cyan]"
            ):
                try:
                    result = analyze_jacoco_report(archive_path=path)
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    return
                except Exception as e:
                    console.print(f"[red]Error analyzing report:[/red] {e}")
                    return
        else:
            with console.status("[cyan]Analyzing JaCoCo report...[/cyan]"):
                try:
                    result = analyze_jacoco_report(report_dir=path)
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    return
                except Exception as e:
                    console.print(f"[red]Error analyzing report:[/red] {e}")
                    return
    else:
        # Auto-detect JaCoCo files in current directory
        archive_files, report_dirs = find_jacoco_files()
        detected_items = []

        for f in archive_files:
            detected_items.append(("archive", f))
        for d in report_dirs:
            detected_items.append(("directory", d))

        if detected_items:
            console.print("[bold]Detected JaCoCo files/directories:[/bold]")
            console.print()

            for i, (item_type, item_path) in enumerate(detected_items, 1):
                type_label = (
                    "[dim](archive)[/dim]"
                    if item_type == "archive"
                    else "[dim](directory)[/dim]"
                )
                console.print(f"    [green][{i}][/green] {item_path} {type_label}")

            console.print(
                f"    [green][{len(detected_items) + 1}][/green] Enter path manually"
            )
            console.print()

            valid_choices = [str(i) for i in range(1, len(detected_items) + 2)]

            try:
                choice = menu_ask(
                    "[bold]Enter choice[/bold]",
                    choices=valid_choices,
                )
                choice_idx = int(choice) - 1

                if choice_idx < len(detected_items):
                    item_type, selected_path = detected_items[choice_idx]
                    console.print()

                    if item_type == "archive":
                        # Check if it's a 7z file
                        if selected_path.endswith(".7z") or selected_path.endswith(
                            ".7zip"
                        ):
                            if not select_7zip_executable():
                                return

                        with console.status(
                            "[cyan]Extracting and analyzing JaCoCo report...[/cyan]"
                        ):
                            try:
                                result = analyze_jacoco_report(
                                    archive_path=selected_path
                                )
                            except ValueError as e:
                                console.print(f"[red]Error:[/red] {e}")
                                return
                            except Exception as e:
                                console.print(
                                    f"[red]Error analyzing report:[/red] {e}"
                                )
                                return
                    else:
                        with console.status("[cyan]Analyzing JaCoCo report...[/cyan]"):
                            try:
                                result = analyze_jacoco_report(report_dir=selected_path)
                            except ValueError as e:
                                console.print(f"[red]Error:[/red] {e}")
                                return
                            except Exception as e:
                                console.print(
                                    f"[red]Error analyzing report:[/red] {e}"
                                )
                                return
                else:
                    # Fall through to manual entry
                    detected_items = None

            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Cancelled.[/dim]")
                return

        # Manual entry (either no files detected or user chose manual)
        if not detected_items or (detected_items and choice_idx >= len(detected_items)):
            console.print()
            console.print("Select input method:")
            console.print("    [green][1][/green] Analyze a zip/7z archive")
            console.print("    [green][2][/green] Analyze an extracted directory")
            console.print()

            try:
                choice = menu_ask(
                    "[bold]Enter choice[/bold]", choices=["1", "2"]
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Cancelled.[/dim]")
                return

            if choice == "1":
                # Archive file
                archive_path = prompt_for_jacoco_path(
                    "Enter path to JaCoCo archive (zip or 7z):", path_type="file"
                )

                if not archive_path:
                    console.print("[dim]Cancelled.[/dim]")
                    return

                if not os.path.exists(archive_path):
                    console.print(f"[red]Error:[/red] File not found: {archive_path}")
                    return

                # Check if it's a 7z file and we need to select executable
                if archive_path.endswith(".7z") or archive_path.endswith(".7zip"):
                    console.print()
                    if not select_7zip_executable():
                        return

                console.print()
                with console.status(
                    "[cyan]Extracting and analyzing JaCoCo report...[/cyan]"
                ):
                    try:
                        result = analyze_jacoco_report(archive_path=archive_path)
                    except ValueError as e:
                        console.print(f"[red]Error:[/red] {e}")
                        return
                    except Exception as e:
                        console.print(f"[red]Error analyzing report:[/red] {e}")
                        return

            else:
                # Directory
                report_dir = prompt_for_jacoco_path(
                    "Enter path to JaCoCo report directory:", path_type="directory"
                )

                if not report_dir:
                    console.print("[dim]Cancelled.[/dim]")
                    return

                if not os.path.isdir(report_dir):
                    console.print(
                        f"[red]Error:[/red] Directory not found: {report_dir}"
                    )
                    return

                console.print()
                with console.status("[cyan]Analyzing JaCoCo report...[/cyan]"):
                    try:
                        result = analyze_jacoco_report(report_dir=report_dir)
                    except ValueError as e:
                        console.print(f"[red]Error:[/red] {e}")
                        return
                    except Exception as e:
                        console.print(f"[red]Error analyzing report:[/red] {e}")
                        return

    # Print the report
    print_jacoco_report(result)

    # Non-interactive mode: save directly and return
    if output_format:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
        run_output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        if output_format == "full":
            output_file = os.path.join(run_output_dir, "jacoco_analysis.json")
            formatted_result = format_analysis_result(result)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(formatted_result, f, indent=2, ensure_ascii=False)
            console.print()
            console.print(f"[green]Results saved to:[/green] [bold]{output_file}[/bold]")
            console.print(f"[bold]Output:[/bold] {os.path.realpath(run_output_dir)}")

        elif output_format == "chunked":
            chunks = format_analysis_result_chunked(result, max_tokens=max_tokens, split_by=split_by)
            console.print()
            console.print(f"[green]Generated {len(chunks)} chunk(s)[/green]")
            for i, chunk in enumerate(chunks, 1):
                chunk_file = os.path.join(run_output_dir, f"jacoco_chunk_{i}_of_{len(chunks)}.json")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)
                console.print(f"  [dim]Saved:[/dim] {chunk_file}")
            console.print()
            console.print("[green]All chunks saved![/green]")
            console.print(f"[bold]Output:[/bold] {os.path.realpath(run_output_dir)}")

        elif output_format == "package":
            available_packages = get_available_packages(result)
            if not available_packages:
                console.print("[yellow]No packages with coverage issues found.[/yellow]")
                return

            if packages.lower() == "all":
                selected_packages = available_packages
            else:
                # Match by package name (comma-separated)
                requested = [p.strip() for p in packages.split(",")]
                selected_packages = [
                    pkg for pkg in available_packages
                    if pkg['package'] in requested
                ]
                if not selected_packages:
                    console.print(f"[red]Error:[/red] No matching packages found for: {packages}")
                    console.print("[dim]Available packages:[/dim]")
                    for pkg in available_packages:
                        console.print(f"  {pkg['package']}")
                    return

            for pkg_info in selected_packages:
                pkg_name = pkg_info['package']
                pkg_result = format_analysis_for_package(result, pkg_name)
                safe_name = pkg_name.replace('/', '_').replace('.', '_')
                pkg_file = os.path.join(run_output_dir, f"jacoco_package_{safe_name}.json")
                with open(pkg_file, "w", encoding="utf-8") as f:
                    json.dump(pkg_result, f, indent=2, ensure_ascii=False)
                console.print(f"  [dim]Saved:[/dim] {pkg_file}")

            console.print()
            console.print(f"[green]Exported {len(selected_packages)} package(s)![/green]")
            console.print(f"[bold]Output:[/bold] {os.path.realpath(run_output_dir)}")

        return

    # Ask if user wants to save to JSON
    console.print()
    try:
        save_choice = menu_ask(
            "Save results to JSON?", choices=["y", "n"], default="y"
        )
        if save_choice.lower() == "y":
            # Ask about output format
            console.print()
            console.print("[bold]Select output format:[/bold]")
            console.print("    [green][1][/green] Full output (single file)")
            console.print("    [green][2][/green] Chunked output (multiple files, <20K tokens each) [bold cyan](Recommended)[/bold cyan]")
            console.print("    [green][3][/green] By package (select specific packages)")
            console.print()

            format_choice = menu_ask(
                "[bold]Enter choice[/bold]", choices=["1", "2", "3"], default="2"
            )

            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if format_choice == "1":
                # Full output (single file)
                timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
                full_output_dir = os.path.join(output_dir, timestamp)
                os.makedirs(full_output_dir, exist_ok=True)

                output_file = os.path.join(full_output_dir, "jacoco_analysis.json")
                formatted_result = format_analysis_result(result)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(formatted_result, f, indent=2, ensure_ascii=False)

                console.print(
                    f"[green]Results saved to:[/green] [bold]{output_file}[/bold]"
                )
                console.print(f"[bold]Output:[/bold] {os.path.realpath(full_output_dir)}")

            elif format_choice == "2":
                # Chunked output
                console.print()
                console.print("[bold]Split report by:[/bold]")
                console.print("    [green][1][/green] Package [bold cyan](Default)[/bold cyan]")
                console.print("    [green][2][/green] Class/File")
                console.print("    [green][3][/green] Package + Class [dim](two-level hierarchy)[/dim]")
                console.print()

                split_choice = menu_ask(
                    "[bold]Enter choice[/bold]", choices=["1", "2", "3"], default="1"
                )
                split_by_map = {"1": "package", "2": "class", "3": "package_class"}
                split_by = split_by_map[split_choice]

                console.print()
                max_tokens = int(Prompt.ask(
                    "Max tokens per chunk",
                    default="18000"
                ))

                chunks = format_analysis_result_chunked(result, max_tokens=max_tokens, split_by=split_by)
                console.print()
                console.print(f"[green]Generated {len(chunks)} chunk(s)[/green]")

                # Create datetime-named output directory
                timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
                chunk_output_dir = os.path.join(output_dir, timestamp)
                os.makedirs(chunk_output_dir, exist_ok=True)

                # Save each chunk
                for i, chunk in enumerate(chunks, 1):
                    chunk_file = os.path.join(chunk_output_dir, f"jacoco_chunk_{i}_of_{len(chunks)}.json")
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, indent=2, ensure_ascii=False)
                    console.print(f"  [dim]Saved:[/dim] {chunk_file}")

                console.print()
                console.print("[green]All chunks saved![/green]")
                console.print(f"[bold]Output:[/bold] {os.path.realpath(chunk_output_dir)}")

            elif format_choice == "3":
                # By package
                packages = get_available_packages(result)

                if not packages:
                    console.print("[yellow]No packages with coverage issues found.[/yellow]")
                else:
                    console.print()
                    console.print("[bold]Available packages with coverage issues:[/bold]")
                    console.print()

                    # Create table for packages
                    pkg_table = Table(show_header=True, header_style="bold")
                    pkg_table.add_column("#", style="green", width=4)
                    pkg_table.add_column("Package", style="bold")
                    pkg_table.add_column("Missed Branches", style="yellow", justify="right")
                    pkg_table.add_column("Uncovered Lines", style="red", justify="right")
                    pkg_table.add_column("Total", justify="right")

                    for i, pkg in enumerate(packages, 1):
                        pkg_table.add_row(
                            str(i),
                            pkg['package'],
                            str(pkg['missed_branches']),
                            str(pkg['uncovered_lines']),
                            str(pkg['total_issues'])
                        )

                    console.print(pkg_table)
                    console.print()

                    # Ask which packages to export
                    pkg_choices = Prompt.ask(
                        "Enter package numbers to export (comma-separated, or 'all')",
                        default="all"
                    )

                    if pkg_choices.lower() == "all":
                        selected_packages = packages
                    else:
                        selected_indices = [int(x.strip()) - 1 for x in pkg_choices.split(",") if x.strip().isdigit()]
                        selected_packages = [packages[i] for i in selected_indices if 0 <= i < len(packages)]

                    # Create datetime-named output directory
                    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
                    pkg_output_dir = os.path.join(output_dir, timestamp)
                    os.makedirs(pkg_output_dir, exist_ok=True)

                    # Export selected packages
                    for pkg_info in selected_packages:
                        pkg_name = pkg_info['package']
                        pkg_result = format_analysis_for_package(result, pkg_name)

                        # Sanitize package name for filename
                        safe_name = pkg_name.replace('/', '_').replace('.', '_')
                        pkg_file = os.path.join(pkg_output_dir, f"jacoco_package_{safe_name}.json")

                        with open(pkg_file, "w", encoding="utf-8") as f:
                            json.dump(pkg_result, f, indent=2, ensure_ascii=False)
                        console.print(f"  [dim]Saved:[/dim] {pkg_file}")

                    console.print()
                    console.print(f"[green]Exported {len(selected_packages)} package(s)![/green]")
                    console.print(f"[bold]Output:[/bold] {os.path.realpath(pkg_output_dir)}")

    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Skipping save.[/dim]")


def run_docs_generation(repo_path: str = None, output_dir: str = None, provider: str = None):
    """Run documentation generation interactively.

    Args:
        repo_path: Optional path to the repository.
        output_dir: Optional output directory for generated docs.
        provider: Optional AI provider ('openai' or 'deepseek').
    """
    console.print("[bold]AI Documentation Generator[/bold]")
    console.print("[dim]Generate documentation updates from git changes using AI[/dim]")
    console.print()

    # Get repository path if not provided
    if not repo_path:
        setup_path_completion()
        console.print("[bold]Enter repository path:[/bold]")
        console.print("[dim]Press Enter to use current directory[/dim]")
        try:
            repo_input = input("Path: ").strip()
            repo_path = expand_path(repo_input) if repo_input else "."
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return
        finally:
            readline.set_completer(None)

    # Get diff source
    console.print()
    console.print("[bold]Select change source to analyze:[/bold]")
    console.print()
    console.print("    [green][1][/green] Last commit [dim](diff between HEAD~1 and HEAD)[/dim]")
    console.print("    [green][2][/green] Staged changes [dim](changes ready to commit)[/dim]")
    console.print("    [green][3][/green] Unstaged changes [dim](working directory changes)[/dim]")
    console.print("    [green][4][/green] All changes [dim](staged + unstaged)[/dim]")
    console.print("    [green][5][/green] Specific commit ID [dim](enter commit SHA)[/dim]")
    console.print("    [green][6][/green] Full commit history [dim](root to HEAD)[/dim]")
    console.print()

    try:
        source_choice = menu_ask(
            "[bold]Enter choice[/bold]",
            choices=["1", "2", "3", "4", "5", "6"],
            default="1",
        )
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return

    # Handle commit ID input for option 5
    commit_id = None
    if source_choice == "5":
        console.print()
        console.print("[bold]Enter commit ID (SHA) to analyze:[/bold]")
        console.print("[dim]You can enter a full SHA or a short SHA (minimum 7 characters)[/dim]")
        try:
            commit_id = menu_ask("[bold]Commit ID[/bold]")
            commit_id = commit_id.strip()
            if not commit_id:
                console.print("[red]Error:[/red] No commit ID provided.")
                return
            if len(commit_id) < 7:
                console.print("[red]Error:[/red] Commit ID must be at least 7 characters.")
                return
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return

    diff_source_map = {
        "1": "commit",
        "2": "staged",
        "3": "unstaged",
        "4": "all",
        "5": "specific_commit",
        "6": "full_history",
    }
    diff_source = diff_source_map[source_choice]

    # Get provider if not provided - with credit display
    if not provider:
        console.print()
        console.print("[bold]Select AI provider:[/bold]")
        console.print()

        provider_list = list(DOCS_PROVIDERS.keys())
        provider_credits = {}

        # Fetch credits for each provider
        for prov in provider_list:
            prov_info = DOCS_PROVIDERS[prov]
            credit_str = "[dim]Not configured[/dim]"
            status_icon = "[red]○[/red]"

            if prov == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    with console.status(f"[cyan]Checking {prov_info['name']} credits...[/cyan]"):
                        credits = fetch_openai_credits(api_key)
                    if credits:
                        remaining = credits.get("remaining")
                        total_used = credits.get("total_used")
                        plan = credits.get("plan", "Active")
                        if remaining is not None:
                            credit_str = f"[green]${remaining:.2f}[/green]"
                            if total_used is not None:
                                credit_str += f" [dim](used: ${total_used:.2f})[/dim]"
                        else:
                            # No billing access - show plan status
                            credit_str = f"[green]{plan}[/green]"
                        status_icon = "[green]●[/green]"
                    else:
                        credit_str = "[red]Invalid key[/red]"
            elif prov == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if api_key:
                    with console.status(f"[cyan]Checking {prov_info['name']} credits...[/cyan]"):
                        credits = fetch_deepseek_credits(api_key)
                    if credits:
                        remaining = credits.get("remaining")
                        if remaining is not None:
                            credit_str = f"[green]${remaining:.2f}[/green]"
                        else:
                            credit_str = "[dim]N/A[/dim]"
                        status_icon = "[green]●[/green]"
                    else:
                        credit_str = "[red]Invalid key[/red]"
            elif prov == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    with console.status(f"[cyan]Checking {prov_info['name']} credits...[/cyan]"):
                        credits = fetch_anthropic_credits(api_key)
                    if credits:
                        credit_str = "[yellow]See console[/yellow]"
                        status_icon = "[green]●[/green]"
                    else:
                        credit_str = "[red]Invalid key[/red]"

            provider_credits[prov] = {"credit_str": credit_str, "status_icon": status_icon}

        # Display providers with credits
        for i, prov in enumerate(provider_list, 1):
            prov_info = DOCS_PROVIDERS[prov]
            cred_info = provider_credits.get(prov, {"credit_str": "[dim]N/A[/dim]", "status_icon": "[red]○[/red]"})
            console.print(
                f"    {cred_info['status_icon']} [green][{i}][/green] {prov_info['name']} "
                f"- Credit: {cred_info['credit_str']}"
            )
        console.print()

        try:
            prov_choice = menu_ask(
                "[bold]Enter choice[/bold]",
                choices=[str(i) for i in range(1, len(provider_list) + 1)],
                default="1",
            )
            provider = provider_list[int(prov_choice) - 1]
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return

    # Get output directory if not provided
    if not output_dir:
        console.print()
        default_output = os.path.join(os.path.abspath(repo_path), "docs")
        try:
            output_dir = Prompt.ask(
                "Output directory",
                default=default_output
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return

    # Run the generator
    console.print()
    console.print(f"[dim]Repository:[/dim]  {os.path.abspath(repo_path)}")
    console.print(f"[dim]Provider:[/dim]    {DOCS_PROVIDERS[provider]['name']}")
    console.print(f"[dim]Diff source:[/dim] {diff_source}")
    if commit_id:
        console.print(f"[dim]Commit ID:[/dim]   {commit_id}")
    console.print(f"[dim]Output:[/dim]      {output_dir}")
    console.print()

    if diff_source == "full_history":
        # Use progress display for chunked full history processing
        def progress_callback(seg_num, total_segs, chars):
            if chars > 0:
                console.print(
                    f"  [green]Segment {seg_num}/{total_segs} complete[/green] "
                    f"[dim]({chars:,} chars)[/dim]"
                )
            else:
                console.print(
                    f"  [yellow]Segment {seg_num}/{total_segs} skipped[/yellow] "
                    f"[dim](no updates)[/dim]"
                )

        console.print("[cyan]Processing full commit history in segments...[/cyan]")
        success, message = run_docs_generator(
            repo_path=repo_path,
            output_dir=output_dir,
            provider=provider,
            diff_source=diff_source,
            commit_id=commit_id,
            progress_callback=progress_callback,
        )
        if success and "saved to" in message.lower():
            console.print("[cyan]Combining segments...[/cyan]")
    else:
        with console.status("[cyan]AI is analyzing changes and generating documentation...[/cyan]"):
            success, message = run_docs_generator(
                repo_path=repo_path,
                output_dir=output_dir,
                provider=provider,
                diff_source=diff_source,
                commit_id=commit_id,
            )

    if success:
        console.print()
        console.print("[green]" + "=" * 50 + "[/green]")
        console.print(f"[green]{message}[/green]")
        console.print("[green]" + "=" * 50 + "[/green]")
    else:
        console.print(f"[red]Error:[/red] {message}")


def print_banner():
    """Print the application banner."""
    console.print()
    console.print(
        Panel(
            "[bold]SONAR JACOCO ANALYZER[/bold]\n"
            "[dim]Analyze SonarCloud issues and JaCoCo coverage reports[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def print_help():
    """Print help message with formatting."""
    print_banner()
    console.print("[bold]USAGE[/bold]")
    console.print("    codecraft-ai")
    console.print()
    console.print("[bold]OPTIONS[/bold]")
    console.print(
        "    [green]--api, -a[/green]              "
        "Fetch data from SonarCloud API (select project)"
    )
    console.print(
        "    [green]--jacoco, -j[/green] [PATH]    "
        "Analyze JaCoCo HTML coverage report"
    )
    console.print("                          PATH can be a zip/7z file or directory")
    console.print()
    console.print("    [bold]JaCoCo non-interactive flags[/bold] (use with --jacoco PATH):")
    console.print(
        "    [green]--output, -o[/green] FORMAT    "
        "Output format: full, chunked, package"
    )
    console.print(
        "    [green]--split-by, -s[/green] STRATEGY "
        "Chunked split: package, class, package_class [dim](default: package)[/dim]"
    )
    console.print(
        "    [green]--max-tokens, -t[/green] N      "
        "Max tokens per chunk [dim](default: 18000)[/dim]"
    )
    console.print(
        "    [green]--packages, -p[/green] PKGS     "
        "Package filter: all or comma-separated names [dim](default: all)[/dim]"
    )
    console.print(
        "    [green]--output-dir[/green] DIR        "
        "Output directory [dim](default: output)[/dim]"
    )
    console.print()
    console.print(
        "    [green]--commit, -c[/green]           "
        "AI-powered commit message generator (interactive)"
    )
    console.print(
        "    [green]--quick-commit, -q[/green]     "
        "Quick commit: auto-detect repo, generate and commit"
    )
    console.print(
        "    [green]--docs, -d[/green]             "
        "Generate documentation from git changes using AI"
    )
    console.print(
        "    [green]--reset, -r[/green]            "
        "Reset saved selections (data source and project)"
    )
    console.print(
        "    [green]--clear-history[/green]        "
        "Clear user input history"
    )
    console.print(
        "    [green]--clear-output[/green]         "
        "Clear the output directory"
    )
    console.print("    [green]--help, -h[/green]             Show this help message")
    console.print()
    console.print("[dim]If no option is provided, you'll be prompted to choose.[/dim]")
    console.print("[dim]Your selections are remembered for future runs.[/dim]")
    console.print()


def main():
    """Main entry point with mode selection."""
    # Initialize input history for arrow key navigation
    setup_input_history()

    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--api", "-a"):
            print_banner()
            run_with_api()
            return
        elif arg in ("--jacoco", "-j"):
            print_banner()
            # Parse --jacoco arguments: PATH and optional flags
            jacoco_path = None
            output_format = None
            split_by = "package"
            max_tokens = 18000
            packages = "all"
            output_dir = "output"

            i = 2
            # First non-flag argument after --jacoco is the PATH
            if i < len(sys.argv) and not sys.argv[i].startswith("-"):
                jacoco_path = sys.argv[i]
                i += 1

            while i < len(sys.argv):
                flag = sys.argv[i]
                if flag in ("--output", "-o") and i + 1 < len(sys.argv):
                    output_format = sys.argv[i + 1]
                    if output_format not in ("full", "chunked", "package"):
                        console.print(f"[red]Error:[/red] Invalid --output value: {output_format}")
                        console.print("[dim]Valid values: full, chunked, package[/dim]")
                        return
                    i += 2
                elif flag in ("--split-by", "-s") and i + 1 < len(sys.argv):
                    split_by = sys.argv[i + 1]
                    if split_by not in ("package", "class", "package_class"):
                        console.print(f"[red]Error:[/red] Invalid --split-by value: {split_by}")
                        console.print("[dim]Valid values: package, class, package_class[/dim]")
                        return
                    i += 2
                elif flag in ("--max-tokens", "-t") and i + 1 < len(sys.argv):
                    try:
                        max_tokens = int(sys.argv[i + 1])
                    except ValueError:
                        console.print(f"[red]Error:[/red] --max-tokens must be an integer")
                        return
                    i += 2
                elif flag in ("--packages", "-p") and i + 1 < len(sys.argv):
                    packages = sys.argv[i + 1]
                    i += 2
                elif flag == "--output-dir" and i + 1 < len(sys.argv):
                    output_dir = sys.argv[i + 1]
                    i += 2
                else:
                    console.print(f"[red]Error:[/red] Unknown flag: {flag}")
                    return

            # When --output is provided, PATH is required
            if output_format and not jacoco_path:
                console.print("[red]Error:[/red] PATH is required when --output is specified")
                console.print("[dim]Usage: codecraft-ai --jacoco PATH --output full|chunked|package[/dim]")
                return

            run_jacoco_analysis(
                path=jacoco_path,
                output_format=output_format,
                split_by=split_by,
                max_tokens=max_tokens,
                packages=packages,
                output_dir=output_dir,
            )
            return
        elif arg in ("--commit", "-c"):
            # Run the commit message generator (interactive mode)
            commit_main()
            return
        elif arg in ("--quick-commit", "-q"):
            # Run quick commit mode
            from .commit_config import CommitConfig, ConfigurationError

            try:
                config = CommitConfig.from_env()
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to load configuration: {e}")
                console.print("[dim]Set OPENAI_API_KEY in your .env file.[/dim]")
                sys.exit(1)

            # Validate OpenAI configuration
            is_valid, errors = config.validate_openai()
            if not is_valid:
                for error in errors:
                    console.print(f"[red]Error:[/red] {error}")
                console.print("[dim]Set OPENAI_API_KEY in your .env file.[/dim]")
                sys.exit(1)

            success = run_quick_commit(config)
            sys.exit(0 if success else 1)
        elif arg in ("--docs", "-d"):
            print_banner()
            run_docs_generation()
            return
        elif arg in ("--reset", "-r"):
            print_banner()
            if reset_config():
                console.print("[green]Saved selections have been reset.[/green]")
            else:
                console.print("[dim]No saved selections to reset.[/dim]")
            return
        elif arg == "--clear-history":
            print_banner()
            if clear_history():
                console.print("[green]User input history has been cleared.[/green]")
            else:
                console.print("[red]Failed to clear history.[/red]")
            return
        elif arg == "--clear-output":
            print_banner()
            success, count = clear_output()
            if success:
                if count > 0:
                    console.print(f"[green]Output directory cleared ({count} file(s) removed).[/green]")
                else:
                    console.print("[dim]Output directory is already empty.[/dim]")
            else:
                console.print("[red]Failed to clear output directory.[/red]")
            return
        elif arg in ("--help", "-h"):
            print_help()
            return

    print_banner()

    # Load saved configuration
    config = load_config()
    saved_mode = config.get("data_source")

    # Check if we have a saved selection (only "api" is valid now)
    if saved_mode == "api":
        console.print(f"[dim]Using saved selection:[/dim] [bold]SonarCloud API[/bold]")
        console.print()

        # Show option to change
        console.print(
            "[dim]Press Enter to continue, or type 'r' to reset and choose again[/dim]"
        )
        try:
            choice = menu_ask("", default="")
            if choice.lower() == "r":
                reset_config()
                console.print("[green]Selection reset.[/green]")
                console.print()
                # Fall through to interactive selection
                saved_mode = None
            else:
                run_with_api()
                return
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            return

    # Interactive mode selection
    console.print("[bold]Select analysis mode:[/bold]")
    console.print()
    console.print(
        "    [green][1][/green] Fetch from SonarCloud API "
        "[dim](select project interactively)[/dim]"
    )
    console.print(
        "    [green][2][/green] Analyze JaCoCo coverage report "
        "[dim](zip/7z or directory)[/dim]"
    )
    console.print(
        "    [green][3][/green] AI-powered commit message generator "
        "[dim](generate conventional commits)[/dim]"
    )
    console.print(
        "    [green][4][/green] AI documentation generator "
        "[dim](generate docs from git changes)[/dim]"
    )
    console.print()

    while True:
        try:
            choice = menu_ask(
                "[bold]Enter choice[/bold]",
                choices=["1", "2", "3", "4"],
            )

            if choice == "1":
                # Save selection
                config["data_source"] = "api"
                save_config(config)
                console.print()
                run_with_api()
                break
            elif choice == "2":
                console.print()
                run_jacoco_analysis()
                break
            elif choice == "3":
                console.print()
                commit_main()
                break
            elif choice == "4":
                console.print()
                run_docs_generation()
                break
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        console.print("[red]Error:[/red] 'input.json' file not found!")
        console.print(
            "[dim]Please ensure you have an 'input.json' file in the current directory.[/dim]"
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON file:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)

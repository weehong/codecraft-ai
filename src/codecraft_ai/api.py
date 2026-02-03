"""
SonarCloud API Client Module

Provides functionality to interact with SonarCloud API for:
- Listing projects in an organization
- Fetching code smell issues for a selected project
"""

import json
import os
import readline
import urllib.request
import urllib.parse
import urllib.error
import base64

# Config file path for saving user selections (in user's home directory)
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".codecraft_ai_config.json")


def load_config():
    """
    Load saved configuration from config file.

    Returns:
        Dictionary with saved configuration or empty dict if not found
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config):
    """
    Save configuration to config file.

    Args:
        config: Dictionary with configuration to save
    """
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save config: {e}")


def reset_config():
    """
    Reset/clear saved configuration.

    Returns:
        True if config was reset, False if no config existed
    """
    if os.path.exists(CONFIG_FILE):
        try:
            os.remove(CONFIG_FILE)
            return True
        except IOError:
            return False
    return False


def is_meaningful_history_entry(entry: str) -> bool:
    """
    Check if a history entry is meaningful and should be saved.

    Filters out menu selections (single digits, single letters) and other
    non-meaningful inputs that clutter the history.
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
        "all", "q", "yes", "no", "y", "n", "c", "r",
        "approve", "edit", "regenerate", "cancel",
    }
    if entry.lower() in menu_patterns:
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
            readline.remove_history_item(history_length - 1)


class SonarCloudAPI:
    """Client for interacting with SonarCloud API."""

    BASE_URL = "https://sonarcloud.io/api"

    def __init__(self, token=None, organization=None, cookies=None, xsrf_token=None):
        """
        Initialize the SonarCloud API client.

        Args:
            token: SonarCloud user token for authentication
            organization: Organization key (required for listing projects)
            cookies: Browser session cookies (alternative auth method)
            xsrf_token: XSRF token (required when using cookies)
        """
        self.token = token
        self.organization = organization
        self.cookies = cookies
        self.xsrf_token = xsrf_token

    @classmethod
    def from_env(cls):
        """
        Create a SonarCloudAPI instance from environment variables.

        Environment variables:
            SONAR_TOKEN: User token
            SONAR_ORGANIZATION: Organization key
            SONAR_COOKIES: Browser session cookies
            SONAR_XSRF_TOKEN: XSRF token
        """
        return cls(
            token=os.getenv('SONAR_TOKEN'),
            organization=os.getenv('SONAR_ORGANIZATION'),
            cookies=os.getenv('SONAR_COOKIES'),
            xsrf_token=os.getenv('SONAR_XSRF_TOKEN')
        )

    def _build_headers(self):
        """Build request headers based on authentication method."""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'codecraft-ai/1.0'
        }

        if self.token:
            # Token authentication uses Basic auth with token as username
            credentials = base64.b64encode(f"{self.token}:".encode()).decode()
            headers['Authorization'] = f"Basic {credentials}"
        elif self.cookies:
            headers['Cookie'] = self.cookies
            if self.xsrf_token:
                headers['X-XSRF-TOKEN'] = self.xsrf_token

        return headers

    def _make_request(self, endpoint, params=None):
        """
        Make an API request to SonarCloud.

        Args:
            endpoint: API endpoint path (e.g., '/projects/search')
            params: Dictionary of query parameters

        Returns:
            Parsed JSON response

        Raises:
            Exception: If the request fails
        """
        url = f"{self.BASE_URL}{endpoint}"

        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"

        headers = self._build_headers()
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            raise Exception(f"API request failed: {e.code} {e.reason}\n{error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def list_projects(self, page_size=100):
        """
        List all projects in the organization.

        Args:
            page_size: Number of projects per page (max 500)

        Returns:
            List of project dictionaries with keys: key, name, qualifier, visibility
        """
        if not self.organization:
            raise ValueError("Organization is required to list projects. Set SONAR_ORGANIZATION.")

        all_projects = []
        page = 1

        while True:
            params = {
                'organization': self.organization,
                'ps': min(page_size, 500),
                'p': page
            }

            response = self._make_request('/projects/search', params)
            projects = response.get('components', [])
            all_projects.extend(projects)

            # Check if there are more pages
            paging = response.get('paging', {})
            total = paging.get('total', 0)

            if len(all_projects) >= total:
                break

            page += 1

        return all_projects

    def get_issues(self, project_key, page_size=500, resolved=False, issue_types=None):
        """
        Fetch code smell issues for a project.

        Args:
            project_key: The project key to fetch issues for
            page_size: Number of issues per page (max 500)
            resolved: Whether to include resolved issues
            issue_types: List of issue types to filter (e.g., ['CODE_SMELL', 'BUG'])

        Returns:
            Dictionary with 'issues' and 'rules' arrays (compatible with analyzer)
        """
        all_issues = []
        all_rules = {}
        page = 1

        while True:
            params = {
                'componentKeys': project_key,
                's': 'FILE_LINE',
                'resolved': 'true' if resolved else 'false',
                'ps': min(page_size, 500),
                'p': page,
                'facets': 'severities,types',
                'additionalFields': '_all'
            }

            if issue_types:
                params['types'] = ','.join(issue_types)

            response = self._make_request('/issues/search', params)

            issues = response.get('issues', [])
            rules = response.get('rules', [])

            all_issues.extend(issues)

            # Collect unique rules
            for rule in rules:
                all_rules[rule['key']] = rule

            # Check if there are more pages
            paging = response.get('paging', {})
            total = paging.get('total', 0)

            if len(all_issues) >= total:
                break

            page += 1

            # Safety limit to prevent infinite loops
            if page > 100:
                print(f"Warning: Reached page limit. Retrieved {len(all_issues)} of {total} issues.")
                break

        return {
            'issues': all_issues,
            'rules': list(all_rules.values()),
            'total': len(all_issues)
        }

    def get_project_status(self, project_key):
        """
        Get the quality gate status for a project.

        Args:
            project_key: The project key

        Returns:
            Quality gate status information
        """
        params = {'projectKey': project_key}
        return self._make_request('/qualitygates/project_status', params)


def load_env_file(filepath='.env'):
    """
    Load environment variables from a .env file.

    Args:
        filepath: Path to the .env file
    """
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse key=value
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]

                os.environ[key] = value

    return True


def select_project_interactive(projects, console=None):
    """
    Display projects and let user select one interactively.
    Remembers the last selected project and offers to reuse it.

    Args:
        projects: List of project dictionaries
        console: Optional rich Console instance for output

    Returns:
        Selected project dictionary or None if cancelled
    """
    # Import rich components here to avoid circular imports
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt

    if console is None:
        from rich.console import Console
        console = Console()

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        return None

    # Check for saved project selection
    config = load_config()
    saved_project_key = config.get('selected_project_key')
    saved_project_name = config.get('selected_project_name')

    # Find saved project in current list
    saved_project = None
    if saved_project_key:
        for p in projects:
            if p['key'] == saved_project_key:
                saved_project = p
                break

    # If saved project exists and is in the list, offer to use it
    if saved_project:
        console.print(f"[dim]Previously selected project:[/dim] [bold]{saved_project_name}[/bold]")
        console.print()
        console.print("[dim]Press Enter to use this project, or type 'c' to choose a different one[/dim]")

        try:
            choice = Prompt.ask("", default="")
            remove_last_history_if_not_meaningful()
            if choice.lower() != 'c':
                return saved_project
            console.print()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return None

    # Create projects table
    table = Table(title="Available Projects", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Visibility", justify="center", width=10)
    table.add_column("Name", style="bold")
    table.add_column("Key", style="dim")

    for i, project in enumerate(projects, 1):
        visibility = project.get('visibility', 'unknown')
        if visibility == 'private':
            visibility_display = "[red]Private[/red]"
        else:
            visibility_display = "[green]Public[/green]"
        table.add_row(str(i), visibility_display, project['name'], project['key'])

    console.print()
    console.print(table)
    console.print()

    while True:
        try:
            choice = Prompt.ask(
                "Enter project number [dim](or 'q' to quit)[/dim]",
                default="q"
            )
            remove_last_history_if_not_meaningful()

            if choice.lower() == 'q':
                return None

            index = int(choice) - 1

            if 0 <= index < len(projects):
                selected = projects[index]
                # Save the selection
                config['selected_project_key'] = selected['key']
                config['selected_project_name'] = selected['name']
                save_config(config)
                return selected
            else:
                console.print(f"[yellow]Please enter a number between 1 and {len(projects)}[/yellow]")
        except ValueError:
            console.print("[yellow]Please enter a valid number or 'q' to quit[/yellow]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return None

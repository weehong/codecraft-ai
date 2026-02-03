"""
JaCoCo HTML Report Analyzer Module

Analyzes JaCoCo HTML coverage reports to extract:
- Missed branches
- Uncovered code lines
- Coverage statistics per class/file
"""

import os
import re
import zipfile
import tempfile
import shutil
from html.parser import HTMLParser
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class MissedBranch:
    """Represents a missed branch in the code."""
    file_path: str
    class_name: str
    line_number: int
    branch_info: str  # e.g., "1 of 2 branches missed"
    source_line: str


@dataclass
class UncoveredLine:
    """Represents an uncovered line of code."""
    file_path: str
    class_name: str
    line_number: int
    source_line: str
    instruction_missed: int = 0
    instruction_covered: int = 0


@dataclass
class CoverageStats:
    """Coverage statistics for a class/file."""
    file_path: str
    class_name: str
    instruction_missed: int = 0
    instruction_covered: int = 0
    branch_missed: int = 0
    branch_covered: int = 0
    line_missed: int = 0
    line_covered: int = 0
    method_missed: int = 0
    method_covered: int = 0


@dataclass
class JaCoCoAnalysisResult:
    """Complete analysis result from JaCoCo report."""
    missed_branches: List[MissedBranch] = field(default_factory=list)
    uncovered_lines: List[UncoveredLine] = field(default_factory=list)
    coverage_stats: List[CoverageStats] = field(default_factory=list)
    total_files_analyzed: int = 0
    source_directory: str = ""


class JaCoCoSourceHTMLParser(HTMLParser):
    """
    Parser for JaCoCo source code HTML files.
    Extracts line-by-line coverage information.
    """

    def __init__(self, file_path: str, class_name: str):
        super().__init__()
        self.file_path = file_path
        self.class_name = class_name
        self.missed_branches: List[MissedBranch] = []
        self.uncovered_lines: List[UncoveredLine] = []

        self.current_line_number = 0
        self.current_line_class = ""
        self.current_line_title = ""
        self.current_line_content = ""
        self.in_span = False
        self.span_class = ""
        self.span_title = ""
        self.in_code = False
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'pre':
            self.in_pre = True

        if tag == 'span' and self.in_pre:
            self.in_span = True
            self.span_class = attrs_dict.get('class', '')
            self.span_title = attrs_dict.get('title', '')
            # Extract line number from id attribute (e.g., "L1", "L2")
            span_id = attrs_dict.get('id', '')
            if span_id.startswith('L') and span_id[1:].isdigit():
                self.current_line_number = int(span_id[1:])
                self.current_line_class = self.span_class
                self.current_line_title = self.span_title
                self.current_line_content = ""

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.in_pre = False

        if tag == 'span' and self.in_span:
            self.in_span = False
            # Process the completed line
            if self.current_line_number > 0:
                self._process_line()

    def handle_data(self, data):
        if self.in_span and self.in_pre:
            self.current_line_content += data

    def _process_line(self):
        """Process a completed source line and extract coverage info."""
        line_content = self.current_line_content.strip()

        # Skip empty lines
        if not line_content:
            return

        # Check for missed branches (partial coverage - yellow/orange background)
        # JaCoCo uses class "pc" for partially covered and has branch info in title
        if 'pc' in self.current_line_class:
            branch_info = self.current_line_title or "Partially covered"
            self.missed_branches.append(MissedBranch(
                file_path=self.file_path,
                class_name=self.class_name,
                line_number=self.current_line_number,
                branch_info=branch_info,
                source_line=line_content
            ))

        # Check for completely uncovered lines (red background)
        # JaCoCo uses class "nc" for not covered
        if 'nc' in self.current_line_class:
            self.uncovered_lines.append(UncoveredLine(
                file_path=self.file_path,
                class_name=self.class_name,
                line_number=self.current_line_number,
                source_line=line_content
            ))


class JaCoCoIndexHTMLParser(HTMLParser):
    """
    Parser for JaCoCo index.html to extract overall coverage statistics
    and links to individual source files.
    """

    def __init__(self):
        super().__init__()
        self.source_files: List[Tuple[str, str]] = []  # (href, display_name)
        self.coverage_stats: List[CoverageStats] = []

        self.in_table = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.current_row_data: List[str] = []
        self.current_cell_content = ""
        self.current_href = ""
        self.cell_index = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'table':
            self.in_table = True
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr' and self.in_tbody:
            self.in_row = True
            self.current_row_data = []
            self.cell_index = 0
        elif tag == 'td' and self.in_row:
            self.in_cell = True
            self.current_cell_content = ""
            self.current_href = ""
        elif tag == 'a' and self.in_cell:
            self.current_href = attrs_dict.get('href', '')

    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            # Process completed row
            if self.current_row_data:
                self._process_row()
        elif tag == 'td' and self.in_cell:
            self.in_cell = False
            self.current_row_data.append((self.current_cell_content.strip(), self.current_href))
            self.cell_index += 1

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell_content += data

    def _process_row(self):
        """Process a completed table row to extract coverage data."""
        if len(self.current_row_data) < 2:
            return

        # First cell usually contains the element name/link
        element_name, href = self.current_row_data[0]

        if href and (href.endswith('.html') or '/' in href):
            self.source_files.append((href, element_name))


def find_7zip_executables() -> List[str]:
    """
    Find all available 7-Zip executables on the system.

    Returns:
        List of paths to 7z executables found
    """
    import platform
    import subprocess

    found_paths = []

    system = platform.system()

    if system == 'Windows':
        # Common Windows installation paths
        common_paths = [
            r'C:\Program Files\7-Zip\7z.exe',
            r'C:\Program Files (x86)\7-Zip\7z.exe',
            r'C:\Program Files\7-Zip\7za.exe',
            r'C:\Program Files (x86)\7-Zip\7za.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\7-Zip\7z.exe'),
            os.path.expandvars(r'%PROGRAMFILES%\7-Zip\7z.exe'),
            os.path.expandvars(r'%PROGRAMFILES(X86)%\7-Zip\7z.exe'),
        ]

        # Check PATH environment variable
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        for path_dir in path_dirs:
            for exe_name in ['7z.exe', '7za.exe']:
                exe_path = os.path.join(path_dir, exe_name)
                if os.path.isfile(exe_path) and exe_path not in common_paths:
                    common_paths.append(exe_path)
    else:
        # Linux/macOS paths
        common_paths = [
            '/usr/bin/7z',
            '/usr/bin/7za',
            '/usr/bin/7zr',
            '/usr/local/bin/7z',
            '/usr/local/bin/7za',
            '/usr/local/bin/7zr',
            '/opt/homebrew/bin/7z',  # macOS Homebrew ARM
            '/opt/homebrew/bin/7za',
            '/home/linuxbrew/.linuxbrew/bin/7z',  # Linuxbrew
            os.path.expanduser('~/.local/bin/7z'),
            os.path.expanduser('~/.local/bin/7za'),
        ]

        # Check PATH using 'which' command
        for cmd in ['7z', '7za', '7zr']:
            try:
                result = subprocess.run(
                    ['which', cmd],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if path and path not in common_paths:
                        common_paths.append(path)
            except (FileNotFoundError, subprocess.SubprocessError):
                pass

    # Check which paths actually exist and are executable
    for path in common_paths:
        # Expand environment variables
        expanded_path = os.path.expandvars(path)
        if os.path.isfile(expanded_path) and os.access(expanded_path, os.X_OK):
            if expanded_path not in found_paths:
                found_paths.append(expanded_path)

    return found_paths


# Module-level variable to store selected 7zip path
_selected_7zip_path: Optional[str] = None


def get_7zip_path() -> Optional[str]:
    """Get the currently selected 7zip path."""
    global _selected_7zip_path
    return _selected_7zip_path


def set_7zip_path(path: str) -> None:
    """Set the 7zip path to use for extraction."""
    global _selected_7zip_path
    _selected_7zip_path = path


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """
    Extract a zip or 7z archive to the specified directory.

    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to

    Returns:
        True if extraction was successful
    """
    archive_path = os.path.abspath(archive_path)

    if archive_path.endswith('.zip'):
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid or corrupted zip file: {archive_path}")

    elif archive_path.endswith('.7z') or archive_path.endswith('.7zip'):
        # Try using py7zr if available, otherwise fall back to system 7z command
        try:
            import py7zr
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(path=extract_to)
            return True
        except ImportError:
            # Fall back to system 7z command
            import subprocess

            # Use the selected 7zip path if available
            seven_zip_cmd = get_7zip_path() or '7z'

            try:
                result = subprocess.run(
                    [seven_zip_cmd, 'x', archive_path, f'-o{extract_to}', '-y'],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            except FileNotFoundError:
                raise ValueError(
                    "Cannot extract 7z file. Please install py7zr (pip install py7zr) "
                    "or 7-Zip command line tool."
                )
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def find_jacoco_index(directory: str) -> Optional[str]:
    """
    Find the JaCoCo index.html file in the extracted directory.

    Args:
        directory: Directory to search in

    Returns:
        Path to index.html or None if not found
    """
    # Common locations for JaCoCo reports
    possible_paths = [
        os.path.join(directory, 'index.html'),
        os.path.join(directory, 'jacoco', 'index.html'),
        os.path.join(directory, 'site', 'jacoco', 'index.html'),
        os.path.join(directory, 'target', 'site', 'jacoco', 'index.html'),
        os.path.join(directory, 'build', 'reports', 'jacoco', 'test', 'html', 'index.html'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Search recursively for index.html
    for root, dirs, files in os.walk(directory):
        if 'index.html' in files:
            # Verify it's a JaCoCo report by checking content
            index_path = os.path.join(root, 'index.html')
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    if 'jacoco' in content.lower() or 'coverage' in content.lower():
                        return index_path
            except (IOError, UnicodeDecodeError):
                continue

    return None


def find_source_html_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    Find all source HTML files in the JaCoCo report directory.

    Args:
        base_dir: Base directory of the JaCoCo report

    Returns:
        List of (file_path, class_name) tuples
    """
    source_files = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # JaCoCo source files end with .java.html or similar
            if file.endswith('.html') and file != 'index.html':
                # Skip package index files
                if file in ('index.html', 'index.source.html'):
                    continue

                file_path = os.path.join(root, file)
                # Extract class name from file name
                class_name = file.replace('.html', '')
                # Get relative path for context
                rel_path = os.path.relpath(file_path, base_dir)
                source_files.append((file_path, class_name, rel_path))

    return source_files


def parse_source_file(file_path: str, class_name: str) -> Tuple[List[MissedBranch], List[UncoveredLine]]:
    """
    Parse a JaCoCo source HTML file to extract coverage information.

    Args:
        file_path: Path to the HTML file
        class_name: Name of the class

    Returns:
        Tuple of (missed_branches, uncovered_lines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return [], []

    parser = JaCoCoSourceHTMLParser(file_path, class_name)
    try:
        parser.feed(content)
    except Exception:
        return [], []

    return parser.missed_branches, parser.uncovered_lines


def analyze_jacoco_report(archive_path: str = None, report_dir: str = None) -> JaCoCoAnalysisResult:
    """
    Analyze a JaCoCo HTML report.

    Args:
        archive_path: Path to a zip or 7z archive containing the JaCoCo report
        report_dir: Path to an already extracted JaCoCo report directory

    Returns:
        JaCoCoAnalysisResult with all missed branches and uncovered lines
    """
    result = JaCoCoAnalysisResult()
    temp_dir = None

    try:
        if archive_path:
            # Expand environment variables and user home directory in archive path
            archive_path = os.path.expandvars(os.path.expanduser(archive_path))
            # Extract archive to temp directory
            temp_dir = tempfile.mkdtemp(prefix='jacoco_')
            extract_archive(archive_path, temp_dir)
            base_dir = temp_dir
        elif report_dir:
            # Expand environment variables and user home directory in report directory
            report_dir = os.path.expandvars(os.path.expanduser(report_dir))
            base_dir = report_dir
        else:
            raise ValueError("Either archive_path or report_dir must be provided")

        # Find the index.html
        index_path = find_jacoco_index(base_dir)
        if not index_path:
            raise ValueError("Could not find JaCoCo index.html in the provided location")

        result.source_directory = os.path.dirname(index_path)

        # Find all source HTML files
        source_files = find_source_html_files(result.source_directory)
        result.total_files_analyzed = len(source_files)

        # Parse each source file
        for file_path, class_name, rel_path in source_files:
            missed_branches, uncovered_lines = parse_source_file(file_path, class_name)

            # Update file paths to be more readable
            for mb in missed_branches:
                mb.file_path = rel_path
            for ul in uncovered_lines:
                ul.file_path = rel_path

            result.missed_branches.extend(missed_branches)
            result.uncovered_lines.extend(uncovered_lines)

    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def format_analysis_result(result: JaCoCoAnalysisResult) -> Dict:
    """
    Format the analysis result as a dictionary for JSON output.

    Args:
        result: JaCoCoAnalysisResult object

    Returns:
        Dictionary representation of the result
    """
    return {
        'summary': {
            'total_files_analyzed': result.total_files_analyzed,
            'total_missed_branches': len(result.missed_branches),
            'total_uncovered_lines': len(result.uncovered_lines),
        },
        'missed_branches': [
            {
                'file': mb.file_path,
                'class': mb.class_name,
                'line': mb.line_number,
                'branch_info': mb.branch_info,
                'source': mb.source_line
            }
            for mb in result.missed_branches
        ],
        'uncovered_lines': [
            {
                'file': ul.file_path,
                'class': ul.class_name,
                'line': ul.line_number,
                'source': ul.source_line
            }
            for ul in result.uncovered_lines
        ],
        'by_file': _group_by_file(result)
    }


def _group_by_file(result: JaCoCoAnalysisResult) -> Dict:
    """Group missed branches and uncovered lines by file."""
    by_file = {}

    for mb in result.missed_branches:
        if mb.file_path not in by_file:
            by_file[mb.file_path] = {'missed_branches': [], 'uncovered_lines': []}
        by_file[mb.file_path]['missed_branches'].append({
            'line': mb.line_number,
            'branch_info': mb.branch_info,
            'source': mb.source_line
        })

    for ul in result.uncovered_lines:
        if ul.file_path not in by_file:
            by_file[ul.file_path] = {'missed_branches': [], 'uncovered_lines': []}
        by_file[ul.file_path]['uncovered_lines'].append({
            'line': ul.line_number,
            'source': ul.source_line
        })

    return by_file


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.
    Rough estimate: ~4 characters per token for code/JSON.
    """
    return len(text) // 4


def _get_package_from_path(file_path: str) -> str:
    """Extract package name from file path."""
    # Remove the .html suffix and file name to get package
    parts = file_path.replace('.java.html', '').replace('.html', '').split('/')
    if len(parts) > 1:
        return parts[0]  # Return the package directory
    return file_path


def _get_class_from_path(file_path: str) -> str:
    """Extract class name from file path (e.g., 'com.example/MyClass.java.html' -> 'MyClass.java')."""
    basename = os.path.basename(file_path)
    # Remove .html suffix to get the class file name
    if basename.endswith('.html'):
        return basename[:-5]  # e.g., MyClass.java
    return basename


def format_analysis_result_chunked(
    result: JaCoCoAnalysisResult,
    max_tokens: int = 18000,
    include_branches: bool = True,
    include_lines: bool = True,
    file_filter: str = None,
    split_by: str = "package"
) -> List[Dict]:
    """
    Format the analysis result as multiple chunks to stay under token limits.

    Args:
        result: JaCoCoAnalysisResult object
        max_tokens: Maximum tokens per chunk (default 18000 to stay under 20000)
        include_branches: Include missed branches in output
        include_lines: Include uncovered lines in output
        file_filter: Optional filter for file paths (substring match)
        split_by: Element to split by - "package", "class", or "package_class"

    Returns:
        List of dictionaries, each representing a chunk of the result
    """
    import json

    # Filter data if file_filter is specified
    missed_branches = result.missed_branches
    uncovered_lines = result.uncovered_lines

    if file_filter:
        missed_branches = [mb for mb in missed_branches if file_filter.lower() in mb.file_path.lower()]
        uncovered_lines = [ul for ul in uncovered_lines if file_filter.lower() in ul.file_path.lower()]

    if not include_branches:
        missed_branches = []
    if not include_lines:
        uncovered_lines = []

    if split_by == "class":
        return _chunk_by_class(result, missed_branches, uncovered_lines, max_tokens)
    elif split_by == "package_class":
        return _chunk_by_package_class(result, missed_branches, uncovered_lines, max_tokens)
    else:
        return _chunk_by_package(result, missed_branches, uncovered_lines, max_tokens)


def _chunk_by_package(
    result: JaCoCoAnalysisResult,
    missed_branches: List[MissedBranch],
    uncovered_lines: List[UncoveredLine],
    max_tokens: int
) -> List[Dict]:
    """Chunk analysis results grouped by package."""
    import json

    # Group by package
    packages = {}
    for mb in missed_branches:
        pkg = _get_package_from_path(mb.file_path)
        if pkg not in packages:
            packages[pkg] = {'missed_branches': [], 'uncovered_lines': []}
        packages[pkg]['missed_branches'].append(mb)

    for ul in uncovered_lines:
        pkg = _get_package_from_path(ul.file_path)
        if pkg not in packages:
            packages[pkg] = {'missed_branches': [], 'uncovered_lines': []}
        packages[pkg]['uncovered_lines'].append(ul)

    # Create chunks
    chunks = []
    current_chunk_data = {}
    current_chunk_branches = 0
    current_chunk_lines = 0

    for pkg in sorted(packages.keys()):
        pkg_data = packages[pkg]
        pkg_dict = _format_group_data(pkg_data)

        pkg_json = json.dumps({pkg: pkg_dict}, indent=2)
        pkg_tokens = _estimate_tokens(pkg_json)

        current_json = json.dumps(current_chunk_data, indent=2) if current_chunk_data else "{}"
        current_tokens = _estimate_tokens(current_json)

        if current_tokens + pkg_tokens > max_tokens and current_chunk_data:
            chunks.append(_create_chunk_dict(
                current_chunk_data, current_chunk_branches, current_chunk_lines,
                len(chunks) + 1, result.total_files_analyzed,
                len(missed_branches), len(uncovered_lines),
                split_by="package"
            ))
            current_chunk_data = {}
            current_chunk_branches = 0
            current_chunk_lines = 0

        current_chunk_data[pkg] = pkg_dict
        current_chunk_branches += len(pkg_data['missed_branches'])
        current_chunk_lines += len(pkg_data['uncovered_lines'])

    if current_chunk_data:
        chunks.append(_create_chunk_dict(
            current_chunk_data, current_chunk_branches, current_chunk_lines,
            len(chunks) + 1, result.total_files_analyzed,
            len(missed_branches), len(uncovered_lines),
            split_by="package"
        ))

    return _finalize_chunks(chunks, result, missed_branches, uncovered_lines, "package")


def _chunk_by_class(
    result: JaCoCoAnalysisResult,
    missed_branches: List[MissedBranch],
    uncovered_lines: List[UncoveredLine],
    max_tokens: int
) -> List[Dict]:
    """Chunk analysis results grouped by individual class/file."""
    import json

    # Group by class (using file_path as unique key)
    classes = {}
    for mb in missed_branches:
        cls_key = mb.file_path
        if cls_key not in classes:
            classes[cls_key] = {'missed_branches': [], 'uncovered_lines': []}
        classes[cls_key]['missed_branches'].append(mb)

    for ul in uncovered_lines:
        cls_key = ul.file_path
        if cls_key not in classes:
            classes[cls_key] = {'missed_branches': [], 'uncovered_lines': []}
        classes[cls_key]['uncovered_lines'].append(ul)

    # Create chunks
    chunks = []
    current_chunk_data = {}
    current_chunk_branches = 0
    current_chunk_lines = 0

    for cls_key in sorted(classes.keys()):
        cls_data = classes[cls_key]
        cls_dict = _format_group_data(cls_data)

        cls_json = json.dumps({cls_key: cls_dict}, indent=2)
        cls_tokens = _estimate_tokens(cls_json)

        current_json = json.dumps(current_chunk_data, indent=2) if current_chunk_data else "{}"
        current_tokens = _estimate_tokens(current_json)

        if current_tokens + cls_tokens > max_tokens and current_chunk_data:
            chunks.append(_create_chunk_dict(
                current_chunk_data, current_chunk_branches, current_chunk_lines,
                len(chunks) + 1, result.total_files_analyzed,
                len(missed_branches), len(uncovered_lines),
                split_by="class"
            ))
            current_chunk_data = {}
            current_chunk_branches = 0
            current_chunk_lines = 0

        current_chunk_data[cls_key] = cls_dict
        current_chunk_branches += len(cls_data['missed_branches'])
        current_chunk_lines += len(cls_data['uncovered_lines'])

    if current_chunk_data:
        chunks.append(_create_chunk_dict(
            current_chunk_data, current_chunk_branches, current_chunk_lines,
            len(chunks) + 1, result.total_files_analyzed,
            len(missed_branches), len(uncovered_lines),
            split_by="class"
        ))

    return _finalize_chunks(chunks, result, missed_branches, uncovered_lines, "class")


def _chunk_by_package_class(
    result: JaCoCoAnalysisResult,
    missed_branches: List[MissedBranch],
    uncovered_lines: List[UncoveredLine],
    max_tokens: int
) -> List[Dict]:
    """Chunk analysis results grouped by package, then by class within each package."""
    import json

    # Group by package -> class
    packages = {}
    for mb in missed_branches:
        pkg = _get_package_from_path(mb.file_path)
        cls = _get_class_from_path(mb.file_path)
        if pkg not in packages:
            packages[pkg] = {}
        if cls not in packages[pkg]:
            packages[pkg][cls] = {'missed_branches': [], 'uncovered_lines': []}
        packages[pkg][cls]['missed_branches'].append(mb)

    for ul in uncovered_lines:
        pkg = _get_package_from_path(ul.file_path)
        cls = _get_class_from_path(ul.file_path)
        if pkg not in packages:
            packages[pkg] = {}
        if cls not in packages[pkg]:
            packages[pkg][cls] = {'missed_branches': [], 'uncovered_lines': []}
        packages[pkg][cls]['uncovered_lines'].append(ul)

    # Create chunks
    chunks = []
    current_chunk_data = {}
    current_chunk_branches = 0
    current_chunk_lines = 0

    for pkg in sorted(packages.keys()):
        pkg_classes = packages[pkg]

        # Build the package dict with class-level grouping
        pkg_dict = {}
        pkg_branches = 0
        pkg_lines = 0
        for cls in sorted(pkg_classes.keys()):
            cls_data = pkg_classes[cls]
            pkg_dict[cls] = _format_group_data(cls_data)
            pkg_branches += len(cls_data['missed_branches'])
            pkg_lines += len(cls_data['uncovered_lines'])

        pkg_json = json.dumps({pkg: pkg_dict}, indent=2)
        pkg_tokens = _estimate_tokens(pkg_json)

        current_json = json.dumps(current_chunk_data, indent=2) if current_chunk_data else "{}"
        current_tokens = _estimate_tokens(current_json)

        if current_tokens + pkg_tokens > max_tokens and current_chunk_data:
            chunks.append(_create_chunk_dict(
                current_chunk_data, current_chunk_branches, current_chunk_lines,
                len(chunks) + 1, result.total_files_analyzed,
                len(missed_branches), len(uncovered_lines),
                split_by="package_class"
            ))
            current_chunk_data = {}
            current_chunk_branches = 0
            current_chunk_lines = 0

        current_chunk_data[pkg] = pkg_dict
        current_chunk_branches += pkg_branches
        current_chunk_lines += pkg_lines

    if current_chunk_data:
        chunks.append(_create_chunk_dict(
            current_chunk_data, current_chunk_branches, current_chunk_lines,
            len(chunks) + 1, result.total_files_analyzed,
            len(missed_branches), len(uncovered_lines),
            split_by="package_class"
        ))

    return _finalize_chunks(chunks, result, missed_branches, uncovered_lines, "package_class")


def _format_group_data(group_data: Dict) -> Dict:
    """Format a group's missed branches and uncovered lines into a serializable dict."""
    return {
        'missed_branches': [
            {
                'file': mb.file_path,
                'class': mb.class_name,
                'line': mb.line_number,
                'branch_info': mb.branch_info,
                'source': mb.source_line
            }
            for mb in group_data['missed_branches']
        ],
        'uncovered_lines': [
            {
                'file': ul.file_path,
                'class': ul.class_name,
                'line': ul.line_number,
                'source': ul.source_line
            }
            for ul in group_data['uncovered_lines']
        ]
    }


def _finalize_chunks(
    chunks: List[Dict],
    result: JaCoCoAnalysisResult,
    missed_branches: list,
    uncovered_lines: list,
    split_by: str
) -> List[Dict]:
    """Finalize chunks by setting total_chunks and handling empty results."""
    group_key = _split_by_to_key(split_by)

    if not chunks:
        chunks.append({
            'chunk_info': {
                'chunk_number': 1,
                'total_chunks': 1,
                'split_by': split_by,
                f'{group_key}_in_chunk': 0,
                'missed_branches_in_chunk': 0,
                'uncovered_lines_in_chunk': 0
            },
            'overall_summary': {
                'total_files_analyzed': result.total_files_analyzed,
                'total_missed_branches': len(missed_branches),
                'total_uncovered_lines': len(uncovered_lines)
            },
            f'by_{group_key}': {}
        })

    total_chunks = len(chunks)
    for chunk in chunks:
        chunk['chunk_info']['total_chunks'] = total_chunks

    return chunks


def _split_by_to_key(split_by: str) -> str:
    """Convert split_by parameter to the JSON key name."""
    if split_by == "class":
        return "class"
    elif split_by == "package_class":
        return "package_class"
    return "package"


def _create_chunk_dict(
    data: Dict,
    branches_count: int,
    lines_count: int,
    chunk_number: int,
    total_files: int,
    total_branches: int,
    total_lines: int,
    split_by: str = "package"
) -> Dict:
    """Create a chunk dictionary with metadata."""
    group_key = _split_by_to_key(split_by)
    return {
        'chunk_info': {
            'chunk_number': chunk_number,
            'total_chunks': 0,  # Will be updated after all chunks are created
            'split_by': split_by,
            f'{group_key}_in_chunk': len(data),
            'missed_branches_in_chunk': branches_count,
            'uncovered_lines_in_chunk': lines_count
        },
        'overall_summary': {
            'total_files_analyzed': total_files,
            'total_missed_branches': total_branches,
            'total_uncovered_lines': total_lines
        },
        f'by_{group_key}': data
    }


def get_available_packages(result: JaCoCoAnalysisResult) -> List[Dict]:
    """
    Get list of packages with their coverage issue counts.

    Args:
        result: JaCoCoAnalysisResult object

    Returns:
        List of package info dictionaries sorted by total issues
    """
    packages = {}

    for mb in result.missed_branches:
        pkg = _get_package_from_path(mb.file_path)
        if pkg not in packages:
            packages[pkg] = {'missed_branches': 0, 'uncovered_lines': 0}
        packages[pkg]['missed_branches'] += 1

    for ul in result.uncovered_lines:
        pkg = _get_package_from_path(ul.file_path)
        if pkg not in packages:
            packages[pkg] = {'missed_branches': 0, 'uncovered_lines': 0}
        packages[pkg]['uncovered_lines'] += 1

    # Convert to list and sort by total issues
    package_list = [
        {
            'package': pkg,
            'missed_branches': data['missed_branches'],
            'uncovered_lines': data['uncovered_lines'],
            'total_issues': data['missed_branches'] + data['uncovered_lines']
        }
        for pkg, data in packages.items()
    ]

    return sorted(package_list, key=lambda x: x['total_issues'], reverse=True)


def format_analysis_for_package(
    result: JaCoCoAnalysisResult,
    package_filter: str
) -> Dict:
    """
    Format analysis result for a specific package only.

    Args:
        result: JaCoCoAnalysisResult object
        package_filter: Package name to filter by

    Returns:
        Dictionary with analysis for the specified package
    """
    missed_branches = [
        mb for mb in result.missed_branches
        if _get_package_from_path(mb.file_path) == package_filter
    ]
    uncovered_lines = [
        ul for ul in result.uncovered_lines
        if _get_package_from_path(ul.file_path) == package_filter
    ]

    return {
        'package': package_filter,
        'summary': {
            'missed_branches': len(missed_branches),
            'uncovered_lines': len(uncovered_lines)
        },
        'missed_branches': [
            {
                'file': mb.file_path,
                'class': mb.class_name,
                'line': mb.line_number,
                'branch_info': mb.branch_info,
                'source': mb.source_line
            }
            for mb in missed_branches
        ],
        'uncovered_lines': [
            {
                'file': ul.file_path,
                'class': ul.class_name,
                'line': ul.line_number,
                'source': ul.source_line
            }
            for ul in uncovered_lines
        ]
    }

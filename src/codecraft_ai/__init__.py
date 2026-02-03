"""
Sonar JaCoCo Analyzer

A Python tool for analyzing SonarCloud/SonarQube code quality issues
and JaCoCo coverage reports.
"""

__version__ = "1.0.0"

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
    JaCoCoAnalysisResult,
    find_7zip_executables,
    set_7zip_path,
)

__all__ = [
    "SonarCloudAPI",
    "load_env_file",
    "select_project_interactive",
    "load_config",
    "save_config",
    "reset_config",
    "analyze_jacoco_report",
    "format_analysis_result",
    "JaCoCoAnalysisResult",
    "find_7zip_executables",
    "set_7zip_path",
]

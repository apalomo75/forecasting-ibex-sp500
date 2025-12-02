"""
utils_paths.py
-----------------------------------
Utility for centralized project path management.

Ensures consistent access to directories like /data/, /figures/, /models/, etc.
Used across all notebooks to avoid hard-coded paths.

Example usage:
>>> from src.utils_paths import get_project_paths
>>> paths = get_project_paths()
>>> paths["data"]
PosixPath('/home/apalo/projects/ibex-sp500-forecasting/data')
"""

from pathlib import Path


def get_project_paths():
    """
    Return key directories for the IBEX–SP500 project.

    Returns
    -------
    dict
        Dictionary with main project paths (root, data, figures, models, reports).
    """
    project_root = Path(__file__).resolve().parents[1]

    paths = {
        "root": project_root,
        "data": project_root / "data",
        "figures": project_root / "figures",
        "models": project_root / "models",
        "reports": project_root / "reports",
        "src": project_root / "src",
    }

    return paths

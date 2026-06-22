"""Caminhos centrais do projeto."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
SELECTED_DATA_DIR = DATA_DIR / "selected"
RESULTS_DIR = PACKAGE_DIR / "results"

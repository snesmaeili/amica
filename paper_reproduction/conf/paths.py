"""Shared path configuration for all benchmark scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
CONF_DIR = REPO_ROOT / "conf"
SLURM_DIR = REPO_ROOT / "slurm"

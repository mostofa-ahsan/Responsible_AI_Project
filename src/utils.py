"""Shared helpers: config loading, per-stage logging, and phase reports."""

import logging
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def load_config(path=REPO / "config.yaml"):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel):
    """Resolve a config path (relative to the repo root)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def get_logger(stage, cfg):
    """Log to logs/<stage>.log (DEBUG) and stderr (INFO)."""
    log_dir = repo_path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Report:
    """Collects report lines; prints them and saves to logs/<stage>_report.txt."""

    def __init__(self, stage, cfg):
        self.path = repo_path(cfg["paths"]["logs"]) / f"{stage}_report.txt"
        self.lines = []

    def __call__(self, line=""):
        self.lines.append(str(line))

    def save(self):
        text = "\n".join(self.lines) + "\n"
        print(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")


def slugify(text, max_len=48):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s or "doc"

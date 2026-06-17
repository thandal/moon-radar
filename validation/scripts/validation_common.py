"""Shared helpers for the moon-radar physics validation scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = REPO_ROOT / "validation"
RESULTS_DIR = VALIDATION_ROOT / "results"
LOGS_DIR = VALIDATION_ROOT / "logs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def write_json(name: str, payload: dict) -> Path:
    ensure_dirs()
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    return path


def finite_stats(values) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95_abs": float(np.quantile(np.abs(arr), 0.95)),
        "max_abs": float(np.max(np.abs(arr))),
        "std": float(arr.std()),
    }


def report_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))

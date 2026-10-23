"""Shared helpers for the moon-radar physics validation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = REPO_ROOT / "validation"
RESULTS_DIR = VALIDATION_ROOT / "results"
LOGS_DIR = VALIDATION_ROOT / "logs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import registration_analysis as ra  # noqa: E402  (light: no SPICE or CuPy)


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


# The lock label and its uncalibrated threshold live with the
# cross-correlation (registration_analysis.LOCK_SIGNIFICANCE, is_lock).
LOCK_SIGNIFICANCE = ra.LOCK_SIGNIFICANCE
is_lock = ra.is_lock


def production_looks(run_dirs, stations, chans, min_snr, min_tx_s=None):
    """The looks a production stack would use and their brightness.

    Looks are selected by stack_maps.select_looks (the production gates,
    with file paths resolved in each run directory) and converted by
    stack_maps.look_brightness (brightness per unit area, noise floor
    removed, normalized by the look's own gain; NaN where masked, including
    the degeneracy mask) -- the same steps as stack_maps.main. Every look
    left out is printed with its reason.

    Returns (rows, brightness, left_out): the brightness as float32 arrays
    in the order of rows; left_out, for a results file, has
    "looks_left_out" (each look the gates or the radiometry refused) and
    "captures_not_mapped" (each capture the batch did not map, from its
    registration_skipped CSV), each entry a dict of rx_file, station, chan
    and reason.
    """
    # Imported here: stack_maps needs CuPy, and the geometry-only scripts
    # that also use this module must not.
    import doppler_equator_alignment as dea
    import stack_maps as sm

    def entry(r, why):
        return {"rx_file": r["rx_file"], "station": r["_station"],
                "chan": r["_chan"], "reason": why}

    good, decisions = sm.select_looks(run_dirs, stations, chans, min_snr, min_tx_s)
    left_out = {"looks_left_out": [], "captures_not_mapped": []}
    for r, why in decisions:
        if why:
            print(f"  - {r['rx_file']}: {why}")
            key = "captures_not_mapped" if why.startswith("not mapped") else "looks_left_out"
            left_out[key].append(entry(r, why))
    npix = len(np.load(good[0]["map_npy"], mmap_mode="r")) if good else 0
    station_cache = {}
    rows, bright = [], []
    for r in good:
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix,
                                station_cache, rx_name=r["_rx"])
        try:
            I, _gain = sm.look_brightness(r, cosi)
        except sm.LookRejected as exc:
            print(f"  - {r['rx_file']}: {exc}")
            left_out["looks_left_out"].append(entry(r, str(exc)))
            continue
        rows.append(r)
        bright.append(I.astype(np.float32))
    print(f"[*] {len(rows)} looks used, {len(left_out['looks_left_out'])} left out by "
          f"the gates or the radiometry, {len(left_out['captures_not_mapped'])} captures "
          f"not mapped by the batch ({stations} x {chans})")
    return rows, bright, left_out

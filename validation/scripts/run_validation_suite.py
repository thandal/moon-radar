"""Run the validation suite with practical default settings.

Only one script in the suite tests its results against acceptance criteria
(validate_srp_velocity.py, which exits 1 when they fail); every other
script reports measurements only and exits 0 whenever it runs to the end.
So a step that "ran" has not "passed": each step is labelled accordingly in
the output. Any step that exits non-zero makes the suite exit non-zero.

Every station-dependent step runs Stockert's geometry or looks, named in
the command where the script takes a station and fixed in the script where
it does not; each step is named after its main results file under
validation/results (<name>.json), and its log is validation/logs/<name>.log.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

import validation_common as vc

CRITERIA = "has acceptance criteria"
REPORT_ONLY = "report only"

# Scripts whose results REPORT.md cites but which this suite does not run
# (they need the stacked or per-look map products; run them by hand, see
# validation/README.md).
NOT_RUN = [
    ("validate_speckle_floor.py --run-dir DIR --stations STATION --chan CHAN",
     "speckle floor (variance vs look count)"),
    ("validate_absolute_registration.py --map DIR/stacked_linear_<label>.npy",
     "absolute registration against LOLA"),
    ("validate_registration_bootstrap.py --run-dir DIR --stations STATION --mode closure",
     "cross-session closure bootstrap"),
    ("validate_mirror_ghosts.py --run-dir DIR --stations STATION --chan CHAN",
     "mirror-ghost leakage of a stack, by feature size"),
    ("validate_edge_offsets.py --run-dir DIR --data-root ROOT --station STATION --looks HH:MM:SS",
     "edge fit recovers injected delay and Doppler offsets"),
]


def run_step(name: str, cmd: list[str]) -> int:
    vc.ensure_dirs()
    log_path = vc.LOGS_DIR / f"{name}.log"
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {datetime.now(timezone.utc).isoformat()}\n")
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=vc.REPO_ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(proc.stdout)
    print(proc.stdout)
    print(f"log: {vc.report_path(log_path)}")
    return proc.returncode


def print_not_run() -> None:
    print("REPORT-cited checks this suite does not run (run them by hand):")
    for script, what in NOT_RUN:
        print(f"  {script}: {what}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the moon-radar validation suite. The default runs every "
                    "check at full settings (including the slow registration "
                    "bootstrap and the GPU/raw-data LOLA A/B check). --quick is a "
                    "fast smoke subset.")
    parser.add_argument("--python", default=str(vc.REPO_ROOT / ".conda/bin/python"))
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke run: smaller settings, geometry-only LOLA, "
                             "and no registration bootstrap")
    parser.add_argument("--run-dir", nargs="+",
                        help="run directories with Stockert's per-look maps and runs "
                             "CSVs (processed since 2026-10-21) for the registration "
                             "bootstrap; required without --quick")
    args = parser.parse_args()
    if not args.quick and not args.run_dir:
        parser.error("--run-dir is required for a full run (the registration bootstrap)")

    py = args.python
    quick = args.quick

    # (name, label, command). Physics-only checks run in both modes (smaller
    # settings under --quick).
    steps = [
        ("doppler_dem_physics_stockert", REPORT_ONLY, [
            py, "validation/scripts/validate_doppler_dem_physics.py",
            "--limit", "600" if quick else "3000",
        ]),
        ("srp_velocity_stockert", CRITERIA, [
            py, "validation/scripts/validate_srp_velocity.py",
            # Kernels-only (no SDR data / GPU): analytic SRP velocity vs the
            # lattice-free finite-difference reference + span cross-check.
            "--rx", "STOCKERT",
            *(["--quick"] if quick else []),
        ]),
        # At its defaults (production length, 30 s; tone strengths from the
        # stacker's threshold to 180) in both modes. Synthetic: no station.
        ("timing_frequency_separability", REPORT_ONLY, [
            py, "validation/scripts/validate_timing_frequency_separability.py",
        ]),
        ("rim_calibration_stress_stockert", REPORT_ONLY, [
            py, "validation/scripts/validate_rim_calibration_stress.py",
            "--n-real", "1" if quick else "3",
        ]),
        ("signal_processing", REPORT_ONLY, [
            py, "validation/scripts/validate_signal_processing.py",
        ]),
        ("lola_dem_projection_stockert", REPORT_ONLY, [
            py, "validation/scripts/validate_lola_dem_projection.py",
            "--nside", "200" if quick else "400",
            # Geometry (Part 1) always runs; the GPU/raw-data A/B check is full-run only.
            *(["--skip-ab"] if quick else []),
        ]),
    ]

    # Heavy steps with external inputs (saved map products / raw SDR + GPU) only
    # in a full run. Bootstrap both channels.
    if not quick:
        for chan in ("chan0", "chan1"):
            steps.append((f"registration_bootstrap_stockert_{chan}", REPORT_ONLY, [
                py, "validation/scripts/validate_registration_bootstrap.py",
                "--run-dir", *args.run_dir,
                "--stations", "stockert",
                "--channel", chan,
                "--n-boot", "20",
            ]))

    print_not_run()
    results = []
    for name, label, cmd in steps:
        results.append((name, label, run_step(name, cmd)))

    print("\n=== Summary ===")
    for name, label, code in results:
        status = f"FAILED (exit {code})" if code else (
            "passed its criteria" if label == CRITERIA else "ran; no criteria to pass")
        print(f"  {name:32s} [{label}] {status}")
    print_not_run()
    failed = [(name, code) for name, _, code in results if code]
    if failed:
        print("\nFAILED:")
        for name, code in failed:
            print(f"  {name}: exit {code}")
        raise SystemExit(1)
    n_criteria = sum(label == CRITERIA for _, label, _ in results)
    print(f"\nAll {len(results)} steps ran without error. {n_criteria} of them test "
          "their results against acceptance criteria and passed; the others "
          "report measurements only, so running is not passing -- read their results.")


if __name__ == "__main__":
    main()

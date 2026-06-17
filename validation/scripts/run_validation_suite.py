"""Run the validation suite with practical default settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

import validation_common as vc


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
    args = parser.parse_args()

    py = args.python
    quick = args.quick

    # Physics-only checks run in both modes (smaller settings under --quick).
    steps = [
        ("doppler_dem_physics", [
            py, "validation/scripts/validate_doppler_dem_physics.py",
            "--limit", "600" if quick else "3000",
        ]),
        ("timing_frequency_separability", [
            py, "validation/scripts/validate_timing_frequency_separability.py",
            "--duration-s", "3" if quick else "6",
        ]),
        ("rim_calibration_stress", [
            py, "validation/scripts/validate_rim_calibration_stress.py",
            "--n-real", "1" if quick else "3",
        ]),
        ("signal_processing", [
            py, "validation/scripts/validate_signal_processing.py",
        ]),
        ("lola_dem_projection", [
            py, "validation/scripts/validate_lola_dem_projection.py",
            "--nside", "200" if quick else "400",
            # Geometry (Part 1) always runs; the GPU/raw-data A/B check is full-run only.
            *(["--skip-ab"] if quick else []),
        ]),
    ]

    # Heavy steps with external inputs (saved map products / raw SDR + GPU) only
    # in a full run.
    if not quick:
        steps.append(("registration_bootstrap_chan1", [
            py, "validation/scripts/validate_registration_bootstrap.py",
            "--channel", "chan1",
            "--n-boot", "20",
        ]))

    failed = []
    for name, cmd in steps:
        code = run_step(name, cmd)
        if code != 0:
            failed.append((name, code))

    if failed:
        print("\nFAILED:")
        for name, code in failed:
            print(f"  {name}: exit {code}")
        raise SystemExit(1)
    print("\nAll validation steps completed.")


if __name__ == "__main__":
    main()

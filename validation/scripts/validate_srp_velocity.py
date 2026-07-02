"""Validate the analytic SRP velocity / Doppler axis (doppler_equator.srp_velocity_analytic).

The production rim arcs (compute_doppler_equator_velocity) and the libration
planning tool need the tangential SRP drift velocity — its *direction* sets
the Doppler axis and the equator/rim geometry, its *magnitude* sets the
limb-to-limb Doppler span. The legacy implementation finite-differenced the
specular zoom over dt=1 s, but the zoom output is quantized on a ~50 m
lattice while the true drift is ~1 m/s: the result relied on lattice-error
cancellation between the two calls and could glitch when the discrete argmin
flipped. This script proves out the closed-form replacement.

Three mutually independent estimates are compared at every epoch:

1. analytic  — srp_velocity_analytic: station states in MOON_ME only
               (bisector model; no solver in the loop).
2. reference — central finite difference of specular_point_refined (the
               zoom followed by a local paraboloid fit of the light-time
               bowl, which removes the lattice: ~1 m position noise), at two
               baselines (dt = 300 s and 150 s). The dt-consistency of the
               reference bounds its own truncation+noise error, so the
               analytic-vs-reference residual is a genuine model error bar.
               The reference minimizes the true two-leg light time, so it
               contains all the physics the analytic form linearizes away
               (finite distance, bistatic angle, light-time epochs).
3. legacy    — forward difference of specular_point_bck over dt=1 s,
               exactly as production used to compute it (characterizes what
               the historical rim arcs actually saw, and hunts for lattice
               glitches over a dense sweep).

Part 2 cross-checks the span formula against the field it predicts: the
full-disk dlt field is evaluated near-side (wide-stencil moonPointDLT_BCK)
and its measured min/max span is compared with the analytic
2*R*|g|*f0/c — this is the quantity the libration planner gates on.

Pass criteria (requirement-derived; printed at the end):
  the rim arcs tolerate ~0.5 deg of axis-direction error (a direction error
  eps biases both rims as cos eps and cancels into the spread; the measured
  spread bounds eps <~ 3-4 deg), and the libration planner needs ~1 deg.
  Tolerances are set 2-10x under those requirements:
    analytic vs reference: axis <= 0.25 deg max AND <= 0.02 deg median,
                           speed <= 0.5% max
    analytic span vs measured field span: <= 2%
  Residuals at the 0.05-deg scale are at the reference's own resolving power
  (dt-halving self-consistency) and cannot be attributed to either method.

Run from the repo root:
  .conda/bin/python validation/scripts/validate_srp_velocity.py [--quick]
Outputs: validation/results/srp_velocity.json, log to stdout.
"""

from __future__ import annotations

import argparse

import numpy as np

from validation_common import REPO_ROOT, finite_stats, write_json, report_path

import sys  # noqa: E402  (path set by validation_common)
import healpy as hp
import cspyce as csp

from spice_setup import furnsh_kernels
import doppler_equator as de

F0_HZ = 1299.5e6

# One mid-session anchor epoch per observing day (all appear in the repo's
# committed validation logs / REPORT).
SESSION_EPOCHS = [
    "2025-06-21T08:59:29",
    "2025-09-10T20:35:24",
    "2025-09-11T08:05:44",
    "2025-09-16T13:23:26",
]


def axis_angle_deg(a, b):
    """Angle between two unsigned axes (great circles), in degrees."""
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def legacy_velocity(et, tx_name, rx_name):
    """The historical production method, byte-for-byte: forward difference
    of the ~50 m-quantized zoom over dt=1 s, tangent-projected."""
    srp = de.specular_point_bck(et, tx_name, rx_name)
    srp_hat = srp / np.linalg.norm(srp)
    v = de.specular_point_bck(et + 1.0, tx_name, rx_name) - srp
    v_t = v - np.dot(v, srp_hat) * srp_hat
    axis = np.cross(srp_hat, v_t / np.linalg.norm(v_t))
    return v_t, axis / np.linalg.norm(axis), srp_hat


def compare_epoch(et, tx_name, rx_name):
    """Full four-way comparison at one epoch.

    The reference is finite-difference, so its axis noise scales as
    sigma_pos / (2 dt |v|): at slow-drift epochs (|v| well under 1 m/s) a
    fixed 300 s baseline would dominate the comparison. Scale the baseline
    with 1/|v| (300 s at >=1 m/s, capped at 1200 s); central-difference
    truncation stays negligible (dt^2/6 * |p'''| ~ 5e-5 m/s at 1200 s)."""
    v_an, ax_an, _ = de.srp_velocity_analytic(et, tx_name, rx_name)
    dt_ref = float(np.clip(300.0 / max(np.linalg.norm(v_an) * 1e3, 1e-3),
                           300.0, 1200.0))
    v_r3, ax_r3, _ = de.srp_velocity_fd(et, tx_name, rx_name, dt=dt_ref, refined=True)
    v_r1, ax_r1, _ = de.srp_velocity_fd(et, tx_name, rx_name, dt=dt_ref / 2,
                                        refined=True)
    v_lg, ax_lg, _ = legacy_velocity(et, tx_name, rx_name)
    s_an, s_r3, s_r1, s_lg = (np.linalg.norm(v) for v in (v_an, v_r3, v_r1, v_lg))
    return {
        "et": float(et),
        "utc": csp.et2utc(et, "ISOC", 0),
        "dt_ref_s": dt_ref,
        "speed_analytic_m_s": s_an * 1e3,
        "speed_ref300_m_s": s_r3 * 1e3,
        "speed_ref150_m_s": s_r1 * 1e3,
        "speed_legacy_m_s": s_lg * 1e3,
        "axis_ref_consistency_deg": axis_angle_deg(ax_r3, ax_r1),
        "speed_ref_consistency": abs(s_r1 / s_r3 - 1.0),
        "axis_analytic_vs_ref_deg": axis_angle_deg(ax_an, ax_r3),
        "speed_analytic_vs_ref": abs(s_an / s_r3 - 1.0),
        "axis_legacy_vs_ref_deg": axis_angle_deg(ax_lg, ax_r3),
        "speed_legacy_vs_ref": abs(s_lg / s_r3 - 1.0),
    }


def measured_field_span_hz(et, tx_name, rx_name, nside=64, dt=60.0):
    """Limb-to-limb Doppler span measured from the actual near-side dlt
    field (wide-stencil derivative of the exact two-leg light time)."""
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    v = v[v[:, 0] > 0]  # near side (MOON_ME +X toward Earth)
    _, dlt = de.moonPointDLT_BCK(et, v, tx_name, rx_name, dt=dt)
    return float((dlt.max() - dlt.min()) * F0_HZ)


def analytic_span_hz(et, tx_name, rx_name):
    t_hat, t_rate = de._station_dir_rate(tx_name, et)
    r_hat, r_rate = de._station_dir_rate(rx_name, et)
    g = t_rate + r_rate
    return float(2.0 * de.moon_radii()[0] * np.linalg.norm(g) * F0_HZ / csp.clight())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tx", default="DWINGELOO")
    ap.add_argument("--rx", default="STOCKERT")
    ap.add_argument("--quick", action="store_true",
                    help="fewer sweep epochs (smoke run)")
    ap.add_argument("--dense-step-min", type=float, default=15.0,
                    help="step of the legacy glitch-hunt sweep (minutes)")
    args = ap.parse_args()

    furnsh_kernels()

    # --- Part 1: four-way comparison ------------------------------------
    epochs = [csp.str2et(u) for u in SESSION_EPOCHS]
    # Year-long sweep for generality (SRP geometry exists at all epochs;
    # no visibility gate needed for the velocity itself).
    t0 = csp.str2et("2025-06-01T00:00:00")
    n_sweep = 6 if args.quick else 37
    epochs += list(t0 + np.linspace(0, 365.0, n_sweep) * 86400.0)

    rows = [compare_epoch(et, args.tx, args.rx) for et in epochs]
    for r in rows:
        print(f"{r['utc']}: |v| analytic {r['speed_analytic_m_s']:7.4f} m/s  "
              f"ref {r['speed_ref300_m_s']:7.4f}  legacy {r['speed_legacy_m_s']:7.4f}  "
              f"axis an-ref {r['axis_analytic_vs_ref_deg']:7.4f} deg  "
              f"legacy-ref {r['axis_legacy_vs_ref_deg']:7.4f} deg")

    def col(k):
        return [r[k] for r in rows]

    ref_ax = finite_stats(col("axis_ref_consistency_deg"))
    ref_sp = finite_stats(col("speed_ref_consistency"))
    an_ax = finite_stats(col("axis_analytic_vs_ref_deg"))
    an_sp = finite_stats(col("speed_analytic_vs_ref"))
    lg_ax = finite_stats(col("axis_legacy_vs_ref_deg"))
    lg_sp = finite_stats(col("speed_legacy_vs_ref"))

    # --- Part 2: span formula vs measured dlt field ---------------------
    span_rows = []
    for utc in SESSION_EPOCHS:
        et = csp.str2et(utc)
        meas = measured_field_span_hz(et, args.tx, args.rx,
                                      nside=32 if args.quick else 64)
        pred = analytic_span_hz(et, args.tx, args.rx)
        span_rows.append({"utc": utc, "measured_hz": meas, "predicted_hz": pred,
                          "rel_err": abs(pred / meas - 1.0)})
        print(f"span {utc}: measured {meas:6.2f} Hz  analytic {pred:6.2f} Hz  "
              f"({100 * span_rows[-1]['rel_err']:.2f}%)")

    # --- Part 3: legacy glitch hunt (dense sweep, cheap pair) ------------
    glitches = []
    step = args.dense_step_min * 60.0
    n_dense = 24 if args.quick else 12 * 3600 // int(step) + 1
    dense_max_deg = 0.0
    for utc in SESSION_EPOCHS:
        base = csp.str2et(utc)
        for k in range(n_dense):
            et = base + (k - n_dense // 2) * step
            _, ax_an, _ = de.srp_velocity_analytic(et, args.tx, args.rx)
            _, ax_lg, _ = legacy_velocity(et, args.tx, args.rx)
            d = axis_angle_deg(ax_an, ax_lg)
            dense_max_deg = max(dense_max_deg, d)
            if d > 5.0:
                glitches.append({"utc": csp.et2utc(et, "ISOC", 0),
                                 "axis_err_deg": d})
    print(f"legacy glitch hunt: {len(glitches)} epochs with axis error > 5 deg "
          f"(max {dense_max_deg:.3f} deg over {4 * n_dense} epochs)")
    for gl in glitches:
        print(f"  LEGACY LATTICE GLITCH at {gl['utc']}: "
              f"{gl['axis_err_deg']:.1f} deg axis error")

    # --- Verdict ----------------------------------------------------------
    # Requirement-derived tolerances (see module docstring): worst case well
    # under what the rim arcs (~0.5 deg) and the planner (~1 deg) tolerate,
    # median demonstrating typical performance an order tighter still.
    ok = (an_ax["max_abs"] <= 0.25 and an_ax["median"] <= 0.02 and
          an_sp["max_abs"] <= 0.005 and
          max(s["rel_err"] for s in span_rows) <= 0.02)
    print(f"reference self-consistency: axis max {ref_ax['max_abs']:.4f} deg, "
          f"speed max {100 * ref_sp['max_abs']:.3f}%")
    print(f"analytic vs reference:      axis max {an_ax['max_abs']:.4f} deg, "
          f"speed max {100 * an_sp['max_abs']:.3f}%")
    print(f"legacy vs reference:        axis max {lg_ax['max_abs']:.4f} deg, "
          f"speed max {100 * lg_sp['max_abs']:.3f}% "
          f"(median axis {lg_ax['median']:.4f} deg)")
    print("VERDICT:", "PASS — srp_velocity_analytic is validated"
          if ok else "FAIL — do not switch production to the analytic method")

    payload = {
        "tx": args.tx, "rx": args.rx,
        "epochs": rows, "span": span_rows, "glitches": glitches,
        "stats": {"ref_axis_deg": ref_ax, "ref_speed": ref_sp,
                  "analytic_axis_deg": an_ax, "analytic_speed": an_sp,
                  "legacy_axis_deg": lg_ax, "legacy_speed": lg_sp},
        "pass": bool(ok),
    }
    out = write_json("srp_velocity.json", payload)
    print(f"wrote {report_path(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

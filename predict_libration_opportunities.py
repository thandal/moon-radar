#!/usr/bin/env python3
"""
Predict optimal times in the Moon's libration cycle to collect new datasets
using Dwingeloo (TX) and Stockert (RX).

The tool finds future epochs when the Doppler equator (the locus of mapping
degeneracy) is maximally displaced from the Doppler equators of the existing
datasets, so new observations cover the surface zones that were previously
degenerate (masked) in the stack.

Method notes (see LIBRATION_ANALYSIS.md):

* The Doppler axis and SRP drift speed come from the closed-form
  `doppler_equator.srp_velocity_analytic` (station states in MOON_ME), not a
  finite difference of the specular-zoom solver — the zoom output is
  quantized on a ~50 m lattice while the true drift is ~1 m/s, so the old
  1 s difference relied on lattice-error cancellation and could glitch.
  Validated in validation/scripts/validate_srp_velocity.py.

* The Doppler axis sweeps tens of degrees WITHIN a session (the diurnal
  parallax term is the same order as the libration term), so existing
  coverage is represented by the axis of EVERY recorded look (from the runs
  CSVs), not one midpoint per session. The displacement metric is the
  minimum angle to any existing look axis. Without the CSVs it falls back
  to one hardcoded mid-session epoch per session and says so.

* Beyond the high-precision Earth PCK the tool switches to the analytic
  IAU_EARTH frame but KEEPS the topocentric station offsets: dropping them
  (geocentric model) would remove the diurnal parallax contribution to the
  apparent rotation — which is comparable to the whole signal. The fallback
  error is an Earth-rotation-phase error (arcminute-scale over a year or
  two), not a missing term.

Usage:
    .conda/bin/python predict_libration_opportunities.py [options]
"""

import os
import csv
import glob
import argparse
import numpy as np
import cspyce as csp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from spice_setup import furnsh_kernels
import doppler_equator as de

F0_HZ = 1299.5e6
RUNS_GLOB = "registration_runs_chan*.csv"

# Fallback session SPANS used when no runs CSVs are available, sampled at
# FALLBACK_SPAN_STEP_S. A single mid-session axis is unusable as a coverage
# summary: the axis sweeps tens of degrees within a session (measured ~87
# deg over the 09-10/11 night; ~70 deg within the 09-11 morning alone, where
# the drift speed is smallest and the axis direction least conditioned), so
# a midpoint metric can misstate a candidate's displacement by ~45 deg.
# Spans are approximate reconstructions from REPORT §5/§10 (43 min, evening
# + overnight-morning, 3.5 h) around known look epochs; per-look CSVs are
# strictly better — use --run-dir on the data machine.
DEFAULT_SESSION_SPANS = {
    "2025-06-21": ("2025-06-21T08:38:00", "2025-06-21T09:21:00"),
    "2025-09-10": ("2025-09-10T18:00:00", "2025-09-10T21:30:00"),
    "2025-09-11": ("2025-09-11T05:30:00", "2025-09-11T08:30:00"),
    "2025-09-16": ("2025-09-16T11:40:00", "2025-09-16T15:10:00"),
}
FALLBACK_SPAN_STEP_S = 900.0


# ---------------------------------------------------------------------------
# Doppler axis / span models
# ---------------------------------------------------------------------------
def axis_span_topocentric(et, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Doppler axis (unit, MOON_ME), SRP drift speed (m/s), and limb-to-limb
    L-band Doppler span (Hz), from the closed-form SRP velocity."""
    v, axis, _ = de.srp_velocity_analytic(et, tx_name, rx_name)
    speed_m_s = np.linalg.norm(v) * 1e3
    # span = 2 R |g| f0/c; with |t_hat+r_hat| ~ 2 that is 4 |v| f0/c.
    span_hz = 4.0 * np.linalg.norm(v) * F0_HZ / csp.clight()
    return axis, speed_m_s, span_hz


def _station_itrf_xyz(lat_deg, lon_deg, alt_km):
    """Earth-fixed rectangular station position (km) from geodetic lat/lon."""
    re, _, rp = csp.bodvrd("EARTH", "RADII")
    return np.asarray(csp.georec(np.radians(lon_deg), np.radians(lat_deg),
                                 alt_km, re, (re - rp) / re))

# Geodetic coordinates from observatories.defs (tracked source of truth).
STATION_GEODETIC = {
    "DWINGELOO": (52.81214958283062, 6.396319071523311, 0.07026),
    "STOCKERT": (50.56944039751571, 6.721943350231514, 0.434),
    "ATA": (40.8178, -121.4733, 1.008),
}


def _fallback_station_dir(et, station):
    """Moon-center -> station unit direction in MOON_ME using the analytic
    IAU_EARTH frame (valid beyond the high-precision PCK), keeping the
    topocentric offset so the diurnal parallax term survives."""
    p_earth, _ = csp.spkpos("EARTH", et, "MOON_ME", de.AB_COR, "MOON")
    m = csp.pxform("IAU_EARTH", "MOON_ME", et)
    p = np.asarray(p_earth) + m @ _station_itrf_xyz(*STATION_GEODETIC[station])
    return p / np.linalg.norm(p)


def axis_span_fallback(et, tx_name="DWINGELOO", rx_name="STOCKERT", ddt=30.0):
    """Same quantities as axis_span_topocentric, from the IAU_EARTH station
    model. The direction functions are smooth (no solver), so a central
    difference over +-ddt is clean."""
    def dir_rate(station):
        u0 = _fallback_station_dir(et - ddt, station)
        u1 = _fallback_station_dir(et + ddt, station)
        u = _fallback_station_dir(et, station)
        return u, (u1 - u0) / (2.0 * ddt)

    t_hat, t_rate = dir_rate(tx_name)
    r_hat, r_rate = dir_rate(rx_name)
    b = t_hat + r_hat
    e_hat = b / np.linalg.norm(b)
    g = t_rate + r_rate
    v = de.moon_radii()[0] * (g - np.dot(g, e_hat) * e_hat) / np.linalg.norm(b)
    axis = np.cross(e_hat, g)
    axis /= np.linalg.norm(axis)
    speed_m_s = np.linalg.norm(v) * 1e3
    span_hz = 2.0 * de.moon_radii()[0] * np.linalg.norm(g) * F0_HZ / csp.clight()
    return axis, speed_m_s, span_hz


# ---------------------------------------------------------------------------
# Elevations
# ---------------------------------------------------------------------------
def get_moon_elevations(et, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Moon elevations (deg) from the PINPOINT topocentric frames (z = up)."""
    elevs = []
    for name in (tx_name, rx_name):
        pos, _ = csp.spkpos("MOON", et, f"{name}_TOPO", "NONE", name)
        elevs.append(np.degrees(np.arcsin(pos[2] / np.linalg.norm(pos))))
    return tuple(elevs)


def get_fallback_elevations(et, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Approximate elevations from the IAU_EARTH analytic frame (station
    offsets included). Adequate for horizon gating (sub-degree)."""
    p_moon, _ = csp.spkpos("MOON", et, "IAU_EARTH", "NONE", "EARTH")
    p_moon = np.asarray(p_moon)
    elevs = []
    for name in (tx_name, rx_name):
        lat, lon, _ = STATION_GEODETIC[name]
        lat, lon = np.radians(lat), np.radians(lon)
        n = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                      np.sin(lat)])
        v = p_moon - _station_itrf_xyz(*STATION_GEODETIC[name])
        elevs.append(np.degrees(np.arcsin(np.dot(v / np.linalg.norm(v), n))))
    return tuple(elevs)


# ---------------------------------------------------------------------------
# Existing coverage
# ---------------------------------------------------------------------------
def get_existing_look_axes(run_dir, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Doppler axis of every recorded look, from the runs CSVs
    (registration_runs_chan*.csv). Falls back to one mid-session epoch per
    session when the CSVs are absent. Returns (axes (N,3), day labels (N,),
    used_fallback)."""
    utcs = set()
    for path in sorted(glob.glob(os.path.join(run_dir, RUNS_GLOB))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("rx_start_utc"):
                    utcs.add(row["rx_start_utc"])

    fallback = not utcs
    if fallback:
        print(f"[*] No {RUNS_GLOB} under {run_dir}; falling back to "
              "reconstructed session SPANS sampled every "
              f"{FALLBACK_SPAN_STEP_S / 60:.0f} min — approximate; per-look "
              "CSVs are strictly better (--run-dir on the data machine).")
        for day, (u0, u1) in DEFAULT_SESSION_SPANS.items():
            et0, et1 = csp.str2et(u0), csp.str2et(u1)
            n = max(2, int((et1 - et0) / FALLBACK_SPAN_STEP_S) + 1)
            for et in np.linspace(et0, et1, n):
                utcs.add(csp.et2utc(float(et), "ISOC", 0))

    axes, days = [], []
    for utc in sorted(utcs):
        axis, _, _ = axis_span_topocentric(csp.str2et(utc), tx_name, rx_name)
        axes.append(axis)
        days.append(utc.split("T")[0])
    print(f"[*] Existing coverage: {len(axes)} look axes over "
          f"{len(set(days))} sessions"
          + (" (fallback midpoints)" if fallback else ""))
    for day in sorted(set(days)):
        sel = [a for a, d in zip(axes, days) if d == day]
        swing = max(axis_angle_deg(sel[0], a) for a in sel) if len(sel) > 1 else 0.0
        print(f"  - {day}: {len(sel)} looks, intra-session axis swing "
              f"{swing:.1f} deg")
    return np.array(axes), days, fallback


def axis_angle_deg(a, b):
    """Angle between two unsigned axes (great circles are unoriented)."""
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def solve_equator_latitude(axis, lon_deg):
    """Latitude (deg) of the Doppler-equator great circle (axis . p = 0)
    at a selenographic longitude."""
    lon = np.radians(lon_deg)
    if abs(axis[2]) < 1e-9:
        return 90.0 if (axis[0] * np.cos(lon) + axis[1] * np.sin(lon)) < 0 else -90.0
    return np.degrees(np.arctan(
        -(axis[0] * np.cos(lon) + axis[1] * np.sin(lon)) / axis[2]))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_opportunities(existing_axes, existing_days, best_opp, output_png):
    """Doppler equators on the lunar surface: per-session envelopes of the
    existing looks (the axis sweeps within a session), plus the target."""
    pl.close("all")
    fig, ax = pl.subplots(figsize=(10, 6), dpi=150)
    lons = np.linspace(-180, 180, 361)

    colors = ["#90A4AE", "#78909C", "#546E7A", "#37474F", "#263238"]
    for idx, day in enumerate(sorted(set(existing_days))):
        sel = np.array([a for a, d in zip(existing_axes, existing_days)
                        if d == day])
        lats = np.array([[solve_equator_latitude(a, lo) for lo in lons]
                         for a in sel])
        c = colors[idx % len(colors)]
        if len(sel) > 1:
            ax.fill_between(lons, lats.min(0), lats.max(0), color=c, alpha=0.25)
        ax.plot(lons, np.median(lats, 0), "--", color=c, alpha=0.9, lw=1.2,
                label=f"Existing: {day} ({len(sel)} looks)")

    new_lats = [solve_equator_latitude(best_opp["axis"], lo) for lo in lons]
    ax.plot(lons, new_lats, "-", color="#E65100", lw=2.5,
            label=f"Optimal target: {best_opp['utc'].split('T')[0]}")

    ax.set_title("Doppler-equator coverage: existing sessions vs optimal target")
    ax.set_xlabel("Selenographic longitude (deg)")
    ax.set_ylabel("Selenographic latitude (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    pl.savefig(output_png, bbox_inches="tight")
    print(f"[*] Saved opportunities plot to: {output_png}")
    pl.close(fig)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Predict optimal times for lunar radar libration coverage.")
    p.add_argument("--start-date", default="2026-06-18")
    p.add_argument("--end-date", default="2026-09-10")
    p.add_argument("--min-tx-elevation", type=float, default=45.0,
                   help="Min Moon elevation at Dwingeloo (TX), deg.")
    p.add_argument("--min-rx-elevation", type=float, default=10.0,
                   help="Min Moon elevation at Stockert (RX), deg.")
    p.add_argument("--min-doppler-span", type=float, default=5.0,
                   help="Min limb-to-limb L-band Doppler span, Hz.")
    p.add_argument("--step-hours", type=float, default=1.0)
    p.add_argument("--run-dir", default="results/LOLA_DEM_REGISTRATION",
                   help="Directory with registration_runs_chan*.csv "
                        "(per-look coverage axes).")
    p.add_argument("--output-csv", default="results/ERRORS/libration_opportunities.csv")
    p.add_argument("--output-plot", default="results/ERRORS/libration_opportunities.png")
    args = p.parse_args()

    furnsh_kernels()

    # High-precision Earth PCK coverage end -> fallback boundary.
    try:
        bpc = os.path.join("spice_kernels", "earth_latest_high_prec.bpc")
        ids = csp.pckfrm(bpc)
        cover = csp.pckcov(bpc, list(ids)[0])
        _, pck_end_et = csp.wnfetd(cover, csp.wncard(cover) - 1)
        pck_end_utc = csp.et2utc(pck_end_et, "ISOC", 0)
    except Exception:
        pck_end_utc = "2026-09-11T00:00:00"
        pck_end_et = csp.str2et(pck_end_utc)
    print(f"[*] High-precision Earth PCK coverage until: {pck_end_utc} UTC "
          "(beyond: IAU_EARTH fallback, stations kept topocentric)")

    existing_axes, existing_days, used_fallback = \
        get_existing_look_axes(args.run_dir)

    start_et = csp.str2et(args.start_date + "T00:00:00")
    end_et = csp.str2et(args.end_date + "T00:00:00")
    step_s = args.step_hours * 3600.0
    n_steps = int((end_et - start_et) / step_s)
    print(f"[*] Scanning {n_steps} steps from {args.start_date} to {args.end_date}...")

    opportunities = []
    for i in range(n_steps):
        et = start_et + i * step_s
        use_fallback = et > pck_end_et
        elev_tx, elev_rx = (get_fallback_elevations(et) if use_fallback
                            else get_moon_elevations(et))
        if elev_tx < args.min_tx_elevation or elev_rx < args.min_rx_elevation:
            continue
        axis, speed, span_hz = (axis_span_fallback(et) if use_fallback
                                else axis_span_topocentric(et))
        if span_hz < args.min_doppler_span:
            continue
        min_dist = min(axis_angle_deg(axis, a) for a in existing_axes)
        opportunities.append({
            "et": et, "utc": csp.et2utc(et, "ISOC", 0),
            "elev_tx": elev_tx, "elev_rx": elev_rx, "axis": axis,
            "speed": speed, "doppler_span": span_hz,
            "min_dist": min_dist, "fallback": use_fallback,
        })
    print(f"[*] Found {len(opportunities)} visible steps passing the gates.")
    if not opportunities:
        print("[!] No visible opportunities in the requested window.")
        return

    # Group contiguous steps into passes (>4 h gap starts a new pass).
    passes, cur = [], []
    for opp in opportunities:
        if cur and (opp["et"] - cur[-1]["et"]) / 3600.0 > 4.0:
            passes.append(cur)
            cur = []
        cur.append(opp)
    passes.append(cur)
    print(f"[*] Grouped into {len(passes)} distinct visible passes.")

    # Per pass: peak by displacement, plus the pass-wide displacement range
    # and axis swing (a real session sweeps the axis — the peak-hour number
    # alone overstates precision).
    best = []
    for pa in passes:
        peak = max(pa, key=lambda x: x["min_dist"])
        peak = dict(peak)
        peak["pass_start"] = pa[0]["utc"]
        peak["pass_end"] = pa[-1]["utc"]
        peak["pass_min_dist"] = min(x["min_dist"] for x in pa)
        peak["pass_axis_swing_deg"] = axis_angle_deg(pa[0]["axis"], pa[-1]["axis"])
        best.append(peak)
    best.sort(key=lambda x: x["min_dist"], reverse=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc", "min_displacement_deg", "pass_start_utc",
                    "pass_end_utc", "pass_min_displacement_deg",
                    "pass_axis_swing_deg", "dwingeloo_elevation_deg",
                    "stockert_elevation_deg", "specular_speed_m_s",
                    "doppler_span_hz", "axis_x", "axis_y", "axis_z",
                    "iau_earth_fallback", "coverage_from_fallback_midpoints"])
        for o in best:
            w.writerow([o["utc"], f"{o['min_dist']:.4f}", o["pass_start"],
                        o["pass_end"], f"{o['pass_min_dist']:.4f}",
                        f"{o['pass_axis_swing_deg']:.4f}",
                        f"{o['elev_tx']:.2f}", f"{o['elev_rx']:.2f}",
                        f"{o['speed']:.4f}", f"{o['doppler_span']:.2f}",
                        f"{o['axis'][0]:.6f}", f"{o['axis'][1]:.6f}",
                        f"{o['axis'][2]:.6f}", o["fallback"], used_fallback])
    print(f"[*] Saved pass peaks to CSV: {args.output_csv}")

    plot_opportunities(existing_axes, existing_days, best[0], args.output_plot)

    print("\n" + "=" * 100)
    print(f"{'TOP LIBRATION OPPORTUNITIES  Dwingeloo (TX) -> Stockert (RX)':^100}")
    print("=" * 100)
    print(f"{'Peak (UTC)':<21}| {'Disp':>7} | {'Pass disp':>9} | {'Swing':>6} | "
          f"{'TX el':>6} | {'RX el':>6} | {'Span':>8} | Model")
    print("-" * 100)
    for o in best[:10]:
        model = "IAU_EARTH" if o["fallback"] else "Topocentric"
        print(f"{o['utc']:<21}| {o['min_dist']:>6.2f}° | "
              f"{o['pass_min_dist']:>6.2f}°+ | {o['pass_axis_swing_deg']:>5.1f}° | "
              f"{o['elev_tx']:>5.1f}° | {o['elev_rx']:>5.1f}° | "
              f"{o['doppler_span']:>6.1f} Hz | {model}")
    print("=" * 100)
    if used_fallback:
        print("NOTE: coverage axes came from reconstructed session spans "
              "(approximate) — re-run with the runs CSVs (--run-dir) for "
              "per-look coverage before committing telescope time.")


if __name__ == "__main__":
    main()

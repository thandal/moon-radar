#!/usr/bin/env python3
"""
Predict optimal times in the Moon's libration cycle to collect new datasets
with Dwingeloo transmitting and a chosen receiver (--station).

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
  coverage is represented by the axis of EVERY look that enters the stack:
  the runs CSVs of every station in --run-dir, channels --chans (chan1
  unless given), gated by stack_maps.select_looks with --min-snr (default
  freq_offset_hunt.MIN_TONE_SNR, the stack's) and --min-tx-s. The
  stacker's later per-look brightness rejection (stack_maps.
  look_brightness), which needs the maps, is not applied, so a look it
  would reject still counts as coverage. Each look's axis uses its own
  receiver. The displacement metric is the minimum angle to any existing
  look axis. Without runs CSVs, --coverage-from-session-spans samples
  reconstructed spans of the 2025 Stockert sessions instead; it is
  approximate, opt-in, and recorded in the outputs.

* Beyond the high-precision Earth PCK the tool switches to the analytic
  IAU_EARTH frame but KEEPS the topocentric station offsets: dropping them
  (geocentric model) would remove the diurnal parallax contribution to the
  apparent rotation — which is comparable to the whole signal. The
  IAU_EARTH model's error is an Earth-rotation-phase error (arcminute-scale
  over a year or two), not a missing term. Every row records which model
  it used.

Usage:
    .conda/bin/python predict_libration_opportunities.py --station stockert \\
        --start-date 2026-11-01 --end-date 2027-11-01 \\
        --run-dir results/STOCKERT_2025 results/EVE2026_CIRC [options]
"""

import os
import re
import csv
import glob
import argparse
import numpy as np
import cspyce as csp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from spice_setup import furnsh_kernels, SPICE_KERNEL_DIR
import doppler_equator as de
import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import stack_maps as sm

F0_HZ = 1299.5e6
RUNS_CSV_RE = re.compile(r"registration_runs_(.+)_(chan\d+)\.csv$")

# Reconstructed session SPANS for --coverage-from-session-spans (no runs
# CSVs), sampled every SESSION_SPAN_STEP_S. A single mid-session axis is
# unusable as a coverage summary: the axis sweeps tens of degrees within a
# session (measured ~87 deg over the 09-10/11 night; ~70 deg within the
# 09-11 morning alone, where the drift speed is smallest and the axis
# direction least conditioned), so a midpoint metric can misstate a
# candidate's displacement by ~45 deg. Spans are approximate reconstructions
# from REPORT §5/§10 (43 min, evening + overnight-morning, 3.5 h) around
# known look epochs, all recorded at Stockert; per-look runs CSVs (--run-dir)
# are strictly better.
SESSION_SPANS = {
    "2025-06-21": ("stockert", "2025-06-21T08:38:00", "2025-06-21T09:21:00"),
    "2025-09-10": ("stockert", "2025-09-10T18:00:00", "2025-09-10T21:30:00"),
    "2025-09-11": ("stockert", "2025-09-11T05:30:00", "2025-09-11T08:30:00"),
    "2025-09-16": ("stockert", "2025-09-16T11:40:00", "2025-09-16T15:10:00"),
}
SESSION_SPAN_STEP_S = 900.0


# ---------------------------------------------------------------------------
# Doppler axis / span models
# ---------------------------------------------------------------------------
def axis_span_topocentric(et, tx_name, rx_name):
    """Doppler axis (unit, MOON_ME), SRP drift speed (m/s), and limb-to-limb
    L-band Doppler span (Hz), from the closed-form SRP velocity."""
    v, axis, _ = de.srp_velocity_analytic(et, tx_name, rx_name)
    speed_m_s = np.linalg.norm(v) * 1e3
    # span = 2 R |g| f0/c; with |t_hat+r_hat| ~ 2 that is 4 |v| f0/c.
    span_hz = 4.0 * np.linalg.norm(v) * F0_HZ / csp.clight()
    return axis, speed_m_s, span_hz


def _station_itrf_xyz(lat_deg, lon_deg, alt_km):
    """Earth-fixed rectangular station position (km) from geodetic lat/lon."""
    re_km, _, rp_km = csp.bodvrd("EARTH", "RADII")
    return np.asarray(csp.georec(np.radians(lon_deg), np.radians(lat_deg),
                                 alt_km, re_km, (re_km - rp_km) / re_km))


def read_station_geodetic(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "observatories.defs")):
    """Geodetic (lat deg, lon deg, alt km) of every site in observatories.defs,
    the tracked source of the station kernels (<NAME>_LATLON = ( lat, lon,
    alt ))."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for name, values in re.findall(r"(\w+)_LATLON\s*=\s*\(([^)]*)\)", text):
        out[name] = tuple(float(v) for v in values.replace(",", " ").split())
    return out


STATION_GEODETIC = read_station_geodetic()


def _iau_station_dir(et, station):
    """Moon-center -> station unit direction in MOON_ME using the analytic
    IAU_EARTH frame (valid beyond the high-precision PCK), keeping the
    topocentric offset so the diurnal parallax term survives."""
    p_earth, _ = csp.spkpos("EARTH", et, "MOON_ME", de.AB_COR, "MOON")
    m = csp.pxform("IAU_EARTH", "MOON_ME", et)
    p = np.asarray(p_earth) + m @ _station_itrf_xyz(*STATION_GEODETIC[station])
    return p / np.linalg.norm(p)


def axis_span_iau_earth(et, tx_name, rx_name, ddt=30.0):
    """Same quantities as axis_span_topocentric, from the IAU_EARTH station
    model. The direction functions are smooth (no solver), so a central
    difference over +-ddt is clean."""
    def dir_rate(station):
        u0 = _iau_station_dir(et - ddt, station)
        u1 = _iau_station_dir(et + ddt, station)
        u = _iau_station_dir(et, station)
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
def get_moon_elevations(et, tx_name, rx_name):
    """Moon elevations (deg) from the PINPOINT topocentric frames (z = up)."""
    elevs = []
    for name in (tx_name, rx_name):
        pos, _ = csp.spkpos("MOON", et, f"{name}_TOPO", "NONE", name)
        elevs.append(np.degrees(np.arcsin(pos[2] / np.linalg.norm(pos))))
    return tuple(elevs)


def get_iau_earth_elevations(et, tx_name, rx_name):
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
def stations_in(run_dirs):
    """Stations with a runs CSV (registration_runs_<station>_<chan>.csv) in
    any of run_dirs. An unknown station, or a CSV written before the station
    was part of the name, is an error rather than silently left out."""
    stations = set()
    for run_dir in run_dirs:
        for path in glob.glob(os.path.join(run_dir, "registration_runs_*.csv")):
            m = RUNS_CSV_RE.search(os.path.basename(path))
            if m is None or m.group(1) not in dea.STATION_RX:
                raise ValueError(f"{path}: not registration_runs_<station>_<chan>.csv "
                                 f"with a station in {sorted(dea.STATION_RX)}; "
                                 "rerun registration_stability.py for it")
            stations.add(m.group(1))
    if not stations:
        raise ValueError(f"no registration_runs_<station>_<chan>.csv in {run_dirs}")
    return sorted(stations)


def axes_of_looks(looks):
    """Doppler axes of (label, utc, rx_name) looks, plus per-label swings."""
    axes, labels = [], []
    for label, utc, rx_name in looks:
        axis, _, _ = axis_span_topocentric(csp.str2et(utc), dea.TX_STATION, rx_name)
        axes.append(axis)
        labels.append(label)
    for label in sorted(set(labels)):
        sel = [a for a, d in zip(axes, labels) if d == label]
        swing = max(axis_angle_deg(sel[0], a) for a in sel) if len(sel) > 1 else 0.0
        print(f"  - {label}: {len(sel)} looks, intra-session axis swing "
              f"{swing:.1f} deg")
    return np.array(axes), labels


def stacked_look_axes(run_dirs, chans, min_snr, min_tx_s):
    """Doppler axis of every capture with a look that passes the stack's
    gates (stack_maps.select_looks with min_snr and min_tx_s) in run_dirs,
    over every station there. The per-look brightness rejection that
    follows in the stacker is not applied (it needs the maps).
    Returns (axes (N,3), session labels (N,))."""
    stations = stations_in(run_dirs)
    good, decisions = sm.select_looks(run_dirs, stations, chans,
                                      min_snr=min_snr, min_tx_s=min_tx_s)
    # One axis per capture: its channels share the epoch and the receiver.
    looks = {}
    for r in good:
        looks[(r["_station"], r["rx_file"].split(".chan")[0])] = (
            f"{sm.session_of(r['rx_file'])} {r['_station']}", r["rx_start_utc"], r["_rx"])
    print(f"[*] Existing coverage from {stations} x {chans}: {len(good)} of "
          f"{len(decisions)} looks pass the stack's gates (tone snr >= {min_snr:g}"
          + (f", transmission >= {min_tx_s:g} s" if min_tx_s is not None else "")
          + f"; per-look brightness rejection not applied), {len(looks)} captures")
    return axes_of_looks(sorted(looks.values(), key=lambda t: t[1]))


def session_span_axes():
    """Doppler axes sampled over the reconstructed SESSION_SPANS."""
    looks = []
    for day, (station, u0, u1) in SESSION_SPANS.items():
        et0, et1 = csp.str2et(u0), csp.str2et(u1)
        n = max(2, int((et1 - et0) / SESSION_SPAN_STEP_S) + 1)
        for et in np.linspace(et0, et1, n):
            looks.append((f"{day} {station}", csp.et2utc(float(et), "ISOC", 0),
                          dea.STATION_RX[station]))
    print(f"[*] Existing coverage from reconstructed session spans sampled every "
          f"{SESSION_SPAN_STEP_S / 60:.0f} min (approximate; per-look runs CSVs "
          "via --run-dir are strictly better): "
          f"{len(looks)} sampled epochs over {len(SESSION_SPANS)} sessions")
    return axes_of_looks(looks)


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


def high_precision_pck_end_et():
    """End (ET) of the high-precision Earth-orientation kernel's coverage,
    read from the kernel furnsh_kernels() loads. A kernel that cannot be
    read is an error: the boundary between the two Earth models must come
    from the kernel actually used."""
    bpc = os.path.join(SPICE_KERNEL_DIR, "earth_latest_high_prec.bpc")
    ids = list(csp.pckfrm(bpc))
    if len(ids) != 1:
        raise ValueError(f"{bpc}: expected one frame, found {ids}")
    cover = csp.pckcov(bpc, ids[0])
    _, end_et = csp.wnfetd(cover, csp.wncard(cover) - 1)
    return end_et


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_opportunities(existing_axes, existing_labels, best_opp, title, output_png):
    """Doppler equators on the lunar surface: per-session envelopes of the
    existing looks (the axis sweeps within a session), plus the target."""
    pl.close("all")
    fig, ax = pl.subplots(figsize=(10, 6), dpi=150)
    lons = np.linspace(-180, 180, 361)

    colors = ["#90A4AE", "#78909C", "#546E7A", "#37474F", "#263238"]
    for idx, label in enumerate(sorted(set(existing_labels))):
        sel = np.array([a for a, d in zip(existing_axes, existing_labels)
                        if d == label])
        lats = np.array([[solve_equator_latitude(a, lo) for lo in lons]
                         for a in sel])
        c = colors[idx % len(colors)]
        if len(sel) > 1:
            ax.fill_between(lons, lats.min(0), lats.max(0), color=c, alpha=0.25)
        ax.plot(lons, np.median(lats, 0), "--", color=c, alpha=0.9, lw=1.2,
                label=f"Existing: {label} ({len(sel)} looks)")

    new_lats = [solve_equator_latitude(best_opp["axis"], lo) for lo in lons]
    ax.plot(lons, new_lats, "-", color="#E65100", lw=2.5,
            label=f"Optimal target: {best_opp['utc'].split('T')[0]}")

    ax.set_title(title)
    ax.set_xlabel("Selenographic longitude (deg)")
    ax.set_ylabel("Selenographic latitude (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    pl.savefig(output_png, bbox_inches="tight")
    print(f"[*] Saved opportunities plot to: {output_png}")
    pl.close(fig)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Predict optimal times for lunar radar libration coverage.")
    p.add_argument("--station", required=True, choices=sorted(dea.STATION_RX),
                   help="receiver of the planned observation (Dwingeloo transmits)")
    p.add_argument("--start-date", required=True,
                   help="first UTC date of the scan (YYYY-MM-DD)")
    p.add_argument("--end-date", required=True,
                   help="UTC date the scan stops at (YYYY-MM-DD)")
    p.add_argument("--min-tx-elevation", type=float, default=45.0,
                   help="Min Moon elevation at Dwingeloo (TX), deg.")
    p.add_argument("--min-rx-elevation", type=float, default=10.0,
                   help="Min Moon elevation at the receiver, deg.")
    p.add_argument("--min-doppler-span", type=float, default=5.0,
                   help="Min limb-to-limb L-band Doppler span, Hz.")
    p.add_argument("--step-hours", type=float, default=1.0)
    coverage = p.add_mutually_exclusive_group(required=True)
    coverage.add_argument("--run-dir", nargs="+",
                          help="run directories with registration_runs_<station>_<chan>.csv "
                               "(per-look coverage axes, every station found): the "
                               "looks that pass the stacker's gates "
                               "(stack_maps.select_looks); its per-look brightness "
                               "rejection, which needs the maps, is not applied")
    coverage.add_argument("--coverage-from-session-spans", action="store_true",
                          help="no runs CSVs: approximate the existing coverage by "
                               "reconstructed 2025 Stockert session spans")
    p.add_argument("--chans", nargs="+", default=None,
                   help="with --run-dir: channels whose stacked looks count as "
                        "coverage (default chan1)")
    p.add_argument("--min-snr", type=float, default=None,
                   help="with --run-dir: weakest tone a covering look may have "
                        f"(default {foh.MIN_TONE_SNR:g}, freq_offset_hunt.MIN_TONE_SNR, "
                        "the stack's)")
    p.add_argument("--min-tx-s", type=float, default=None,
                   help="with --run-dir: count only looks whose transmission lasted "
                        "at least this many seconds (as stack_maps.py --min-tx-s; "
                        "default: no limit)")
    p.add_argument("--output-csv", default=None,
                   help="default results/ERRORS/libration_opportunities_<station>.csv")
    p.add_argument("--output-plot", default=None,
                   help="default results/ERRORS/libration_opportunities_<station>.png")
    args = p.parse_args()
    if args.coverage_from_session_spans and (args.min_snr is not None
                                             or args.min_tx_s is not None
                                             or args.chans is not None):
        p.error("--chans, --min-snr and --min-tx-s select runs-CSV looks; they do "
                "not apply to --coverage-from-session-spans")
    chans = ["chan1"] if args.chans is None else args.chans
    min_snr = foh.MIN_TONE_SNR if args.min_snr is None else args.min_snr
    tx_name, rx_name = dea.TX_STATION, dea.STATION_RX[args.station]
    output_csv = args.output_csv or os.path.join(
        "results", "ERRORS", f"libration_opportunities_{args.station}.csv")
    output_plot = args.output_plot or os.path.join(
        "results", "ERRORS", f"libration_opportunities_{args.station}.png")

    furnsh_kernels()

    # High-precision Earth PCK coverage end -> IAU_EARTH model beyond it.
    pck_end_et = high_precision_pck_end_et()
    print(f"[*] High-precision Earth PCK coverage until: "
          f"{csp.et2utc(pck_end_et, 'ISOC', 0)} UTC "
          "(beyond: IAU_EARTH model, stations kept topocentric)")

    if args.coverage_from_session_spans:
        existing_axes, existing_labels = session_span_axes()
    else:
        existing_axes, existing_labels = stacked_look_axes(args.run_dir, chans,
                                                           min_snr, args.min_tx_s)

    start_et = csp.str2et(args.start_date + "T00:00:00")
    end_et = csp.str2et(args.end_date + "T00:00:00")
    step_s = args.step_hours * 3600.0
    n_steps = int((end_et - start_et) / step_s)
    print(f"[*] Scanning {n_steps} steps from {args.start_date} to {args.end_date} "
          f"({tx_name} -> {rx_name})...")

    opportunities = []
    for i in range(n_steps):
        et = start_et + i * step_s
        iau_earth = et > pck_end_et
        elev_tx, elev_rx = (get_iau_earth_elevations(et, tx_name, rx_name) if iau_earth
                            else get_moon_elevations(et, tx_name, rx_name))
        if elev_tx < args.min_tx_elevation or elev_rx < args.min_rx_elevation:
            continue
        axis, speed, span_hz = (axis_span_iau_earth(et, tx_name, rx_name) if iau_earth
                                else axis_span_topocentric(et, tx_name, rx_name))
        if span_hz < args.min_doppler_span:
            continue
        min_dist = min(axis_angle_deg(axis, a) for a in existing_axes)
        opportunities.append({
            "et": et, "utc": csp.et2utc(et, "ISOC", 0),
            "elev_tx": elev_tx, "elev_rx": elev_rx, "axis": axis,
            "speed": speed, "doppler_span": span_hz,
            "min_dist": min_dist, "iau_earth": iau_earth,
        })
    print(f"[*] Found {len(opportunities)} visible steps passing the gates.")
    if not opportunities:
        # An earlier scan's outputs under the same names would read as this
        # scan's result: remove them, and fail.
        for path in (output_csv, output_plot):
            if os.path.exists(path):
                os.remove(path)
                print(f"[!] Removed the earlier {path}")
        raise SystemExit(f"no step from {args.start_date} to {args.end_date} passes the "
                         "elevation and Doppler-span cuts; nothing written")

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

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc", "min_displacement_deg", "pass_start_utc",
                    "pass_end_utc", "pass_min_displacement_deg",
                    "pass_axis_swing_deg", "tx_station", "rx_station",
                    "tx_elevation_deg", "rx_elevation_deg", "specular_speed_m_s",
                    "doppler_span_hz", "axis_x", "axis_y", "axis_z",
                    "iau_earth_model", "coverage_from_session_spans",
                    "coverage_chans", "coverage_min_snr", "coverage_min_tx_s"])
        for o in best:
            w.writerow([o["utc"], f"{o['min_dist']:.4f}", o["pass_start"],
                        o["pass_end"], f"{o['pass_min_dist']:.4f}",
                        f"{o['pass_axis_swing_deg']:.4f}", tx_name, rx_name,
                        f"{o['elev_tx']:.2f}", f"{o['elev_rx']:.2f}",
                        f"{o['speed']:.4f}", f"{o['doppler_span']:.2f}",
                        f"{o['axis'][0]:.6f}", f"{o['axis'][1]:.6f}",
                        f"{o['axis'][2]:.6f}", o["iau_earth"],
                        args.coverage_from_session_spans,
                        # The runs-CSV coverage selection (blank for session spans).
                        "" if args.coverage_from_session_spans else " ".join(chans),
                        "" if args.coverage_from_session_spans else min_snr,
                        "" if args.min_tx_s is None else args.min_tx_s])
    print(f"[*] Saved pass peaks to CSV: {output_csv}")

    plot_opportunities(existing_axes, existing_labels, best[0],
                       f"Doppler-equator coverage ({tx_name} -> {rx_name}): "
                       "existing sessions vs optimal target", output_plot)

    print("\n" + "=" * 100)
    print(f"{f'TOP LIBRATION OPPORTUNITIES  {tx_name} (TX) -> {rx_name} (RX)':^100}")
    print("=" * 100)
    print(f"{'Peak (UTC)':<21}| {'Disp':>7} | {'Pass disp':>9} | {'Swing':>6} | "
          f"{'TX el':>6} | {'RX el':>6} | {'Span':>8} | Model")
    print("-" * 100)
    for o in best[:10]:
        model = "IAU_EARTH" if o["iau_earth"] else "Topocentric"
        print(f"{o['utc']:<21}| {o['min_dist']:>6.2f}° | "
              f"{o['pass_min_dist']:>6.2f}°+ | {o['pass_axis_swing_deg']:>5.1f}° | "
              f"{o['elev_tx']:>5.1f}° | {o['elev_rx']:>5.1f}° | "
              f"{o['doppler_span']:>6.1f} Hz | {model}")
    print("=" * 100)
    if args.coverage_from_session_spans:
        print("NOTE: coverage axes came from reconstructed session spans "
              "(approximate) — re-run with the runs CSVs (--run-dir) for "
              "per-look coverage before committing telescope time.")


if __name__ == "__main__":
    main()

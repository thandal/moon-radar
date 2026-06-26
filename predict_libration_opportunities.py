#!/usr/bin/env python3
"""
Predict optimal times in the Moon's libration cycle to collect new datasets
using Dwingeloo (TX) and Stockert (RX).

The tool finds future epochs when the Doppler equator (the locus of mapping
degeneracy) is maximally displaced from the Doppler equators of the existing
datasets. This ensures that new observations will cover the regions of the
Moon that were previously degenerate (and hence masked out in the stack).

Usage:
    .conda/bin/python predict_libration_opportunities.py [options]
"""

import os
import csv
import argparse
import datetime
import numpy as np
import cspyce as csp
from astropy import time as at
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from spice_setup import furnsh_kernels
import doppler_equator as de

# Hardcoded fallback sessions in case the runs CSV is missing
DEFAULT_SESSIONS = {
    "2025-06-21": [-0.01294650,  0.70426258, -0.70982154],
    "2025-09-10": [ 0.02949176,  0.85167124,  0.52324596],
    "2025-09-11": [-0.03865091,  0.85211881, -0.52191919],
    "2025-09-16": [ 0.11948707, -0.89262674,  0.43467269]
}

def get_existing_doppler_axes(csv_path):
    """
    Parse unique sessions from the registration runs CSV and compute their
    Doppler axes using SPICE. Fall back to DEFAULT_SESSIONS if the CSV is missing.
    """
    if not os.path.exists(csv_path):
        print(f"[*] Info: Runs CSV not found at {csv_path}. Using default sessions.")
        return [np.array(axis) for axis in DEFAULT_SESSIONS.values()], list(DEFAULT_SESSIONS.keys())

    # Load unique timestamps from the CSV
    unique_dates = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("rx_start_utc"):
                unique_dates.add(row["rx_start_utc"])

    # Group timestamps by day (session)
    sessions = {}
    for date_str in sorted(unique_dates):
        day = date_str.split("T")[0]
        sessions.setdefault(day, []).append(date_str)

    axes = []
    days = []
    print(f"[*] Loaded existing runs from {csv_path}:")
    for day, dates in sorted(sessions.items()):
        # Compute Doppler axis at the midpoint of each session
        mid_idx = len(dates) // 2
        t = at.Time(dates[mid_idx])
        et = csp.str2et(t.utc.value)
        axis, _ = compute_doppler_axis(et, geocentric=False)
        axes.append(axis)
        days.append(day)
        print(f"  - {day} ({len(dates)} looks): axis = {axis}")
    
    return axes, days

def compute_doppler_axis(et, geocentric=False, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the unit Doppler axis in MOON_ME coordinates for a given ET.
    If geocentric=True, computes the axis from the Earth's center rather than
    topocentric station coordinates, which avoids the need for Earth rotation kernels.
    
    Returns (doppler_axis, speed_m_s)
    """
    if geocentric:
        # Apparent position of Earth center in MOON_ME
        pos, _ = csp.spkpos("EARTH", et, "MOON_ME", "LT+S", "MOON")
        srp_hat = pos / np.linalg.norm(pos)
        
        # Specular point motion approximated by Earth motion
        dt = 1.0
        pos2, _ = csp.spkpos("EARTH", et + dt, "MOON_ME", "LT+S", "MOON")
        srp2_hat = pos2 / np.linalg.norm(pos2)
        v_tangent = (srp2_hat - srp_hat) / dt
        # Convert angular speed to linear surface speed on the Moon (1737.4 km radius)
        speed_m_s = np.linalg.norm(v_tangent) * 1737.4 * 1000.0
    else:
        # Topocentric bistatic specular point and its motion
        srp = de.specular_point_bck(et, tx_name, rx_name)
        srp_hat = srp / np.linalg.norm(srp)

        dt = 1.0
        srp2 = de.specular_point_bck(et + dt, tx_name, rx_name)
        v_srp = (srp2 - srp) / dt
        v_tangent = v_srp - np.dot(v_srp, srp_hat) * srp_hat
        speed_m_s = np.linalg.norm(v_tangent) * 1000.0

    v_tangent_hat = v_tangent / np.linalg.norm(v_tangent)
    doppler_axis = np.cross(srp_hat, v_tangent_hat)
    doppler_axis /= np.linalg.norm(doppler_axis)
    return doppler_axis, speed_m_s

def get_moon_elevations(et, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute Moon elevation angles (degrees) for Dwingeloo and Stockert topocentric sites.
    """
    # Dwingeloo topocentric elevation
    pos_tx, _ = csp.spkpos("MOON", et, "DWINGELOO_TOPO", "NONE", "DWINGELOO")
    elev_tx = np.degrees(np.arcsin(pos_tx[2] / np.linalg.norm(pos_tx)))
    
    # Stockert topocentric elevation
    pos_rx, _ = csp.spkpos("MOON", et, "STOCKERT_TOPO", "NONE", "STOCKERT")
    elev_rx = np.degrees(np.arcsin(pos_rx[2] / np.linalg.norm(pos_rx)))
    
    return elev_tx, elev_rx

def get_geocentric_elevations(et, tx_lat=52.81215, tx_lon=6.39632, rx_lat=50.56944, rx_lon=6.72194):
    """
    Compute approximate elevations using the analytical IAU_EARTH frame when high-precision
    Earth binary kernels are expired.
    """
    pos_moon, _ = csp.spkpos("MOON", et, "IAU_EARTH", "NONE", "EARTH")
    pos_moon = np.array(pos_moon)
    r_moon = np.linalg.norm(pos_moon)
    
    def get_elev(lat_deg, lon_deg):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        # Normal vector to Earth sphere
        n = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        
        # Approximate station vector (assuming Earth radius ~ 6378.1 km)
        r_stn = 6378.1 * n
        v_stn_moon = pos_moon - r_stn
        v_hat = v_stn_moon / np.linalg.norm(v_stn_moon)
        
        elev = np.degrees(np.arcsin(np.dot(v_hat, n)))
        return elev

    return get_elev(tx_lat, tx_lon), get_elev(rx_lat, rx_lon)

def solve_equator_latitude(axis, lon_deg):
    """
    Solve for the latitude (degrees) of the Doppler equator great circle
    at a given selenographic longitude.
    """
    lon = np.radians(lon_deg)
    # Equation: axis_x * cos(lat) * cos(lon) + axis_y * cos(lat) * sin(lon) + axis_z * sin(lat) = 0
    # Tan(lat) = -(axis_x * cos(lon) + axis_y * sin(lon)) / axis_z
    if abs(axis[2]) < 1e-9:
        return 90.0 if (axis[0]*np.cos(lon) + axis[1]*np.sin(lon)) < 0 else -90.0
    tan_lat = -(axis[0]*np.cos(lon) + axis[1]*np.sin(lon)) / axis[2]
    return np.degrees(np.arctan(tan_lat))

def plot_opportunities(existing_axes, existing_days, best_opp, output_png):
    """
    Create a visualization of the Doppler equators on the Moon.
    """
    pl.close('all')
    fig, ax = pl.subplots(figsize=(10, 6), dpi=150)
    
    lons = np.linspace(-180, 180, 360)
    
    # Plot existing equators
    styles = ['--', ':', '-.', (0, (3, 1, 1, 1))]
    colors = ['#90A4AE', '#78909C', '#546E7A', '#37474F']
    for idx, (axis, day) in enumerate(zip(existing_axes, existing_days)):
        lats = [solve_equator_latitude(axis, lon) for lon in lons]
        style = styles[idx % len(styles)]
        color = colors[idx % len(colors)]
        ax.plot(lons, lats, linestyle=style, color=color, alpha=0.8, 
                label=f"Existing: {day}")
        
    # Plot new optimal equator
    new_axis = best_opp['axis']
    new_lats = [solve_equator_latitude(new_axis, lon) for lon in lons]
    ax.plot(lons, new_lats, '-', color='#E65100', linewidth=2.5,
            label=f"Optimal Target: {best_opp['utc'].split('T')[0]}")
    
    ax.set_title("Doppler Equator Shift on Lunar Surface", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Selenographic Longitude (degrees)", fontsize=11)
    ax.set_ylabel("Selenographic Latitude (degrees)", fontsize=11)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend formatting
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    
    # Styled plot borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    pl.savefig(output_png, bbox_inches='tight')
    print(f"[*] Saved opportunities plot to: {output_png}")
    pl.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Predict optimal times for lunar radar libration coverage.")
    parser.add_argument("--start-date", type=str, default="2026-06-18",
                        help="Start date for future search (YYYY-MM-DD). Default: 2026-06-18.")
    parser.add_argument("--end-date", type=str, default="2026-09-10",
                        help="End date for future search (YYYY-MM-DD). Default: 2026-09-10.")
    parser.add_argument("--min-tx-elevation", type=float, default=45.0,
                        help="Minimum Moon elevation at Dwingeloo (TX) (degrees). Default: 45.0.")
    parser.add_argument("--min-rx-elevation", type=float, default=10.0,
                        help="Minimum Moon elevation at Stockert (RX) (degrees). Default: 10.0.")
    parser.add_argument("--min-doppler-span", type=float, default=5.0,
                        help="Minimum estimated L-band Doppler bandwidth/span (Hz). Default: 5.0.")
    parser.add_argument("--step-hours", type=float, default=1.0,
                        help="Search step size (hours). Default: 1.0.")
    parser.add_argument("--output-csv", type=str, default="results/ERRORS/libration_opportunities.csv",
                        help="Path to save the output CSV. Default: results/ERRORS/libration_opportunities.csv.")
    parser.add_argument("--output-plot", type=str, default="results/ERRORS/libration_opportunities.png",
                        help="Path to save the visualization plot. Default: results/ERRORS/libration_opportunities.png.")
    
    args = parser.parse_args()
    
    # Load SPICE kernels
    furnsh_kernels()
    
    # Query SPICE for the high-precision PCK end time
    try:
        bpc_path = os.path.join("spice_kernels", "earth_latest_high_prec.bpc")
        ids = csp.pckfrm(bpc_path)
        cover = csp.pckcov(bpc_path, list(ids)[0])
        _, pck_end_et = csp.wnfetd(cover, csp.wncard(cover) - 1)
        pck_end_utc = csp.et2utc(pck_end_et, "ISOC", 0)
    except Exception:
        # Fallback if cell functions are unsupported or file is missing
        pck_end_et = csp.str2et("2026-09-11T00:00:00")
        pck_end_utc = "2026-09-11T00:00:00"

    print(f"[*] Loaded Earth PCK with valid high-precision coverage until: {pck_end_utc} UTC")
    
    # Load existing runs Doppler axes
    csv_path = "results/LOLA_DEM_REGISTRATION/registration_runs.csv"
    existing_axes, existing_days = get_existing_doppler_axes(csv_path)
    
    # Set search range
    start_et = csp.str2et(at.Time(args.start_date + "T00:00:00").utc.value)
    end_et = csp.str2et(at.Time(args.end_date + "T00:00:00").utc.value)
    
    step_s = args.step_hours * 3600.0
    n_steps = int((end_et - start_et) / step_s)
    
    print(f"[*] Scanning {n_steps} steps from {args.start_date} to {args.end_date}...")
    
    opportunities = []
    
    for i in range(n_steps):
        et = start_et + i * step_s
        
        # Decide whether we use topocentric or geocentric model
        # If outside high-precision PCK coverage, use geocentric model
        use_fallback = (et > pck_end_et)
        
        # Check elevation
        if use_fallback:
            elev_tx, elev_rx = get_geocentric_elevations(et)
        else:
            elev_tx, elev_rx = get_moon_elevations(et)
            
        if elev_tx < args.min_tx_elevation or elev_rx < args.min_rx_elevation:
            continue
        try:
            axis, speed = compute_doppler_axis(et, geocentric=use_fallback)
            # Total Doppler bandwidth at L-band (1299.5 MHz, lambda ~0.2307m)
            # B = 4 * v / lambda
            doppler_span_hz = 4.0 * speed / 0.230688
        except Exception as e:
            continue
            
        if doppler_span_hz < args.min_doppler_span:
            continue
            
        # Compute displacement angle to all existing axes
        min_dist = 180.0
        for ext_axis in existing_axes:
            cos_angle = abs(np.dot(axis, ext_axis))
            cos_angle = min(1.0, max(-1.0, cos_angle))
            dist = np.degrees(np.arccos(cos_angle))
            if dist < min_dist:
                min_dist = dist
                
        opportunities.append({
            "et": et,
            "utc": csp.et2utc(et, "ISOC", 0),
            "elev_tx": elev_tx,
            "elev_rx": elev_rx,
            "axis": axis,
            "speed": speed,
            "doppler_span": doppler_span_hz,
            "min_dist": min_dist,
            "fallback": use_fallback
        })
        
    print(f"[*] Found {len(opportunities)} visible time steps matching elevation requirements.")
    
    if not opportunities:
        print("[!] No visible opportunities found in the requested window.")
        return

    # Group consecutive visible steps into passes (gap of > 4 hours starts a new pass)
    passes = []
    current_pass = []
    for opp in opportunities:
        if not current_pass:
            current_pass.append(opp)
        else:
            time_diff = (opp['et'] - current_pass[-1]['et']) / 3600.0
            if time_diff > 4.0:
                passes.append(current_pass)
                current_pass = [opp]
            else:
                current_pass.append(opp)
    if current_pass:
        passes.append(current_pass)
        
    print(f"[*] Grouped steps into {len(passes)} distinct visible passes.")
    
    # For each pass, find the peak opportunity (maximal displacement)
    best_opportunities = []
    for p in passes:
        peak = max(p, key=lambda x: x['min_dist'])
        best_opportunities.append(peak)
        
    # Sort the passes by displacement descending
    best_opportunities.sort(key=lambda x: x['min_dist'], reverse=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utc", "min_displacement_deg", "dwingeloo_elevation_deg", 
                         "stockert_elevation_deg", "specular_speed_m_s", "doppler_span_hz",
                         "axis_x", "axis_y", "axis_z", "geocentric_fallback"])
        for opp in best_opportunities:
            writer.writerow([
                opp['utc'],
                f"{opp['min_dist']:.4f}",
                f"{opp['elev_tx']:.2f}",
                f"{opp['elev_rx']:.2f}",
                f"{opp['speed']:.4f}",
                f"{opp['doppler_span']:.2f}",
                f"{opp['axis'][0]:.6f}",
                f"{opp['axis'][1]:.6f}",
                f"{opp['axis'][2]:.6f}",
                opp['fallback']
            ])
            
    print(f"[*] Saved pass peaks to CSV: {args.output_csv}")
    
    # Plot the best opportunity
    plot_opportunities(existing_axes, existing_days, best_opportunities[0], args.output_plot)
    
    # Display results
    print("\n" + "="*80)
    print(f"{'TOP LIBRATION OPPORTUNITIES FOR Dwingeloo -> Stockert':^80}")
    print("="*80)
    print(f"{'Date & Time (UTC)':<24} | {'Displacement':<12} | {'TX Elev':<8} | {'RX Elev':<8} | {'Doppler Span':<12} | {'Model':<10}")
    print("-"*95)
    for opp in best_opportunities[:10]:
        model_str = "Geocentric" if opp['fallback'] else "Topocentric"
        span_str = f"{opp['doppler_span']:.1f} Hz"
        print(f"{opp['utc']:<24} | {opp['min_dist']:>10.2f}° | {opp['elev_tx']:>6.1f}° | {opp['elev_rx']:>6.1f}° | {span_str:>12} | {model_str:<10}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

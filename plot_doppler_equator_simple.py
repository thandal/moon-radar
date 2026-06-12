"""
Simple Doppler Equator Plotter

Quick visualization of nominal doppler equator curves (no uncertainties).

Usage:
    python plot_doppler_equator_simple.py
"""

import os
import cspyce as csp
from astropy import time as at

from doppler_equator_errors import plot_equator_nominal

# ---------------------------------------------------------------------------
# SPICE setup
# ---------------------------------------------------------------------------
from spice_setup import furnsh_kernels
furnsh_kernels()


# ---------------------------------------------------------------------------
# Plot doppler equator
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOPPLER EQUATOR - NOMINAL CURVES")
    print("="*70)

    # Use example observation time
    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    print(f"\nObservation time: {obs_time.iso}")
    print(f"Ephemeris time: {rx_time:.3f} s past J2000")
    print(f"Computing doppler equator...")

    # Create plot
    fig, ax = plot_equator_nominal(
        rx_time,
        tx_name="DWINGELOO",
        rx_name="STOCKERT",
        n_points=500
    )

    # Save
    os.makedirs("results/ERRORS", exist_ok=True)
    output_file = "results/ERRORS/doppler_equator_clean.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"\nSaved: {output_file}")
    print("\n" + "="*70 + "\n")

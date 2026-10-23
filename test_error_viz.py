"""Quick test of error visualization to debug issues."""

import os
import numpy as np
import cspyce as csp
from astropy import time as at
from matplotlib import pyplot as plt

# SPICE setup
from spice_setup import furnsh_kernels
furnsh_kernels()

from doppler_equator import compute_doppler_equator_velocity

# The example epoch is in the 2025-09-16 session: Dwingeloo transmitting,
# Stockert receiving.
TX_NAME, RX_NAME = "DWINGELOO", "STOCKERT"

# Test with example time
obs_time = at.Time("2025-09-16T13:22:02")
rx_time = csp.str2et(obs_time.utc.value)

print(f"Computing doppler equator for {obs_time.iso}")
lt_min, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(
    rx_time, n_points=500, tx_name=TX_NAME, rx_name=RX_NAME)

print(f"\nResults:")
print(f"  lt_min: {lt_min:.6f} s")
print(f"  delay_up range: [{delay_up.min():.6f}, {delay_up.max():.6f}] s")
print(f"  dlt_up range: [{dlt_up.min():.9f}, {dlt_up.max():.9f}]")
print(f"  delay_down range: [{delay_down.min():.6f}, {delay_down.max():.6f}] s")
print(f"  dlt_down range: [{dlt_down.min():.9f}, {dlt_down.max():.9f}]")

# Plot the actual curves
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(dlt_up, delay_up, 'b-', linewidth=2, label='Up-Doppler')
ax.plot(dlt_down, delay_down, 'r-', linewidth=2, label='Down-Doppler')
ax.set_xlabel('Fractional Doppler Shift', fontsize=12)
ax.set_ylabel('Delay (s)', fontsize=12)
ax.set_title('Doppler Equator (Velocity Method)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
os.makedirs('results/VERIFY', exist_ok=True)
plt.savefig('results/VERIFY/test_curves.png', dpi=150)
print(f"\nSaved results/VERIFY/test_curves.png")

# Now test uncertainty calculation
from doppler_equator_errors import EphemerisUncertainty

ephem_unc = EphemerisUncertainty()
sigma_pos = ephem_unc.position_uncertainty()
sigma_vel = ephem_unc.velocity_uncertainty()

print(f"\nEphemeris uncertainties:")
print(f"  Position: {sigma_pos*1000:.3f} mm = {sigma_pos*100:.3f} cm")
print(f"  Velocity: {sigma_vel*1000:.3f} mm/s")

# Test DLT uncertainty at a single point
from doppler_equator import AB_COR
import cspyce as csp

# Get SRP
srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time, "MOON_ME", AB_COR, RX_NAME)
srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time, "MOON_ME", AB_COR, TX_NAME)
srp = (srp_rx + srp_tx) / 2.0

print(f"\nSRP position: {srp} km")
print(f"SRP norm: {np.linalg.norm(srp):.1f} km")

# Compute nominal DLT at SRP
from doppler_equator import moonPointDLT_BCK
lt_srp, dlt_srp = moonPointDLT_BCK(rx_time, srp, TX_NAME, RX_NAME)
print(f"\nSRP values:")
print(f"  Light time: {lt_srp:.6f} s")
print(f"  DLT: {dlt_srp:.12f}")

# A radial perturbation of a surface point moves nothing: moonPointDLT_BCK
# projects every point back onto the surface. srp is in km.
delta_r_km = 0.02  # 20 m
srp_perturbed = srp * (1 + delta_r_km / np.linalg.norm(srp))
lt_pert, dlt_pert = moonPointDLT_BCK(rx_time, srp_perturbed, TX_NAME, RX_NAME)

print(f"\nPerturbed by {delta_r_km*1e3:.0f} m radially (projected back onto the surface):")
print(f"  Light time change: {(lt_pert - lt_srp)*1e9:.3f} ns")
print(f"  DLT change: {(dlt_pert - dlt_srp):.12e}")

# Expected DLT uncertainty from the Moon's position: an error in it turns
# the lines of sight (doppler_equator_errors.ephemeris_position_dlt).
from doppler_equator_errors import ephemeris_position_dlt, ephemeris_velocity_dlt
dlt_unc_expected = ephemeris_position_dlt(rx_time, srp, sigma_pos,
                                          tx_name=TX_NAME, rx_name=RX_NAME)
print(f"\nExpected DLT uncertainty (from position only, {sigma_pos*100:.0f} cm):")
print(f"  {dlt_unc_expected:.12e}")
print(f"  As parts per trillion: {dlt_unc_expected*1e12:.3f}")

# Velocity contribution
dlt_unc_vel = ephemeris_velocity_dlt(sigma_vel)
print(f"\nExpected DLT uncertainty (from velocity):")
print(f"  {dlt_unc_vel:.12e}")
print(f"  As parts per trillion: {dlt_unc_vel*1e12:.3f}")

# Total
dlt_unc_total = np.sqrt(dlt_unc_expected**2 + dlt_unc_vel**2)
print(f"\nTotal DLT uncertainty (RSS):")
print(f"  {dlt_unc_total:.12e}")
print(f"  As parts per trillion: {dlt_unc_total*1e12:.3f}")

# Compare to the width of the doppler equator: limb to limb, from the up
# branch's lowest dlt to the down branch's highest (both start at the SRP).
dlt_range = dlt_down.max() - dlt_up.min()
print(f"\nDoppler equator width:")
print(f"  Total DLT range: {dlt_range:.9f}")
print(f"  Uncertainty as fraction of width: {dlt_unc_total/dlt_range*100:.6f}%")
print(f"  3-sigma as fraction of width: {3*dlt_unc_total/dlt_range*100:.6f}%")

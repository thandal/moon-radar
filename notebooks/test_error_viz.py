"""Quick test of error visualization to debug issues."""

import numpy as np
import cspyce as csp
from astropy import time as at
from matplotlib import pyplot as plt

# SPICE setup
SPICE_KERNEL_DIR = "spice_kernels"
csp.kclear()
for k in ["naif0012.tls", "de440s.bsp", "pck00011.tpc",
           "earth_latest_high_prec.bpc", "moon_pa_de440_200625.bpc",
           "moon_de440_250416.tf", "observatories.bsp", "observatories.tf"]:
    csp.furnsh(f"{SPICE_KERNEL_DIR}/{k}")

from doppler_equator import compute_doppler_equator_velocity

# Test with example time
obs_time = at.Time("2025-09-16T13:22:02")
rx_time = csp.str2et(obs_time.utc.value)

print(f"Computing doppler equator for {obs_time.iso}")
lt_min, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(rx_time, n_points=500)

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
plt.savefig('test_curves.png', dpi=150)
print(f"\nSaved test_curves.png")

# Now test uncertainty calculation
from doppler_equator_errors import EphemerisUncertainty, compute_dlt_uncertainty, compute_delay_uncertainty

ephem_unc = EphemerisUncertainty()
sigma_pos = ephem_unc.position_uncertainty(rx_time)
sigma_vel = ephem_unc.velocity_uncertainty(rx_time)

print(f"\nEphemeris uncertainties:")
print(f"  Position: {sigma_pos*1000:.3f} mm = {sigma_pos*100:.3f} cm")
print(f"  Velocity: {sigma_vel*1000:.3f} mm/s")

# Test DLT uncertainty at a single point
from doppler_equator import moonSRP_DLT_BCK, AB_COR, EARTH_FRAME
import cspyce as csp

# Get SRP
srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time, "MOON_ME", AB_COR, "STOCKERT")
srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time, "MOON_ME", AB_COR, "DWINGELOO")
srp = (srp_rx + srp_tx) / 2.0

print(f"\nSRP position: {srp}")
print(f"SRP norm: {np.linalg.norm(srp):.1f} m")

# Compute nominal DLT at SRP
from doppler_equator import moonPointDLT_BCK
lt_srp, dlt_srp = moonPointDLT_BCK(rx_time, srp, "DWINGELOO", "STOCKERT")
print(f"\nSRP values:")
print(f"  Light time: {lt_srp:.6f} s")
print(f"  DLT: {dlt_srp:.12f}")

# Test sensitivity by perturbing position
c = csp.clight()
delta_r = 0.02  # 2 cm perturbation
srp_perturbed = srp * (1 + delta_r / np.linalg.norm(srp))
lt_pert, dlt_pert = moonPointDLT_BCK(rx_time, srp_perturbed, "DWINGELOO", "STOCKERT")

print(f"\nPerturbed by {delta_r*100:.1f} cm radially:")
print(f"  Light time change: {(lt_pert - lt_srp)*1e9:.3f} ns")
print(f"  DLT change: {(dlt_pert - dlt_srp):.12e}")
print(f"  DLT sensitivity: {(dlt_pert - dlt_srp)/delta_r:.6e} per meter")

# Expected DLT uncertainty
dlt_unc_expected = abs(dlt_pert - dlt_srp) / delta_r * sigma_pos
print(f"\nExpected DLT uncertainty (from position only):")
print(f"  {dlt_unc_expected:.12e}")
print(f"  As parts per trillion: {dlt_unc_expected*1e12:.3f}")

# Velocity contribution
dlt_unc_vel = sigma_vel / c
print(f"\nExpected DLT uncertainty (from velocity):")
print(f"  {dlt_unc_vel:.12e}")
print(f"  As parts per trillion: {dlt_unc_vel*1e12:.3f}")

# Total
dlt_unc_total = np.sqrt(dlt_unc_expected**2 + dlt_unc_vel**2)
print(f"\nTotal DLT uncertainty (RSS):")
print(f"  {dlt_unc_total:.12e}")
print(f"  As parts per trillion: {dlt_unc_total*1e12:.3f}")

# Compare to the width of the doppler equator
dlt_range = dlt_up.max() - dlt_down.min()
print(f"\nDoppler equator width:")
print(f"  Total DLT range: {dlt_range:.9f}")
print(f"  Uncertainty as fraction of width: {dlt_unc_total/dlt_range*100:.6f}%")
print(f"  3-sigma as fraction of width: {3*dlt_unc_total/dlt_range*100:.6f}%")

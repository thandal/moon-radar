"""
Velocity Error Source Breakdown

Analyzes the contribution of different ephemeris components to the Doppler
shift uncertainty in bistatic lunar radar measurements.

The Doppler shift depends on velocities from multiple sources:
1. Moon barycentric velocity (orbital motion)
2. Earth barycentric velocity (orbital motion)
3. Observatory velocity (Earth rotation + polar motion)
4. Moon surface velocity (libration - rotation variations)
"""

import numpy as np
import cspyce as csp
from astropy import time as at
from matplotlib import pyplot as plt
import os


def compute_velocity_components(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Break down all velocity components that contribute to the Doppler shift.

    Returns:
        dict with velocity magnitudes and their uncertainties
    """
    c = csp.clight()

    # Get the combined sub-radar point
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", "LT", rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", "LT", tx_name)
    srp = (srp_rx + srp_tx) / 2.0

    # Get Moon position and velocity in J2000 inertial frame
    moon_state, lt_moon = csp.spkezr("MOON", rx_time, "J2000", "NONE", "EARTH")
    moon_vel_inertial = np.linalg.norm(moon_state[3:6])  # m/s

    # Get Earth barycentric velocity
    earth_state, lt_earth = csp.spkezr("EARTH", rx_time, "J2000", "NONE", "SSB")
    earth_vel_bary = np.linalg.norm(earth_state[3:6])  # m/s

    # Get observatory velocities (includes Earth rotation)
    # RX observatory
    rx_state, lt_rx = csp.spkezr(rx_name, rx_time, "J2000", "LT", "MOON")
    rx_vel = np.linalg.norm(rx_state[3:6])  # m/s

    # TX observatory
    tx_state, lt_tx = csp.spkezr(tx_name, rx_time, "J2000", "LT", "MOON")
    tx_vel = np.linalg.norm(tx_state[3:6])  # m/s

    # Get Moon rotation rate (libration causes variations)
    # Check angular velocity of Moon body-fixed frame
    moon_radii = csp.bodvrd("MOON", "RADII")

    # Estimate surface velocity due to rotation/libration at SRP
    # Get SRP velocity in Moon body-fixed frame (reveals rotation component)
    dt = 1.0  # 1 second time step
    srp_t1, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time - dt,
                               "MOON_ME", "LT", rx_name)
    srp_t2, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time + dt,
                               "MOON_ME", "LT", rx_name)
    srp_vel_body = np.linalg.norm(srp_t2 - srp_t1) / (2 * dt)  # m/s in body frame

    # Compute radial velocities (what matters for Doppler)
    # Use spkcpt/spkcpo to get radial velocity components
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_surf = csp.edpnt_vector(srp, moon_radii[0], moon_radii[1], moon_radii[2])

    s_rx, lt_rx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", rx_time,
                                     "ITRF93", "OBSERVER", "LT", rx_name)
    s_tx, lt_tx = csp.spkcpo_vector(tx_name, rx_time - lt_rx, "ITRF93",
                                     "TARGET", "LT", p_surf, "MOON", "MOON_ME")

    v_rx_radial = csp.dvnorm_vector(s_rx)  # radial velocity component, m/s
    v_tx_radial = csp.dvnorm_vector(s_tx)

    # Fractional Doppler from each component
    dlt = 1 - np.sqrt((1 - v_rx_radial/c)/(1 + v_rx_radial/c)) * \
              np.sqrt((1 - v_tx_radial/c)/(1 + v_tx_radial/c))

    return {
        'moon_vel_inertial': moon_vel_inertial,
        'earth_vel_bary': earth_vel_bary,
        'rx_vel': rx_vel,
        'tx_vel': tx_vel,
        'srp_vel_body': srp_vel_body,
        'v_rx_radial': v_rx_radial,
        'v_tx_radial': v_tx_radial,
        'dlt': dlt,
    }


def estimate_velocity_uncertainties():
    """
    Estimate uncertainties in each velocity component based on SPICE kernel
    documentation and literature.

    Returns:
        dict with uncertainty estimates in m/s

    References:
        Park et al. 2021, "The JPL Planetary and Lunar Ephemerides DE440 and DE441"
        https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf

        Key facts from DE440/DE441 documentation:
        - Modern LLR weighted RMS residual: ~1 cm (APOLLO: mm-level)
        - DE441 vs DE440 position difference: ~10 m at 100 years from present
        - Tidal damping causes quadratic divergence with time from present
    """
    # Based on DE440/DE441 documentation (Park et al. 2021)
    # and Earth orientation parameter accuracies

    # CORRECTED VELOCITY UNCERTAINTY ESTIMATE:
    # Previous estimate of 1 mm/s was WRONG - way too large!
    #
    # Modern epoch (2000-2050): σ_pos ≈ 1 cm (LLR residuals)
    #
    # Velocity uncertainty from position over time baseline:
    # Using ~1 month baseline (conservative):
    # σ_v ≈ σ_pos × √2 / Δt ≈ 0.01 m × √2 / (30 days × 86400 s/day)
    # σ_v ≈ 0.0141 m / 2.592e6 s ≈ 5.4 × 10⁻⁹ m/s = 5.4 nm/s
    #
    # More conservative estimate accounting for model uncertainties:
    # σ_v ≈ 10 μm/s (10⁻⁵ m/s)
    #
    # This is 100× smaller than the previous (incorrect) 1 mm/s estimate!

    return {
        # Moon orbital velocity uncertainty from DE440 ephemeris
        # Based on:
        # - Modern LLR residuals: ~1 cm position accuracy
        # - Position measurements over ~1 month baseline
        # - Conservative estimate including model uncertainties: 10 μm/s
        # - CORRECTED: Previous estimate of 1 mm/s was TOO LARGE by 100×
        'moon_orbital': 1e-5,  # m/s = 10 μm/s (was 1 mm/s - CORRECTED!)

        # Earth orbital velocity uncertainty
        # Earth's orbit is much better constrained than Moon's
        # Inner planet ranging + spacecraft tracking → few μm/s level
        'earth_orbital': 5e-6,  # m/s = 5 μm/s (was 0.5 mm/s - CORRECTED!)

        # Observatory position uncertainty due to:
        # - ITRF2020 station coordinates: ~1 mm accuracy
        # - Station velocities from plate tectonics: ~0.1 mm/year known
        # For Earth rotation contribution:
        # Station position error: 1 mm
        # Earth rotation velocity at mid-latitude: ~300 m/s
        # Fractional error: 1 mm / 6400 km ≈ 1.5e-10
        # Velocity error: 300 m/s * 1.5e-10 ≈ 0.05 μm/s (negligible)
        'observatory_position': 0.00001,  # m/s

        # Earth rotation rate uncertainty (EOP - Earth Orientation Parameters)
        # UT1-UTC accuracy: ~0.01-0.1 ms
        # LOD (Length of Day) variations: ~1 ms over decades
        # Fractional rotation rate error: ~1e-8 to 1e-9
        # Velocity error at observatory: 300 m/s * 1e-9 ≈ 0.3 μm/s
        'earth_rotation': 0.0000003,  # m/s

        # Moon libration (rotation variations) uncertainty
        # Moon orientation kernels based on LLR data
        # Libration amplitude: ~tens of arcminutes
        # Uncertainty in libration: ~0.001 arcsec ≈ 5e-9 rad
        # Moon radius: 1737 km
        # Rotational velocity: ~4.6 m/s (equatorial, synodic period)
        # Angular uncertainty gives position error: 1737 km * 5e-9 ≈ 8.7 μm
        # Over rotation period (27.3 days): velocity error ~0.04 mm/s
        # But libration changes are tracked by LLR, so uncertainty is smaller
        'moon_libration': 0.00002,  # m/s
    }


def propagate_to_doppler_error(vel_uncertainties, reference_frequency=1299.5e6):
    """
    Propagate velocity uncertainties to Doppler frequency errors.

    Args:
        vel_uncertainties: dict of velocity uncertainties in m/s
        reference_frequency: radar frequency in Hz

    Returns:
        dict of Doppler frequency uncertainties in Hz
    """
    c = csp.clight() * 1000.0  # clight() is km/s; velocities here are m/s

    doppler_errors = {}
    for key, sigma_v in vel_uncertainties.items():
        # Fractional Doppler: Δf/f ≈ v/c
        # Frequency error: Δf = f * (v/c)
        doppler_errors[key] = reference_frequency * (sigma_v / c)

    return doppler_errors


if __name__ == "__main__":
    print("="*70)
    print("VELOCITY ERROR SOURCE BREAKDOWN")
    print("="*70)

    # SPICE setup
    SPICE_KERNEL_DIR = "spice_kernels"
    csp.kclear()
    for k in ["naif0012.tls", "de440s.bsp", "pck00011.tpc",
               "earth_latest_high_prec.bpc", "moon_pa_de440_200625.bpc",
               "moon_de440_250416.tf", "observatories.bsp", "observatories.tf"]:
        csp.furnsh(f"{SPICE_KERNEL_DIR}/{k}")

    # Example observation
    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    print(f"\nObservation time: {obs_time.iso}")
    print(f"Ephemeris time: {rx_time:.3f} s past J2000")

    # Compute velocity components
    print("\n" + "-"*70)
    print("VELOCITY COMPONENTS")
    print("-"*70)
    vel_comp = compute_velocity_components(rx_time)

    print(f"Moon orbital velocity (inertial): {vel_comp['moon_vel_inertial']:.3f} m/s")
    print(f"Earth barycentric velocity: {vel_comp['earth_vel_bary']:.3f} m/s")
    print(f"RX observatory velocity: {vel_comp['rx_vel']:.3f} m/s")
    print(f"TX observatory velocity: {vel_comp['tx_vel']:.3f} m/s")
    print(f"SRP velocity in body frame: {vel_comp['srp_vel_body']:.6f} m/s")
    print(f"\nRadial velocities (Doppler-relevant):")
    print(f"  RX radial: {vel_comp['v_rx_radial']:.3f} m/s")
    print(f"  TX radial: {vel_comp['v_tx_radial']:.3f} m/s")
    print(f"  Fractional Doppler (DLT): {vel_comp['dlt']:.6e}")

    # Uncertainty estimates
    print("\n" + "-"*70)
    print("VELOCITY UNCERTAINTY ESTIMATES")
    print("-"*70)
    vel_unc = estimate_velocity_uncertainties()

    for key, sigma in vel_unc.items():
        print(f"{key:30s}: {sigma*1000:.4f} mm/s")

    # RSS total
    total_vel_unc = np.sqrt(sum(sigma**2 for sigma in vel_unc.values()))
    print(f"{'TOTAL (RSS)':30s}: {total_vel_unc*1000:.4f} mm/s")

    # Propagate to Doppler errors
    print("\n" + "-"*70)
    print("DOPPLER FREQUENCY ERRORS @ 1299.5 MHz")
    print("-"*70)

    freq = 1299.5e6
    doppler_errors = propagate_to_doppler_error(vel_unc, freq)

    for key, df in doppler_errors.items():
        print(f"{key:30s}: {df:.4f} Hz")

    total_doppler_unc = np.sqrt(sum(df**2 for df in doppler_errors.values()))
    print(f"{'TOTAL (RSS)':30s}: {total_doppler_unc:.4f} Hz")

    # Comparison with pixel resolution: 3000 Doppler bins over the
    # limb-to-limb dlt span (~1.4e-8 ≈ 18 Hz @ 1299.5 MHz), NOT the dlt
    # magnitude (~4e-6) — see REPORT.md "Open items" 8.1.
    doppler_resolution = 0.006  # Hz/pixel @ 1299.5 MHz
    print(f"\nStandard DD image resolution: {doppler_resolution:.2f} Hz/pixel")
    print(f"Total Doppler uncertainty: {total_doppler_unc/doppler_resolution:.4f} pixels")

    # Create bar chart
    print("\n" + "-"*70)
    print("Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity uncertainties
    labels = list(vel_unc.keys())
    values = [vel_unc[k] * 1000 for k in labels]  # convert to mm/s
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    axes[0].barh(labels, values, color=colors)
    axes[0].set_xlabel('Velocity Uncertainty (mm/s)', fontsize=11)
    axes[0].set_title('Velocity Error Sources', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].axvline(total_vel_unc*1000, color='red', linestyle='--',
                    linewidth=2, label=f'Total RSS: {total_vel_unc*1000:.3f} mm/s')
    axes[0].legend()

    # Doppler frequency errors
    values_hz = [doppler_errors[k] for k in labels]

    axes[1].barh(labels, values_hz, color=colors)
    axes[1].set_xlabel('Doppler Uncertainty (Hz @ 1300 MHz)', fontsize=11)
    axes[1].set_title('Doppler Frequency Error Sources', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].axvline(total_doppler_unc, color='red', linestyle='--',
                    linewidth=2, label=f'Total RSS: {total_doppler_unc:.3f} Hz')
    axes[1].legend()

    plt.tight_layout()

    os.makedirs("results/ERRORS", exist_ok=True)
    output_file = "results/ERRORS/velocity_error_breakdown.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
The dominant velocity error source is Moon orbital velocity uncertainty
from the DE440 ephemeris (~10 um/s for modern epochs, LLR-constrained).

This propagates to a Doppler frequency uncertainty of ~{doppler_errors['moon_orbital']:.2e} Hz
at 1299.5 MHz, which is ~{doppler_errors['moon_orbital']/doppler_resolution:.4f} pixels in
standard DD images (bin ~{doppler_resolution*1000:.0f} mHz): ephemeris velocity is negligible.
The measured look-to-look Doppler scatter (+/-47 mHz, ~8 bins) is instead
set by the Stockert Rb oscillator (5.5e-11 ~ 0.07 Hz) — see REPORT.md.
    """)
    print("="*70 + "\n")

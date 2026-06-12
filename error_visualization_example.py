"""
Example: Doppler Equator Error Visualization

Demonstrates how to visualize uncertainties in bistatic radar doppler equator
calculations due to ephemeris errors and computational approximations.

Usage:
    python error_visualization_example.py
"""

import os
import numpy as np
import cspyce as csp
from astropy import time as at
from matplotlib import pyplot as plt

from doppler_equator_errors import (
    EphemerisUncertainty,
    ComputationalErrors,
    HardwareErrors,
    plot_equator_nominal,
    plot_equator_with_errors,
    plot_error_breakdown,
    compare_ephemeris_quality,
)

# ---------------------------------------------------------------------------
# SPICE setup
# ---------------------------------------------------------------------------
from spice_setup import furnsh_kernels
furnsh_kernels()


# ---------------------------------------------------------------------------
# Example 1: Simple nominal doppler equator plot
# ---------------------------------------------------------------------------
def example_nominal_plot():
    """Plot doppler equator curves without uncertainty bounds."""
    print("\n" + "="*70)
    print("Example 1: Nominal Doppler Equator (No Uncertainty)")
    print("="*70)

    # Use example observation time
    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    print(f"Observation time: {obs_time.iso}")
    print(f"Ephemeris time: {rx_time:.3f} s past J2000")

    # Create simple plot
    fig, ax = plot_equator_nominal(rx_time)

    # Save
    os.makedirs("results/ERRORS", exist_ok=True)
    fig.savefig("results/ERRORS/doppler_equator_nominal.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/doppler_equator_nominal.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example 2: Doppler equator with uncertainty bounds
# ---------------------------------------------------------------------------
def example_with_uncertainty():
    """Plot doppler equator with uncertainty bounds for a single observation."""
    print("\n" + "="*70)
    print("Example 2: Doppler Equator with Uncertainty Bounds")
    print("="*70)

    # Use example observation time
    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    print(f"Observation time: {obs_time.iso}")
    print(f"Ephemeris time: {rx_time:.3f} s past J2000")

    # Create error models
    ephem_unc = EphemerisUncertainty()
    comp_err = ComputationalErrors()
    hw_err = HardwareErrors()

    # Print error estimates
    sigma_pos = ephem_unc.position_uncertainty(rx_time)
    sigma_vel = ephem_unc.velocity_uncertainty(rx_time)
    print(f"\nEphemeris uncertainties:")
    print(f"  Position: {sigma_pos*100:.2f} cm")
    print(f"  Velocity: {sigma_vel*1000:.2f} mm/s")

    # Calculate expected DLT uncertainty
    c_m_s = csp.clight() * 1000.0
    dlt_unc_approx = sigma_vel / c_m_s
    print(f"\nExpected DLT uncertainty:")
    print(f"  ~{dlt_unc_approx:.3e} (fractional)")
    print(f"  ~{dlt_unc_approx*1e12:.2f} parts per trillion")
    print(f"\n** NOTE: These uncertainties are EXTREMELY SMALL! **")
    print(f"   Scaling by 10^6 for visualization...")

    # Create plot with realistic uncertainties (scaled for visibility)
    fig, ax = plot_equator_with_errors(
        rx_time,
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err,
        n_sigma=3,
        include_model_errors=False,  # Only ephemeris uncertainties
    )

    # Save
    os.makedirs("results/ERRORS", exist_ok=True)
    fig.savefig("results/ERRORS/equator_measurement_uncertainty.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/equator_measurement_uncertainty.png")
    print(f"   -> RELEVANCE: This plot visualizes RANDOM measurement uncertainty")
    print(f"      (due to SDR hardware variations, clock stability, and ephemeris).")
    print(f"      This represents the fundamental limit / 'blurriness' of our tracking lines.")
    plt.close(fig)

    # Also create version with model errors included
    print(f"\nCreating plot with systematic model errors included...")
    fig2, ax2 = plot_equator_with_errors(
        rx_time,
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err,
        n_sigma=1,  # Use 1-sigma for model errors (they're large!)
        include_model_errors=True,  # Include systematic model biases
        scale_factor=1.0  # No scaling needed - model errors are large
    )

    fig2.savefig("results/ERRORS/equator_systematic_bias.png", dpi=150, bbox_inches='tight')
    print(f"Saved: results/ERRORS/equator_systematic_bias.png")
    print(f"   (Includes ellipsoid approx ±4km, SRP averaging ±50m)")
    print(f"   -> RELEVANCE: This plot visualizes SYSTEMATIC geometric biases.")
    print(f"      Unlike random 'blurriness', assuming the moon is a perfect ellipsoid")
    print(f"      shifts the entire measurement line by several kilometers.")
    print(f"      This bound must be fixed via 3D DEM (LOLA) mapping.")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Example 2: Error source breakdown
# ---------------------------------------------------------------------------
def example_error_breakdown():
    """Visualize contribution of different error sources."""
    print("\n" + "="*70)
    print("Example 2: Error Source Breakdown")
    print("="*70)

    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    ephem_unc = EphemerisUncertainty()
    comp_err = ComputationalErrors()
    hw_err = HardwareErrors()

    # Print individual error sources
    print(f"\nDelay error sources (two-way light time):")
    c_m_s = csp.clight() * 1000.0
    sigma_pos = ephem_unc.position_uncertainty(rx_time)
    print(f"  Ephemeris position: {2*sigma_pos/c_m_s * 1e9:.3f} ns")
    print(f"  Dwingeloo TX/RX Pipeline: {hw_err.pipeline_delay('DWINGELOO_TX_RX_OFFSET') * 1e9:.3f} ns")
    print(f"  PPS Sampling Ambiguity: {hw_err.pipeline_delay('PPS_AMBIGUITY_250KHZ') * 1e9:.3f} ns")
    print(f"  Stockert GPS Cable Offset: {hw_err.pipeline_delay('STOCKERT_GPS_OFFSET') * 1e9:.3f} ns")
    print(f"  Light-time iteration: {comp_err.light_time_iteration_error() * 1e9:.3f} ns")
    print(f"  SRP averaging: {2*comp_err.srp_averaging_error()/c_m_s * 1e9:.3f} ns")
    print(f"  Ellipsoid approx: {2*comp_err.ellipsoid_approximation_error()/c_m_s * 1e9:.3f} ns")

    print(f"\nDoppler shift error sources:")
    sigma_vel = ephem_unc.velocity_uncertainty(rx_time)
    print(f"  Ephemeris velocity: {sigma_vel/c_m_s * 1e12:.3f} × 10⁻¹²")
    print(f"  Finite difference: {comp_err.finite_difference_error()/c_m_s * 1e12:.3f} × 10⁻¹²")
    print(f"  Stockert Rubidium Stability: {hw_err.oscillator_stability('STOCKERT_RUBIDIUM') * 1e12:.3f} × 10⁻¹²")
    print(f"  Dwingeloo H-Maser Stability: {hw_err.oscillator_stability('DWINGELOO_HMASER') * 1e12:.3f} × 10⁻¹²")

    # Create plot
    fig, axes = plot_error_breakdown(
        rx_time,
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err
    )

    # Save
    fig.savefig("results/ERRORS/error_breakdown.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/error_breakdown.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example 3: Ephemeris quality over time
# ---------------------------------------------------------------------------
def example_ephemeris_quality_over_time():
    """Show how ephemeris uncertainty grows with time from present."""
    print("\n" + "="*70)
    print("Example 3: Ephemeris Quality vs Time from Present")
    print("="*70)

    # Reference epoch (approximately now)
    ref_time = at.Time("2025-01-01T00:00:00")
    reference_et = csp.str2et(ref_time.utc.value)

    # Time range: -100 to +100 years from reference
    years = np.linspace(-100, 100, 201)
    rx_times = reference_et + years * 365.25 * 86400.0

    print(f"Reference epoch: {ref_time.iso}")
    print(f"Time range: {years[0]:.0f} to {years[-1]:.0f} years from reference")

    # Create plot
    fig, axes = compare_ephemeris_quality(
        rx_times,
        reference_et=reference_et
    )

    # Save
    fig.savefig("results/ERRORS/ephemeris_quality_vs_time.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/ephemeris_quality_vs_time.png")
    plt.close(fig)

    # Print some key values
    ephem = EphemerisUncertainty()
    print(f"\nKey uncertainty values:")
    print(f"  Modern (within 50 yr): {ephem.position_uncertainty(reference_et)*100:.2f} cm")
    print(f"  At +50 years: {ephem.position_uncertainty(reference_et + 50*365.25*86400)*100:.2f} cm")
    print(f"  At +100 years: {ephem.position_uncertainty(reference_et + 100*365.25*86400):.2f} m")


# ---------------------------------------------------------------------------
# Example 4: Sensitivity analysis - varying uncertainty parameters
# ---------------------------------------------------------------------------
def example_sensitivity_analysis():
    """Show how results change with different uncertainty assumptions."""
    print("\n" + "="*70)
    print("Example 4: Sensitivity to Uncertainty Parameters")
    print("="*70)

    obs_time = at.Time("2025-09-16T13:22:02")
    rx_time = csp.str2et(obs_time.utc.value)

    # Create scenarios with different uncertainty levels
    scenarios = {
        'Optimistic (1 cm)': EphemerisUncertainty(sigma_pos_modern=0.01),
        'Nominal (2 cm)': EphemerisUncertainty(sigma_pos_modern=0.02),
        'Conservative (5 cm)': EphemerisUncertainty(sigma_pos_modern=0.05),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (label, ephem_unc) in zip(axes, scenarios.items()):
        # Import here to avoid circular dependency
        from doppler_equator_errors import compute_equator_with_uncertainty

        (lt_min, delay_up, dlt_up, delay_down, dlt_down,
         delay_up_std, dlt_up_std, delay_down_std, dlt_down_std) = \
            compute_equator_with_uncertainty(rx_time, ephem_uncertainty=ephem_unc)

        # Plot
        ax.plot(dlt_up, delay_up, 'b-', linewidth=2, label='Up-Doppler')
        ax.plot(dlt_down, delay_down, 'r-', linewidth=2, label='Down-Doppler')

        ax.fill_betweenx(delay_up,
                         dlt_up - 3*dlt_up_std,
                         dlt_up + 3*dlt_up_std,
                         alpha=0.3, color='blue')
        ax.fill_betweenx(delay_down,
                         dlt_down - 3*dlt_down_std,
                         dlt_down + 3*dlt_down_std,
                         alpha=0.3, color='red')

        ax.set_xlabel('Fractional Doppler Shift', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        if ax == axes[0]:
            ax.set_ylabel('Delay (s)', fontsize=11)
            ax.legend(fontsize=9)

    plt.suptitle('Sensitivity to Position Uncertainty (±3σ bounds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig("results/ERRORS/sensitivity_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/sensitivity_analysis.png")
    plt.close(fig)

    print("\nScenarios compared:")
    for label, ephem_unc in scenarios.items():
        sigma_pos = ephem_unc.position_uncertainty(rx_time)
        print(f"  {label}: σ_pos = {sigma_pos*100:.1f} cm")


# ---------------------------------------------------------------------------
# Example 5: Multiple observations comparison
# ---------------------------------------------------------------------------
def example_multiple_observations():
    """Compare uncertainty for observations at different times."""
    print("\n" + "="*70)
    print("Example 5: Uncertainty Across Multiple Observations")
    print("="*70)

    # Example observation times spanning several months
    obs_times = [
        at.Time("2025-06-15T12:00:00"),
        at.Time("2025-09-16T13:22:02"),
        at.Time("2025-12-20T08:30:00"),
    ]

    ephem_unc = EphemerisUncertainty()

    print("\nObservation times and uncertainties:")
    for obs_time in obs_times:
        rx_time = csp.str2et(obs_time.utc.value)
        sigma_pos = ephem_unc.position_uncertainty(rx_time)
        sigma_vel = ephem_unc.velocity_uncertainty(rx_time)

        print(f"\n  {obs_time.iso}")
        print(f"    Position unc: {sigma_pos*100:.2f} cm")
        print(f"    Velocity unc: {sigma_vel*1000:.2f} mm/s")

    print("\nFor modern observations (within ~50 years of present),")
    print("uncertainties are dominated by LLR measurement accuracy (~1-2 cm)")
    print("and remain relatively constant.")


# ---------------------------------------------------------------------------
# Main: Run all examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOPPLER EQUATOR ERROR VISUALIZATION EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various error analysis and visualization")
    print("capabilities for bistatic radar doppler equator calculations.")

    # Run examples
    example_nominal_plot()  # Simple plot without uncertainties
    example_with_uncertainty()  # With uncertainty bounds
    example_error_breakdown()
    example_ephemeris_quality_over_time()
    example_sensitivity_analysis()
    example_multiple_observations()

    print("\n" + "="*70)
    print("All examples completed! Check results/ERRORS/ for output plots.")
    print("="*70 + "\n")

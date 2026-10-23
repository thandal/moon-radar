"""
Example: Doppler Equator Error Visualization

Demonstrates how to visualize uncertainties in bistatic radar doppler equator
calculations due to ephemeris errors and computational approximations. The
values are doppler_equator_errors' stated per-source figures, not measured
uncertainties of processed maps (REPORT.md section 6 has those).

Usage:
    python error_visualization_example.py
"""

import os
import cspyce as csp
from astropy import time as at
from matplotlib import pyplot as plt

from doppler_equator import specular_point_bck
from doppler_equator_errors import (
    EphemerisUncertainty,
    ComputationalErrors,
    HardwareErrors,
    plot_equator_nominal,
    plot_equator_with_errors,
    plot_error_breakdown,
    compute_equator_with_uncertainty,
    ephemeris_position_dlt,
    ephemeris_velocity_dlt,
)

# ---------------------------------------------------------------------------
# SPICE setup
# ---------------------------------------------------------------------------
from spice_setup import furnsh_kernels
furnsh_kernels()

# The example epoch, 2025-09-16T13:22:02, is in the 2025-09-16 session:
# Dwingeloo transmitting, Stockert receiving. The hardware terms are those
# two stations' entries in doppler_equator_errors.HardwareErrors; the
# stations time-tag their samples from GPS (REPORT.md 3.1: a one-sample PPS
# ambiguity and a GPS cable-delay difference).
TX_NAME, RX_NAME = "DWINGELOO", "STOCKERT"
OSCILLATOR_SOURCE = "STOCKERT_RUBIDIUM"
TIMING_SOURCE = "GPS"
STATION_KW = dict(tx_name=TX_NAME, rx_name=RX_NAME,
                  oscillator_source=OSCILLATOR_SOURCE)


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
    fig, ax = plot_equator_nominal(rx_time, tx_name=TX_NAME, rx_name=RX_NAME)

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
    sigma_pos = ephem_unc.position_uncertainty()
    sigma_vel = ephem_unc.velocity_uncertainty()
    print(f"\nEphemeris uncertainties (constant model, no epoch dependence):")
    print(f"  Position: {sigma_pos*100:.2f} cm")
    print(f"  Velocity: {sigma_vel*1e6:.1f} um/s")

    # Calculate expected DLT uncertainty
    c_m_s = csp.clight() * 1000.0
    dlt_unc_approx = ephemeris_velocity_dlt(sigma_vel)
    print(f"\nExpected DLT uncertainty from the ephemeris velocity alone:")
    print(f"  ~{dlt_unc_approx:.3e} (fractional)")
    print(f"  ~{dlt_unc_approx*1e12:.2f} parts per trillion")
    print(f"Drawn: a Doppler band (ephemeris at the SRP + the {OSCILLATOR_SOURCE} "
          f"oscillator, {hw_err.oscillator_stability(OSCILLATOR_SOURCE):.1e}, against "
          f"the {dlt_unc_approx:.1e} above) and a delay band (ephemeris position + "
          f"light-time iteration, {comp_err.light_time_iteration_error() * 1e9:.0f} ns: "
          "invisible at this scale).")
    print("Not drawn: fixed station delays such as the "
          f"{hw_err.pipeline_delay('DWINGELOO_TX_RX_OFFSET') * 1e6:.0f} us Dwingeloo "
          "TX/RX offset -- the per-look timing measurement absorbs them.")

    # Bounds at their computed size (no scaling).
    fig, ax = plot_equator_with_errors(
        rx_time,
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err,
        n_sigma=3,
        include_model_errors=False,  # ephemeris, oscillator, iteration only
        **STATION_KW,
    )

    # Save
    os.makedirs("results/ERRORS", exist_ok=True)
    fig.savefig("results/ERRORS/equator_measurement_uncertainty.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/equator_measurement_uncertainty.png")
    print("   -> These are this module's stated per-source values at the SRP,")
    print("      not a measured uncertainty of a processed map (REPORT.md section 6),")
    print("      and the per-look frequency calibration removes the part of the")
    print("      oscillator term that is constant over a look.")
    plt.close(fig)

    # Also create version with model errors included
    print(f"\nCreating plot with systematic model errors included...")
    fig2, ax2 = plot_equator_with_errors(
        rx_time,
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err,
        n_sigma=1,  # 1-sigma: the model terms are large
        include_model_errors=True,  # adds the systematic terms to the delay band
        scale_factor=1.0,  # No scaling needed - model errors are large
        **STATION_KW,
    )

    fig2.savefig("results/ERRORS/equator_systematic_bias.png", dpi=150, bbox_inches='tight')
    print(f"Saved: results/ERRORS/equator_systematic_bias.png")
    print(f"   (delay band adds the ellipsoid ±{comp_err.ellipsoid_approximation_error() / 1e3:g} km "
          f"and SRP averaging {comp_err.srp_averaging_error():g} m terms: "
          f"2 x {comp_err.ellipsoid_approximation_error() / 1e3:g} km / c = "
          f"{2 * comp_err.ellipsoid_approximation_error() / c_m_s * 1e6:.0f} us; "
          "the Doppler band is unchanged)")
    print("   -> RELEVANCE: a systematic geometric bias of the smooth-ellipsoid")
    print("      curves drawn here. Production maps project on the LOLA terrain")
    print("      (lunar_projection, use_dem=True), which takes this term out of the maps.")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Example 3: Error source breakdown
# ---------------------------------------------------------------------------
def example_error_breakdown():
    """Visualize contribution of different error sources."""
    print("\n" + "="*70)
    print("Example 3: Error Source Breakdown")
    print("="*70)

    ephem_unc = EphemerisUncertainty()
    comp_err = ComputationalErrors()
    hw_err = HardwareErrors()

    # Print individual error sources
    print(f"\nDelay error sources (two-way light time):")
    c_m_s = csp.clight() * 1000.0
    sigma_pos = ephem_unc.position_uncertainty()
    print(f"  Ephemeris position: {2*sigma_pos/c_m_s * 1e9:.3f} ns")
    print(f"  Light-time iteration: {comp_err.light_time_iteration_error() * 1e9:.3f} ns")
    print(f"  SRP averaging: {2*comp_err.srp_averaging_error()/c_m_s * 1e9:.3f} ns")
    print(f"  Ellipsoid approx: {2*comp_err.ellipsoid_approximation_error()/c_m_s * 1e9:.3f} ns")
    print("Fixed station delays (absorbed by the per-look timing measurement, "
          "not an uncertainty of the delay axis):")
    print(f"  Dwingeloo TX/RX Pipeline: {hw_err.pipeline_delay('DWINGELOO_TX_RX_OFFSET') * 1e9:.3f} ns")
    print(f"  PPS Sampling Ambiguity: {hw_err.pipeline_delay('PPS_AMBIGUITY_250KHZ') * 1e9:.3f} ns")
    print(f"  Stockert GPS Cable Offset: {hw_err.pipeline_delay('STOCKERT_GPS_OFFSET') * 1e9:.3f} ns")

    print(f"\nDoppler shift error sources:")
    sigma_vel = ephem_unc.velocity_uncertainty()
    print(f"  Ephemeris velocity: {ephemeris_velocity_dlt(sigma_vel) * 1e12:.3f} × 10⁻¹²")
    print(f"  Finite difference: {comp_err.finite_difference_error()/c_m_s * 1e12:.3f} × 10⁻¹²")
    print(f"  Stockert Rubidium Stability: {hw_err.oscillator_stability('STOCKERT_RUBIDIUM') * 1e12:.3f} × 10⁻¹²")
    print(f"  Dwingeloo H-Maser Stability: {hw_err.oscillator_stability('DWINGELOO_HMASER') * 1e12:.3f} × 10⁻¹²")

    # Create plot
    fig, axes = plot_error_breakdown(
        ephem_uncertainty=ephem_unc,
        computational_errors=comp_err,
        hardware_errors=hw_err,
        timing_source=TIMING_SOURCE,
    )

    # Save
    fig.savefig("results/ERRORS/error_breakdown.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/ERRORS/error_breakdown.png")
    plt.close(fig)


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
        'Optimistic (1 cm)': EphemerisUncertainty(sigma_pos=0.01),
        'Nominal (2 cm)': EphemerisUncertainty(sigma_pos=0.02),
        'Conservative (5 cm)': EphemerisUncertainty(sigma_pos=0.05),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    dlt_std_of, delay_std_of = {}, {}
    for ax, (label, ephem_unc) in zip(axes, scenarios.items()):
        (lt_min, delay_up, dlt_up, delay_down, dlt_down,
         delay_up_std, dlt_up_std, delay_down_std, dlt_down_std) = \
            compute_equator_with_uncertainty(rx_time, ephem_uncertainty=ephem_unc,
                                             **STATION_KW)
        dlt_std_of[label] = dlt_up_std[0]
        delay_std_of[label] = delay_up_std[0]

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

    srp = specular_point_bck(rx_time, TX_NAME, RX_NAME)
    print("\nScenarios compared (DLT terms at the SRP of specular_point_bck):")
    for label, ephem_unc in scenarios.items():
        sigma_pos = ephem_unc.position_uncertainty()
        pos_term = ephemeris_position_dlt(rx_time, srp, sigma_pos,
                                          tx_name=TX_NAME, rx_name=RX_NAME)
        print(f"  {label}: σ_pos = {sigma_pos*100:.1f} cm; DLT terms: position "
              f"{pos_term:.2e}, velocity {ephemeris_velocity_dlt(ephem_unc.velocity_uncertainty()):.2e}; "
              f"plotted DLT σ {dlt_std_of[label]:.9e}; "
              f"delay σ {delay_std_of[label] * 1e9:.3f} ns")
    print("  The three panels look the same, for a physical reason: the Moon's position")
    print("  enters the Doppler only by turning the lines of sight (ephemeris_position_dlt),")
    print("  a term far below the ephemeris velocity term and the "
          f"{OSCILLATOR_SOURCE} oscillator's")
    print(f"  {HardwareErrors.oscillator_stability(OSCILLATOR_SOURCE):.1e}, which sets the "
          "band. The position uncertainty does change")
    print("  the delay σ (2 σ_pos / c, above), which these panels do not draw.")


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
    example_sensitivity_analysis()

    print("\n" + "="*70)
    print("All examples completed! Check results/ERRORS/ for output plots.")
    print("="*70 + "\n")

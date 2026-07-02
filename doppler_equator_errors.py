"""
Doppler Equator Error Analysis

Quantifies and visualizes errors in bistatic radar Doppler equator calculations
arising from:
1. Ephemeris uncertainties in SPICE kernels (DE440/DE441)
2. Computational simplifications and approximations
3. Light-time iteration convergence
4. Finite difference approximations

Based on DE440/DE441 documentation:
- Modern lunar positions: ~1-2 cm accuracy (from LLR residuals)
- DE441 vs DE440 difference: ~10 m at 100 years from present
- Position uncertainty grows quadratically with time from present epoch

References:
- Park et al. 2021, The JPL Planetary and Lunar Ephemerides DE440 and DE441
  https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf
- SPICE Toolkit: https://naif.jpl.nasa.gov/naif/toolkit.html
"""

import numpy as np
import cspyce as csp
from astropy import units as au
from matplotlib import pyplot as plt

from doppler_equator import (
    moonPointDLT_BCK,
    moonSRP_DLT_BCK,
    compute_doppler_equator_velocity,
    AB_COR,
    EARTH_FRAME,
)


# ---------------------------------------------------------------------------
# Error source definitions
# ---------------------------------------------------------------------------

class EphemerisUncertainty:
    """
    Ephemeris uncertainty model based on DE440/DE441 characteristics.

    For modern epoch (within ~50 years of 2020):
        - Position uncertainty: ~1-2 cm (LLR residuals)
        - Velocity uncertainty: ~1-2 mm/s (estimated from position derivatives)

    For times far from present:
        - Uncertainty grows quadratically: ~10 m at 100 years
        - Position error ~ sigma_0 * (1 + (t/t_ref)^2)
    """

    def __init__(self,
                 sigma_pos_modern=0.01,      # meters, modern LLR accuracy (~1 cm)
                 sigma_vel_modern=1e-5,      # m/s, conservative estimate (10 μm/s)
                 t_ref_years=100,            # reference time for quadratic growth
                 sigma_pos_century=10.0):    # meters at t_ref
        """
        Initialize ephemeris uncertainty model.

        Args:
            sigma_pos_modern: Position uncertainty at present epoch (meters)
                            Default: 1 cm (modern LLR weighted RMS residual)
            sigma_vel_modern: Velocity uncertainty at present epoch (m/s)
                            Default: 10 μm/s (conservative estimate from LLR constraints)
                            Note: True value likely < 10 μm/s for modern epochs
            t_ref_years: Reference time for quadratic scaling (years)
            sigma_pos_century: Position uncertainty at t_ref (meters)
                             Based on DE441-DE440 difference (~10 m at 100 years)
        """
        self.sigma_pos_modern = sigma_pos_modern
        self.sigma_vel_modern = sigma_vel_modern
        self.t_ref_years = t_ref_years
        self.sigma_pos_century = sigma_pos_century

        # Compute quadratic growth coefficient
        # sigma(t) = sigma_0 * sqrt(1 + (t/t_ref)^2 * k)
        # At t=t_ref: sigma(t_ref) = sigma_0 * sqrt(1 + k)
        # Solve: (sigma_century / sigma_modern)^2 = 1 + k
        self.k = (sigma_pos_century / sigma_pos_modern)**2 - 1

    def position_uncertainty(self, et_time, reference_et=None):
        """
        Compute position uncertainty at given ephemeris time.

        Args:
            et_time: Ephemeris time (seconds past J2000)
            reference_et: Reference epoch for modern accuracy (default: J2000)

        Returns:
            Position uncertainty in meters
        """
        if reference_et is None:
            reference_et = 0.0  # J2000 epoch

        # Time difference in years
        dt_years = (et_time - reference_et) / (365.25 * 86400.0)

        # Quadratic growth model
        scale_factor = np.sqrt(1 + self.k * (dt_years / self.t_ref_years)**2)
        return self.sigma_pos_modern * scale_factor

    def velocity_uncertainty(self, et_time, reference_et=None):
        """
        Compute velocity uncertainty at given ephemeris time.

        Args:
            et_time: Ephemeris time (seconds past J2000)
            reference_et: Reference epoch for modern accuracy (default: J2000)

        Returns:
            Velocity uncertainty in m/s
        """
        if reference_et is None:
            reference_et = 0.0

        dt_years = (et_time - reference_et) / (365.25 * 86400.0)
        scale_factor = np.sqrt(1 + self.k * (dt_years / self.t_ref_years)**2)
        return self.sigma_vel_modern * scale_factor


class ComputationalErrors:
    """
    Computational approximation errors in doppler equator calculations.
    """

    @staticmethod
    def finite_difference_error(dt=1.0, order=1):
        """
        Estimate error from finite difference velocity approximation.

        Args:
            dt: Time step for finite differences (seconds)
            order: Order of the finite difference scheme (1 or 2)

        Returns:
            Fractional error estimate
        """
        # For first-order FD: error ~ O(dt * acceleration)
        # Moon orbital acceleration ~ v^2/r ~ (1022 m/s)^2 / 384400 km ~ 0.0027 m/s^2
        moon_accel = 0.0027  # m/s^2

        if order == 1:
            # Truncation error: (dt/2) * d^2x/dt^2
            return dt * moon_accel / 2.0  # m/s error
        else:
            # Second-order: (dt^2/6) * d^3x/dt^3
            # Jerk ~ 1e-9 m/s^3 (estimated)
            moon_jerk = 1e-9
            return (dt**2 / 6.0) * moon_jerk

    @staticmethod
    def ellipsoid_approximation_error():
        """
        Error from using ellipsoid instead of detailed shape model.

        The Moon is approximated as an ellipsoid, but has topographic
        variations up to ~8 km peak-to-peak.

        Returns:
            Maximum position error in meters
        """
        # Lunar topography range: ±4 km from mean radius
        return 4000.0  # meters

    @staticmethod
    def light_time_iteration_error(n_iterations=2, tolerance=1e-6):
        """
        Error from light-time iteration convergence.

        SPICE typically converges to sub-millimeter accuracy in 2-3 iterations.

        Args:
            n_iterations: Number of iterations used
            tolerance: Convergence tolerance (seconds)

        Returns:
            Light time error in seconds
        """
        # Typical convergence: ~1e-9 s after 2 iterations
        if n_iterations >= 2:
            return 1e-9  # seconds
        else:
            return tolerance

    @staticmethod
    def srp_averaging_error(tx_name="DWINGELOO", rx_name="STOCKERT"):
        """
        Error from averaging TX and RX sub-radar points.

        For bistatic radar, we approximate the specular point as the
        average of TX and RX sub-points. The true bistatic reflection
        point can differ by up to ~tens of meters.

        Returns:
            Position error in meters
        """
        # Estimated from geometry: for Earth-Moon-Earth, separation ~few km
        # Bistatic error: ~1-50 m depending on bistatic angle
        return 50.0  # meters (conservative estimate)

    @staticmethod
    def timing_error(timing_source="GPS"):
        """
        Clock timing uncertainty.

        Args:
            timing_source: "GPS", "NTP", or "atomic"

        Returns:
            Timing uncertainty in seconds

        Notes:
            - GPS timing: ~10-100 ns (with good reception)
            - NTP over internet: ~1-100 ms (variable)
            - Atomic clock: ~1-10 ns (disciplined)
            - Free-running crystal: ~10-100 ppm (10-100 μs over 1 sec)

            For radar observations, typical uncertainties:
            - Best case (GPS + atomic): 10 ns
            - Typical (GPS): 100 ns
            - Poor (NTP): 1-10 ms
        """
        timing_uncertainties = {
            "atomic": 10e-9,   # 10 ns
            "GPS": 100e-9,     # 100 ns
            "NTP": 1e-3,       # 1 ms (conservative)
        }
        return timing_uncertainties.get(timing_source, 100e-9)


class HardwareErrors:
    """
    Hardware-specific errors, particularly for the Ettus USRP B210 SDR.
    """

    @staticmethod
    def pipeline_delay(source="DWINGELOO_TX_RX_OFFSET"):
        """
        Delay in the SDR TX/RX pipeline.
        USRP B210 via USB 3.0 typical latency is ~100 microseconds.
        Dwingeloo TX/RX auto-correlation offset: ~30 microseconds
        PPS sampling ambiguity (250 kHz): ~4 microseconds
        Stockert GPS cable offset: 735 ns
        Returns: Light time offset in seconds.
        """
        delays = {
            "DWINGELOO_TX_RX_OFFSET": 3.0e-5,
            "PPS_AMBIGUITY_250KHZ": 4.0e-6,
            "STOCKERT_GPS_OFFSET": 7.35e-7,
            "USRP_B210_GENERIC": 1e-4,
        }
        return delays.get(source, 0.0)

    @staticmethod
    def oscillator_stability(source="STOCKERT_RUBIDIUM"):
        r"""
        Oscillator frequency stability (fractional offset \Delta f / f).
        USRP B210 standard TCXO: +/- 2.0 ppm
        USRP B210 with GPSDO: < 1.0 ppb
        Dwingeloo H-Maser: ~1e-13
        Stockert Rubidium: ~5.5e-11
        """
        stabilities = {
            "DWINGELOO_HMASER": 1.0e-13,
            "STOCKERT_RUBIDIUM": 5.5e-11,
            "USRP_B210_TCXO": 2.0e-6,
            "USRP_B210_GPSDO": 1.0e-9,
        }
        return stabilities.get(source, 5.5e-11)


# ---------------------------------------------------------------------------
# Error propagation functions
# ---------------------------------------------------------------------------

def compute_dlt_uncertainty(rx_time, p_moon,
                           ephem_uncertainty=None,
                           tx_name="DWINGELOO",
                           rx_name="STOCKERT"):
    """
    Compute uncertainty in DLT (fractional Doppler) due to position/velocity errors.

    Uses linear sensitivity analysis:
        δ(DLT) ≈ |∂DLT/∂r_moon| * δr + |∂DLT/∂v_moon| * δv

    Args:
        rx_time: RX ephemeris time
        p_moon: Moon surface point(s) in Moon body-fixed frame
        ephem_uncertainty: EphemerisUncertainty instance
        tx_name: TX station name
        rx_name: RX station name

    Returns:
        dlt_uncertainty: Uncertainty in DLT (fractional Doppler shift)
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()

    # Nominal computation
    lt_nom, dlt_nom = moonPointDLT_BCK(rx_time, p_moon, tx_name, rx_name)

    # Position perturbation (use ephemeris uncertainty)
    sigma_pos = ephem_uncertainty.position_uncertainty(rx_time)

    # Compute DLT sensitivity to position by finite differences
    # Perturb in radial direction
    if p_moon.ndim == 1:
        p_perturbed = p_moon * (1 + sigma_pos / np.linalg.norm(p_moon))
        lt_pert, dlt_pert = moonPointDLT_BCK(rx_time, p_perturbed, tx_name, rx_name)
        dlt_sensitivity_pos = np.abs(dlt_pert - dlt_nom) / sigma_pos
    else:
        # Handle multiple points
        norms = np.linalg.norm(p_moon, axis=1, keepdims=True)
        p_perturbed = p_moon * (1 + sigma_pos / norms)
        lt_pert, dlt_pert = moonPointDLT_BCK(rx_time, p_perturbed, tx_name, rx_name)
        dlt_sensitivity_pos = np.abs(dlt_pert - dlt_nom) / sigma_pos

    # Velocity uncertainty contribution
    # DLT = 1 - sqrt((1-v_rx/c)/(1+v_rx/c)) * sqrt((1-v_tx/c)/(1+v_tx/c))
    # For small v/c: DLT ≈ (v_rx + v_tx)/c
    # Sensitivity: ∂DLT/∂v ≈ 1/c ≈ 3.3e-9 s/m
    c_m_s = csp.clight() * 1000.0  # c in m/s
    sigma_vel = ephem_uncertainty.velocity_uncertainty(rx_time)
    dlt_sensitivity_vel = 1.0 / c_m_s

    # Total uncertainty (RSS combination)
    dlt_uncertainty = np.sqrt(
        (dlt_sensitivity_pos * sigma_pos)**2 +
        (dlt_sensitivity_vel * sigma_vel)**2
    )

    return dlt_uncertainty


def compute_delay_uncertainty(rx_time, p_moon,
                              ephem_uncertainty=None,
                              computational_errors=None,
                              hardware_errors=None,
                              include_model_errors=False,
                              tx_name="DWINGELOO",
                              rx_name="STOCKERT"):
    """
    Compute uncertainty in delay (light time) due to ephemeris errors.

    Args:
        rx_time: RX ephemeris time
        p_moon: Moon surface point(s)
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance (for model errors)
        include_model_errors: If True, include systematic model errors
                            (ellipsoid approx, SRP averaging). Default False.
        tx_name: TX station name
        rx_name: RX station name

    Returns:
        delay_uncertainty: Uncertainty in delay (seconds)

    Note:
        By default, only ephemeris uncertainties are included (~0.1 ns).
        Model errors like ellipsoid approximation are systematic biases,
        not random uncertainties, so they are excluded by default.
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()
    if computational_errors is None:
        computational_errors = ComputationalErrors()
    if hardware_errors is None:
        hardware_errors = HardwareErrors()

    c_m_s = csp.clight() * 1000.0

    # Ephemeris position uncertainty contribution
    sigma_pos = ephem_uncertainty.position_uncertainty(rx_time)
    # Two-way light time: uncertainty ~ 2 * sigma_pos / c
    lt_uncertainty_ephem = 2.0 * sigma_pos / c_m_s

    # Light-time iteration error
    lt_uncertainty_iter = computational_errors.light_time_iteration_error()

    if include_model_errors:
        # These are systematic biases, not random uncertainties
        # Include only if specifically requested
        srp_error = computational_errors.srp_averaging_error()
        lt_uncertainty_srp = 2.0 * srp_error / c_m_s

        ellipsoid_error = computational_errors.ellipsoid_approximation_error()
        lt_uncertainty_ellipsoid = 2.0 * ellipsoid_error / c_m_s

        # Total uncertainty (RSS combination)
        delay_uncertainty = np.sqrt(
            lt_uncertainty_ephem**2 +
            lt_uncertainty_iter**2 +
            lt_uncertainty_srp**2 +
            lt_uncertainty_ellipsoid**2 +
            hardware_errors.pipeline_delay()**2
        )
    else:
        # Only random uncertainties
        delay_uncertainty = np.sqrt(
            lt_uncertainty_ephem**2 +
            lt_uncertainty_iter**2 +
            hardware_errors.pipeline_delay()**2
        )

    return delay_uncertainty


def compute_equator_with_uncertainty(rx_time, n_points=500,
                                     n_samples=100,
                                     ephem_uncertainty=None,
                                     computational_errors=None,
                                     hardware_errors=None,
                                     include_model_errors=False,
                                     tx_name="DWINGELOO",
                                     rx_name="STOCKERT"):
    """
    Compute doppler equator with uncertainty bounds using Monte Carlo sampling.

    Args:
        rx_time: RX ephemeris time
        n_points: Number of points along equator curves
        n_samples: Number of Monte Carlo samples for uncertainty
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        tx_name: TX station name
        rx_name: RX station name

    Returns:
        lt_min: Minimum light time
        delay_up: Delay values for up_doppler branch
        dlt_up: DLT values for up_doppler branch
        delay_down: Delay values for down_doppler branch
        dlt_down: DLT values for down_doppler branch
        delay_up_std: Standard deviation in delay (up branch)
        dlt_up_std: Standard deviation in DLT (up branch)
        delay_down_std: Standard deviation in delay (down branch)
        dlt_down_std: Standard deviation in DLT (down branch)
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()
    if computational_errors is None:
        computational_errors = ComputationalErrors()
    if hardware_errors is None:
        hardware_errors = HardwareErrors()

    # Nominal computation
    lt_min, delay_up, dlt_up, delay_down, dlt_down = \
        compute_doppler_equator_velocity(rx_time, n_points, tx_name, rx_name)

    # Get uncertainty estimates for each point
    moon_radii = csp.bodvrd("MOON", "RADII")

    # Reconstruct surface points from the velocity method
    # NOTE: legacy midpoint-subpnt SRP (predates specular_point_bck);
    # display-only -- it anchors the representative uncertainty values, not
    # the nominal curves computed above.
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, tx_name)
    srp = (srp_rx + srp_tx) / 2.0
    srp_hat = srp / np.linalg.norm(srp)

    # Compute uncertainties (simplified: use representative values)
    dlt_uncertainty_up = compute_dlt_uncertainty(
        rx_time, srp, ephem_uncertainty, tx_name, rx_name
    ) + hardware_errors.oscillator_stability()
    delay_uncertainty_up = compute_delay_uncertainty(
        rx_time, srp, ephem_uncertainty, computational_errors, hardware_errors,
        include_model_errors, tx_name, rx_name
    )

    # For simplicity, use same uncertainty for up and down branches
    # In reality, could compute separately for each point
    delay_up_std = np.full_like(delay_up, delay_uncertainty_up)
    dlt_up_std = np.full_like(dlt_up, dlt_uncertainty_up)
    delay_down_std = np.full_like(delay_down, delay_uncertainty_up)
    dlt_down_std = np.full_like(dlt_down, dlt_uncertainty_up)

    return (lt_min, delay_up, dlt_up, delay_down, dlt_down,
            delay_up_std, dlt_up_std, delay_down_std, dlt_down_std)



# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def plot_equator_with_errors(rx_time,
                             ephem_uncertainty=None,
                             computational_errors=None,
                             hardware_errors=None,
                             tx_name="DWINGELOO",
                             rx_name="STOCKERT",
                             n_points=500,
                             n_sigma=3,
                             include_model_errors=False,
                             scale_factor=1.0,
                             figsize=(12, 8)):
    """
    Plot doppler equator curves with error bounds.

    Args:
        rx_time: RX ephemeris time
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        tx_name: TX station name
        rx_name: RX station name
        n_points: Number of points along curves
        n_sigma: Number of standard deviations for error bounds
        include_model_errors: If True, include systematic model errors
        scale_factor: Multiplicative factor to exaggerate uncertainties for visualization
        figsize: Figure size

    Returns:
        fig, ax: Matplotlib figure and axes

    Note:
        Ephemeris uncertainties are VERY small (~10^-12 fractional Doppler).
        Use scale_factor > 1 to make error bounds visible in plots.
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()
    if computational_errors is None:
        computational_errors = ComputationalErrors()
    if hardware_errors is None:
        hardware_errors = HardwareErrors()

    # Compute equator with uncertainties
    (lt_min, delay_up, dlt_up, delay_down, dlt_down,
     delay_up_std, dlt_up_std, delay_down_std, dlt_down_std) = \
        compute_equator_with_uncertainty(
            rx_time, n_points, 100, ephem_uncertainty,
            computational_errors, hardware_errors, include_model_errors, tx_name, rx_name
        )

    # Apply scale factor to uncertainties for visualization
    dlt_up_std_scaled = dlt_up_std * scale_factor
    dlt_down_std_scaled = dlt_down_std * scale_factor
    delay_up_std_scaled = delay_up_std * scale_factor
    delay_down_std_scaled = delay_down_std * scale_factor

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot nominal curves
    ax.plot(dlt_up, delay_up, 'b-', linewidth=2, label='Up-Doppler (nominal)')
    ax.plot(dlt_down, delay_down, 'r-', linewidth=2, label='Down-Doppler (nominal)')

    # Plot uncertainty bounds as filled regions
    scale_label = f" ×{scale_factor:.0e}" if scale_factor != 1.0 else ""
    ax.fill_betweenx(delay_up,
                     dlt_up - n_sigma * dlt_up_std_scaled,
                     dlt_up + n_sigma * dlt_up_std_scaled,
                     alpha=0.3, color='blue',
                     label=f'Up-Doppler (±{n_sigma}σ{scale_label})')

    ax.fill_betweenx(delay_down,
                     dlt_down - n_sigma * dlt_down_std_scaled,
                     dlt_down + n_sigma * dlt_down_std_scaled,
                     alpha=0.3, color='red',
                     label=f'Down-Doppler (±{n_sigma}σ{scale_label})')

    ax.set_xlabel('Fractional Doppler Shift', fontsize=12)
    ax.set_ylabel('Delay (s)', fontsize=12)

    if include_model_errors:
        title = 'Doppler Equator: Systematic Bias Bounds\n[Source: Ellipsoid Topography & SRP Approximations]'
    else:
        title = 'Doppler Equator: Measurement Uncertainty Bounds\n[Source: SDR Hardware Latency/TCXO & Ephemeris Noise]'
        
    if scale_factor != 1.0:
        title += f'\n(Uncertainties ×{scale_factor:.0e} for visibility)'
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def plot_equator_nominal(rx_time,
                        tx_name="DWINGELOO",
                        rx_name="STOCKERT",
                        n_points=500,
                        figsize=(10, 8)):
    """
    Plot doppler equator curves (nominal, no uncertainty bounds).

    Args:
        rx_time: RX ephemeris time
        tx_name: TX station name
        rx_name: RX station name
        n_points: Number of points along curves
        figsize: Figure size

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Compute equator
    lt_min, delay_up, dlt_up, delay_down, dlt_down = \
        compute_doppler_equator_velocity(rx_time, n_points, tx_name, rx_name)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves
    ax.plot(dlt_up, delay_up, 'b-', linewidth=2, label='Up-Doppler')
    ax.plot(dlt_down, delay_down, 'r-', linewidth=2, label='Down-Doppler')

    ax.set_xlabel('Fractional Doppler Shift', fontsize=12)
    ax.set_ylabel('Delay (s)', fontsize=12)
    ax.set_title('Doppler Equator (Velocity Method)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def plot_error_breakdown(rx_time,
                        ephem_uncertainty=None,
                        computational_errors=None,
                        hardware_errors=None,
                        tx_name="DWINGELOO",
                        rx_name="STOCKERT",
                        reference_frequency=1299.5e6,
                        range_resolution=None,
                        doppler_resolution=None,
                        figsize=(14, 5)):
    """
    Create a breakdown of different error sources.

    Args:
        rx_time: RX ephemeris time
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        tx_name: TX station name
        rx_name: RX station name
        reference_frequency: Reference frequency for Doppler error conversion (Hz)
                           Default: 1299.5 MHz (CAMRAS L-band radar frequency)
        range_resolution: Range resolution per pixel (m). Default: 600 m/pixel
                         (based on 0.25 Msps sample rate in standard DD images)
        doppler_resolution: Doppler frequency resolution per pixel (Hz).
                           Default: 0.006 Hz/pixel @ 1299.5 MHz
                           (3000 Doppler bins spanning the limb-to-limb dlt
                           range of ~1.4e-8, i.e. ~18 Hz; the dlt *magnitude*
                           ~4e-6 is the bulk Doppler offset, not the axis span)
        figsize: Figure size

    Returns:
        fig, axes: Matplotlib figure and axes

    Notes:
        Default resolution values match the standard CAMRAS DD images:
        - Sample rate: 0.25 Msps → range resolution ~600 m
        - Doppler bins: 3000 over the limb-to-limb dlt span (edterm
          terminator points), ~1.4-1.9e-8 ≈ 18-24 Hz @ 1299.5 MHz for the
          2025 sessions → ~6-8 mHz/px. With bins this fine, oscillator
          stability is NOT negligible: the Stockert Rb (5.5e-11 ≈ 0.07 Hz)
          is ~10 pixels — this is the look-to-look δ that rim calibration
          measures and removes.
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()
    if computational_errors is None:
        computational_errors = ComputationalErrors()
    if hardware_errors is None:
        hardware_errors = HardwareErrors()
    if range_resolution is None:
        # Standard CAMRAS DD image: 0.25 Msps sample rate
        # range_resolution = (1/sample_rate) * c / 2 ≈ 600 m
        range_resolution = 600.0  # meters per pixel
    if doppler_resolution is None:
        # Standard CAMRAS DD image: 3000 Doppler bins over the limb-to-limb
        # dlt span (~1.4e-8 fractional ≈ 18 Hz @ 1299.5 MHz), not the dlt
        # magnitude (~4e-6 ≈ 5 kHz, the bulk offset already removed by SRP
        # compensation). (1.4e-8 / 3000) * 1299.5e6 ≈ 0.006 Hz/pixel.
        doppler_resolution = 0.006  # Hz per pixel @ 1299.5 MHz

    c_m_s = csp.clight() * 1000.0

    # Compute individual error contributions
    sigma_pos = ephem_uncertainty.position_uncertainty(rx_time)
    sigma_vel = ephem_uncertainty.velocity_uncertainty(rx_time)

    # Clock timing uncertainty
    # Typical atomic clock stability: 1e-12 to 1e-13 over hours
    # GPS timing: ~10-100 ns
    # Assume conservative 100 ns timing uncertainty
    sigma_time = computational_errors.timing_error()

    # Moon velocity for clock error propagation
    # Moon orbital velocity ~1 km/s → position error = v * Δt
    moon_velocity = 1000.0  # m/s (approximate)

    # Delay errors (seconds)
    delay_errors = {
        'SDR Pipeline\n(Dwingeloo TX/RX)': hardware_errors.pipeline_delay("DWINGELOO_TX_RX_OFFSET"),
        'SDR Pipeline\n(PPS Ambiguity)': hardware_errors.pipeline_delay("PPS_AMBIGUITY_250KHZ"),
        'SDR Pipeline\n(Stockert GPS)': hardware_errors.pipeline_delay("STOCKERT_GPS_OFFSET"),
        'Ephemeris\nPosition': 2.0 * sigma_pos / c_m_s,
        'Clock\nTiming': sigma_time,
        'Light-Time\nIteration': computational_errors.light_time_iteration_error(),
        'SRP\nAveraging': 2.0 * computational_errors.srp_averaging_error() / c_m_s,
        'Ellipsoid\nApprox': 2.0 * computational_errors.ellipsoid_approximation_error() / c_m_s,
    }

    # DLT errors (fractional Doppler) - break down ephemeris velocity by source
    # Moon orbital velocity uncertainty: ~10 μm/s (from LLR position constraints)
    # Earth orbital velocity uncertainty: ~5 μm/s (better constrained than Moon)
    # Note: These are MUCH smaller than previously estimated (was 1 mm/s, too large!)
    # Velocity uncertainty ≈ position_uncertainty / time_baseline
    # For 1 cm position over ~1 month: ~10 μm/s is conservative
    sigma_vel_moon = 1e-5  # m/s = 10 μm/s (Moon orbital from LLR)
    sigma_vel_earth = 5e-6  # m/s = 5 μm/s (Earth orbital, better constrained)

    # Clock timing error propagates to Doppler via acceleration
    # Doppler error: Δ(DLT) ≈ a_los * Δt / c
    moon_accel = 0.0027  # m/s^2
    dlt_clock_error = moon_accel * sigma_time / c_m_s

    # Length-scale errors map to dlt (dimensionless) two different ways:
    #  - A MOON POSITION error dx rotates the line of sight by dx/d, changing
    #    the projection of the ~1 km/s transverse velocity onto it:
    #    d(dlt) ~ (v_trans/c) * (dx/d).
    #  - A SURFACE-POINT mislocation dx samples the dlt field at the wrong
    #    place: d(dlt) ~ |grad dlt| * dx = (limb-to-limb span / 2R) * dx.
    #    (This reproduces the measured ~22 mHz topographic Doppler for the
    #    +-4 km ellipsoid term -- see REPORT.md section 6.)
    # The previous forms divided a delay error by the Moon distance (s/m,
    # dimensionally incoherent).
    moon_distance_m = 384400e3
    moon_radius_m = 1.7374e6
    dlt_span = 1.4e-8  # limb-to-limb fractional span (see doppler_resolution note)
    dlt_grad = dlt_span / (2.0 * moon_radius_m)  # per meter of surface offset

    dlt_errors = {
        'Oscillator\n(Stockert Rubidium)': hardware_errors.oscillator_stability("STOCKERT_RUBIDIUM"),
        'Oscillator\n(Dwingeloo H-Maser)': hardware_errors.oscillator_stability("DWINGELOO_HMASER"),
        'Ephemeris\nPosition': (moon_velocity / c_m_s) * (sigma_pos / moon_distance_m),
        'Moon Orbital\nVelocity (LLR)': sigma_vel_moon / c_m_s,
        'Earth Orbital\nVelocity': sigma_vel_earth / c_m_s,
        'Clock\nTiming': dlt_clock_error,
        'Finite\nDifference': computational_errors.finite_difference_error() / c_m_s,
        'SRP\nAveraging': dlt_grad * computational_errors.srp_averaging_error(),
        'Ellipsoid\nApprox': dlt_grad * computational_errors.ellipsoid_approximation_error(),
    }

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Delay errors - convert to meters (one-way range)
    # Two-way delay uncertainty -> one-way range uncertainty = (delay/2) * c
    labels = list(delay_errors.keys())
    values = [(delay_errors[k] * c_m_s / 2.0) for k in labels]  # convert to meters
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    axes[0].barh(labels, values, color=colors)
    axes[0].set_xlabel('Range Uncertainty (m)', fontsize=11)
    axes[0].set_title('Delay Error Sources', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xscale('log')

    # Add secondary x-axis for pixels (range)
    ax0_top = axes[0].twiny()
    ax0_top.set_xscale('log')
    ax0_top.set_xlim(axes[0].get_xlim()[0] / range_resolution,
                     axes[0].get_xlim()[1] / range_resolution)
    ax0_top.set_xlabel(f'Range Uncertainty (pixels @ {range_resolution:.1f} m/px)',
                       fontsize=11, color='gray')
    ax0_top.tick_params(axis='x', labelcolor='gray')

    # DLT errors - convert to Hz at reference frequency
    # Fractional Doppler shift -> Hz: Δf = f_0 * (fractional shift)
    labels = list(dlt_errors.keys())
    values = [dlt_errors[k] * reference_frequency for k in labels]  # convert to Hz

    axes[1].barh(labels, values, color=colors)
    axes[1].set_xlabel(f'Doppler Uncertainty (Hz @ {reference_frequency/1e6:.0f} MHz)', fontsize=11)
    axes[1].set_title('Doppler Shift Error Sources', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xscale('log')

    # Add secondary x-axis for pixels (Doppler)
    ax1_top = axes[1].twiny()
    ax1_top.set_xscale('log')
    ax1_top.set_xlim(axes[1].get_xlim()[0] / doppler_resolution,
                     axes[1].get_xlim()[1] / doppler_resolution)
    ax1_top.set_xlabel(f'Doppler Uncertainty (pixels @ {doppler_resolution:.2f} Hz/px)',
                       fontsize=11, color='gray')
    ax1_top.tick_params(axis='x', labelcolor='gray')

    plt.tight_layout()
    return fig, axes


def compare_ephemeris_quality(rx_times,
                              reference_et=None,
                              figsize=(10, 6)):
    """
    Visualize how ephemeris uncertainty varies with time from present epoch.

    Args:
        rx_times: Array of RX ephemeris times
        reference_et: Reference epoch (default: J2000)
        figsize: Figure size

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    ephem = EphemerisUncertainty()

    if reference_et is None:
        reference_et = 0.0  # J2000

    # Convert times to years from reference
    years_from_ref = (rx_times - reference_et) / (365.25 * 86400.0)

    # Compute uncertainties
    pos_unc = [ephem.position_uncertainty(t, reference_et) for t in rx_times]
    vel_unc = [ephem.velocity_uncertainty(t, reference_et) for t in rx_times]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Position uncertainty
    axes[0].semilogy(years_from_ref, pos_unc, 'b-', linewidth=2)
    axes[0].axhline(0.02, color='g', linestyle='--', label='Modern LLR accuracy (2 cm)')
    axes[0].axhline(10.0, color='r', linestyle='--', label='DE441-DE440 diff at 100 yr')
    axes[0].set_ylabel('Position Uncertainty (m)', fontsize=11)
    axes[0].set_title('Ephemeris Quality vs Time from Present', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Velocity uncertainty
    axes[1].semilogy(years_from_ref, vel_unc, 'r-', linewidth=2)
    axes[1].set_xlabel('Years from Reference Epoch', fontsize=11)
    axes[1].set_ylabel('Velocity Uncertainty (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes

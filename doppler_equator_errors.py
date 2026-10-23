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
- DE441 vs DE440 difference: ~10 m at 100 years from present (a difference
  between two ephemerides, not an uncertainty; not used here)
The ephemeris model below holds its values constant: it is meant for the
observing sessions (2025-2026), a few years from DE440's fitted data, and
does not model how the error grows far from them.

Every value here is this module's stated per-source figure, not a measured
uncertainty of a processed map (REPORT.md section 6 has the measured
limits). The per-look calibrations -- the timing measurement and the
frequency calibration (tone centroid plus rim correction) -- remove
whatever part of a term is constant over a look.

References:
- Park et al. 2021, The JPL Planetary and Lunar Ephemerides DE440 and DE441
  https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf
- SPICE Toolkit: https://naif.jpl.nasa.gov/naif/toolkit.html
"""

import numpy as np
import cspyce as csp
from matplotlib import pyplot as plt

from doppler_equator import (
    moon_surface_points,
    compute_doppler_equator_velocity,
    AB_COR,
)


# ---------------------------------------------------------------------------
# Error source definitions
# ---------------------------------------------------------------------------

def _lookup(table, key, what):
    """table[key]; an unknown key is an error naming the known ones (no
    default value stands in for a source that is not in the table)."""
    if key not in table:
        raise ValueError(f"unknown {what} {key!r}; known: {sorted(table)}")
    return table[key]


class EphemerisUncertainty:
    """
    Ephemeris uncertainty model, constant over the observing epochs.

    Position: 1 cm, the modern LLR weighted RMS residual (Park et al. 2021).
    Moon velocity: 10 um/s, a conservative stated value (1 cm over a ~1 month
    baseline alone gives ~5 nm/s). Earth barycentric velocity: 5 um/s,
    stated (the Earth's orbit is better constrained than the Moon's).

    No growth with time is modelled (see the module note). An earlier
    version grew the error from J2000 towards the ~10 m DE441-DE440
    difference at 100 years; at 2025-09-16 that gave 2.6 m and 2.6 mm/s
    while its docstrings said 1 cm and 10 um/s.
    """

    def __init__(self,
                 sigma_pos=0.01,          # m, modern LLR accuracy (~1 cm)
                 sigma_vel_moon=1e-5,     # m/s, conservative (10 um/s)
                 sigma_vel_earth=5e-6):   # m/s, Earth orbital (5 um/s)
        """
        Args:
            sigma_pos: Moon position uncertainty (m)
            sigma_vel_moon: Moon velocity uncertainty (m/s)
            sigma_vel_earth: Earth barycentric velocity uncertainty (m/s)
        """
        self.sigma_pos = sigma_pos
        self.sigma_vel_moon = sigma_vel_moon
        self.sigma_vel_earth = sigma_vel_earth

    def position_uncertainty(self):
        """Moon position uncertainty (m)."""
        return self.sigma_pos

    def velocity_uncertainty(self):
        """Moon velocity uncertainty (m/s)."""
        return self.sigma_vel_moon


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
    def srp_averaging_error():
        """
        Error from averaging TX and RX sub-radar points.

        For bistatic radar, we approximate the specular point as the
        average of TX and RX sub-points. The true bistatic reflection
        point can differ by up to ~tens of meters. One conservative value
        for every station pair (not computed from the geometry).

        Returns:
            Position error in meters
        """
        # Estimated from geometry: for Earth-Moon-Earth, separation ~few km
        # Bistatic error: ~1-50 m depending on bistatic angle
        return 50.0  # meters (conservative estimate)

    @staticmethod
    def timing_error(timing_source):
        """
        Clock timing uncertainty.

        Args:
            timing_source: "GPS", "NTP", or "atomic" (required: it depends
                           on how the station time-tags its samples)

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
        return _lookup(timing_uncertainties, timing_source, "timing source")


class HardwareErrors:
    """
    Hardware-specific errors, particularly for the Ettus USRP B210 SDR.
    """

    @staticmethod
    def pipeline_delay(source):
        """
        Delay in the SDR TX/RX pipeline: a fixed offset, not a random
        uncertainty. The per-look timing measurement
        (freq_offset_hunt.measure_offset) puts the specular echo's leading
        edge at lag 0, so it absorbs any such delay.
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
        return _lookup(delays, source, "pipeline delay source")

    @staticmethod
    def oscillator_stability(source):
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
        return _lookup(stabilities, source, "oscillator")


# ---------------------------------------------------------------------------
# Error propagation functions
# ---------------------------------------------------------------------------

def ephemeris_velocity_dlt(sigma_vel):
    """Doppler (dlt) error from an error sigma_vel (m/s) in the Moon's
    velocity. dlt is the rate of change of the two-way light time, the sum of
    the two legs' range rates over c; a Moon velocity error changes both
    legs' range rates (the stations see the Moon in nearly the same
    direction), so dlt changes by about 2 sigma_vel / c."""
    return 2.0 * sigma_vel / (csp.clight() * 1000.0)


def ephemeris_position_dlt(rx_time, p_moon, sigma_pos, *, tx_name, rx_name):
    """
    Doppler (dlt) error from an error sigma_pos (m) in the Moon's position,
    at surface point(s) p_moon (km, MOON_ME).

    The ephemeris error moves the whole Moon, centre and surface together,
    relative to the stations. A constant displacement dx changes a leg's
    range rate only by turning its line of sight, by up to dx/d, which
    changes the projection of the leg's relative velocity across the line
    of sight, v_perp, by up to v_perp * dx/d. With both legs turned the same
    way (the stations see the Moon in nearly the same direction):
        d(dlt) = dx * (v_perp_rx / d_rx + v_perp_tx / d_tx) / c,
    with the velocities in an inertial frame (J2000): in a frame turning
    with the Earth the displaced Moon would itself move. The same dx
    changes the two-way light time by up to 2 dx/c, which is a delay term
    (compute_delay_uncertainty); a constant change of light time has no
    Doppler.

    (An earlier version scaled the surface point radially instead; but
    moon_surface_points projects every point back onto the surface, so that
    term was always zero, and it divided metres by kilometres.)
    """
    p_surf = moon_surface_points(p_moon)
    state_rx, lt_rx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", rx_time,
                                        "J2000", "OBSERVER", AB_COR, rx_name)
    state_tx, _ = csp.spkcpo_vector(tx_name, rx_time - lt_rx, "J2000", "TARGET",
                                    AB_COR, p_surf, "MOON", "MOON_ME")
    turn_rate = 0.0                                   # sum of v_perp / d, 1/s
    for state in (np.asarray(state_rx), np.asarray(state_tx)):
        pos, vel = state[..., :3], state[..., 3:]     # km, km/s
        d = np.linalg.norm(pos, axis=-1)
        u = pos / d[..., np.newaxis]
        v_along = np.sum(vel * u, axis=-1)[..., np.newaxis] * u
        turn_rate = turn_rate + np.linalg.norm(vel - v_along, axis=-1) / d
    c_m_s = csp.clight() * 1000.0
    return sigma_pos * turn_rate / c_m_s              # m * (1/s) / (m/s)


def compute_dlt_uncertainty(rx_time, p_moon,
                           ephem_uncertainty=None, *,
                           tx_name, rx_name):
    """
    Uncertainty in dlt (fractional Doppler) from the ephemeris: the Moon
    position term (ephemeris_position_dlt) and the Moon velocity term
    (2 sigma_vel / c, both legs: ephemeris_velocity_dlt), combined in quadrature.

    Args:
        rx_time: RX ephemeris time
        p_moon: Moon surface point(s) in the Moon body-fixed frame (km)
        ephem_uncertainty: EphemerisUncertainty instance
        tx_name: TX station name
        rx_name: RX station name

    Returns:
        dlt_uncertainty: Uncertainty in dlt (fractional Doppler shift)
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()

    dlt_pos = ephemeris_position_dlt(rx_time, p_moon,
                                     ephem_uncertainty.position_uncertainty(),
                                     tx_name=tx_name, rx_name=rx_name)

    # Velocity uncertainty contribution: both legs (ephemeris_velocity_dlt).
    dlt_vel = ephemeris_velocity_dlt(ephem_uncertainty.velocity_uncertainty())

    # Total uncertainty (RSS combination)
    return np.sqrt(dlt_pos ** 2 + dlt_vel ** 2)


def compute_delay_uncertainty(ephem_uncertainty=None,
                              computational_errors=None,
                              include_model_errors=False):
    """
    Uncertainty in delay (two-way light time), seconds.

    Args:
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance (for model errors)
        include_model_errors: If True, include systematic model errors
                            (ellipsoid approx, SRP averaging). Default False.

    Returns:
        delay_uncertainty: Uncertainty in delay (seconds)

    Note:
        Always included: ephemeris position (2 x sigma_pos / c, ~0.07 ns
        for 1 cm) and light-time iteration (1 ns). Model errors like the
        ellipsoid approximation are systematic biases, not random
        uncertainties, so they are excluded by default; production maps
        project on the LOLA terrain (lunar_projection, use_dem=True), which
        takes the ellipsoid term out of the maps.

        Not included: fixed station and pipeline delays
        (HardwareErrors.pipeline_delay, e.g. the ~30 us Dwingeloo TX/RX
        offset). The per-look timing measurement absorbs any delay common
        to the whole image, so they are not an uncertainty of the image's
        delay axis. (It absorbs the parts of the two terms above that are
        common to the whole image too.)
    """
    if ephem_uncertainty is None:
        ephem_uncertainty = EphemerisUncertainty()
    if computational_errors is None:
        computational_errors = ComputationalErrors()

    c_m_s = csp.clight() * 1000.0

    # Ephemeris position uncertainty contribution
    sigma_pos = ephem_uncertainty.position_uncertainty()
    # Two-way light time: uncertainty ~ 2 * sigma_pos / c
    lt_uncertainty_ephem = 2.0 * sigma_pos / c_m_s

    # Light-time iteration error
    lt_uncertainty_iter = computational_errors.light_time_iteration_error()

    terms = [lt_uncertainty_ephem, lt_uncertainty_iter]
    if include_model_errors:
        # These are systematic biases, not random uncertainties
        # Include only if specifically requested
        terms.append(2.0 * computational_errors.srp_averaging_error() / c_m_s)
        terms.append(2.0 * computational_errors.ellipsoid_approximation_error() / c_m_s)

    # Total uncertainty (RSS combination)
    return np.sqrt(sum(t ** 2 for t in terms))


def compute_equator_with_uncertainty(rx_time, n_points=500,
                                     n_samples=100,
                                     ephem_uncertainty=None,
                                     computational_errors=None,
                                     hardware_errors=None,
                                     include_model_errors=False, *,
                                     tx_name, rx_name,
                                     oscillator_source):
    """
    Compute doppler equator with uncertainty bounds (one representative
    uncertainty, evaluated at the sub-radar point, applied to every point).

    Doppler (dlt) uncertainty: the ephemeris terms at the SRP
    (compute_dlt_uncertainty) plus the named oscillator's stated stability
    (added linearly). Delay uncertainty: compute_delay_uncertainty (no
    fixed station delays; see there). The per-look frequency calibration
    removes the part of the oscillator term that is constant over a look,
    so this is the stated figure, not what remains in a calibrated image.

    Args:
        rx_time: RX ephemeris time
        n_points: Number of points along equator curves
        n_samples: unused (no Monte Carlo sampling is done)
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        tx_name: TX station name
        rx_name: RX station name
        oscillator_source: HardwareErrors.oscillator_stability key added to
                           the Doppler uncertainty (e.g. "STOCKERT_RUBIDIUM")

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
        compute_doppler_equator_velocity(rx_time, n_points,
                                         tx_name=tx_name, rx_name=rx_name)

    # Reconstruct surface points from the velocity method
    # NOTE: legacy midpoint-subpnt SRP (predates specular_point_bck);
    # display-only -- it anchors the representative uncertainty values, not
    # the nominal curves computed above.
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, tx_name)
    srp = (srp_rx + srp_tx) / 2.0

    # Compute uncertainties (simplified: use representative values)
    dlt_uncertainty_up = compute_dlt_uncertainty(
        rx_time, srp, ephem_uncertainty, tx_name=tx_name, rx_name=rx_name
    ) + hardware_errors.oscillator_stability(oscillator_source)
    delay_uncertainty_up = compute_delay_uncertainty(
        ephem_uncertainty, computational_errors, include_model_errors
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
                             hardware_errors=None, *,
                             tx_name, rx_name,
                             oscillator_source,
                             n_points=500,
                             n_sigma=3,
                             include_model_errors=False,
                             scale_factor=1.0,
                             figsize=(12, 8)):
    """
    Plot doppler equator curves with error bounds: a Doppler band (each
    curve widened by +-n_sigma Doppler sigma, shaded) and a delay band (each
    curve shifted by -n_sigma and +n_sigma delay sigma, dotted), the sigmas
    of compute_equator_with_uncertainty. The title lists what each band
    holds.

    Args:
        rx_time: RX ephemeris time
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        tx_name: TX station name
        rx_name: RX station name
        oscillator_source: HardwareErrors.oscillator_stability key
            (see compute_equator_with_uncertainty)
        n_points: Number of points along curves
        n_sigma: Number of standard deviations for error bounds
        include_model_errors: If True, the delay band includes the
            systematic ellipsoid and SRP-averaging terms (the Doppler band
            does not change)
        scale_factor: Multiplicative factor to exaggerate uncertainties for visualization
        figsize: Figure size

    Returns:
        fig, ax: Matplotlib figure and axes

    Note:
        The oscillator dominates the Doppler band (STOCKERT_RUBIDIUM:
        5.5e-11, about 1 % of the ~1.4e-8 limb-to-limb dlt span at 3 sigma);
        the ephemeris terms are far smaller (2 x 10 um/s / c = 7e-14). The delay
        band without model errors is ~1 ns, invisible on the ~11.6 ms delay
        axis unless scale_factor is large.
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
            computational_errors, hardware_errors, include_model_errors,
            tx_name=tx_name, rx_name=rx_name,
            oscillator_source=oscillator_source
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

    # Doppler band: each curve widened by +-n_sigma Doppler sigma.
    scale_label = f" ×{scale_factor:.0e}" if scale_factor != 1.0 else ""
    ax.fill_betweenx(delay_up,
                     dlt_up - n_sigma * dlt_up_std_scaled,
                     dlt_up + n_sigma * dlt_up_std_scaled,
                     alpha=0.3, color='blue',
                     label=f'Doppler band (±{n_sigma}σ{scale_label})')
    ax.fill_betweenx(delay_down,
                     dlt_down - n_sigma * dlt_down_std_scaled,
                     dlt_down + n_sigma * dlt_down_std_scaled,
                     alpha=0.3, color='red')

    # Delay band: each curve shifted by -n_sigma and +n_sigma delay sigma,
    # drawn as its two edge curves (the curves' dlt is not monotonic at the
    # sample level, so a filled polygon over dlt would fold).
    for sign in (-1, +1):
        ax.plot(dlt_up, delay_up + sign * n_sigma * delay_up_std_scaled, ':',
                color='blue', linewidth=1,
                label=f'Delay band edges (±{n_sigma}σ{scale_label})' if sign < 0 else None)
        ax.plot(dlt_down, delay_down + sign * n_sigma * delay_down_std_scaled, ':',
                color='red', linewidth=1)

    ax.set_xlabel('Fractional Doppler Shift', fontsize=12)
    ax.set_ylabel('Delay (s)', fontsize=12)

    # What each band holds, from the same values that set them.
    doppler_sources = f'ephemeris at the SRP, {oscillator_source} oscillator'
    delay_sources = 'ephemeris, light-time iteration'
    if include_model_errors:
        delay_sources += (
            f', ellipsoid ±{computational_errors.ellipsoid_approximation_error() / 1e3:g} km'
            f', SRP averaging {computational_errors.srp_averaging_error():g} m')
    title = ('Doppler Equator with stated per-source uncertainties\n'
             f'Doppler band: {doppler_sources}\n'
             f'Delay band: {delay_sources}\n'
             '(fixed station delays not drawn: the per-look timing measurement absorbs them)')

    if scale_factor != 1.0:
        title += f'\n(Uncertainties ×{scale_factor:.0e} for visibility)'
    ax.set_title(title, fontsize=11, fontweight='bold')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def plot_equator_nominal(rx_time, *,
                        tx_name, rx_name,
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
        compute_doppler_equator_velocity(rx_time, n_points,
                                         tx_name=tx_name, rx_name=rx_name)

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


def plot_error_breakdown(ephem_uncertainty=None,
                         computational_errors=None,
                         hardware_errors=None,
                         reference_frequency=1299.5e6,
                         range_resolution=None,
                         doppler_resolution=None,
                         figsize=(14, 5), *,
                         timing_source):
    """
    Create a breakdown of different error sources.

    Args:
        ephem_uncertainty: EphemerisUncertainty instance
        computational_errors: ComputationalErrors instance
        timing_source: ComputationalErrors.timing_error key of the stations'
                       time-tagging (required)
        reference_frequency: Reference frequency for Doppler error conversion (Hz)
                           Default: 1299.5 MHz (CAMRAS L-band radar frequency)
        range_resolution: Range resolution per pixel (m). Default: 600 m/pixel
                         (based on 0.25 Msps sample rate in standard DD images)
        doppler_resolution: Doppler frequency resolution per pixel (Hz).
                           Default: 0.006 Hz/pixel @ 1299.5 MHz
                           (3000 Doppler bins spanning the limb-to-limb dlt
                           range of ~1.4e-8, i.e. ~18 Hz, the widest span of
                           the 2025 looks; the dlt *magnitude* ~4e-6 is the
                           bulk Doppler offset, not the axis span)
        figsize: Figure size

    Returns:
        fig, axes: Matplotlib figure and axes

    Notes:
        Default resolution values match the standard CAMRAS DD images:
        - Sample rate: 0.25 Msps → range resolution ~600 m
        - Doppler bins: 3000 over the limb-to-limb dlt span (edterm
          terminator points), 3.3e-9 to 1.4e-8 ≈ 4.3-18.8 Hz @ 1299.5 MHz
          over the 2025 look epochs → 1.4-6.3 mHz/px (REPORT.md section 1).
          With bins this fine, oscillator stability is NOT negligible: the
          Stockert Rb (5.5e-11 ≈ 0.07 Hz) is 11-50 pixels, comparable to
          the spread of the session-median chain offsets (+5/+54/−1/−20
          mHz, REPORT.md section 4). Each look's
          chain offset is measured, whatever its cause, as the tone
          centroid (applied_df_hz) plus the rim correction (rim_delta_hz),
          which removes what the centroid leaves (REPORT.md 3.2); which
          oscillator or receiver sets it is not established (section 4).
        The bars are this module's stated per-source values, shown for both
        stations of the 2025 Dwingeloo -> Stockert link; none depends on
        the epoch. The fixed station delays are marked: the per-look timing
        measurement absorbs any delay common to the whole image, so they
        are not an uncertainty of the image's delay axis.
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
    sigma_pos = ephem_uncertainty.position_uncertainty()

    # Clock timing uncertainty of the named time-tagging source
    sigma_time = computational_errors.timing_error(timing_source)

    # Moon velocity for clock error propagation
    # Moon orbital velocity ~1 km/s → position error = v * Δt
    moon_velocity = 1000.0  # m/s (approximate)

    # Delay errors (seconds)
    delay_errors = {
        'Dwingeloo TX/RX\n(fixed, absorbed)': hardware_errors.pipeline_delay("DWINGELOO_TX_RX_OFFSET"),
        'PPS Ambiguity\n(fixed, absorbed)': hardware_errors.pipeline_delay("PPS_AMBIGUITY_250KHZ"),
        'Stockert GPS\n(fixed, absorbed)': hardware_errors.pipeline_delay("STOCKERT_GPS_OFFSET"),
        'Ephemeris\nPosition': 2.0 * sigma_pos / c_m_s,
        'Clock\nTiming': sigma_time,
        'Light-Time\nIteration': computational_errors.light_time_iteration_error(),
        'SRP\nAveraging': 2.0 * computational_errors.srp_averaging_error() / c_m_s,
        'Ellipsoid\nApprox': 2.0 * computational_errors.ellipsoid_approximation_error() / c_m_s,
    }

    # DLT errors (fractional Doppler) - ephemeris velocity by source, the
    # EphemerisUncertainty values (see there).
    sigma_vel_moon = ephem_uncertainty.sigma_vel_moon
    sigma_vel_earth = ephem_uncertainty.sigma_vel_earth

    # Clock timing error propagates to Doppler via acceleration
    # Doppler error: Δ(DLT) ≈ a_los * Δt / c
    moon_accel = 0.0027  # m/s^2
    dlt_clock_error = moon_accel * sigma_time / c_m_s

    # Length-scale errors map to dlt (dimensionless) two different ways:
    #  - A MOON POSITION error dx rotates each leg's line of sight by dx/d,
    #    changing the projection of the ~1 km/s transverse velocity onto it:
    #    d(dlt) ~ 2 (v_trans/c) * (dx/d) over the two legs
    #    (ephemeris_position_dlt computes it from the geometry).
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
        'Ephemeris\nPosition': 2.0 * (moon_velocity / c_m_s) * (sigma_pos / moon_distance_m),
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
    axes[0].set_title('Delay Error Sources\n(fixed: absorbed by the per-look timing measurement)',
                      fontsize=12, fontweight='bold')
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
    ax1_top.set_xlabel(f'Doppler Uncertainty (pixels @ {doppler_resolution:.3g} Hz/px)',
                       fontsize=11, color='gray')
    ax1_top.tick_params(axis='x', labelcolor='gray')

    plt.tight_layout()
    return fig, axes

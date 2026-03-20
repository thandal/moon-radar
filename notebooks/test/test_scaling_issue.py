import os
import numpy as np
import scipy.signal
import scipy.interpolate
import cupy
import healpy as hp
from astropy import units as au
from astropy import time as at
import cspyce as csp
import sigmf
from tqdm import tqdm

csp.kclear()
SPICE_KERNEL_DIR = "spice_kernels"
csp.furnsh(f"{SPICE_KERNEL_DIR}/naif0012.tls")
csp.furnsh(f"{SPICE_KERNEL_DIR}/de440s.bsp")
csp.furnsh(f"{SPICE_KERNEL_DIR}/pck00011.tpc")
csp.furnsh(f"{SPICE_KERNEL_DIR}/earth_latest_high_prec.bpc")
csp.furnsh(f"{SPICE_KERNEL_DIR}/moon_pa_de440_200625.bpc")
csp.furnsh(f"{SPICE_KERNEL_DIR}/moon_de440_250416.tf")
csp.furnsh(f"{SPICE_KERNEL_DIR}/observatories.bsp")
csp.furnsh(f"{SPICE_KERNEL_DIR}/observatories.tf")

TX_START_OFFSET = 1.0 * au.s
RX_START_OFFSET = 0 * au.s
MOON_RADIUS = 1_737_400.0 * au.m
AB_COR = "LT"
EARTH_FRAME = "ITRF93"

def moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_astrotime, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT"):
    try: rx_time = float(rx_astrotime)
    except: rx_time = csp.str2et(rx_astrotime.utc.value)
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_moon_surf = csp.edpnt_vector(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])
    surf_p_moon_from_rx, lt_rx = csp.spkcpt_vector(p_moon_surf, "MOON", "MOON_ME", rx_time, EARTH_FRAME, "OBSERVER", AB_COR, rx_name)
    p_tx_from_sr, lt_tx = csp.spkcpo_vector(tx_name, rx_time - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_moon_surf, "MOON", "MOON_ME") 
    v_sr_from_rx = csp.dvnorm_vector(surf_p_moon_from_rx)
    v_tx_from_sr = csp.dvnorm_vector(p_tx_from_sr)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_sr_from_rx / c) / (1 + v_sr_from_rx / c)) * np.sqrt((1 - v_tx_from_sr / c) / (1 + v_tx_from_sr / c))
    return lt_rx + lt_tx, dlt

def moonLightTimeAndDeltaLightTime_spice_BCK(rx_astrotime):
    try: rx_time = float(rx_astrotime)
    except: rx_time = csp.str2et(rx_astrotime.utc.value)
    srp_rx, trgepc_rx, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time, "MOON_ME", AB_COR, "STOCKERT")
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trgepc_rx, "MOON_ME", AB_COR, "DWINGELOO")
    srp = (srp_rx + srp_tx) / 2.0
    return moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_astrotime, srp)
    
def moonLightTimeAndDeltaLightTime_spice_FWD(tx_astrotime):
    try: tx_time = float(tx_astrotime)
    except: tx_time = csp.str2et(tx_astrotime.utc.value)
    srp_tx, trgepc_tx, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", tx_time, "MOON_ME", "X" + AB_COR, "DWINGELOO")
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trgepc_tx, "MOON_ME", AB_COR, "STOCKERT")
    srp = (srp_tx + srp_rx) / 2.0
    return moonPointLightTimeAndDeltaLightTime_spice_FWD(tx_astrotime, srp)

def moonPointLightTimeAndDeltaLightTime_spice_FWD(tx_astrotime, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT"):
    try: tx_time = float(tx_astrotime)
    except: tx_time = csp.str2et(tx_astrotime.utc.value)
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_moon_surf = csp.edpnt_vector(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])
    surf_p_tx_to_moon, lt_tx = csp.spkcpt_vector(p_moon_surf, "MOON", "MOON_ME", tx_time, EARTH_FRAME, "TARGET", "X"+AB_COR, tx_name)
    p_sr_to_rx, lt_rx = csp.spkcpo_vector(rx_name, tx_time + lt_tx, EARTH_FRAME, "OBSERVER", "X"+AB_COR, p_moon_surf, "MOON", "MOON_ME")
    v_tx_to_moon = csp.dvnorm_vector(surf_p_tx_to_moon)
    v_sr_to_rx = csp.dvnorm_vector(p_sr_to_rx)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_tx_to_moon / c) / (1 + v_tx_to_moon / c)) * np.sqrt((1 - v_sr_to_rx / c) / (1 + v_sr_to_rx / c))
    return lt_tx + lt_rx, dlt

from astropy import constants as ak
def calculateDelayWindowIndices(sample_rate, delay_start, len_rx_samples, len_tx_samples):
    t_start = delay_start
    t_end = delay_start + MOON_RADIUS / ak.c * 2
    cor_lags = scipy.signal.correlation_lags(len_rx_samples, len_tx_samples, mode="same") / sample_rate
    cor_i_start = np.argwhere(cor_lags >= t_start)[0][0]
    cor_i_end = np.argwhere(cor_lags >= t_end)[0][0]
    return cor_lags, cor_i_start, cor_i_end

DATA_PREFIX = "data.camras.nl/lunar-radar/2025-06-21/"
rx_filename = "stockert_eme_2025_06_21_08_48_35_1299.500MHz_0.25Msps_ci16_le.chan0.sigmf-meta"
rx_chan0_sigmf_file = sigmf.sigmffile.fromfile(DATA_PREFIX + rx_filename, skip_checksum=True)
rx_samples = rx_chan0_sigmf_file.read_samples().astype("complex64")

rx_info = rx_chan0_sigmf_file.get_global_info()
sample_rate = rx_info['core:sample_rate'] * au.Hz
tx_filename = rx_info['core:description'].split(';')[0]
rx_captures = rx_chan0_sigmf_file.get_captures()
frequency = rx_captures[0]['core:frequency'] * au.Hz
rx_start_astrotime = at.Time(rx_captures[0]['core:datetime'])

tx_sigmf_filename = f"data.camras.nl/lunar-radar/tx_signals/{tx_filename}"
tx_sigmf_file = sigmf.sigmffile.fromfile(tx_sigmf_filename, skip_checksum=True)
tx_samples = tx_sigmf_file.read_samples().astype("complex64")

print(f"Sample rate: {sample_rate.value}, Frequency: {frequency.value}")
rx_duration = len(rx_samples) / sample_rate
tx_duration = len(tx_samples) / sample_rate

rx_start_time_s = csp.str2et(rx_start_astrotime.utc.value) + RX_START_OFFSET.to(au.s).value
rx_end_time_s = rx_start_time_s + rx_duration.to(au.s).value
tx_start_time_s = rx_start_time_s + TX_START_OFFSET.to(au.s).value
tx_end_time_s = tx_start_time_s + tx_duration.to(au.s).value

lt_tx_start, dlt_tx_start = moonLightTimeAndDeltaLightTime_spice_FWD(tx_start_time_s)
lt_tx_end, dlt_tx_end = moonLightTimeAndDeltaLightTime_spice_FWD(tx_end_time_s)

rx_sample_times0 = np.arange(len(rx_samples)) / sample_rate.to(au.Hz).value
adjusted_tx_sample_times0 = np.linspace(TX_START_OFFSET.to(au.s).value + lt_tx_start, (TX_START_OFFSET + tx_duration).to(au.s).value + lt_tx_end, len(tx_samples), endpoint=False)

tx_phase_interp = scipy.interpolate.interp1d(adjusted_tx_sample_times0, np.unwrap(np.angle(tx_samples)), kind='linear', fill_value=np.nan, bounds_error=False)
tx_samples_resampled = np.exp(1j * tx_phase_interp(rx_sample_times0))
np.nan_to_num(tx_samples_resampled, copy=False)

lt_rx_start, dlt_rx_start = moonLightTimeAndDeltaLightTime_spice_BCK(rx_start_time_s)
lt_rx_end, dlt_rx_end = moonLightTimeAndDeltaLightTime_spice_BCK(rx_end_time_s)
doppler_start = -dlt_rx_start * frequency
doppler_end = -dlt_rx_end * frequency
doppler_rate = (doppler_end - doppler_start) / rx_duration

t_s = np.arange(len(rx_samples)) / sample_rate
phi_Hz = -(doppler_start + doppler_rate * t_s / 2)
tx_samples_resampled_compensated = tx_samples_resampled * np.exp(-1j * 2 * np.pi * (phi_Hz * t_s).value).T  

cor_lags, cor_i_start, cor_i_end = calculateDelayWindowIndices(sample_rate.value, 0, len(rx_samples), len(tx_samples_resampled_compensated))

RX_OBSERVER = "STOCKERT"
NTERMINATOR = 1000
target_epoch, observer_pos, v_term = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_start_time_s, "MOON_ME", AB_COR, RX_OBSERVER, NTERMINATOR)
lt_term, dlt_term = moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_start_time_s, v_term)

dlt_shifts = np.linspace(dlt_term.min(), dlt_term.max(), 3000)  
f_shifts = -dlt_shifts * frequency.to(au.Hz).value - doppler_start.to(au.Hz).value

print(f"dlt_term.min(): {dlt_term.min()}")
print(f"dlt_term.max(): {dlt_term.max()}")
print(f"dlt_shifts[0]: {dlt_shifts[0]}")
print(f"dlt_shifts[-1]: {dlt_shifts[-1]}")
print(f"doppler_start: {doppler_start.to(au.Hz).value} Hz")
print(f"f_shifts[0]: {f_shifts[0]} Hz (corresponding to dlt_term.min())")
print(f"f_shifts[-1]: {f_shifts[-1]} Hz (corresponding to dlt_term.max())")


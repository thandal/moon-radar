import cspyce as csp
import numpy as np
import healpy as hp

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

import sigmf
import os
from astropy import units as au

DATA_PREFIX = "data.camras.nl/lunar-radar/2025-06-21/"
rx_filename = "stockert_eme_2025_06_21_08_48_35_1299.500MHz_0.25Msps_ci16_le.chan0.sigmf-meta"
sigmf_file = sigmf.sigmffile.fromfile(DATA_PREFIX + rx_filename, skip_checksum=True)
rx_info = sigmf_file.get_global_info()
rx_captures = sigmf_file.get_captures()
rx_start_astrotime = rx_captures[0]['core:datetime']
sample_rate = rx_info['core:sample_rate']

rx_start_time_s = csp.str2et(str(rx_start_astrotime)) + 0.0
rx_duration = 66 # roughly 66s
rx_end_time_s = rx_start_time_s + rx_duration

AB_COR = "LT"
EARTH_FRAME = "ITRF93"

def moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT"):
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_moon_surf = csp.edpnt_vector(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])
    surf_p_moon_from_rx, lt_rx = csp.spkcpt_vector(p_moon_surf, "MOON", "MOON_ME", rx_time, EARTH_FRAME, "OBSERVER", AB_COR, rx_name)
    p_tx_from_sr, lt_tx = csp.spkcpo_vector(tx_name, rx_time - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_moon_surf, "MOON", "MOON_ME") 
    v_sr_from_rx = csp.dvnorm_vector(surf_p_moon_from_rx)
    v_tx_from_sr = csp.dvnorm_vector(p_tx_from_sr)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_sr_from_rx / c) / (1 + v_sr_from_rx / c)) * np.sqrt((1 - v_tx_from_sr / c) / (1 + v_tx_from_sr / c))
    return lt_rx + lt_tx, dlt

def get_bck_dlts(t):
    srp_rx, trgepc_rx, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", t, "MOON_ME", AB_COR, "STOCKERT")
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trgepc_rx, "MOON_ME", AB_COR, "DWINGELOO")
    srp = (srp_rx + srp_tx) / 2.0
    return moonPointLightTimeAndDeltaLightTime_spice_BCK(t, srp)

lt_rx_start, dlt_rx_start = get_bck_dlts(rx_start_time_s)
lt_rx_end, dlt_rx_end = get_bck_dlts(rx_end_time_s)

freq = 1299.5e6
doppler_start = -dlt_rx_start * freq
doppler_end = -dlt_rx_end * freq
doppler_rate = (doppler_end - doppler_start) / rx_duration

delay_tau = np.min(lt_rx_start)

print(f"doppler_rate: {doppler_rate} Hz/s")
print(f"delay tau (lt_rx_start): {delay_tau} s")
print(f"Expected shift error (doppler_rate * tau): {doppler_rate * delay_tau} Hz")

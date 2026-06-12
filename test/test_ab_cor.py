import cspyce as csp
import numpy as np

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

rx_start_time_s = csp.str2et("2025-09-16 13:23:26")

def check_dlt(cor):
    srp_rx, trgepc_rx, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_start_time_s, "MOON_ME", cor, "STOCKERT")
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trgepc_rx, "MOON_ME", cor, "DWINGELOO")
    srp = (srp_rx + srp_tx) / 2.0
    
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_moon_surf = csp.edpnt(srp, moon_radii[0], moon_radii[1], moon_radii[2])
    surf_p_moon_from_rx, lt_rx = csp.spkcpt(p_moon_surf, "MOON", "MOON_ME", rx_start_time_s, "ITRF93", "OBSERVER", cor, "STOCKERT")
    p_tx_from_sr, lt_tx = csp.spkcpo("DWINGELOO", rx_start_time_s - lt_rx, "ITRF93", "TARGET", cor, p_moon_surf, "MOON", "MOON_ME") 
    v_sr_from_rx = csp.dvnorm(surf_p_moon_from_rx)
    v_tx_from_sr = csp.dvnorm(p_tx_from_sr)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_sr_from_rx / c) / (1 + v_sr_from_rx / c)) * np.sqrt((1 - v_tx_from_sr / c) / (1 + v_tx_from_sr / c))
    return dlt

dlt_lt = check_dlt("LT")
dlt_lts = check_dlt("LT+S")
dlt_cn = check_dlt("CN")
dlt_cns = check_dlt("CN+S")

freq = 1299.5e6
print(f"LT:   {-dlt_lt * freq} Hz")
print(f"LT+S: {-dlt_lts * freq} Hz  | diff = {-(dlt_lts - dlt_lt) * freq} Hz")
print(f"CN:   {-dlt_cn * freq} Hz   | diff = {-(dlt_cn - dlt_lt) * freq} Hz")
print(f"CN+S: {-dlt_cns * freq} Hz  | diff = {-(dlt_cns - dlt_lt) * freq} Hz")

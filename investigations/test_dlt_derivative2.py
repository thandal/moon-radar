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

rx_start_time_s = csp.str2et("2025-06-21 09:32:02")
AB_COR = "LT"
EARTH_FRAME = "ITRF93"

srp_rx, trgepc_rx, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_start_time_s, "MOON_ME", AB_COR, "STOCKERT")
srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trgepc_rx, "MOON_ME", AB_COR, "DWINGELOO")
srp = (srp_rx + srp_tx) / 2.0
moon_radii = csp.bodvrd("MOON", "RADII")
p_moon_surf = csp.edpnt(srp, moon_radii[0], moon_radii[1], moon_radii[2])

def get_lt(rx_time):
    surf_p_moon_from_rx, lt_rx = csp.spkcpt(p_moon_surf, "MOON", "MOON_ME", rx_time, EARTH_FRAME, "OBSERVER", AB_COR, "STOCKERT")
    p_tx_from_sr, lt_tx = csp.spkcpo("DWINGELOO", rx_time - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_moon_surf, "MOON", "MOON_ME") 
    return lt_rx + lt_tx

def get_analytic_dlt(rx_time):
    surf_p_moon_from_rx, lt_rx = csp.spkcpt(p_moon_surf, "MOON", "MOON_ME", rx_time, EARTH_FRAME, "OBSERVER", AB_COR, "STOCKERT")
    p_tx_from_sr, lt_tx = csp.spkcpo("DWINGELOO", rx_time - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_moon_surf, "MOON", "MOON_ME") 
    v_sr_from_rx = csp.dvnorm(surf_p_moon_from_rx)
    v_tx_from_sr = csp.dvnorm(p_tx_from_sr)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_sr_from_rx / c) / (1 + v_sr_from_rx / c)) * np.sqrt((1 - v_tx_from_sr / c) / (1 + v_tx_from_sr / c))
    return dlt

freq = 1299.5e6
analytic_dlt = get_analytic_dlt(rx_start_time_s)
print(f"analytic_dlt = {analytic_dlt}")

for dt in [1.0, 0.1, 0.01, 0.001]:
    lt_plus = get_lt(rx_start_time_s + dt)
    lt_minus = get_lt(rx_start_time_s - dt)
    numeric_dlt = (lt_plus - lt_minus) / (2 * dt)
    diff_dlt = analytic_dlt - numeric_dlt
    print(f"dt={dt}: numeric={numeric_dlt} | diff={diff_dlt} | Freq Error={-diff_dlt * freq} Hz")

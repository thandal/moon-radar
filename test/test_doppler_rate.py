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

rx_time = csp.str2et("2025-06-21 08:48:35")
AB_COR = "LT"
EARTH_FRAME = "ITRF93"

moon_radii = csp.bodvrd("MOON", "RADII")
p_moon = [1.0, 0.0, 0.0]
p_moon_surf = csp.edpnt(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])

def get_dvnorm(t):
    surf_p_moon_from_rx, lt_rx = csp.spkcpt(p_moon_surf, "MOON", "MOON_ME", t, EARTH_FRAME, "OBSERVER", AB_COR, "STOCKERT")
    p_tx_from_sr, lt_tx = csp.spkcpo("DWINGELOO", t - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_moon_surf, "MOON", "MOON_ME") 
    v_sr_from_rx = csp.dvnorm(surf_p_moon_from_rx)
    v_tx_from_sr = csp.dvnorm(p_tx_from_sr)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v_sr_from_rx / c) / (1 + v_sr_from_rx / c)) * np.sqrt((1 - v_tx_from_sr / c) / (1 + v_tx_from_sr / c))
    return dlt

freq = 1299.5e6
dlt0 = get_dvnorm(rx_time)
dlt1 = get_dvnorm(rx_time + 1.0)
print(f"doppler0 = {-dlt0 * freq} Hz")
print(f"doppler1 = {-dlt1 * freq} Hz")
print(f"Drift per sec = {-(dlt1 - dlt0) * freq} Hz")

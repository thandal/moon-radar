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

rx_time = csp.str2et("2025-06-21 08:48:35")
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

NTERMINATOR = 1000
target_epoch, observer_pos, v_term = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_time, "MOON_ME", AB_COR, "STOCKERT", NTERMINATOR)
lt_term, dlt_term = moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_time, v_term)

NSIDE = 200
NPIX = hp.nside2npix(NSIDE)
v = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))).T
v_near = v[:, 0] > 0
v = v[v_near, :]
lt, dlt = moonPointLightTimeAndDeltaLightTime_spice_BCK(rx_time, v)

freq = 1299.5e6

print(f"dlt_term.max() * freq = {dlt_term.max() * freq}")
print(f"dlt.max()      * freq = {dlt.max() * freq}")
print(f"Difference (Hz)       = {(dlt.max() - dlt_term.max()) * freq}")

print(f"dlt_term.min() * freq = {dlt_term.min() * freq}")
print(f"dlt.min()      * freq = {dlt.min() * freq}")
print(f"Difference (Hz)       = {(dlt.min() - dlt_term.min()) * freq}")


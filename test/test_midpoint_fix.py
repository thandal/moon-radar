import cspyce as csp
import numpy as np
import healpy as hp

csp.kclear()
SPICE_KERNEL_DIR = "spice_kernels"
for k in ["naif0012.tls","de440s.bsp","pck00011.tpc","earth_latest_high_prec.bpc",
           "moon_pa_de440_200625.bpc","moon_de440_250416.tf","observatories.bsp","observatories.tf"]:
    csp.furnsh(f"{SPICE_KERNEL_DIR}/{k}")

AB_COR = "LT"
EARTH_FRAME = "ITRF93"

def bck_dlt(rx_time, p_moon):
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_surf = csp.edpnt_vector(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])
    s_rx, lt_rx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", rx_time, EARTH_FRAME, "OBSERVER", AB_COR, "STOCKERT")
    s_tx, lt_tx = csp.spkcpo_vector("DWINGELOO", rx_time - lt_rx, EARTH_FRAME, "TARGET", AB_COR, p_surf, "MOON", "MOON_ME")
    v1 = csp.dvnorm_vector(s_rx)
    v2 = csp.dvnorm_vector(s_tx)
    c = csp.clight()
    dlt = 1 - np.sqrt((1-v1/c)/(1+v1/c)) * np.sqrt((1-v2/c)/(1+v2/c))
    return lt_rx + lt_tx, dlt

def srp_dlt(t):
    srp_rx, trg, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", t, "MOON_ME", AB_COR, "STOCKERT")
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trg, "MOON_ME", AB_COR, "DWINGELOO")
    return bck_dlt(t, (srp_rx + srp_tx) / 2.0)

# Use 2025-06-21 data timing
rx_start = csp.str2et("2025-06-21 09:32:02")
T = 66.0  # rx_duration in seconds
rx_mid = rx_start + T/2
freq = 1299.5e6

# Get SRP doppler rate
_, dlt_srp_start = srp_dlt(rx_start)
_, dlt_srp_end = srp_dlt(rx_start + T)
dlt_rate_srp = (dlt_srp_end - dlt_srp_start) / T
print(f"SRP Doppler rate: {-dlt_rate_srp * freq:.6f} Hz/s")

# Get terminator points
NTERMINATOR = 1000
_, _, v_term = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_start, "MOON_ME", AB_COR, "STOCKERT", NTERMINATOR)
_, dlt_term_start = bck_dlt(rx_start, v_term)
_, dlt_term_end = bck_dlt(rx_start + T, v_term)

# For each terminator point, compute the differential Doppler rate
dlt_rate_term = (dlt_term_end - dlt_term_start) / T
diff_rate = dlt_rate_term - dlt_rate_srp

# The predicted peak offset due to time averaging
peak_offset_hz = -diff_rate * freq * T / 2

# DLT at start vs midpoint
_, dlt_term_mid = bck_dlt(rx_mid, v_term)

# Find the limb points with max and min DLT (highest and lowest Doppler)
i_max = np.argmax(dlt_term_start)
i_min = np.argmin(dlt_term_start)

print(f"\n--- LIMB WITH MAX DLT (one side) ---")
print(f"DLT at rx_start:  {dlt_term_start[i_max]:.12e}")
print(f"DLT at rx_mid:    {dlt_term_mid[i_max]:.12e}")
print(f"Diff Doppler rate: {diff_rate[i_max]*freq:.6f} Hz/s")
print(f"Predicted peak offset: {peak_offset_hz[i_max]:.4f} Hz")
print(f"DLT(mid) - DLT(start) in Hz: {(dlt_term_mid[i_max]-dlt_term_start[i_max])*freq:.4f} Hz")

print(f"\n--- LIMB WITH MIN DLT (other side) ---")
print(f"DLT at rx_start:  {dlt_term_start[i_min]:.12e}")
print(f"DLT at rx_mid:    {dlt_term_mid[i_min]:.12e}")
print(f"Diff Doppler rate: {diff_rate[i_min]*freq:.6f} Hz/s")
print(f"Predicted peak offset: {peak_offset_hz[i_min]:.4f} Hz")
print(f"DLT(mid) - DLT(start) in Hz: {(dlt_term_mid[i_min]-dlt_term_start[i_min])*freq:.4f} Hz")

print(f"\n--- STATISTICS ---")
print(f"Peak offset range: {peak_offset_hz.min():.4f} to {peak_offset_hz.max():.4f} Hz")
print(f"Peak offset at SRP (should be ~0): {-0 * freq * T/2:.4f} Hz")

# Also check the 2025-09-16 data
rx_start_09 = csp.str2et("2025-09-16 13:23:26")
T_09 = 30.0
_, dlt_srp_start_09 = srp_dlt(rx_start_09)
_, dlt_srp_end_09 = srp_dlt(rx_start_09 + T_09)
dlt_rate_srp_09 = (dlt_srp_end_09 - dlt_srp_start_09) / T_09

_, _, v_term_09 = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_start_09, "MOON_ME", AB_COR, "STOCKERT", NTERMINATOR)
_, dlt_term_start_09 = bck_dlt(rx_start_09, v_term_09)
_, dlt_term_end_09 = bck_dlt(rx_start_09 + T_09, v_term_09)
dlt_rate_term_09 = (dlt_term_end_09 - dlt_term_start_09) / T_09
diff_rate_09 = dlt_rate_term_09 - dlt_rate_srp_09
peak_offset_hz_09 = -diff_rate_09 * freq * T_09 / 2

i_max_09 = np.argmax(dlt_term_start_09)
i_min_09 = np.argmin(dlt_term_start_09)
print(f"\n=== 2025-09-16 (T={T_09}s) ===")
print(f"SRP Doppler rate: {-dlt_rate_srp_09 * freq:.6f} Hz/s")
print(f"Max DLT limb offset: {peak_offset_hz_09[i_max_09]:.4f} Hz")
print(f"Min DLT limb offset: {peak_offset_hz_09[i_min_09]:.4f} Hz")
print(f"Peak offset range: {peak_offset_hz_09.min():.4f} to {peak_offset_hz_09.max():.4f} Hz")

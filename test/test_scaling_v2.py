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

rx_start = csp.str2et("2025-06-21 09:32:02")
T = 66.0
freq = 1299.5e6

# SRP doppler rate
_, dlt_srp_s = srp_dlt(rx_start)
_, dlt_srp_e = srp_dlt(rx_start + T)
dlt_rate_srp = (dlt_srp_e - dlt_srp_s) / T

# Terminator at rx_start (matches the image axis)
NTERMINATOR = 1000
_, _, v_term = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_start, "MOON_ME", AB_COR, "STOCKERT", NTERMINATOR)
_, dlt_term_s = bck_dlt(rx_start, v_term)
_, dlt_term_e = bck_dlt(rx_start + T, v_term)

dlt_rate_term = (dlt_term_e - dlt_term_s) / T

# The effective DLT in the image for each terminator point:
# dlt_eff = dlt_start + (dlt_rate_p - dlt_rate_srp) * T/2
dlt_eff_term = dlt_term_s + (dlt_rate_term - dlt_rate_srp) * T/2

# Error = dlt_eff - dlt_start (the mismatch caused by using dlt_start for mapping)
error_hz = (dlt_eff_term - dlt_term_s) * freq

i_max = np.argmax(dlt_term_s)
i_min = np.argmin(dlt_term_s)

print(f"=== Differential Doppler Rate Correction ===")
print(f"Max DLT limb: error = {error_hz[i_max]:.4f} Hz")
print(f"Min DLT limb: error = {error_hz[i_min]:.4f} Hz")
print(f"Error range:  {error_hz.min():.4f} to {error_hz.max():.4f} Hz")
print()

# Now check: does dlt_eff align with the image?
# The dd image axis is dlt_shifts = linspace(dlt_term_s.min(), dlt_term_s.max(), 3000)
# Using dlt_eff for mapping:
#   index = (dlt_eff - dlt_term_s.min()) / ddlt
# vs original:
#   index = (dlt_term_s - dlt_term_s.min()) / ddlt
ddlt = (dlt_term_s.max() - dlt_term_s.min()) / 2999.0
orig_index_max = (dlt_term_s[i_max] - dlt_term_s.min()) / ddlt
corr_index_max = (dlt_eff_term[i_max] - dlt_term_s.min()) / ddlt
print(f"Max limb: orig_index={orig_index_max:.1f}, corrected_index={corr_index_max:.1f}, diff={corr_index_max-orig_index_max:.2f} rows")

orig_index_min = (dlt_term_s[i_min] - dlt_term_s.min()) / ddlt
corr_index_min = (dlt_eff_term[i_min] - dlt_term_s.min()) / ddlt
print(f"Min limb: orig_index={orig_index_min:.1f}, corrected_index={corr_index_min:.1f}, diff={corr_index_min-orig_index_min:.2f} rows")

# Also check 2025-09-16
rx09 = csp.str2et("2025-09-16 13:23:26")
T09 = 30.0
_, dlt_srp_s09 = srp_dlt(rx09)
_, dlt_srp_e09 = srp_dlt(rx09 + T09)
dlt_rate_srp09 = (dlt_srp_e09 - dlt_srp_s09) / T09
_, _, v_term09 = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx09, "MOON_ME", AB_COR, "STOCKERT", NTERMINATOR)
_, dlt_term_s09 = bck_dlt(rx09, v_term09)
_, dlt_term_e09 = bck_dlt(rx09 + T09, v_term09)
dlt_rate_term09 = (dlt_term_e09 - dlt_term_s09) / T09
error_hz09 = (dlt_rate_term09 - dlt_rate_srp09) * T09/2 * freq
print(f"\n=== 2025-09-16 (T={T09}s) ===")
print(f"Error range: {error_hz09.min():.4f} to {error_hz09.max():.4f} Hz")

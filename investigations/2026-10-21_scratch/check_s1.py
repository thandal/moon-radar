"""New lunar_projection with supersample=1 vs the old one, same input."""
import csv, json, os, sys, time
import numpy as np, healpy as hp
from astropy import time as at
sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator as de
import doppler_equator_alignment as dea
from doppler_equator_alignment import (moon_surface_points, apparent_station_positions,
                                       srp_dlt_rate_bck, moonPointDLT_BCK)


def old_projection(log_A, dlt_shifts, delay_values_s, lt_min_image,
                   rx_time_s, nside, rx_duration_s=None, dlt_rate_srp=None,
                   tx_name="DWINGELOO", rx_name="STOCKERT", use_dem=True):
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix))).T
    p = moon_surface_points(v, use_dem=use_dem)
    def lt_field(t):
        R_rx, R_tx, c = apparent_station_positions(t, tx_name, rx_name)
        return (np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / c
    lt = lt_field(rx_time_s)
    lt_end = lt_field(rx_time_s + rx_duration_s)
    dlt = (lt_end - lt) / rx_duration_s - dlt_rate_srp * rx_duration_s / 2
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    ddelay = delay_values_s[1] - delay_values_s[0]
    doppler_index = np.rint((dlt - dlt_shifts[0]) / ddlt).astype(int)
    delay_index = np.rint((lt - lt_min_image - delay_values_s[0]) / ddelay).astype(int)
    valid = ((doppler_index >= 0) & (doppler_index < log_A.shape[0]) &
             (delay_index >= 0) & (delay_index < log_A.shape[1]))
    val_surface = np.full(npix, hp.UNSEEN, dtype=np.float32)
    val_surface[valid] = log_A[doppler_index[valid], delay_index[valid]]
    multiplicity = np.zeros(npix, dtype=np.float32)
    cell = doppler_index[valid] * log_A.shape[1] + delay_index[valid]
    counts = np.bincount(cell)
    multiplicity[valid] = counts[cell]
    return val_surface, valid.mean(), multiplicity


de.load_lola_dem()
DATA = "data/eve2026_circ/2026-10-11"
for csvp, idx in (("results/EVE2026_CIRC/registration_runs_stockert_chan1.csv", 4),
                  ("results/EVE2026_CIRC/registration_runs_wsrt_chan0.csv", 0)):
    r = list(csv.DictReader(open(csvp)))[idx]
    base = r["rx_file"]; rx_name = base.split("_")[0].upper()
    meta = json.load(open(os.path.join(DATA, base)))
    rx_time_s = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    fs, f_hz, T, tx_extra = (float(r[k]) for k in ("sample_rate_hz", "frequency_hz", "capture_s", "tx_extra_offset_s"))
    dd = np.load(r["dd_npz"]); log_A = dd["log_A"].astype(np.float32)
    _, dlt_srp = de.moonSRP_DLT_BCK(rx_time_s, "DWINGELOO", rx_name, dt=T / 2)
    dlt_cal = dlt_srp - dd["doppler_hz"] / f_hz
    delays = (dd["delay_s"][0] - tx_extra) + np.arange(log_A.shape[1]) / fs
    rate = de.srp_dlt_rate_bck(rx_time_s, T, "DWINGELOO", rx_name)
    args = (log_A, dlt_cal, delays, 0.0, rx_time_s, 400)
    kw = dict(rx_duration_s=T, dlt_rate_srp=rate, rx_name=rx_name)
    t0 = time.time(); vo, fo, mo = old_projection(*args, **kw); t1 = time.time()
    vn, fn, mn = dea.lunar_projection(*args, **kw, supersample=1); t2 = time.time()
    d = np.abs(vo.astype(np.float64) - vn)
    print(f"{base[:45]} ({r['rx_start_utc'][11:19]}, {T:.0f} s): valid {fo:.6f} vs {fn:.6f}; "
          f"same valid pixels {np.array_equal(vo == hp.UNSEEN, vn == hp.UNSEEN)}; "
          f"values identical {np.array_equal(vo, vn)} (max diff {d[vo != hp.UNSEEN].max():.2e}); "
          f"multiplicity identical {np.array_equal(mo, mn)}; old {t1 - t0:.1f} s, new {t2 - t1:.1f} s")

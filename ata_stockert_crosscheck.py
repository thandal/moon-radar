"""
ATA <-> Stockert bistatic registration cross-check (2025-09-16).

The 2025-09-16 session was recorded simultaneously by two independent bistatic
receivers: Stockert (the canonical RX) and ATA (a second dish). Each has its own
SDR, clock chain, and pointing, but both observe the same illuminated Moon lit by
the same TX (Dwingeloo). If the registration pipeline is sound, the two maps,
built completely independently, should land on the same surface features with
sub-degree agreement.

This is a *validation* of the registration, not a science map: ATA alone has only
8 co-pol looks from one ~12-min pass and is speckle-limited (see REPORT). The
question here is solely whether ATA and Stockert AGREE on 2025-09-16.

Method: build a single-session co-pol (chan1) stack for each receiver, band-pass
both at the same speckle/trend scales used by the cross-session offset solve, and
cross-correlate to measure the residual (lon, lat) offset, its significance, and
the band-passed correlation coefficient.

Usage (from the repo root):
    .conda/bin/python ata_stockert_crosscheck.py
"""

import csv
import os

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import doppler_equator_alignment as dea  # noqa: F401  (SPICE side effects)
import registration_analysis as ra
import stack_maps as sm

RUN_DIR = os.path.join(os.path.dirname(__file__), "results/REGISTRATION")
SESSION = "2025_09_16"
MIN_SNR = 15.0
MIN_COUNT = 3

# Same grid / band-pass geometry as the cross-session solve in stack_maps.main.
STEP = 0.075
LON_AXIS = np.arange(-55, 55 + STEP / 2, STEP)
LAT_AXIS = np.arange(-55, 55 + STEP / 2, STEP)
LO_PX, HI_PX = 0.3 / STEP, 2.5 / STEP
SEARCH_PX, EXCLUDE_PX = int(1.5 / STEP), int(0.5 / STEP)


def gated_session_rows(run_prefix):
    """2025-09-16 chan1 looks from a runs CSV that pass the stack gating."""
    path = os.path.join(RUN_DIR, f"{run_prefix}_chan1.csv")
    rows = []
    for r in csv.DictReader(open(path)):
        if SESSION not in r["rx_file"]:
            continue
        gate_snr = float(r.get("pair_snr") or r["tone_snr"])
        shift = float(r.get("applied_shift_samples") or r["shift_refined"])
        if gate_snr < MIN_SNR or abs(shift) >= 39.5:
            continue
        rows.append(r)
    return rows


def build_stack(rows):
    """Incoherent co-pol stack (log map, UNSEEN-masked) for one receiver."""
    npix = len(np.load(rows[0]["map_npy"]))
    acc = sm.Accumulator(npix)
    for r in rows:
        acc.add(sm.look_linear_intensity(r["map_npy"], r["mult_npy"]))
    return acc.stacked_log(MIN_COUNT)


def main():
    st_rows = gated_session_rows("registration_runs")
    at_rows = gated_session_rows("registration_runs_ata")
    print(f"Stockert {SESSION}: {len(st_rows)} looks pass gating")
    print(f"ATA      {SESSION}: {len(at_rows)} looks pass gating")

    st_map = build_stack(st_rows)
    at_map = build_stack(at_rows)
    st_grid = ra.grid_map(st_map, LON_AXIS, LAT_AXIS)
    at_grid = ra.grid_map(at_map, LON_AXIS, LAT_AXIS)

    def corr(a, b):
        m = (a != 0) & (b != 0)
        return np.corrcoef(a[m], b[m])[0, 1]

    def sweep_corr(grid_a, grid_b):
        """Best-aligned correlation across the low-pass sweep (per-scale rows)."""
        rows_out, best_r = [], -1.0
        for lo_deg in (0.3, 0.5, 0.8, 1.2, 1.8):
            sb = ra.bandpass(grid_a, lo_deg / STEP, HI_PX)
            ab = ra.bandpass(grid_b, lo_deg / STEP, HI_PX)
            dy_, dx_, _, sig_ = ra.xcorr_offset(sb, ab, SEARCH_PX, EXCLUDE_PX)
            ab_al = np.roll(np.roll(ab, int(round(dy_)), 0), int(round(dx_)), 1)
            r_ = corr(sb, ab_al)
            rows_out.append((lo_deg, np.hypot(dx_, dy_) * STEP, sig_, r_))
            best_r = max(best_r, r_)
        return rows_out, best_r

    # Control: split Stockert's own looks (even/odd by time) into two independent
    # sub-stacks of comparable look count. Same receiver, so this is the *ceiling*
    # -- how well any ~10-look co-pol stack agrees with an independent twin. ATA's
    # cross-receiver correlation is only interpretable against this baseline.
    st_sorted = sorted(st_rows, key=lambda r: r["rx_start_utc"])
    half_a = build_stack(st_sorted[0::2])
    half_b = build_stack(st_sorted[1::2])
    _, ctrl_best = sweep_corr(ra.grid_map(half_a, LON_AXIS, LAT_AXIS),
                              ra.grid_map(half_b, LON_AXIS, LAT_AXIS))
    print(f"\nControl (Stockert split-half, {len(st_sorted[0::2])} vs "
          f"{len(st_sorted[1::2])} looks): best corr {ctrl_best:+.3f}")

    # Speckle is receiver-specific: ATA and Stockert sit at different sites, so
    # their speckle realizations differ and SHOULD decorrelate. What the two
    # independent receivers share is the larger-scale reflectivity. ATA's 8
    # looks barely average speckle at the fine (0.3 deg) band used for the
    # cross-session solve, so we sweep the low-pass scale: agreement that
    # emerges only as the speckle is smoothed away is the registration check.
    print("\nATA vs Stockert agreement vs smoothing scale (low-pass sigma):")
    print(f"  {'lo_deg':>7} {'offset_deg':>11} {'signif':>7} {'corr_aligned':>13}")
    sweep, _ = sweep_corr(st_grid, at_grid)
    best = None
    for (lo_deg, off_, sig_, r_) in sweep:
        print(f"  {lo_deg:7.1f} {off_:11.3f} {sig_:7.2f} {r_:13.3f}")
        if best is None or r_ > best[0]:
            best = (r_, lo_deg)

    # Report the registration offset at the scale where the receivers agree best.
    _, lo_best = best
    st_band = ra.bandpass(st_grid, lo_best / STEP, HI_PX)
    at_band = ra.bandpass(at_grid, lo_best / STEP, HI_PX)
    dy, dx, peak, sig = ra.xcorr_offset(st_band, at_band, SEARCH_PX, EXCLUDE_PX)
    dlon, dlat = dx * STEP, dy * STEP
    offset_deg = np.hypot(dlon, dlat)
    print(f"\nBest agreement at low-pass {lo_best:.1f} deg:")

    # Shift ATA's band-passed map onto Stockert for the "after" correlation/figure.
    at_band_aligned = np.roll(np.roll(at_band, int(round(dy)), axis=0),
                              int(round(dx)), axis=1)

    r_before = corr(st_band, at_band)
    r_after = corr(st_band, at_band_aligned)

    print(f"  dlon {dlon:+.3f} deg, dlat {dlat:+.3f} deg  -> {offset_deg:.3f} deg")
    print(f"  xcorr peak {peak:.3f}, significance {sig:.2f} "
          f"({'genuine lock' if sig > 1.5 else 'NO lock -- noise floor'})")
    print(f"  band-passed correlation: {r_before:+.3f} (raw) "
          f"-> {r_after:+.3f} (aligned)")
    frac = r_after / ctrl_best if ctrl_best > 0 else float("nan")
    print(f"  vs same-receiver control {ctrl_best:+.3f}: "
          f"ATA recovers {100 * frac:.0f}% of the achievable agreement")
    print("  VERDICT: " + (
        "consistent -- no gross misregistration; ATA agrees with Stockert "
        "as well as Stockert agrees with itself at this look count."
        if offset_deg < 0.5 and frac > 0.6 else
        "weak -- offset small but agreement below the same-receiver control."))

    # --- Figure: the two independent band-passed maps + aligned overlay ---
    extent = [LON_AXIS[0], LON_AXIS[-1], LAT_AXIS[0], LAT_AXIS[-1]]
    vmax = np.nanpercentile(np.abs(st_band[st_band != 0]), 99)
    fig, ax = pl.subplots(1, 3, figsize=(15, 5.2))
    for a, b, t in [(ax[0], st_band, f"Stockert ({len(st_rows)} looks)"),
                    (ax[1], at_band, f"ATA ({len(at_rows)} looks)")]:
        a.imshow(b, origin="lower", extent=extent, cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax)
        a.set_title(t)
        a.set_xlabel("lon (deg)")
    ax[0].set_ylabel("lat (deg)")
    # Overlay: Stockert in red channel, ATA-aligned in blue -> purple where both.
    ov = np.zeros(st_band.shape + (3,))
    norm = lambda x: np.clip(0.5 + 0.5 * x / max(vmax, 1e-9), 0, 1)
    ov[..., 0] = norm(st_band)
    ov[..., 2] = norm(at_band_aligned)
    ov[..., 1] = 0.5 * (ov[..., 0] + ov[..., 2])
    ax[2].imshow(ov, origin="lower", extent=extent)
    ax[2].set_title(f"overlay (offset {offset_deg:.2f} deg, sig {sig:.1f}, "
                    f"r={r_after:+.2f})")
    ax[2].set_xlabel("lon (deg)")
    fig.suptitle("ATA <-> Stockert bistatic registration cross-check, 2025-09-16 co-pol")
    fig.tight_layout()
    out_png = os.path.join(RUN_DIR, "ata_stockert_crosscheck.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()

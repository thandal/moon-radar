"""
ATA <-> Stockert bistatic registration cross-check (2025-09-16).

The 2025-09-16 session was recorded simultaneously by two independent bistatic
receivers: Stockert and ATA (a second dish). Each has its own SDR, clock
chain, and pointing, but both observe the same illuminated Moon lit by the
same TX (Dwingeloo). If the registration pipeline is sound, the two maps,
built completely independently, should land on the same surface features with
sub-degree agreement.

This is a *validation* of the registration, not a science map: ATA alone has only
8 opposite-sense looks from one ~12-min pass and is speckle-limited (see REPORT). The
question here is solely whether ATA and Stockert AGREE on 2025-09-16.

Method: select each receiver's opposite-sense (chan1) looks of the session with the
stack's own gates (stack_maps.select_looks) and radiometry
(stack_maps.look_brightness). Every stack has the same number of looks, n,
ATA's usable looks. Stockert's looks are paired with ATA's by time -- for
each ATA look in time order, the Stockert look nearest in time not yet taken
(the earlier on a tie) -- giving the set A; the same rule over the Stockert
looks left gives a second, disjoint set B. Each stack's mean brightness
(stack_maps.Accumulator.mean) is band-passed as the cross-session solve
band-passes it (stack_maps.band_of) and cross-correlated, for the residual
(lon, lat) offset, its significance (the lock label) and the band-passed
correlation coefficient:
  control: Stockert A vs Stockert B -- same receiver, same look count,
           nearly the same time: how well two such stacks can agree;
  test:    Stockert A vs ATA -- the same comparison across receivers.

The smoothing scale is chosen once, on the control: the scale where A and B
agree best, among scales where that comparison locks
(registration_analysis.is_lock, an uncalibrated label). ATA vs Stockert is
measured at that same scale, against the same stack A with the same look
count, so the ratio of its correlation to the control's compares like with
like.

If the control locks at no scale, there is no reference for judging ATA and
the check stops with a non-zero exit. The figure is written only when the
check reaches a verdict; a previous figure is removed when the check
starts, so a run that stops leaves none.

It needs an opposite-sense runs CSV for each station, made with the 2026-10-21
pipeline: Stockert's (registration_runs_stockert_chan1.csv) and ATA's
(registration_runs_ata_chan1.csv), in one run directory or several, mapped
with the same terrain model, SPICE kernels and supersampling
(stack_maps.select_looks refuses a mix). results/STOCKERT_2025 holds
Stockert's looks only; ATA's come from a run of registration_stability.py
on ATA's 2025-09-16 captures (with --data-root where they are, if not
under the default), as below.

Usage (from the repo root):
    .conda/bin/python registration_stability.py --station ata --chan chan1 \\
        --dates 2025-09-16 --per-date 0 --out-dir results/ATA_2025
    .conda/bin/python ata_stockert_crosscheck.py \\
        --run-dir results/STOCKERT_2025 results/ATA_2025 --out-dir results/ATA_2025
Output: <out-dir>/ata_stockert_crosscheck.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import doppler_equator_alignment as dea
import registration_analysis as ra
import stack_maps as sm

SESSION = "2025-09-16"          # stack_maps.session_of label
STATIONS = ("stockert", "ata")
MIN_COUNT = 3

# Same grid / band-pass geometry as the cross-session solve in stack_maps.run.
STEP = 0.075
LON_AXIS = np.arange(-55, 55 + STEP / 2, STEP)
LAT_AXIS = np.arange(-55, 55 + STEP / 2, STEP)
HI_PX = 2.5 / STEP
SEARCH_PX = int(1.5 / STEP)
LO_DEG_SWEEP = (0.3, 0.5, 0.8, 1.2, 1.8)

# Lock label: registration_analysis.is_lock (an uncalibrated reporting
# convention; see LOCK_SIGNIFICANCE there).
LOCK_SIGNIFICANCE = ra.LOCK_SIGNIFICANCE
# Verdict thresholds (unchanged from the original check): an offset below
# half a degree is "no gross misregistration", and ATA should reach 60 % of
# the agreement Stockert has with itself.
MAX_OFFSET_DEG = 0.5
MIN_CONTROL_FRACTION = 0.6


def session_looks(run_dirs):
    """Per station, the session's chan1 looks that pass the stack's gates,
    in time order: dicts with the look's epoch (et), rx_file and brightness
    map (stack_maps.look_brightness). Stops unless ATA has MIN_COUNT looks
    and Stockert twice as many as ATA (two disjoint sets of ATA's size)."""
    good, decisions = sm.select_looks(run_dirs, list(STATIONS), ["chan1"])
    for station in STATIONS:
        in_session = [(r, why) for r, why in decisions
                      if r["_station"] == station and sm.session_of(r["rx_file"]) == SESSION]
        not_mapped = [(r, why) for r, why in in_session if why.startswith("not mapped")]
        excluded = [(r, why) for r, why in in_session
                    if why and not why.startswith("not mapped")]
        print(f"{station}: {len(in_session) - len(not_mapped)} {SESSION} looks, "
              f"{len(excluded)} excluded by the stack's gates; {len(not_mapped)} "
              "captures not mapped by the batch")
        for r, why in excluded + not_mapped:
            print(f"  - {r['rx_file']}: {why}")
    npix = None
    cache, looks = {}, {s: [] for s in STATIONS}
    for r in sorted(good, key=lambda r: r["rx_start_utc"]):
        if sm.session_of(r["rx_file"]) != SESSION:
            continue
        if npix is None:
            npix = len(np.load(r["map_npy"], mmap_mode="r"))
        et = dea.csp.str2et(r["rx_start_utc"])
        cosi = sm.incidence_cos(et, npix, cache, rx_name=r["_rx"])
        try:
            brightness, _ = sm.look_brightness(r, cosi)
        except sm.LookRejected as exc:
            print(f"  - {r['rx_file']}: {exc}")
            continue
        looks[r["_station"]].append({"et": et, "rx_file": r["rx_file"],
                                     "brightness": brightness})
    n_ata, n_st = len(looks["ata"]), len(looks["stockert"])
    if n_ata < MIN_COUNT:
        raise SystemExit(f"ata: {n_ata} usable {SESSION} looks; a stack needs at "
                         f"least {MIN_COUNT}")
    if n_st < 2 * n_ata:
        raise SystemExit(f"stockert: {n_st} usable {SESSION} looks; comparing equal "
                         f"look counts needs two disjoint sets of ATA's {n_ata}, "
                         f"{2 * n_ata} looks")
    return looks


def nearest_in_time(candidates, targets):
    """For each target in time order, the candidate nearest in time not yet
    taken (the earlier one on a tie). Returns the (candidate, target) pairs
    and the candidates left."""
    left = list(candidates)
    pairs = []
    for t in sorted(targets, key=lambda x: x["et"]):
        i = min(range(len(left)), key=lambda j: (abs(left[j]["et"] - t["et"]), left[j]["et"]))
        pairs.append((left.pop(i), t))
    return pairs, left


def build_stack(looks):
    """Mean brightness of the looks (stack_maps.Accumulator.mean; NaN below
    MIN_COUNT looks), as a healpix map."""
    acc = sm.Accumulator(len(looks[0]["brightness"]))
    for look in looks:
        acc.add(look["brightness"])
    return acc.mean(MIN_COUNT)


def corr(a, b):
    """Correlation over pixels where both band-passed maps have a value
    (the band-pass returns 0 where a map has none)."""
    m = (a != 0) & (b != 0)
    return np.corrcoef(a[m], b[m])[0, 1]


def compare(mean_a, mean_b, lo_deg, keep_maps=False):
    """Band-pass both mean brightness maps at one smoothing scale
    (stack_maps.band_of) and cross-correlate. The band-passed maps (~17 MB
    each) are kept only if asked for."""
    a = sm.band_of(mean_a, LON_AXIS, LAT_AXIS, lo_deg / STEP, HI_PX)
    b = sm.band_of(mean_b, LON_AXIS, LAT_AXIS, lo_deg / STEP, HI_PX)
    x = ra.xcorr_offset(a, b, SEARCH_PX)
    b_aligned = np.roll(np.roll(b, int(round(x.dy)), axis=0), int(round(x.dx)), axis=1)
    locked = ra.is_lock(x)
    out = {"lo_deg": lo_deg, "x": x, "offset_deg": np.hypot(x.dx, x.dy) * STEP,
           "locked": locked, "r_raw": corr(a, b), "r_aligned": corr(a, b_aligned)}
    if keep_maps:
        out.update(a=a, b=b, b_aligned=b_aligned)
    return out


def print_sweep(title, results):
    print(title)
    print(f"  {'lo_deg':>7} {'offset_deg':>11} {'signif':>7} {'lock':>5} {'corr_aligned':>13}")
    for c in results:
        state = "edge" if c["x"].at_edge else ("yes" if c["locked"] else "no")
        print(f"  {c['lo_deg']:7.1f} {c['offset_deg']:11.3f} "
              f"{c['x'].significance:7.2f} {state:>5} {c['r_aligned']:13.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="run directories with the stockert and ata runs CSVs")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_png = os.path.join(args.out_dir, "ata_stockert_crosscheck.png")
    if os.path.exists(out_png):
        os.remove(out_png)
        print(f"Removed the previous {out_png}: it is written again only if this "
              "run reaches a verdict")

    looks = session_looks(args.run_dir)
    n = len(looks["ata"])
    pairs_a, rest = nearest_in_time(looks["stockert"], looks["ata"])
    pairs_b, _ = nearest_in_time(rest, looks["ata"])
    for name, pairs in (("A", pairs_a), ("B", pairs_b)):
        dt = [abs(s["et"] - t["et"]) for s, t in pairs]
        print(f"Stockert set {name} ({n} looks, {min(dt):.0f}-{max(dt):.0f} s from the "
              "ATA look each is paired with):")
        for s, t in pairs:
            print(f"  {s['rx_file']}  with  {t['rx_file']}")
    st_a = build_stack([s for s, _ in pairs_a])
    st_b = build_stack([s for s, _ in pairs_b])
    at = build_stack(looks["ata"])
    del looks, pairs_a, pairs_b, rest

    # Control: two disjoint Stockert stacks of ATA's look count, from the
    # same time. Same receiver, so this is the *ceiling* -- how well a stack
    # of this many looks agrees with an independent twin. ATA's
    # cross-receiver correlation is only interpretable against it.
    control = [compare(st_a, st_b, lo) for lo in LO_DEG_SWEEP]
    print_sweep(f"\nControl (Stockert A vs Stockert B, {n} vs {n} looks) vs smoothing "
                "scale (low-pass sigma):", control)
    # Speckle is receiver-specific: ATA and Stockert sit at different sites,
    # so their speckle realizations differ and SHOULD decorrelate. What the
    # two receivers share is the larger-scale reflectivity, so the
    # comparison is swept over the low-pass scale too (shown for reference);
    # the verdict uses the scale chosen on the control.
    test = [compare(st_a, at, lo) for lo in LO_DEG_SWEEP]
    print_sweep(f"\nATA ({n} looks) vs Stockert A ({n} looks) vs smoothing scale:", test)

    control_locked = [c for c in control if c["locked"]]
    if not control_locked:
        raise SystemExit("\nNo verdict: Stockert A does not lock onto Stockert B at any "
                         f"scale (significance below {LOCK_SIGNIFICANCE:g} or peak at the "
                         "search edge), so there is no reference for judging ATA at "
                         f"{n} looks. No figure written.")
    ctrl = max(control_locked, key=lambda c: c["r_aligned"])
    lo_best = ctrl["lo_deg"]
    t = compare(st_a, at, lo_best, keep_maps=True)
    x = t["x"]
    dlon, dlat = x.dx * STEP, x.dy * STEP
    print(f"\nScale chosen on the control: low-pass {lo_best:.1f} deg "
          f"(control corr {ctrl['r_aligned']:+.3f}, significance "
          f"{ctrl['x'].significance:.2f})")
    print(f"ATA vs Stockert A at {lo_best:.1f} deg:")
    print(f"  dlon {dlon:+.3f} deg, dlat {dlat:+.3f} deg  -> {t['offset_deg']:.3f} deg"
          + ("  AT THE SEARCH EDGE (not a measurement)" if x.at_edge else ""))
    print(f"  xcorr peak {x.peak:.3f}, significance {x.significance:.2f} "
          f"({'lock' if t['locked'] else 'NO lock'}; lock needs >= {LOCK_SIGNIFICANCE:g} "
          "and a peak inside the search window)")
    print(f"  band-passed correlation: {t['r_raw']:+.3f} (raw) "
          f"-> {t['r_aligned']:+.3f} (aligned)")
    if ctrl["r_aligned"] > 0:
        frac = t["r_aligned"] / ctrl["r_aligned"]
        print(f"  vs same-receiver control {ctrl['r_aligned']:+.3f}: "
              f"ATA reaches {100 * frac:.0f}% of the control's agreement")
    else:
        frac = float("nan")
        print(f"  same-receiver control correlation {ctrl['r_aligned']:+.3f} is not "
              "positive: the ratio to it is undefined")

    if x.at_edge:
        verdict = ("no measurement -- the cross-correlation peak is at the edge of the "
                   f"+-{SEARCH_PX * STEP:g} deg search window.")
    elif not t["locked"]:
        verdict = ("no lock -- the ATA/Stockert cross-correlation peak is not "
                   f"significantly above its sidelobes (significance {x.significance:.2f}); "
                   "the offset above is not a measurement, and agreement cannot be "
                   "judged at this look count.")
    elif t["offset_deg"] >= MAX_OFFSET_DEG:
        verdict = (f"misregistered -- locked at an offset of {t['offset_deg']:.2f} deg, "
                   f"at least {MAX_OFFSET_DEG:g} deg.")
    elif not np.isfinite(frac):
        verdict = (f"locked, offset {t['offset_deg']:.2f} deg; agreement cannot be "
                   "compared with the control.")
    elif frac >= MIN_CONTROL_FRACTION:
        verdict = (f"consistent -- locked, offset {t['offset_deg']:.2f} deg; ATA agrees "
                   "with Stockert about as well as Stockert agrees with itself at "
                   "this look count.")
    else:
        verdict = (f"weak -- locked, offset {t['offset_deg']:.2f} deg, but agreement "
                   f"{100 * frac:.0f}% of the same-receiver control (below "
                   f"{100 * MIN_CONTROL_FRACTION:.0f}%).")
    print("  VERDICT: " + verdict)

    # --- Figure: the two independent band-passed maps + aligned overlay ---
    st_band, at_band, at_band_aligned = t["a"], t["b"], t["b_aligned"]
    extent = [LON_AXIS[0], LON_AXIS[-1], LAT_AXIS[0], LAT_AXIS[-1]]
    nonzero = st_band[st_band != 0]
    if nonzero.size == 0:
        raise SystemExit("the band-passed Stockert map is empty")
    vmax = np.percentile(np.abs(nonzero), 99)
    fig, ax = pl.subplots(1, 3, figsize=(15, 5.2))
    for a, b, title in [(ax[0], st_band, f"Stockert A ({n} looks)"),
                        (ax[1], at_band, f"ATA ({n} looks)")]:
        a.imshow(b, origin="lower", extent=extent, cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_xlabel("lon (deg)")
    ax[0].set_ylabel("lat (deg)")
    # Overlay: Stockert in red channel, ATA-aligned in blue -> purple where
    # both. Values beyond +-vmax (the top 1 % of Stockert's) saturate.
    ov = np.zeros(st_band.shape + (3,))

    def norm(v):
        return np.clip(0.5 + 0.5 * v / vmax, 0, 1)
    ov[..., 0] = norm(st_band)
    ov[..., 2] = norm(at_band_aligned)
    ov[..., 1] = 0.5 * (ov[..., 0] + ov[..., 2])
    ax[2].imshow(ov, origin="lower", extent=extent)
    ax[2].set_title(f"overlay (offset {t['offset_deg']:.2f} deg, sig "
                    f"{x.significance:.1f}{'' if t['locked'] else ', NO lock'}, "
                    f"r={t['r_aligned']:+.2f})")
    ax[2].set_xlabel("lon (deg)")
    fig.suptitle(f"ATA <-> Stockert bistatic registration cross-check, {SESSION} opposite sense, "
                 f"low-pass {lo_best:.1f} deg")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()

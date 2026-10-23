"""
Intra-look Doppler drift experiment.

Splits looks into two half-windows, computes an independent DD image and an
independent rim calibration for each half, and compares the within-look
drift (delta_2 - delta_1) against (a) the rim measurement noise, estimated
from even/odd delay-column splits, and (b) the look-to-look scatter of the
full-look rim offsets (rim_delta_hz) of the same session's looks.

The statistics are like for like, all root-mean-square:
  - drift: its mean (a drift common to every look, e.g. a rate error in the
    Doppler model, which a scatter about the mean would hide), its rms about
    the mean (what varies look to look) and its rms about zero;
  - noise: each look's even/odd split gives one estimate of its drift's
    noise variance; their rms is the drift rms that noise alone would give.
An rms that does not exceed the noise is reported as such, with the
(negative) difference of squares, rather than as zero wander.

Looks: the co-pol looks of --date that pass the stacker's gates
(stack_maps.select_looks with --min-snr), among them a resolved timing
peak, a tone off the band edge and a converged rim calibration. That
includes the stacker's session rim-spread test, which drops looks whose
smeared rims mark intra-look wander (the three 2025-06-21 looks of REPORT
section 4), so this measures the drift left in the looks the stack uses.
The stacker's later per-look brightness rejection (stack_maps.
look_brightness) needs the maps' incidence and is not applied. Every look
of the date the gates drop is listed with its reason.

Each half is calibrated as process_file calibrates a look: dea.calibrate_rim
with lag 0 at the terrain under the SRP (the smooth-sphere lt_min minus
extract_srp_elevation's 2h/c, at the half's start time), and a half is used
only if its calibration converged. The noise estimate re-measures the rims
over the even and over the odd delay columns at the half's applied offset
(measure_rim_offset with col_parity, production's gates).

What still differs from production: each half is half the capture -- half
the integration time, so twice the Doppler resolution width and a weaker
rim; both halves use the full look's measured timing and frequency offsets
(applied_shift_samples, applied_df_hz); and the even/odd noise estimate has
no production counterpart.

Usage (from the repo root):
    .conda/bin/python intra_look_drift.py --run-dir results/STOCKERT_2025 \\
        --station stockert --date 2025-06-21 --n-files 6
"""

import argparse
import os

import numpy as np
from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import stack_maps as sm

# The rim scan-window anchor process_file uses (Hz).
RIM_CAPTURE_HZ = dea.RIM_CAPTURE_HZ


def calibrate_half(log_A, dlt_shifts, dvs, lt_ref, eq, f_hz):
    """One half-window's rim calibration as process_file does it, plus the
    even/odd column re-measure at its applied offset. Returns (delta_dlt,
    noise_dlt, None), or (None, None, reason) when it cannot be used."""
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = eq
    capture = RIM_CAPTURE_HZ / f_hz
    cal = dea.calibrate_rim(log_A, dlt_shifts, dvs, lt_ref, lt_min_eq,
                            delay_up, dlt_up, delay_down, dlt_down,
                            delta_capture_dlt=capture)
    if cal["status"] != "converged":
        return None, None, f"rim calibration {cal['status']}"
    parity = {}
    for p, name in ((0, "even"), (1, "odd")):
        parity[name] = dea.measure_rim_offset(
            log_A, dlt_shifts - cal["delta_dlt"], dvs, lt_ref, lt_min_eq,
            delay_up, dlt_up, delay_down, dlt_down, col_parity=p,
            delta_capture_dlt=capture)
        if parity[name] is None:
            return None, None, f"rims fail their gates over the {name} columns"
    # One estimate of the half's rim-offset noise (rms sense): the even and
    # odd halves of the columns each have twice the variance of the full
    # estimate, so |d_even - d_odd| / 2 has its variance in expectation.
    noise = abs(parity["even"]["delta_dlt"] - parity["odd"]["delta_dlt"]) / 2
    return cal["delta_dlt"], noise, None


def report_excess(label, rms, noise_rms):
    """Wander beyond the noise, or the statement that there is none."""
    diff2 = rms ** 2 - noise_rms ** 2
    if diff2 > 0:
        print(f"  {label}: {rms * 1e3:.1f} mHz rms; beyond the noise: "
              f"{np.sqrt(diff2) * 1e3:.1f} mHz")
    else:
        print(f"  {label}: {rms * 1e3:.1f} mHz rms, not above the noise rms "
              f"{noise_rms * 1e3:.1f} mHz (difference of squares "
              f"{diff2 * 1e6:+.1f} mHz^2): no wander beyond the measurement "
              "noise is detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="run directory with registration_runs_<station>_chan1.csv")
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX),
                        help="RX file prefix, as registration_stability.py --station")
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"),
                        help="the data root the run was made from")
    parser.add_argument("--date", default="2025-06-21",
                        help="UTC date of the looks to split (default 2025-06-21: "
                             "66 s looks, the longest of 2025)")
    parser.add_argument("--n-files", type=int, default=6)
    parser.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR,
                        help="weakest tone to use, passed to the stacker's gates "
                             f"(stack_maps.select_looks; default {foh.MIN_TONE_SNR:g}, "
                             "the stack's)")
    args = parser.parse_args()

    rx_name = dea.STATION_RX[args.station]
    # Terrain under the SRP sets lag 0 (as process_file); a missing terrain
    # model is an error.
    dea.load_lola_dem()
    _, decisions = sm.select_looks([args.run_dir], [args.station], ["chan1"],
                                   min_snr=args.min_snr)
    # Captures the batch did not map carry only their names: date them by
    # the file name, as the stacker's sessions are.
    on_date = [(r, why) for r, why in decisions
               if args.date.replace("-", "_") in r["rx_file"]]
    dropped = [(r, why) for r, why in on_date if why]
    selected = [r for r, why in on_date if not why]
    print(f"{args.date}: {len(on_date)} co-pol looks in {args.run_dir}, "
          f"{len(selected)} pass the stacker's gates (tone snr >= {args.min_snr:g})")
    for r, why in dropped:
        print(f"  - {r['rx_file']}: {why}")
    if len(selected) < 2:
        raise SystemExit(f"{len(selected)} looks on {args.date} pass the stacker's "
                         "gates: too few")
    # The look-to-look scale: scatter of the full-look rim offsets over every
    # selected look of the session (not only the ones split below).
    fulls_all = np.array([float(r["rim_delta_hz"]) for r in selected])
    rows = selected[:args.n_files]
    print(f"{len(rows)} of {len(selected)} looks on {args.date} "
          f"({args.station} -> {rx_name}), each split into two halves")

    print(f"{'look':44s} {'d_full':>8s} {'d_h1':>8s} {'d_h2':>8s} "
          f"{'drift':>8s} {'noise':>7s}")
    drifts, noises, unmeasured = [], [], []
    for r in rows:
        path = os.path.join(args.data_root, r["rx_meta_path"])
        (rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(path, args.data_root)
        fs = fs_q.to_value(au.Hz)
        f_hz = freq.to_value(au.Hz)
        et0 = dea.et_from_astropy(rx_start)
        tx_emit_et = dea.et_from_astropy(dea.nominal_tx_emit_start(rx_start))
        shift_s = float(r["applied_shift_samples"]) / fs
        df = float(r["applied_df_hz"])
        half = len(rx) // 2

        deltas, sigmas, failed = [], [], None
        for h in (0, 1):
            rx_h = rx[h * half:(h + 1) * half]
            et_h = et0 + h * half / fs
            (log_A, dlt_shifts, dvs, lt_min, rate,
             _tx_overlap_s) = dea.compute_dd_image(
                rx_h, tx, fs_q, freq, et_h, tx_emit_et, shift_s, 0.0,
                dea.TX_STATION, rx_name, freq_offset_hz=df)
            T_h = half / fs
            eq = dea.compute_doppler_equator_velocity(
                et_h, n_points=500, rx_duration_s=T_h, dlt_rate_srp=rate,
                tx_name=dea.TX_STATION, rx_name=rx_name)
            # As process_file: lag 0 at the terrain under the SRP, 2h/c
            # ahead of the smooth-sphere SRP light time lt_min.
            _, topo_s = dea.extract_srp_elevation(et_h, dea.TX_STATION, rx_name)
            d, noise, why = calibrate_half(log_A, dlt_shifts, dvs, lt_min - topo_s,
                                           eq, f_hz)
            if why is not None:
                failed = f"half {h + 1}: {why}"
                break
            deltas.append(-d * f_hz)
            sigmas.append(noise * f_hz)
        del rx, tx
        if failed is not None:
            unmeasured.append((r["rx_file"], failed))
            print(f"{r['rx_file'][:44]:44s}  {failed}: not used")
            continue
        d_full = float(r["rim_delta_hz"])
        drift = deltas[1] - deltas[0]
        noise = np.hypot(*sigmas)
        drifts.append(drift)
        noises.append(noise)
        print(f"{r['rx_file'][:44]:44s} {d_full:+8.4f} {deltas[0]:+8.4f} "
              f"{deltas[1]:+8.4f} {drift:+8.4f} {noise:7.4f}")

    n = len(drifts)
    if unmeasured:
        print(f"\n{len(unmeasured)} looks not used (a half's rim calibration did "
              "not converge, or its even/odd re-measure failed the gates)")
    if n < 2:
        raise SystemExit(f"{n} looks measured in both halves: too few for statistics")
    drifts = np.array(drifts)
    noises = np.array(noises)
    mean = drifts.mean()
    scatter = drifts.std(ddof=1)
    rms_zero = np.sqrt(np.mean(drifts ** 2))
    noise_rms = np.sqrt(np.mean(noises ** 2))
    print(f"\nintra-look drift over T/2, {n} looks:")
    print(f"  mean (common to all looks): {mean * 1e3:+.1f} mHz "
          f"+- {scatter / np.sqrt(n) * 1e3:.1f} (standard error)")
    print(f"  measurement noise (even/odd column split): {noise_rms * 1e3:.1f} mHz rms")
    report_excess("rms about the mean (look to look)", scatter, noise_rms)
    report_excess("rms about zero (including the mean)", rms_zero, noise_rms)
    print(f"look-to-look scatter of the full-look rim offsets, {len(fulls_all)} "
          f"looks on {args.date}: {fulls_all.std(ddof=1) * 1e3:.1f} mHz rms about "
          f"their mean")


if __name__ == "__main__":
    main()

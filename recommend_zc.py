#!/usr/bin/env python3
"""Recommend Zadoff-Chu length N and root q for a lunar radar look.

The search uses the odd-length convention

    x[n] = exp(-j*pi*q*n*(n+1)/N).

Two checks decide PASS, both over the lunar Doppler span ``doppler_span_hz``
(the full limb-to-limb interval: a DD-image row and an echo component can
differ by up to that much):

1. Local ridge motion. For small Doppler offsets the correlation ridge
   moves by

       delta_chips = N * delta_f / (q * chip_rate)

   across a Doppler interval ``delta_f``; it must stay below the requested
   fraction of one delay cell. q is searched from the smallest root that
   meets this, so every listed root meets it by construction -- it sets
   where the search starts, it is not what can fail.

2. Whole-bin Doppler ghosts. A Doppler offset of exactly j bins
   (j * chip_rate / N Hz, one bin per code period) turns the code into a
   cyclic shift of itself: the full-strength correlation peak moves from the
   true delay to j * q^-1 (mod N) chips (q^-1 the inverse of q modulo N; see
   doppler_ghost_lag for the sign convention). The DD image's delay window
   runs from a lead before lag 0 -- DD_LEAD_SAMPLES samples at the recorded
   sample rate, where the image's noise floor is read -- to the Moon's
   delay depth 2R/c (~11.6 ms) after it, and the echo itself spans 0..2R/c.
   Echo moved by a ghost lag l therefore lands inside the image, at the
   wrong delay, for some part of the disk when -(2R/c + lead) <= l <= 2R/c.
   Offsets come in both Doppler signs (a lag of k chips for +j is N - k for
   -j), so the nearer of k and N - k must exceed 2R/c + lead. Every whole
   bin up to the span is checked, and the nearest ghost is reported.
   Fractional-bin offsets spread their energy over the neighbouring whole
   bins with sinc weights, so ghosts a bin or two beyond the span still
   carry some echo (one bin beyond: about -13 dB); those are not checked.

All reported roots are coprime with N. By default N is chosen prime near
``look_duration * chip_rate`` and q is the smallest root from the ridge
limit upward that passes both checks.

This is an ideal discrete-time design check. Verify the selected pair through
the actual DAC/RF/ADC filters before production use.
"""

from __future__ import annotations

import argparse
import math

# Mean lunar radius, the value doppler_equator_alignment.MOON_RADIUS uses;
# the echo spans 2R/c of round-trip delay from the nearest point to the limb,
# where compute_dd_image's delay window ends.
MOON_RADIUS_M = 1_737_400.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
LUNAR_DELAY_DEPTH_S = 2.0 * MOON_RADIUS_M / SPEED_OF_LIGHT_M_S

# Samples the DD image's delay window leads lag 0 by: the value of
# doppler_equator_alignment.LEAD_SAMPLES (delay_window). Not imported, which
# would load CuPy and SPICE; test/test_recommend_zc.py checks both values
# above against that module's source.
DD_LEAD_SAMPLES = 20


def dd_lead_s(sample_rate: float) -> float:
    """The DD window's lead before lag 0 (s) at a recorded sample rate (Hz)."""
    return DD_LEAD_SAMPLES / sample_rate


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for unsigned 64-bit n."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2
    # Deterministic for n < 2**64.
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def nearby_primes(target: int, count: int, radius: int) -> list[int]:
    candidates: list[int] = []
    for distance in range(radius + 1):
        values = (target,) if distance == 0 else (target - distance, target + distance)
        for value in values:
            if value > 2 and value % 2 and is_prime(value):
                candidates.append(value)
        if len(candidates) >= count:
            break
    if not candidates:
        raise ValueError(f"no prime N found within {radius} chips of {target}")
    return sorted(set(candidates), key=lambda value: (abs(value - target), value))[:count]


def next_coprime(start: int, n: int) -> int:
    q = max(1, start)
    while q < n and math.gcd(q, n) != 1:
        q += 1
    if q >= n:
        raise ValueError("no valid root below N")
    return q


def root_metrics(n: int, q: int, chip_rate: float, doppler_span: float) -> tuple[float, float]:
    shift_chips = n * doppler_span / (q * chip_rate)
    slope_us_hz = 1e6 * n / (q * chip_rate**2)
    return shift_chips, slope_us_hz


def doppler_ghost_lag(n: int, q: int, j: int) -> int:
    """Lag (chips, 0..n-1) at which an echo offset by j whole Doppler bins
    correlates at full strength with the code.

    Conventions (those of compute_dd_image): the code is
    x[m] = exp(-j*pi*q*m*(m+1)/n); an echo offset by +j bins is
    x[m] * exp(+2*pi*i*j*m/n); the correlation is
    c[l] = sum_m r[m] * conj(x[m - l]), so a positive lag is a later arrival.
    Then x[m] * exp(2*pi*i*j*m/n) equals x[m - l] times a constant for
    l = j * q^-1 (mod n): all of the peak sits there and none at lag 0. A
    negative j gives n - l. test/test_recommend_zc.py checks this against a
    direct correlation.
    """
    return (j * pow(q, -1, n)) % n


def nearest_doppler_ghost(n: int, q: int, chip_rate: float,
                          doppler_span_hz: float) -> tuple[int, int]:
    """The whole-bin Doppler ghost closest to the true delay, over every
    offset of 1..ceil(span / bin) bins (bin = chip_rate / n Hz; -j gives the
    same distance). Returns (j, ghost_chips), ghost_chips being the smaller
    of the ghost lag and n minus it."""
    j_max = math.ceil(doppler_span_hz * n / chip_rate)
    best = None
    for j in range(1, j_max + 1):
        k = doppler_ghost_lag(n, q, j)
        distance = min(k, n - k)
        if best is None or distance < best[1]:
            best = (j, distance)
    return best


def evaluate_root(n: int, q: int, chip_rate: float, doppler_span_hz: float,
                  max_coupling_cell: float, *, lead_s: float) -> dict:
    """Both checks for one (N, q): ridge motion within max_coupling_cell
    and every whole-bin Doppler ghost farther from the true delay than the
    lunar delay depth plus the DD window's lead lead_s (dd_lead_s; see the
    module docstring)."""
    shift, slope = root_metrics(n, q, chip_rate, doppler_span_hz)
    ghost_j, ghost_chips = nearest_doppler_ghost(n, q, chip_rate, doppler_span_hz)
    ghost_s = ghost_chips / chip_rate
    ghost_limit_s = LUNAR_DELAY_DEPTH_S + lead_s
    coupling_ok = shift <= max_coupling_cell
    ghost_ok = ghost_s > ghost_limit_s
    return {"n": n, "q": q, "shift_cell": shift, "slope_us_hz": slope,
            "ghost_j": ghost_j, "ghost_chips": ghost_chips, "ghost_s": ghost_s,
            "ghost_limit_s": ghost_limit_s,
            "coupling_ok": coupling_ok, "ghost_ok": ghost_ok,
            "passed": coupling_ok and ghost_ok}


def status_of(result: dict) -> str:
    if result["passed"]:
        return "PASS"
    failed = [name for name, ok in (("ridge", result["coupling_ok"]),
                                    ("ghost", result["ghost_ok"])) if not ok]
    return "FAIL:" + "+".join(failed)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chip-rate", type=float, default=250_000, help="chips/s (default: 250000)")
    p.add_argument("--sample-rate", type=float, default=250_000,
                   help="the receiver's recorded sample rate, samples/s; sets the DD "
                        f"window's lead of {DD_LEAD_SAMPLES} samples before lag 0 "
                        "(default: 250000, the 0.25 Msps of the 2025-2026 captures)")
    p.add_argument("--look-duration", type=float, default=30, help="coherent look seconds (default: 30)")
    p.add_argument("--doppler-span-hz", type=float, default=19,
                   help="full limb-to-limb lunar Doppler interval to cover, Hz "
                        "(default: 19)")
    p.add_argument("--max-coupling-cell", type=float, default=0.05,
                   help="maximum ridge motion across the span, in delay cells (default: 0.05)")
    p.add_argument("--lengths", type=int, default=5, help="number of nearby prime N values to show")
    p.add_argument("--roots", type=int, default=5, help="number of q values per N to show")
    p.add_argument("--root-search", type=int, default=10_000,
                   help="roots tried per N, from the ridge limit upward, when "
                        "looking for one that passes (default: 10000)")
    p.add_argument("--prime-radius", type=int, default=10000,
                   help="maximum distance from nominal N searched for primes")
    return p


def main() -> None:
    args = parser().parse_args()
    if min(args.chip_rate, args.sample_rate, args.look_duration, args.doppler_span_hz,
           args.max_coupling_cell) <= 0:
        raise SystemExit("rates, durations, span, and coupling tolerance must be positive")
    if args.lengths < 1 or args.roots < 1 or args.root_search < 1:
        raise SystemExit("--lengths, --roots and --root-search must be positive")

    nominal = round(args.chip_rate * args.look_duration)
    lengths = nearby_primes(nominal, args.lengths, args.prime_radius)
    range_resolution_m = SPEED_OF_LIGHT_M_S / (2 * args.chip_rate)
    lead_s = dd_lead_s(args.sample_rate)

    def evaluate(n, q):
        return evaluate_root(n, q, args.chip_rate, args.doppler_span_hz,
                             args.max_coupling_cell, lead_s=lead_s)

    print(f"Nominal chips: {nominal:,} ({args.look_duration:g} s at {args.chip_rate:g} chip/s)")
    print(f"Delay cell: {1e6 / args.chip_rate:.3f} us = {range_resolution_m:.1f} m one-way")
    print(f"Doppler span: {args.doppler_span_hz:g} Hz; ridge limit: "
          f"{args.max_coupling_cell:g} cell (met by every listed root: the "
          "search starts at the smallest such q)")
    print(f"Ghost limit: whole-bin Doppler ghosts must lie beyond the lunar delay "
          f"depth 2R/c = {LUNAR_DELAY_DEPTH_S * 1e3:.3f} ms plus the DD window's lead "
          f"({DD_LEAD_SAMPLES} samples at {args.sample_rate:g}/s = {lead_s * 1e6:.0f} us): "
          f"{(LUNAR_DELAY_DEPTH_S + lead_s) * 1e3:.3f} ms")
    print()
    print(f"{'N':>12} {'duration(s)':>12} {'q':>9} {'shift(cell)':>12} "
          f"{'slope(us/Hz)':>13} {'ghost j':>8} {'ghost(ms)':>10} {'status':>14}")

    recommendations, no_root = [], []
    for n in lengths:
        q_min = math.ceil(n * args.doppler_span_hz /
                          (args.chip_rate * args.max_coupling_cell))
        q = next_coprime(q_min, n)
        first_pass = None
        for tried in range(args.root_search):
            result = evaluate(n, q)
            if tried < args.roots:
                print(f"{n:12d} {n / args.chip_rate:12.6f} {q:9d} "
                      f"{result['shift_cell']:12.6f} {result['slope_us_hz']:13.6f} "
                      f"{result['ghost_j']:8d} {result['ghost_s'] * 1e3:10.3f} "
                      f"{status_of(result):>14}")
            if result["passed"] and first_pass is None:
                first_pass = result
            if first_pass is not None and tried >= args.roots - 1:
                break
            if q + 1 >= n:
                break
            q = next_coprime(q + 1, n)
        if first_pass is None:
            no_root.append(n)
            print(f"  N={n}: no root passes within {args.root_search} roots from q={q_min}")
        else:
            recommendations.append((abs(n - nominal), first_pass))
            print(f"  N={n}: first passing root q={first_pass['q']}")
        print()

    if not recommendations:
        raise SystemExit(f"no (N, q) passes both checks; N tried: {no_root}")
    best = min(recommendations, key=lambda t: (t[0], t[1]["n"]))[1]
    print(f"Recommended: N={best['n']}, q={best['q']}")
    print(f"  duration={best['n'] / args.chip_rate:.9f} s; "
          f"ridge motion={best['shift_cell']:.6f} cell across "
          f"{args.doppler_span_hz:g} Hz ({best['slope_us_hz']:.6f} us/Hz)")
    print(f"  nearest whole-bin Doppler ghost: {best['ghost_j']} bin(s) "
          f"({best['ghost_j'] * args.chip_rate / best['n']:.4f} Hz) -> "
          f"{best['ghost_chips']} chips = {best['ghost_s'] * 1e3:.3f} ms from the "
          f"true delay (limit {best['ghost_limit_s'] * 1e3:.3f} ms: lunar depth plus "
          "the DD window's lead)")
    print("  gcd(N,q)=1; N is prime. Validate this pair through the real RF chain.")


if __name__ == "__main__":
    main()

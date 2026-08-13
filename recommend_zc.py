#!/usr/bin/env python3
"""Recommend Zadoff-Chu length N and root q for a lunar radar look.

The search uses the odd-length convention

    x[n] = exp(-j*pi*q*n*(n+1)/N).

For this convention, the local delay-Doppler ambiguity ridge moves by

    delta_chips = N * delta_f / (q * chip_rate)

across a Doppler interval ``delta_f``.  All reported roots are coprime with N.
By default N is chosen prime near ``look_duration * chip_rate`` and q is the
smallest root that keeps the ridge motion below the requested fraction of one
delay-resolution cell.  A smaller q is preferred only to avoid choosing an
arbitrarily huge root; roots meeting the tolerance have equivalent ideal
zero-Doppler periodic autocorrelation.

This is an ideal discrete-time design check.  Verify the selected pair through
the actual DAC/RF/ADC filters before production use.
"""

from __future__ import annotations

import argparse
import math


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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip-rate", type=float, default=250_000, help="chips/s (default: 250000)")
    p.add_argument("--look-duration", type=float, default=30, help="coherent look seconds (default: 30)")
    p.add_argument("--doppler-span-hz", type=float, default=19,
                   help="full lunar Doppler interval to cover, Hz (default: 19)")
    p.add_argument("--max-coupling-cell", type=float, default=0.05,
                   help="maximum ridge motion across the span, in delay cells (default: 0.05)")
    p.add_argument("--lengths", type=int, default=5, help="number of nearby prime N values to show")
    p.add_argument("--roots", type=int, default=5, help="number of q values per N to show")
    p.add_argument("--prime-radius", type=int, default=10000,
                   help="maximum distance from nominal N searched for primes")
    return p


def main() -> None:
    args = parser().parse_args()
    if min(args.chip_rate, args.look_duration, args.doppler_span_hz,
           args.max_coupling_cell) <= 0:
        raise SystemExit("rates, durations, span, and coupling tolerance must be positive")
    if args.lengths < 1 or args.roots < 1:
        raise SystemExit("--lengths and --roots must be positive")

    nominal = round(args.chip_rate * args.look_duration)
    lengths = nearby_primes(nominal, args.lengths, args.prime_radius)
    range_resolution_m = 299_792_458.0 / (2 * args.chip_rate)

    print(f"Nominal chips: {nominal:,} ({args.look_duration:g} s at {args.chip_rate:g} chip/s)")
    print(f"Delay cell: {1e6 / args.chip_rate:.3f} us = {range_resolution_m:.1f} m one-way")
    print(f"Doppler span: {args.doppler_span_hz:g} Hz; coupling limit: "
          f"{args.max_coupling_cell:g} cell")
    print()
    print(f"{'N':>12} {'duration(s)':>12} {'q':>9} {'shift(cell)':>12} "
          f"{'slope(us/Hz)':>13} {'status':>8}")

    recommendations = []
    for n in lengths:
        q_min = math.ceil(n * args.doppler_span_hz /
                          (args.chip_rate * args.max_coupling_cell))
        q0 = next_coprime(q_min, n)
        recommendations.append((abs(n - nominal), n, q0))
        for offset in range(args.roots):
            q = next_coprime(q0 + offset, n)
            shift, slope = root_metrics(n, q, args.chip_rate, args.doppler_span_hz)
            status = "PASS" if shift <= args.max_coupling_cell else "FAIL"
            print(f"{n:12d} {n / args.chip_rate:12.6f} {q:9d} "
                  f"{shift:12.6f} {slope:13.6f} {status:>8}")
        print()

    _, best_n, best_q = min(recommendations)
    shift, slope = root_metrics(best_n, best_q, args.chip_rate, args.doppler_span_hz)
    print(f"Recommended: N={best_n}, q={best_q}")
    print(f"  duration={best_n / args.chip_rate:.9f} s; "
          f"coupling={shift:.6f} cell across {args.doppler_span_hz:g} Hz "
          f"({slope:.6f} us/Hz)")
    print("  gcd(N,q)=1; N is prime. Validate this pair through the real RF chain.")


if __name__ == "__main__":
    main()

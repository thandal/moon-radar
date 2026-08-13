import math

import recommend_zc as rz


def test_primality_and_nearby_search():
    assert rz.is_prime(7_500_013)
    assert not rz.is_prime(7_500_000)
    values = rz.nearby_primes(100, 2, 10)
    assert values == [101, 97]


def test_recommended_root_meets_coupling_limit():
    n, rate, span, limit = 7_500_013, 250_000.0, 19.0, 0.05
    q_min = math.ceil(n * span / (rate * limit))
    q = rz.next_coprime(q_min, n)
    shift, slope = rz.root_metrics(n, q, rate, span)
    assert math.gcd(n, q) == 1
    assert shift <= limit
    assert math.isclose(slope * span * rate / 1e6, shift)

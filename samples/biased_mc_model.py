"""A Monte Carlo Model Under Test with a bias that no path count removes.

This is the model that motivates Check Type 4.  It has one defect: the drift of
the simulated log price is ``r - q`` instead of ``r - q - sigma**2 / 2``, so the
simulated asset has the wrong expected value under the risk-neutral measure.

The defect is invisible to a spot check.  At a short maturity and a low
volatility the price is close enough to look right, and the number moves around
between runs, so a reviewer attributes the difference to sampling noise.  Raise
the path count and the noise falls away, leaving the bias standing on its own.

Do not fix this file. Its bias is the test fixture.
"""

import math

import numpy as np

DEFAULT_PATHS = 200_000


def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
    """Price a European option. Converges to the wrong number."""
    if instrument != "european":
        raise ValueError("this model prices European options only")
    phi = 1.0 if option_type == "call" else -1.0

    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        forward = S * math.exp((r - q) * T)
        if T <= 0.0:
            return max(phi * (S - K), 0.0)
        return math.exp(-r * T) * max(phi * (forward - K), 0.0)

    paths = int(kwargs.get("precision") or DEFAULT_PATHS)
    pairs = max(paths // 2, 1)
    rng = np.random.default_rng(20260816)

    z = rng.standard_normal(pairs)
    z = np.concatenate((z, -z))
    # The Ito correction term is missing from the drift.
    terminal = S * np.exp((r - q) * T + sigma * math.sqrt(T) * z)
    payoff = np.maximum(phi * (terminal - K), 0.0)
    return float(math.exp(-r * T) * payoff.mean())

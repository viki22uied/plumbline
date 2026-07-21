"""A correct Monte Carlo Model Under Test with a working precision knob.

This sample exists for Check Type 4.  It is right, it is noisy, and its noise
falls as the path count rises, which is what a convergent method looks like.
Compare it with ``biased_mc_model.py``, which is just as noisy and never gets
closer to the right answer.
"""

import math

import numpy as np

DEFAULT_PATHS = 200_000


def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
    """Price a European option by simulating the terminal price."""
    if instrument != "european":
        raise ValueError("this model prices European options only")
    phi = 1.0 if option_type == "call" else -1.0

    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        # No randomness left, so the payoff is certain.
        forward = S * math.exp((r - q) * T)
        if T <= 0.0:
            return max(phi * (S - K), 0.0)
        return math.exp(-r * T) * max(phi * (forward - K), 0.0)

    paths = int(kwargs.get("precision") or DEFAULT_PATHS)
    pairs = max(paths // 2, 1)
    rng = np.random.default_rng(20260816)

    z = rng.standard_normal(pairs)
    z = np.concatenate((z, -z))  # antithetic variates
    terminal = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * z)
    payoff = np.maximum(phi * (terminal - K), 0.0)
    return float(math.exp(-r * T) * payoff.mean())

"""A deliberately broken Model Under Test, with five seeded pricing errors.

This is the AC-04 sample.  It looks like a competent Black-Scholes
implementation, it runs without raising, and it returns plausible numbers.
Every one of those properties is what makes a wrong model dangerous, and none
of them is evidence that it is right.

The five seeded errors, and the check type that must catch each one:

======  ===========================================================  ==========
Error   What is wrong                                                Check type
======  ===========================================================  ==========
E1      the volatility term uses ``sigma * T`` instead of              1 and 3
        ``sigma * sqrt(T)``, so every maturity except one year is
        priced with the wrong amount of diffusion
E2      the put branch does not discount the strike                    2
E3      zero time to expiry returns 0.0 instead of the intrinsic       5
        value
E4      zero volatility returns 0.0 instead of the discounted          5
        deterministic payoff
E5      a "skew adjustment" proportional to the strike is added to     6
        every call, which makes the call price rise with the strike
======  ===========================================================  ==========

Do not fix this file. Its errors are the test fixture.
"""

import math


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
    """Price a European option. Contains five deliberate errors."""
    # E3: an option at expiry is worth its payoff, not nothing.
    if T <= 0.0:
        return 0.0
    # E4: at zero volatility the payoff is certain, not worthless.
    if sigma <= 0.0:
        return 0.0
    if S <= 0.0 or K <= 0.0:
        return 0.0

    # E1: the diffusion term of a Brownian motion scales with sqrt(T), not T.
    vol_time = sigma * T

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_time
    d2 = d1 - vol_time

    if option_type == "call":
        value = S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        # E5: a strike-proportional "skew adjustment" with no basis in any
        # arbitrage-free model. It makes the call price increase with strike.
        value += 0.02 * K * math.exp(-r * T)
    else:
        # E2: the strike is not discounted on the put branch.
        value = K * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

    return value

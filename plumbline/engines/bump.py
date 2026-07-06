"""Bump-and-reprice Greeks (FR-B-07, FR-C-07).

One shared implementation serves two callers:

* a Ground Truth Engine that has no closed-form Greek for its instrument, and
* the Validation Engine, which must derive the Model Under Test's Greeks from
  the Model Under Test's *own* price function and nothing else.

Central differences are used everywhere, so the truncation error is O(h^2).
The default steps are relative for the two multiplicative parameters (spot and
volatility) and absolute for the two additive ones (rate and time).
"""

from __future__ import annotations

import math
from typing import Callable

from plumbline.contracts import Greeks, OptionSpec

PriceFn = Callable[[OptionSpec], float]

#: Relative bump for spot, floored so a tiny spot still gets a usable step.
SPOT_BUMP = 1e-4
#: Absolute bumps.
VOL_BUMP = 1e-4
RATE_BUMP = 1e-4
TIME_BUMP = 1e-4


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def bump_greeks(
    price_fn: PriceFn,
    spec: OptionSpec,
    bump: float | None = None,
) -> Greeks:
    """Return the five core Greeks of ``price_fn`` around ``spec``.

    ``bump`` overrides the relative spot step, which is the one a caller may
    want to widen when ``price_fn`` is noisy (a Monte Carlo model, say).
    Any sensitivity whose repricing fails or leaves the valid parameter domain
    comes back as NaN rather than as a wrong number.
    """
    spot_bump = bump if bump is not None else SPOT_BUMP

    def value(**changes) -> float:
        try:
            out = price_fn(spec.with_(**changes))
        except Exception:
            return math.nan
        return float(out) if _finite(out) else math.nan

    base = value()

    # --- delta and gamma: central and second central difference in S ---------
    h_s = max(spot_bump * spec.S, 1e-6)
    up, down = value(S=spec.S + h_s), value(S=max(spec.S - h_s, 0.0))
    delta = (up - down) / (2.0 * h_s)
    gamma = (up - 2.0 * base + down) / (h_s * h_s)

    # --- vega: central difference in sigma, one-sided at the zero-vol wall ---
    h_v = VOL_BUMP
    if spec.sigma - h_v < 0.0:
        vega = (value(sigma=spec.sigma + h_v) - base) / h_v
    else:
        vega = (value(sigma=spec.sigma + h_v) - value(sigma=spec.sigma - h_v)) / (2.0 * h_v)

    # --- theta: minus the derivative in T, since calendar time runs down -----
    h_t = TIME_BUMP
    if spec.T - h_t <= 0.0:
        theta = -(value(T=spec.T + h_t) - base) / h_t
    else:
        theta = -(value(T=spec.T + h_t) - value(T=spec.T - h_t)) / (2.0 * h_t)

    # --- rho: central difference in r ---------------------------------------
    h_r = RATE_BUMP
    rho = (value(r=spec.r + h_r) - value(r=spec.r - h_r)) / (2.0 * h_r)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

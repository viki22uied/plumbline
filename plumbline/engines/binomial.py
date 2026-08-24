"""Cox-Ross-Rubinstein binomial tree engine (FR-B-02).

Reference: Cox, Ross & Rubinstein (1979); Hull, *Options, Futures, and Other
Derivatives*, ch. 21.  This is the Ground Truth Engine for American exercise,
where no closed form exists.

The tree is built in log space with ``u = exp(sigma * sqrt(dt))`` and
``d = 1 / u``, so the lattice recombines exactly and the node prices are
``S * u**(2j - i)`` at step ``i``.

Two corrections from Broadie & Detemple (1996) sit on top of that lattice, and
neither is optional for an engine other models are judged against.

**Binomial Black-Scholes (BBS).**  A plain tree puts the kink of the payoff
somewhere between two terminal nodes, and exactly where it falls depends on
the step count.  The result is the well-known CRR sawtooth: the error changes
sign between one step count and the next and does not shrink between them.  At
800 steps that oscillation reaches 2.5e-3 in absolute price, which is larger
than the tolerance this engine's own answers are used to enforce.  Replacing
the final step with the Black-Scholes value at each node integrates the payoff
exactly over that last interval and removes the oscillation.

**Richardson extrapolation (BBSR).**  What survives BBS is a smooth error of
order ``1/steps``.  Two runs, at ``steps`` and ``steps / 2``, combine as
``2 V(n) - V(n/2)`` to cancel it.

Together these take the worst relative error over the default audit grid from
4.0e-3 to 4.2e-5, and the standard American put benchmark from 6.08940 to
6.09046 against a true 6.0903.  Without them this engine could not meet NFR-01
at its own production settings, and a correct American model would be failed
by an audit that used it as the reference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from plumbline.contracts import Greeks, OptionSpec, PriceResult, UnsupportedInstrument

DEFAULT_STEPS = 800
#: Below this many steps the lattice Greeks are meaningless.
MIN_STEPS_FOR_GREEKS = 8
#: Richardson needs a half-size run as well, so a usable tree needs this many.
MIN_STEPS_FOR_RICHARDSON = 16

SUPPORTED = ("european", "american")


def _degenerate_value(spec: OptionSpec) -> float | None:
    """Exact value for the corners where a tree cannot be built."""
    if spec.T <= 0.0:
        return max(spec.phi * (spec.S - spec.K), 0.0)
    if spec.S <= 0.0:
        # Zero is absorbing under GBM.  An American put is exercised at once.
        if spec.phi > 0:
            return 0.0
        return spec.K if spec.instrument == "american" else spec.K * spec.discount
    if spec.sigma <= 0.0:
        european = spec.discount * max(spec.phi * (spec.forward - spec.K), 0.0)
        if spec.instrument == "european":
            return european
        # American: the best of exercising now and holding to expiry, because
        # the deterministic path is monotone in time.
        return max(european, max(spec.phi * (spec.S - spec.K), 0.0))
    return None


def _black_scholes_layer(
    prices: np.ndarray, K: float, dt: float, r: float, q: float, sigma: float, phi: float
) -> np.ndarray:
    """Black-Scholes value at each node, one step from expiry.

    Vectorised on purpose: this replaces the terminal payoff layer, so it runs
    once per tree over ``steps`` nodes.
    """
    vol_time = sigma * math.sqrt(dt)
    d1 = (np.log(prices / K) + (r - q + 0.5 * sigma * sigma) * dt) / vol_time
    d2 = d1 - vol_time
    return phi * (
        prices * math.exp(-q * dt) * norm.cdf(phi * d1)
        - K * math.exp(-r * dt) * norm.cdf(phi * d2)
    )


def _rollback(spec: OptionSpec, steps: int) -> tuple[float, float, float, float]:
    """One BBS tree. Returns (value, delta, gamma, theta)."""
    S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma
    phi = spec.phi
    american = spec.instrument == "american"

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        # dt too coarse for this drift: the tree would admit arbitrage.
        raise ValueError(
            f"CRR risk-neutral probability p={p:.6f} left [0, 1]; "
            f"increase the step count above {steps}"
        )

    # Every node price on the lattice is S * u**k for an integer k in
    # [-steps, steps], so one table of powers serves every time level. Building
    # it once turns the American exercise test from an array exponentiation per
    # step -- O(steps^2) calls to pow -- into a slice of precomputed values.
    exponents = np.arange(-steps, steps + 1, dtype=float)
    price_table = S * np.power(u, exponents)
    intrinsic_table = np.maximum(phi * (price_table - K), 0.0) if american else None

    def level_prices(level: int) -> np.ndarray:
        """Node prices at ``level``, as a stride-2 view of the table."""
        return price_table[steps - level : steps + level + 1 : 2]

    def level_intrinsic(level: int) -> np.ndarray:
        return intrinsic_table[steps - level : steps + level + 1 : 2]

    # Binomial Black-Scholes start-up: the last step is not rolled back from a
    # kinked payoff, it is valued exactly. This is what removes the sawtooth.
    node_prices = level_prices(steps - 1)
    values = _black_scholes_layer(node_prices, K, dt, r, q, sigma, phi)
    if american:
        np.maximum(values, level_intrinsic(steps - 1), out=values)

    snapshots: dict[int, np.ndarray] = {}
    for i in range(steps - 2, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            np.maximum(values, level_intrinsic(i), out=values)
        if i <= 2:
            snapshots[i] = values.copy()

    value = float(snapshots[0][0])

    if steps < MIN_STEPS_FOR_GREEKS:
        return value, math.nan, math.nan, math.nan

    v1 = snapshots[1]
    v2 = snapshots[2]
    s_up, s_down = S * u, S * d
    delta = (v1[1] - v1[0]) / (s_up - s_down)

    s_uu, s_ud, s_dd = S * u * u, S, S * d * d
    upper = (v2[2] - v2[1]) / (s_uu - s_ud)
    lower = (v2[1] - v2[0]) / (s_ud - s_dd)
    gamma = (upper - lower) / (0.5 * (s_uu - s_dd))

    theta = (v2[1] - value) / (2.0 * dt)
    return value, float(delta), float(gamma), float(theta)


def binomial_price(spec: OptionSpec, steps: int | None = None) -> float:
    if spec.instrument not in SUPPORTED:
        raise UnsupportedInstrument(
            f"the binomial engine prices {SUPPORTED}, not {spec.instrument!r}"
        )
    corner = _degenerate_value(spec)
    if corner is not None:
        return corner
    steps = max(int(steps or spec.precision or DEFAULT_STEPS), 2)

    if steps < MIN_STEPS_FOR_RICHARDSON:
        # Too coarse to extrapolate from; one tree is all there is.
        return _rollback(spec, steps)[0]

    # Richardson: the BBS error is smooth and of order 1 / steps, so two runs
    # cancel it. Convergence tests drive this engine at rising step counts and
    # must still see the error fall, which it does.
    #
    # Both counts are forced even, and the fine one is exactly twice the
    # coarse. An even tree with S = K puts the strike on a terminal node and an
    # odd one puts it between two, so letting the parity of either count vary
    # leaves a residual wobble with a period of four in the requested steps.
    # Pinning both parities removes it.
    coarse_steps = max(2 * round(steps / 4), 2)
    fine = _rollback(spec, 2 * coarse_steps)[0]
    coarse = _rollback(spec, coarse_steps)[0]
    return 2.0 * fine - coarse


def binomial_greeks(spec: OptionSpec, steps: int | None = None) -> Greeks:
    """Lattice Greeks for delta, gamma and theta; bumped values for vega and rho.

    Delta, gamma and theta are read straight off the first two tree levels,
    which is both cheaper and more stable than repricing the whole tree.
    Vega and rho need a rebuild, so they use a wide bump: a narrow one would
    only measure the lattice's own discretisation wobble.
    """
    corner = _degenerate_value(spec)
    if corner is not None:
        from plumbline.engines.bump import bump_greeks

        return bump_greeks(lambda s: binomial_price(s, steps), spec)

    steps = int(steps or spec.precision or DEFAULT_STEPS)
    _, delta, gamma, theta = _rollback(spec, max(steps, 1))

    h_v, h_r = 1e-2, 1e-3
    vega = (
        binomial_price(spec.with_(sigma=spec.sigma + h_v), steps)
        - binomial_price(spec.with_(sigma=max(spec.sigma - h_v, 1e-8)), steps)
    ) / (2.0 * h_v)
    rho = (
        binomial_price(spec.with_(r=spec.r + h_r), steps)
        - binomial_price(spec.with_(r=spec.r - h_r), steps)
    ) / (2.0 * h_r)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def price(spec: OptionSpec, with_greeks: bool = True) -> PriceResult:
    steps = int(spec.precision or DEFAULT_STEPS)
    return PriceResult(
        price=binomial_price(spec, steps),
        greeks=binomial_greeks(spec, steps) if with_greeks else None,
        engine="binomial_crr",
        diagnostics={"steps": steps},
    )

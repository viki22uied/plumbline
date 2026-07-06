"""Cox-Ross-Rubinstein binomial tree engine (FR-B-02).

Reference: Cox, Ross & Rubinstein (1979); Hull, *Options, Futures, and Other
Derivatives*, ch. 21.  This is the Ground Truth Engine for American exercise,
where no closed form exists.

The tree is built in log space with ``u = exp(sigma * sqrt(dt))`` and
``d = 1 / u``, so the lattice recombines exactly and the node prices are
``S * u**(2j - i)`` at step ``i``.
"""

from __future__ import annotations

import math

import numpy as np

from plumbline.contracts import Greeks, OptionSpec, PriceResult, UnsupportedInstrument

DEFAULT_STEPS = 800
#: Below this many steps the lattice Greeks are meaningless.
MIN_STEPS_FOR_GREEKS = 4

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


def _rollback(spec: OptionSpec, steps: int) -> tuple[float, float, float, float]:
    """Backward induction. Returns (value, delta, gamma, theta)."""
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

    j = np.arange(steps + 1)
    prices = S * u ** (2.0 * j - steps)
    values = np.maximum(phi * (prices - K), 0.0)

    snapshots: dict[int, np.ndarray] = {}
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            node_prices = S * u ** (2.0 * np.arange(i + 1) - i)
            values = np.maximum(values, phi * (node_prices - K))
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
    steps = int(steps or spec.precision or DEFAULT_STEPS)
    return _rollback(spec, max(steps, 1))[0]


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


def price(spec: OptionSpec) -> PriceResult:
    steps = int(spec.precision or DEFAULT_STEPS)
    return PriceResult(
        price=binomial_price(spec, steps),
        greeks=binomial_greeks(spec, steps),
        engine="binomial_crr",
        diagnostics={"steps": steps},
    )

"""Monte Carlo engine with variance reduction (FR-B-03).

Two variance reduction techniques are always available and both are on by
default:

* **Antithetic variates** -- every normal draw ``z`` is used together with
  ``-z``.  A pair is averaged into one sample before any statistic is taken,
  so the reported standard error stays honest.
* **Control variates** -- a correlated payoff with a known expectation is
  subtracted, with the variance-minimising coefficient estimated from the same
  sample (Glasserman, *Monte Carlo Methods in Financial Engineering*, ch. 4).

Path-dependent instruments use exact Brownian-bridge corrections rather than
naive discrete monitoring:

* barrier survival uses the bridge no-hit probability
  ``1 - exp(-2 (x_i - h)(x_{i+1} - h) / (sigma^2 dt))``;
* lookback extremes are sampled exactly from the bridge maximum law.

Without those corrections a discretely monitored path would price a continuous
barrier or lookback with an O(1/sqrt(steps)) bias that no path count can fix.

Memory is O(paths), not O(paths * steps): the simulation carries running state
forward one step at a time and never stores a full path matrix.
"""

from __future__ import annotations

import math

import numpy as np

from plumbline.contracts import (
    OptionSpec,
    PriceResult,
    UnsupportedInstrument,
    PlumblineError,
)
from plumbline.engines.analytic import (
    black_scholes_price,
    geometric_asian_price,
)

DEFAULT_PATHS = 200_000
DEFAULT_STEPS = 252

SUPPORTED = ("european", "asian", "barrier", "digital", "lookback")


class MCResult:
    """Price plus its Monte Carlo standard error."""

    __slots__ = ("price", "stderr", "paths", "steps", "control_beta")

    def __init__(self, price: float, stderr: float, paths: int, steps: int, beta: float):
        self.price = price
        self.stderr = stderr
        self.paths = paths
        self.steps = steps
        self.control_beta = beta

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "stderr": self.stderr,
            "paths": self.paths,
            "steps": self.steps,
            "control_beta": self.control_beta,
        }


# ---------------------------------------------------------------------------
# path simulation
# ---------------------------------------------------------------------------


def _needs_path(spec: OptionSpec) -> bool:
    return spec.instrument in ("asian", "barrier", "lookback")


def _simulate(spec: OptionSpec, n_pairs: int, n_steps: int, rng: np.random.Generator):
    """Run the simulation and return the per-path payoff and control arrays.

    Returns ``(payoff, control, control_mean)``; ``control`` is ``None`` when
    the instrument has no usable control variate.
    """
    S, T, r, q, sigma = spec.S, spec.T, spec.r, spec.q, spec.sigma
    drift_total = (r - q - 0.5 * sigma * sigma) * T

    if not _needs_path(spec):
        z = rng.standard_normal(n_pairs)
        z = np.concatenate((z, -z))
        log_ST = math.log(S) + drift_total + sigma * math.sqrt(T) * z
        ST = np.exp(log_ST)
        payoff = _terminal_payoff(spec, ST)
        # Control: the discounted terminal spot, whose mean is S * exp(-q T).
        control = math.exp(-r * T) * ST
        return payoff, control, S * math.exp(-q * T)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)
    var_step = sigma * sigma * dt

    n_paths = 2 * n_pairs
    x = np.full(n_paths, math.log(S))
    x_prev = np.empty(n_paths)

    arith_sum = np.zeros(n_paths) if spec.instrument == "asian" else None
    geo_sum = np.zeros(n_paths) if spec.instrument == "asian" else None
    run_max = np.full(n_paths, math.log(S)) if spec.instrument == "lookback" else None
    run_min = np.full(n_paths, math.log(S)) if spec.instrument == "lookback" else None
    survival = np.ones(n_paths) if spec.instrument == "barrier" else None

    if spec.instrument == "barrier":
        log_h = math.log(float(spec.barrier))
        is_down = spec.barrier_kind.startswith("down")

    for _ in range(n_steps):
        z = rng.standard_normal(n_pairs)
        z = np.concatenate((z, -z))
        np.copyto(x_prev, x)
        x += drift + vol_step * z

        if spec.instrument == "asian":
            np.add(arith_sum, np.exp(x), out=arith_sum)
            np.add(geo_sum, x, out=geo_sum)

        elif spec.instrument == "lookback":
            # Exact bridge extremes for the interval just simulated.
            u = rng.random(n_paths)
            np.maximum(u, 1e-300, out=u)  # log(0) would blow the span up
            span = np.sqrt((x - x_prev) ** 2 - 2.0 * var_step * np.log(u))
            np.maximum(run_max, 0.5 * (x_prev + x + span), out=run_max)
            np.minimum(run_min, 0.5 * (x_prev + x - span), out=run_min)

        elif spec.instrument == "barrier":
            gap_prev = x_prev - log_h
            gap_now = x - log_h
            breached = (gap_now <= 0.0) if is_down else (gap_now >= 0.0)
            step_survival = 1.0 - np.exp(
                -2.0 * np.maximum(gap_prev * gap_now, 0.0) / var_step
            )
            step_survival[breached] = 0.0
            survival *= step_survival

    ST = np.exp(x)
    disc = math.exp(-r * T)

    if spec.instrument == "asian":
        arith_avg = arith_sum / n_steps
        geo_avg = np.exp(geo_sum / n_steps)
        average = arith_avg if spec.averaging == "arithmetic" else geo_avg
        payoff = disc * np.maximum(spec.phi * (average - spec.K), 0.0)
        # The geometric Asian has a closed form, so it is the natural control
        # for the arithmetic one (Kemna & Vorst 1990).
        control = disc * np.maximum(spec.phi * (geo_avg - spec.K), 0.0)
        control_mean = _discrete_geometric_asian(spec, n_steps)
        return payoff, control, control_mean

    if spec.instrument == "lookback":
        if spec.strike_type == "floating":
            if spec.option_type == "call":
                payoff = disc * (ST - np.exp(run_min))
            else:
                payoff = disc * (np.exp(run_max) - ST)
        else:
            extreme = np.exp(run_max) if spec.option_type == "call" else np.exp(run_min)
            payoff = disc * np.maximum(spec.phi * (extreme - spec.K), 0.0)
        control = disc * np.maximum(spec.phi * (ST - spec.K), 0.0)
        return payoff, control, black_scholes_price(spec.with_(instrument="european"))

    # barrier
    if spec.rebate != 0.0:
        raise PlumblineError(
            "the Monte Carlo barrier engine prices a zero-rebate contract only; "
            "use the analytic engine for a rebate"
        )
    vanilla = np.maximum(spec.phi * (ST - spec.K), 0.0)
    weight = survival if spec.barrier_kind.endswith("out") else (1.0 - survival)
    payoff = disc * vanilla * weight
    control = disc * vanilla
    return payoff, control, black_scholes_price(spec.with_(instrument="european"))


def _terminal_payoff(spec: OptionSpec, ST: np.ndarray) -> np.ndarray:
    disc = math.exp(-spec.r * spec.T)
    if spec.instrument == "european":
        return disc * np.maximum(spec.phi * (ST - spec.K), 0.0)
    if spec.instrument == "digital":
        hit = (ST > spec.K) if spec.phi > 0 else (ST < spec.K)
        if spec.payout == "cash":
            return disc * spec.cash_amount * hit
        return disc * ST * hit
    raise UnsupportedInstrument(
        f"the Monte Carlo engine has no terminal payoff for {spec.instrument!r}"
    )


def _discrete_geometric_asian(spec: OptionSpec, n_fixings: int) -> float:
    """Closed form for a geometric Asian averaged over ``n_fixings`` points.

    The simulation averages at discrete times, so the control's expectation
    must be the discrete-average price, not the continuous Kemna-Vorst value.
    """
    S, K, T, r, sigma = spec.S, spec.K, spec.T, spec.r, spec.sigma
    b = spec.carry
    n = n_fixings
    dt = T / n
    # log of the geometric average is normal with these moments.
    mean = math.log(S) + (b - 0.5 * sigma * sigma) * dt * (n + 1) / 2.0
    var = sigma * sigma * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    if var <= 0.0 or K <= 0.0:
        return math.exp(-r * T) * max(spec.phi * (math.exp(mean) - K), 0.0)
    sd = math.sqrt(var)
    d1 = (mean - math.log(K) + var) / sd
    d2 = d1 - sd
    from plumbline.engines.analytic import _N

    return math.exp(-r * T) * spec.phi * (
        math.exp(mean + 0.5 * var) * _N(spec.phi * d1) - K * _N(spec.phi * d2)
    )


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------


def monte_carlo(
    spec: OptionSpec,
    paths: int | None = None,
    steps: int | None = None,
    seed: int = 12345,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Price ``spec`` by simulation and report the standard error."""
    if spec.instrument not in SUPPORTED:
        raise UnsupportedInstrument(
            f"the Monte Carlo engine prices {SUPPORTED}, not {spec.instrument!r}"
        )
    if spec.T <= 0.0 or spec.sigma <= 0.0 or spec.S <= 0.0:
        # No randomness left: fall back to the exact deterministic value.
        from plumbline.engines.registry import deterministic_fallback

        return MCResult(deterministic_fallback(spec), 0.0, 0, 0, 0.0)

    paths = int(paths or spec.precision or DEFAULT_PATHS)
    steps = int(steps or DEFAULT_STEPS)
    n_pairs = max(paths // 2, 1)
    rng = np.random.default_rng(seed)

    payoff, control, control_mean = _simulate(spec, n_pairs, steps, rng)

    if antithetic:
        # Average each (z, -z) pair into a single sample.
        payoff = 0.5 * (payoff[:n_pairs] + payoff[n_pairs:])
        if control is not None:
            control = 0.5 * (control[:n_pairs] + control[n_pairs:])

    beta = 0.0
    sample = payoff
    if control_variate and control is not None:
        control_var = float(np.var(control, ddof=1))
        if control_var > 1e-16:
            beta = float(np.cov(payoff, control, ddof=1)[0, 1] / control_var)
            sample = payoff - beta * (control - control_mean)

    n = sample.size
    mean = float(np.mean(sample))
    stderr = float(np.std(sample, ddof=1) / math.sqrt(n)) if n > 1 else math.inf
    return MCResult(mean, stderr, 2 * n_pairs if antithetic else n, steps, beta)


def mc_price(spec: OptionSpec, **kwargs) -> float:
    return monte_carlo(spec, **kwargs).price


def price(spec: OptionSpec) -> PriceResult:
    result = monte_carlo(spec)
    from plumbline.engines.bump import bump_greeks

    # A wide bump keeps the finite difference above the simulation noise; the
    # common random numbers (fixed seed) make the difference far less noisy
    # than the raw standard error suggests.
    greeks = bump_greeks(lambda s: mc_price(s), spec, bump=1e-2)
    return PriceResult(
        price=result.price,
        greeks=greeks,
        engine="monte_carlo",
        diagnostics=result.to_dict(),
    )


__all__ = [
    "MCResult",
    "monte_carlo",
    "mc_price",
    "price",
    "geometric_asian_price",
    "SUPPORTED",
]

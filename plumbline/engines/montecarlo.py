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


#: Backends this engine can run on. ``"numpy"`` is the documented default and
#: the reference for what the estimator means. ``"cpp"`` is the optional native
#: library; ``"auto"`` uses it when it is built and covers the contract, and
#: falls back to NumPy otherwise.
BACKENDS = ("numpy", "cpp", "auto")
DEFAULT_BACKEND = "numpy"


class MCResult:
    """Price plus its Monte Carlo standard error."""

    __slots__ = ("price", "stderr", "paths", "steps", "control_beta", "backend", "threads")

    def __init__(
        self,
        price: float,
        stderr: float,
        paths: int,
        steps: int,
        beta: float,
        backend: str = "numpy",
        threads: int = 1,
    ):
        self.price = price
        self.stderr = stderr
        self.paths = paths
        self.steps = steps
        self.control_beta = beta
        self.backend = backend
        self.threads = threads

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "stderr": self.stderr,
            "paths": self.paths,
            "steps": self.steps,
            "control_beta": self.control_beta,
            "backend": self.backend,
            "threads": self.threads,
        }


# ---------------------------------------------------------------------------
# path simulation
# ---------------------------------------------------------------------------


def _needs_path(spec: OptionSpec) -> bool:
    return spec.instrument in ("asian", "barrier", "lookback")


def control_mean(spec: OptionSpec, steps: int) -> float:
    """``E[control]`` for the control variate this engine uses.

    Both backends take the expectation from here rather than computing their
    own. The control variate is only unbiased if the mean subtracted is the
    exact one, so there must be a single implementation of these closed forms
    and it must be this one.
    """
    if not _needs_path(spec):
        # Control is the discounted terminal spot.
        return spec.S * math.exp(-spec.q * spec.T)
    if spec.instrument == "asian":
        # The simulation averages at discrete times, so the control's mean is
        # the discrete-average price, not the continuous Kemna-Vorst value.
        return _discrete_geometric_asian(spec, steps)
    # Barrier and lookback both control on the vanilla European payoff.
    return black_scholes_price(spec.with_(instrument="european"))


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
        return payoff, control, control_mean(spec, n_steps)

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
        return payoff, control, control_mean(spec, n_steps)

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
        return payoff, control, control_mean(spec, n_steps)

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
    return payoff, control, control_mean(spec, n_steps)


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


def _native_monte_carlo(
    spec: OptionSpec,
    paths: int,
    steps: int,
    seed: int,
    antithetic: bool,
    control_variate: bool,
    threads: int,
    required: bool,
) -> MCResult | None:
    """Try the native backend. Return None to fall back to NumPy.

    ``required`` is True when the caller asked for ``backend="cpp"`` by name,
    in which case a missing or refusing library is an error rather than a
    reason to quietly use something else. A caller who asked for the fast path
    and silently got the slow one would draw the wrong conclusion from a
    benchmark.
    """
    from plumbline.engines import native

    try:
        price, stderr, beta, used_paths, used_threads = native.price(
            spec,
            control_mean=control_mean(spec, steps),
            paths=paths,
            steps=steps,
            seed=seed,
            antithetic=antithetic,
            control_variate=control_variate,
            threads=threads,
        )
    except native.NativeBackendError:
        if required:
            raise
        return None

    return MCResult(
        price, stderr, used_paths, steps, beta, backend="cpp", threads=used_threads
    )


def monte_carlo(
    spec: OptionSpec,
    paths: int | None = None,
    steps: int | None = None,
    seed: int = 12345,
    antithetic: bool = True,
    control_variate: bool = True,
    backend: str | None = None,
    threads: int = 0,
) -> MCResult:
    """Price ``spec`` by simulation and report the standard error.

    ``backend`` selects the implementation. ``"numpy"`` is the default and the
    reference. ``"cpp"`` requires the native library and raises if it is not
    there. ``"auto"`` prefers the native library and falls back silently, which
    is the only sensible behaviour for an optional component.

    The two backends draw from different random streams on purpose, so they
    are two independent estimators of the same expectation rather than one
    checking its own arithmetic. Their prices agree within the combined
    standard error, not to the last bit.
    """
    # Resolved here, not in the signature default, so a caller that sets
    # DEFAULT_BACKEND once for the process is obeyed by every later call.
    backend = backend or DEFAULT_BACKEND
    if spec.instrument not in SUPPORTED:
        raise UnsupportedInstrument(
            f"the Monte Carlo engine prices {SUPPORTED}, not {spec.instrument!r}"
        )
    if backend not in BACKENDS:
        raise PlumblineError(f"backend must be one of {BACKENDS}, not {backend!r}")
    if spec.T <= 0.0 or spec.sigma <= 0.0 or spec.S <= 0.0:
        # No randomness left: fall back to the exact deterministic value.
        from plumbline.engines.registry import deterministic_fallback

        return MCResult(deterministic_fallback(spec), 0.0, 0, 0, 0.0, backend="closed_form")

    paths = int(paths or spec.precision or DEFAULT_PATHS)
    steps = int(steps or DEFAULT_STEPS)

    if backend in ("cpp", "auto"):
        native_result = _native_monte_carlo(
            spec, paths, steps, seed, antithetic, control_variate, threads,
            required=(backend == "cpp"),
        )
        if native_result is not None:
            return native_result

    n_pairs = max(paths // 2, 1)
    rng = np.random.default_rng(seed)

    payoff, control, expectation = _simulate(spec, n_pairs, steps, rng)

    if antithetic:
        # Average each (z, -z) pair into a single sample.
        payoff = 0.5 * (payoff[:n_pairs] + payoff[n_pairs:])
        if control is not None:
            control = 0.5 * (control[:n_pairs] + control[n_pairs:])

    beta = 0.0
    sample = payoff
    # A single sample has no variance to estimate, and asking NumPy for one
    # with ddof=1 warns and returns NaN rather than saying so.
    if control_variate and control is not None and payoff.size > 1:
        control_var = float(np.var(control, ddof=1))
        if control_var > 1e-16:
            beta = float(np.cov(payoff, control, ddof=1)[0, 1] / control_var)
            sample = payoff - beta * (control - expectation)

    n = sample.size
    mean = float(np.mean(sample))
    stderr = float(np.std(sample, ddof=1) / math.sqrt(n)) if n > 1 else math.inf
    return MCResult(
        mean, stderr, 2 * n_pairs if antithetic else n, steps, beta, backend="numpy"
    )


def mc_price(spec: OptionSpec, **kwargs) -> float:
    return monte_carlo(spec, **kwargs).price


def price(spec: OptionSpec, with_greeks: bool = True) -> PriceResult:
    result = monte_carlo(spec)
    from plumbline.engines.bump import bump_greeks

    # A wide bump keeps the finite difference above the simulation noise; the
    # common random numbers (fixed seed) make the difference far less noisy
    # than the raw standard error suggests. Nine more simulations is a real
    # cost, so a caller that only needs the price does not pay it.
    greeks = bump_greeks(lambda s: mc_price(s), spec, bump=1e-2) if with_greeks else None
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

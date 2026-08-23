"""Crank-Nicolson finite difference engine (FR-B-04).

The Black-Scholes PDE is solved on a uniform grid in ``x = log S``, where the
diffusion coefficient is constant for the Black-Scholes-Merton model and the
grid spacing is therefore uniform in the variable that actually matters:

    dV/dtau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V

with ``tau = T - t`` running forward from expiry.

Two details are not optional:

* **Rannacher start-up.**  The first two steps run fully implicit before the
  scheme switches to Crank-Nicolson.  Crank-Nicolson is only A-stable, not
  L-stable, so a payoff with a kink or a jump (every option here, and a digital
  especially) produces oscillations in the Greeks that never damp out.  Two
  implicit steps kill them (Rannacher 1984; Giles & Carter 2006).
* **Grid alignment.**  The spot, and the barrier when there is one, are placed
  exactly on grid nodes, so no interpolation error contaminates the reported
  price or the barrier condition.

American exercise uses the explicit projection ``V = max(V, payoff)`` after
each time step (operator splitting).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded

from plumbline.contracts import Greeks, OptionSpec, PriceResult, UnsupportedInstrument

DEFAULT_SPACE_STEPS = 800
DEFAULT_TIME_STEPS = 400
#: Half-width of the log-space domain, in standard deviations of log S.
DOMAIN_WIDTH_SDS = 7.0
#: Number of fully implicit start-up steps (Rannacher).
RANNACHER_STEPS = 2

SUPPORTED = ("european", "american", "digital", "barrier")

LocalVolFn = Callable[[np.ndarray, float], np.ndarray]


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------


def _build_grid(spec: OptionSpec, space_steps: int) -> tuple[np.ndarray, int]:
    """Return the log-space grid and the index of the spot node."""
    x0 = math.log(spec.S)
    sigma_ref = max(spec.sigma, 0.05)
    half_width = DOMAIN_WIDTH_SDS * sigma_ref * math.sqrt(max(spec.T, 1e-4))
    half_width = max(half_width, abs(math.log(spec.K / spec.S)) + 0.5)

    lo, hi = x0 - half_width, x0 + half_width
    barrier_x: float | None = None
    if spec.instrument == "barrier":
        barrier_x = math.log(float(spec.barrier))
        if spec.barrier_kind.startswith("up"):
            hi = barrier_x
        else:
            lo = barrier_x
        if not lo < x0 < hi:
            raise ValueError("the spot must sit strictly inside the barrier domain")

    dx_target = (hi - lo) / space_steps

    if barrier_x is not None:
        # Force both the spot and the barrier onto nodes.
        gap = abs(barrier_x - x0)
        n_between = max(int(round(gap / dx_target)), 1)
        dx = gap / n_between
    else:
        dx = dx_target

    n_lo = max(int(math.ceil((x0 - lo) / dx)), 1)
    n_hi = max(int(math.ceil((hi - x0) / dx)), 1)
    grid = x0 + dx * np.arange(-n_lo, n_hi + 1)
    return grid, n_lo


def _terminal_payoff(spec: OptionSpec, S: np.ndarray, dx: float = 0.0) -> np.ndarray:
    """Payoff at expiry, cell-averaged across a jump.

    A digital payoff jumps inside one grid cell.  Sampling it pointwise makes
    the answer depend on which side of the jump the nearest node happens to
    fall, which is a first-order error that no amount of time stepping removes.
    Averaging the payoff over each cell (Tavella & Randall 2000) restores
    second-order convergence, and costs one extra line per payout type.
    """
    if spec.instrument != "digital" or dx <= 0.0:
        if spec.instrument == "digital":
            hit = (S > spec.K) if spec.phi > 0 else (S < spec.K)
            return spec.cash_amount * hit.astype(float) if spec.payout == "cash" else S * hit
        return np.maximum(spec.phi * (S - spec.K), 0.0)

    x = np.log(S)
    lo, hi = x - 0.5 * dx, x + 0.5 * dx
    log_k = math.log(spec.K)
    # In-the-money sub-interval of each cell.
    itm_lo = np.maximum(lo, log_k) if spec.phi > 0 else lo
    itm_hi = hi if spec.phi > 0 else np.minimum(hi, log_k)
    width = np.maximum(itm_hi - itm_lo, 0.0)
    if spec.payout == "cash":
        return spec.cash_amount * width / dx
    covered = width > 0.0
    return np.where(covered, (np.exp(itm_hi) - np.exp(itm_lo)) / dx, 0.0)


def _boundary_values(spec: OptionSpec, S_lo: float, S_hi: float, tau: float):
    """Dirichlet values at the two domain edges, ``tau`` before expiry."""
    disc_r = math.exp(-spec.r * tau)
    disc_q = math.exp(-spec.q * tau)

    if spec.instrument == "digital":
        if spec.payout == "cash":
            deep_itm = spec.cash_amount * disc_r
            return (0.0, deep_itm) if spec.phi > 0 else (deep_itm, 0.0)
        return (0.0, S_hi * disc_q) if spec.phi > 0 else (S_lo * disc_q, 0.0)

    if spec.phi > 0:  # call
        low = 0.0
        high = S_hi * disc_q - spec.K * disc_r
        if spec.instrument == "american":
            high = max(high, S_hi - spec.K)
        return low, max(high, 0.0)

    low = spec.K * disc_r - S_lo * disc_q  # put
    if spec.instrument == "american":
        low = max(low, spec.K - S_lo)
    return max(low, 0.0), 0.0


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def _solve(
    spec: OptionSpec,
    space_steps: int,
    time_steps: int,
    local_vol: LocalVolFn | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Roll the PDE back to today. Returns (grid_S, values, spot_index)."""
    grid_x, spot_index = _build_grid(spec, space_steps)
    S = np.exp(grid_x)
    dx = grid_x[1] - grid_x[0]
    n = grid_x.size
    dtau = spec.T / time_steps

    values = _terminal_payoff(spec, S, dx)
    exercise = _terminal_payoff(spec, S) if spec.instrument == "american" else None

    knock_out = spec.instrument == "barrier" and spec.barrier_kind.endswith("out")
    barrier_is_up = spec.instrument == "barrier" and spec.barrier_kind.startswith("up")
    if spec.instrument == "barrier":
        # The dead edge is worth the rebate from the moment it is touched.
        if barrier_is_up:
            values[-1] = spec.rebate
        else:
            values[0] = spec.rebate

    interior = slice(1, n - 1)
    S_int = S[interior]

    for step in range(time_steps):
        tau_new = (step + 1) * dtau
        theta = 1.0 if step < RANNACHER_STEPS else 0.5

        if local_vol is None:
            var = np.full(n - 2, spec.sigma * spec.sigma)
        else:
            var = np.asarray(local_vol(S_int, spec.T - tau_new)) ** 2

        drift = spec.r - spec.q - 0.5 * var
        lower = 0.5 * var / (dx * dx) - drift / (2.0 * dx)
        diag = -var / (dx * dx) - spec.r
        upper = 0.5 * var / (dx * dx) + drift / (2.0 * dx)

        # Right-hand side: (I + (1 - theta) dtau L) V_old
        rhs = (
            values[interior]
            + (1.0 - theta)
            * dtau
            * (lower * values[:-2] + diag * values[interior] + upper * values[2:])
        )

        lo_val, hi_val = _boundary_values(spec, S[0], S[-1], tau_new)
        if knock_out:
            # The barrier edge is dead and pays the rebate; the other edge
            # keeps its ordinary far-field value.
            if barrier_is_up:
                hi_val = spec.rebate
            else:
                lo_val = spec.rebate

        # Move the known boundary values of the new time level to the RHS.
        rhs[0] += theta * dtau * lower[0] * lo_val
        rhs[-1] += theta * dtau * upper[-1] * hi_val

        banded = np.zeros((3, n - 2))
        banded[0, 1:] = -theta * dtau * upper[:-1]
        banded[1, :] = 1.0 - theta * dtau * diag
        banded[2, :-1] = -theta * dtau * lower[1:]

        values = np.empty(n)
        values[interior] = solve_banded((1, 1), banded, rhs)
        values[0] = lo_val
        values[-1] = hi_val

        if exercise is not None:
            np.maximum(values, exercise, out=values)

    return S, values, spot_index


def _local_vol_fn(spec: OptionSpec) -> LocalVolFn | None:
    if spec.model != "localvol":
        return None

    def surface(S: np.ndarray, t: float) -> np.ndarray:
        return np.clip(
            spec.lv_a + spec.lv_b * np.log(S / spec.lv_ref) + spec.lv_c * t, 0.01, 3.0
        )

    return surface


def _degenerate(spec: OptionSpec) -> float | None:
    from plumbline.engines.registry import deterministic_fallback

    if spec.T <= 0.0 or spec.sigma <= 0.0 or spec.S <= 0.0 or spec.K <= 0.0:
        return deterministic_fallback(spec)
    return None


def fdm_price(
    spec: OptionSpec,
    space_steps: int | None = None,
    time_steps: int | None = None,
) -> float:
    if spec.instrument not in SUPPORTED:
        raise UnsupportedInstrument(
            f"the finite difference engine prices {SUPPORTED}, not {spec.instrument!r}"
        )
    corner = _degenerate(spec)
    if corner is not None:
        return corner

    if spec.instrument == "barrier" and spec.barrier_kind.endswith("in"):
        # In-out parity: a knock-in is the vanilla minus the matching knock-out.
        out_kind = spec.barrier_kind.replace("-in", "-out")
        knock_out = fdm_price(
            spec.with_(barrier_kind=out_kind, rebate=0.0), space_steps, time_steps
        )
        european = spec.with_(instrument="european")
        if spec.model == "bsm":
            # Constant volatility has an exact vanilla, so use it and keep the
            # only discretisation error in the knock-out leg.
            from plumbline.engines.analytic import black_scholes_price

            vanilla = black_scholes_price(european)
        else:
            # Under a local volatility surface there is no closed-form vanilla.
            # Taking the Black-Scholes one would complete the knock-in with a
            # contract priced off a different volatility, and in-out parity
            # would fail against this engine's own European price.
            vanilla = fdm_price(european, space_steps, time_steps)
        return vanilla - knock_out

    space_steps = int(space_steps or spec.precision or DEFAULT_SPACE_STEPS)
    time_steps = int(time_steps or DEFAULT_TIME_STEPS)
    _, values, index = _solve(spec, space_steps, time_steps, _local_vol_fn(spec))
    return float(values[index])


def fdm_greeks(
    spec: OptionSpec,
    space_steps: int | None = None,
    time_steps: int | None = None,
) -> Greeks:
    """Delta, gamma and theta straight off the grid; vega and rho by bump.

    Reading delta and gamma from the solved grid costs nothing extra, and theta
    follows from the PDE itself: on the solution, ``V_t = r V - 0.5 sigma^2
    V_xx - (r - q - 0.5 sigma^2) V_x``.
    """
    corner = _degenerate(spec)
    if corner is not None or spec.instrument == "barrier":
        from plumbline.engines.bump import bump_greeks

        return bump_greeks(lambda s: fdm_price(s, space_steps, time_steps), spec)

    space_steps = int(space_steps or spec.precision or DEFAULT_SPACE_STEPS)
    time_steps = int(time_steps or DEFAULT_TIME_STEPS)
    S, values, i = _solve(spec, space_steps, time_steps, _local_vol_fn(spec))
    dx = math.log(S[1]) - math.log(S[0])
    S0 = S[i]

    v_x = (values[i + 1] - values[i - 1]) / (2.0 * dx)
    v_xx = (values[i + 1] - 2.0 * values[i] + values[i - 1]) / (dx * dx)
    delta = v_x / S0
    gamma = (v_xx - v_x) / (S0 * S0)

    var = spec.sigma * spec.sigma
    theta = spec.r * values[i] - 0.5 * var * v_xx - (spec.r - spec.q - 0.5 * var) * v_x

    h_v, h_r = 1e-3, 1e-4
    vega = (
        fdm_price(spec.with_(sigma=spec.sigma + h_v), space_steps, time_steps)
        - fdm_price(spec.with_(sigma=max(spec.sigma - h_v, 1e-6)), space_steps, time_steps)
    ) / (2.0 * h_v)
    rho = (
        fdm_price(spec.with_(r=spec.r + h_r), space_steps, time_steps)
        - fdm_price(spec.with_(r=spec.r - h_r), space_steps, time_steps)
    ) / (2.0 * h_r)
    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def price(spec: OptionSpec) -> PriceResult:
    return PriceResult(
        price=fdm_price(spec),
        greeks=fdm_greeks(spec),
        engine="fdm_crank_nicolson",
        diagnostics={"space_steps": DEFAULT_SPACE_STEPS, "time_steps": DEFAULT_TIME_STEPS},
    )

"""Closed-form Ground Truth Engines (FR-B-01, FR-B-05, FR-B-07).

Every formula in this module comes from a published reference:

* Black-Scholes-Merton with continuous dividend yield -- Merton (1973);
  Hull, *Options, Futures, and Other Derivatives*, ch. 15 and 17.
* Cash-or-nothing and asset-or-nothing digitals -- Reiner & Rubinstein (1991b).
* Standard barrier options -- Reiner & Rubinstein (1991a), in the tabulated
  form given by Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.
* Geometric-average Asian options -- Kemna & Vorst (1990).
* Floating-strike lookbacks -- Goldman, Sosin & Gatto (1979).
* Fixed-strike lookbacks -- Conze & Viswanathan (1991).

All prices are per one contract, in units of the underlying's currency.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from plumbline.contracts import Greeks, OptionSpec, PriceResult, UnsupportedInstrument

SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Carry values closer to zero than this make the lookback formulas singular.
#: ponytail: nudge b off zero instead of coding the b -> 0 limit; the induced
#: error is O(1e-8) in price, upgrade to the analytic limit if that ever bites.
_CARRY_FLOOR = 1e-8


def _N(x: float) -> float:
    return float(norm.cdf(x))


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _safe_carry(b: float) -> float:
    if abs(b) < _CARRY_FLOOR:
        return _CARRY_FLOOR if b >= 0.0 else -_CARRY_FLOOR
    return b


# ---------------------------------------------------------------------------
# Black-Scholes-Merton vanilla European
# ---------------------------------------------------------------------------


def bs_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """The two Black-Scholes arguments. Both are NaN in a degenerate case."""
    vol_time = sigma * math.sqrt(T)
    if vol_time <= 0.0 or S <= 0.0 or K <= 0.0:
        return math.nan, math.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_time
    return d1, d1 - vol_time


def black_scholes_price(spec: OptionSpec) -> float:
    """European call/put under Black-Scholes-Merton, with dividend yield.

    Degenerate inputs (``T == 0``, ``sigma == 0``, ``S == 0``) fall back to the
    exact deterministic limit rather than to NaN, because Check Type 5 tests
    exactly those corners.
    """
    S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma
    phi = spec.phi

    if T <= 0.0:
        return max(phi * (S - K), 0.0)
    if S <= 0.0:
        # Spot pinned at zero is absorbing: a call is worthless, a put pays K.
        return 0.0 if phi > 0 else K * math.exp(-r * T)
    if sigma <= 0.0:
        # Deterministic forward: discounted payoff on the forward price.
        return math.exp(-r * T) * max(phi * (spec.forward - K), 0.0)
    if K <= 0.0:
        # A zero-strike call is the discounted forward; a zero-strike put is 0.
        return S * math.exp(-q * T) if phi > 0 else 0.0

    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    return phi * (
        S * math.exp(-q * T) * _N(phi * d1) - K * math.exp(-r * T) * _N(phi * d2)
    )


def black_scholes_greeks(spec: OptionSpec) -> Greeks:
    """Closed-form Greeks for a European option (FR-B-07)."""
    S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma
    phi = spec.phi

    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        # At the boundary the option is a forward or nothing; delta is the only
        # sensitivity that survives and it is a step function.
        in_money = phi * (S - K) > 0.0
        return Greeks(
            delta=(phi if in_money else 0.0),
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
        )

    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    sqrt_T = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = _n(d1)

    delta = phi * disc_q * _N(phi * d1)
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T
    theta = (
        -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
        + phi * q * S * disc_q * _N(phi * d1)
        - phi * r * K * disc_r * _N(phi * d2)
    )
    rho = phi * K * T * disc_r * _N(phi * d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Digital / binary (Reiner & Rubinstein 1991b)
# ---------------------------------------------------------------------------


def digital_price(spec: OptionSpec) -> float:
    """Cash-or-nothing or asset-or-nothing digital.

    Cash-or-nothing pays ``cash_amount`` if the option expires in the money.
    Asset-or-nothing delivers one unit of the asset in the same event.
    """
    S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma
    phi = spec.phi

    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if sigma <= 0.0 and T > 0.0 and S > 0.0 and K > 0.0:
            # Deterministic: the forward decides the payoff.
            in_money = phi * (spec.forward - K) > 0.0
            if not in_money:
                return 0.0
            if spec.payout == "cash":
                return spec.cash_amount * math.exp(-r * T)
            return S * math.exp(-q * T)
        in_money = phi * (S - K) > 0.0
        if not in_money:
            return 0.0
        return spec.cash_amount if spec.payout == "cash" else S

    _, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if spec.payout == "cash":
        return spec.cash_amount * math.exp(-r * T) * _N(phi * d2)
    d1 = d2 + sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _N(phi * d1)


# ---------------------------------------------------------------------------
# Asian, geometric average (Kemna & Vorst 1990)
# ---------------------------------------------------------------------------


def geometric_asian_price(spec: OptionSpec) -> float:
    """Continuously sampled geometric-average-price Asian option.

    The geometric average of a lognormal path is itself lognormal, so the
    option prices as a Black-Scholes option on an asset with volatility
    ``sigma / sqrt(3)`` and cost of carry ``(b - sigma**2 / 6) / 2``.
    """
    S, K, T, r, sigma = spec.S, spec.K, spec.T, spec.r, spec.sigma
    phi = spec.phi

    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(phi * (S - K), 0.0)

    b = spec.carry
    sigma_a = sigma / math.sqrt(3.0)
    b_a = 0.5 * (b - sigma * sigma / 6.0)

    if sigma <= 0.0:
        forward_avg = S * math.exp(b_a * T)
        return math.exp(-r * T) * max(phi * (forward_avg - K), 0.0)

    vol_time = sigma_a * math.sqrt(T)
    d1 = (math.log(S / K) + (b_a + 0.5 * sigma_a * sigma_a) * T) / vol_time
    d2 = d1 - vol_time
    return phi * (
        S * math.exp((b_a - r) * T) * _N(phi * d1) - K * math.exp(-r * T) * _N(phi * d2)
    )


# ---------------------------------------------------------------------------
# Standard barrier options (Reiner & Rubinstein 1991a; Haug 2nd ed. table 4-13)
# ---------------------------------------------------------------------------


def barrier_price(spec: OptionSpec) -> float:
    """Single-barrier continuously monitored option, with optional rebate.

    ``eta`` is +1 for a down-barrier and -1 for an up-barrier.
    ``phi`` is +1 for a call and -1 for a put.
    The rebate is paid at expiry for a knock-in and at the hit for a knock-out,
    which is the convention of the Reiner-Rubinstein table.
    """
    S, K, T, r, q, sigma = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma
    H, R = float(spec.barrier), spec.rebate
    kind = spec.barrier_kind
    phi = spec.phi
    is_down = kind.startswith("down")
    is_out = kind.endswith("out")
    eta = 1.0 if is_down else -1.0

    # Already-dead / already-alive states are settled before the formulas run.
    breached = (S <= H) if is_down else (S >= H)
    if breached:
        if is_out:
            return R
        return black_scholes_price(spec.with_(instrument="european"))

    if T <= 0.0:
        return max(phi * (S - K), 0.0) if is_out else R
    if sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        # Deterministic path S(t) = S * exp(b t): monitor it on a fine grid.
        return _deterministic_barrier(spec, H, R, is_down, is_out)

    b = spec.carry
    v2 = sigma * sigma
    sqrt_T = math.sqrt(T)
    vol_time = sigma * sqrt_T

    mu = (b - 0.5 * v2) / v2
    lam = math.sqrt(mu * mu + 2.0 * r / v2)

    x1 = math.log(S / K) / vol_time + (1.0 + mu) * vol_time
    x2 = math.log(S / H) / vol_time + (1.0 + mu) * vol_time
    y1 = math.log(H * H / (S * K)) / vol_time + (1.0 + mu) * vol_time
    y2 = math.log(H / S) / vol_time + (1.0 + mu) * vol_time
    z = math.log(H / S) / vol_time + lam * vol_time

    disc_r = math.exp(-r * T)
    carry_disc = math.exp((b - r) * T)
    hs = H / S

    A = phi * S * carry_disc * _N(phi * x1) - phi * K * disc_r * _N(phi * (x1 - vol_time))
    B = phi * S * carry_disc * _N(phi * x2) - phi * K * disc_r * _N(phi * (x2 - vol_time))
    C = phi * S * carry_disc * hs ** (2.0 * (mu + 1.0)) * _N(eta * y1) - phi * K * disc_r * hs ** (
        2.0 * mu
    ) * _N(eta * (y1 - vol_time))
    D = phi * S * carry_disc * hs ** (2.0 * (mu + 1.0)) * _N(eta * y2) - phi * K * disc_r * hs ** (
        2.0 * mu
    ) * _N(eta * (y2 - vol_time))
    E = R * disc_r * (
        _N(eta * (x2 - vol_time)) - hs ** (2.0 * mu) * _N(eta * (y2 - vol_time))
    )
    F = R * (
        hs ** (mu + lam) * _N(eta * z)
        + hs ** (mu - lam) * _N(eta * (z - 2.0 * lam * vol_time))
    )

    strike_above_barrier = K > H
    is_call = phi > 0

    if not is_out:  # knock-in
        if is_down and is_call:
            value = (C + E) if strike_above_barrier else (A - B + D + E)
        elif (not is_down) and is_call:
            value = (A + E) if strike_above_barrier else (B - C + D + E)
        elif is_down and (not is_call):
            value = (B - C + D + E) if strike_above_barrier else (A + E)
        else:  # up-and-in put
            value = (A - B + D + E) if strike_above_barrier else (C + E)
    else:  # knock-out
        if is_down and is_call:
            value = (A - C + F) if strike_above_barrier else (B - D + F)
        elif (not is_down) and is_call:
            value = F if strike_above_barrier else (A - B + C - D + F)
        elif is_down and (not is_call):
            value = (A - B + C - D + F) if strike_above_barrier else F
        else:  # up-and-out put
            value = (B - D + F) if strike_above_barrier else (A - C + F)

    return max(value, 0.0)


def _deterministic_barrier(
    spec: OptionSpec, H: float, R: float, is_down: bool, is_out: bool
) -> float:
    """Zero-volatility barrier value: the path is the deterministic forward."""
    steps = 2000
    hit_time = None
    for i in range(steps + 1):
        t = spec.T * i / steps
        S_t = spec.S * math.exp(spec.carry * t)
        if (S_t <= H) if is_down else (S_t >= H):
            hit_time = t
            break
    payoff = max(spec.phi * (spec.S * math.exp(spec.carry * spec.T) - spec.K), 0.0)
    if hit_time is None:
        return R * spec.discount if not is_out else payoff * spec.discount
    return R * math.exp(-spec.r * hit_time) if is_out else payoff * spec.discount


# ---------------------------------------------------------------------------
# Lookbacks
# ---------------------------------------------------------------------------


def lookback_price(spec: OptionSpec) -> float:
    """Continuously monitored lookback, fixed or floating strike."""
    if spec.strike_type == "floating":
        return _floating_lookback(spec)
    return _fixed_lookback(spec)


def _floating_lookback(spec: OptionSpec) -> float:
    """Goldman, Sosin & Gatto (1979).

    Call payoff ``S_T - min(S)``, put payoff ``max(S) - S_T``.  The observed
    running extreme defaults to the current spot (a fresh contract).
    """
    S, T, r, q, sigma = spec.S, spec.T, spec.r, spec.q, spec.sigma
    b = _safe_carry(spec.carry)
    v2 = sigma * sigma
    disc_r = math.exp(-r * T)
    carry_disc = math.exp((b - r) * T)

    if spec.option_type == "call":
        m = spec.running_min if spec.running_min is not None else S
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
            return max(S - m, 0.0) if T <= 0.0 else disc_r * max(spec.forward - m, 0.0)
        vol_time = sigma * math.sqrt(T)
        a1 = (math.log(S / m) + (b + 0.5 * v2) * T) / vol_time
        a2 = a1 - vol_time
        return (
            S * carry_disc * _N(a1)
            - m * disc_r * _N(a2)
            + S
            * disc_r
            * (v2 / (2.0 * b))
            * (
                (S / m) ** (-2.0 * b / v2) * _N(-a1 + 2.0 * b * math.sqrt(T) / sigma)
                - math.exp(b * T) * _N(-a1)
            )
        )

    M = spec.running_max if spec.running_max is not None else S
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(M - S, 0.0) if T <= 0.0 else disc_r * max(M - spec.forward, 0.0)
    vol_time = sigma * math.sqrt(T)
    b1 = (math.log(S / M) + (b + 0.5 * v2) * T) / vol_time
    b2 = b1 - vol_time
    return (
        M * disc_r * _N(-b2)
        - S * carry_disc * _N(-b1)
        + S
        * disc_r
        * (v2 / (2.0 * b))
        * (
            -((S / M) ** (-2.0 * b / v2)) * _N(b1 - 2.0 * b * math.sqrt(T) / sigma)
            + math.exp(b * T) * _N(b1)
        )
    )


def _fixed_lookback(spec: OptionSpec) -> float:
    """Conze & Viswanathan (1991).

    Call payoff ``max(max(S) - K, 0)``, put payoff ``max(K - min(S), 0)``.
    """
    S, K, T, r, sigma = spec.S, spec.K, spec.T, spec.r, spec.sigma
    b = _safe_carry(spec.carry)
    v2 = sigma * sigma
    disc_r = math.exp(-r * T)
    carry_disc = math.exp((b - r) * T)
    sqrt_T = math.sqrt(T)

    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        M = spec.running_max if spec.running_max is not None else S
        m = spec.running_min if spec.running_min is not None else S
        if T <= 0.0:
            return max(M - K, 0.0) if spec.option_type == "call" else max(K - m, 0.0)
        # Zero volatility: the extreme of a monotone forward path is an endpoint.
        F = spec.forward
        M = max(M, F, S)
        m = min(m, F, S)
        return disc_r * (max(M - K, 0.0) if spec.option_type == "call" else max(K - m, 0.0))

    vol_time = sigma * sqrt_T
    tail = 2.0 * b * sqrt_T / sigma

    if spec.option_type == "call":
        M = spec.running_max if spec.running_max is not None else S
        if K > M:
            d1 = (math.log(S / K) + (b + 0.5 * v2) * T) / vol_time
            d2 = d1 - vol_time
            return (
                S * carry_disc * _N(d1)
                - K * disc_r * _N(d2)
                + S
                * disc_r
                * (v2 / (2.0 * b))
                * (-((S / K) ** (-2.0 * b / v2)) * _N(d1 - tail) + math.exp(b * T) * _N(d1))
            )
        e1 = (math.log(S / M) + (b + 0.5 * v2) * T) / vol_time
        e2 = e1 - vol_time
        return (
            disc_r * (M - K)
            + S * carry_disc * _N(e1)
            - M * disc_r * _N(e2)
            + S
            * disc_r
            * (v2 / (2.0 * b))
            * (-((S / M) ** (-2.0 * b / v2)) * _N(e1 - tail) + math.exp(b * T) * _N(e1))
        )

    m = spec.running_min if spec.running_min is not None else S
    if K < m:
        d1 = (math.log(S / K) + (b + 0.5 * v2) * T) / vol_time
        d2 = d1 - vol_time
        return (
            K * disc_r * _N(-d2)
            - S * carry_disc * _N(-d1)
            + S
            * disc_r
            * (v2 / (2.0 * b))
            * ((S / K) ** (-2.0 * b / v2) * _N(-d1 + tail) - math.exp(b * T) * _N(-d1))
        )
    f1 = (math.log(S / m) + (b + 0.5 * v2) * T) / vol_time
    f2 = f1 - vol_time
    return (
        disc_r * (K - m)
        - S * carry_disc * _N(-f1)
        + m * disc_r * _N(-f2)
        + S
        * disc_r
        * (v2 / (2.0 * b))
        * ((S / m) ** (-2.0 * b / v2) * _N(-f1 + tail) - math.exp(b * T) * _N(-f1))
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Instruments this engine prices in closed form.
SUPPORTED = ("european", "digital", "asian", "barrier", "lookback")


def analytic_price(spec: OptionSpec) -> float:
    """Closed-form price for any instrument in :data:`SUPPORTED`."""
    if spec.instrument == "european":
        return black_scholes_price(spec)
    if spec.instrument == "digital":
        return digital_price(spec)
    if spec.instrument == "asian":
        if spec.averaging != "geometric":
            raise UnsupportedInstrument(
                "no closed form exists for an arithmetic-average Asian option; "
                "use the Monte Carlo engine"
            )
        return geometric_asian_price(spec)
    if spec.instrument == "barrier":
        return barrier_price(spec)
    if spec.instrument == "lookback":
        return lookback_price(spec)
    raise UnsupportedInstrument(
        f"the analytic engine does not price {spec.instrument!r}"
    )


def analytic_greeks(spec: OptionSpec, bump: float | None = None) -> Greeks:
    """Greeks in closed form for European options, bump-and-reprice elsewhere.

    FR-B-07 asks for closed form where a formula exists and finite differences
    where it does not.  Only the vanilla European case has hand-coded Greeks;
    every exotic reprices the closed-form value at bumped parameters, so the
    result is still free of any Monte Carlo noise.
    """
    if spec.instrument == "european":
        return black_scholes_greeks(spec)
    from plumbline.engines.bump import bump_greeks

    return bump_greeks(analytic_price, spec, bump=bump)


def price(spec: OptionSpec) -> PriceResult:
    """Engine entry point used by the registry."""
    return PriceResult(
        price=analytic_price(spec),
        greeks=analytic_greeks(spec),
        engine="analytic",
    )

"""Heston stochastic volatility engine (FR-B-06).

Semi-analytical pricing by the characteristic function of Heston (1993), in
the "little trap" formulation of Albrecher, Mayer, Schoutens & Tistaert (2007).
The little-trap form replaces ``g`` with ``1/g`` in the exponent, which keeps
the complex logarithm on its principal branch for long maturities.  The naive
1993 form crosses a branch cut and returns visibly wrong prices past roughly
one year, so it is not used here.

The variance process is

    dv = kappa (theta_v - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho_sv

A full-truncation Euler simulator is included as an independent cross-check of
the integral (Lord et al. 2010); it is not the production pricing path.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.integrate import quad

from plumbline.contracts import Greeks, OptionSpec, PriceResult, UnsupportedInstrument

SUPPORTED = ("european",)

#: Upper limit of the Fourier integral. The integrand is below 1e-14 well
#: before this for every parameter set in the self-validation suite.
INTEGRATION_LIMIT = 250.0

#: Below this volatility of variance the model is treated as its exact
#: deterministic-variance limit rather than integrated numerically.
DETERMINISTIC_VARIANCE_XI = 1e-4


def _char_func(
    phi: complex,
    j: int,
    spec: OptionSpec,
) -> complex:
    """``f_j(phi)`` from Heston (1993), little-trap form."""
    S, T, r, q = spec.S, spec.T, spec.r, spec.q
    kappa, theta, xi, rho, v0 = (
        spec.kappa,
        spec.theta_v,
        spec.xi,
        spec.rho_sv,
        spec.v0,
    )
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa

    i = 1j
    rho_xi_phi_i = rho * xi * phi * i
    d = cmath.sqrt((rho_xi_phi_i - b) ** 2 - xi * xi * (2.0 * u * phi * i - phi * phi))
    numer = b - rho_xi_phi_i - d
    denom = b - rho_xi_phi_i + d
    c = numer / denom  # this is 1/g: the little-trap substitution

    exp_dt = cmath.exp(-d * T)
    D = (numer / (xi * xi)) * (1.0 - exp_dt) / (1.0 - c * exp_dt)
    C = (r - q) * phi * i * T + (kappa * theta / (xi * xi)) * (
        numer * T - 2.0 * cmath.log((1.0 - c * exp_dt) / (1.0 - c))
    )
    return cmath.exp(C + D * v0 + phi * i * math.log(S))


def _probability(j: int, spec: OptionSpec) -> float:
    log_K = math.log(spec.K)

    def integrand(phi: float) -> float:
        value = cmath.exp(-1j * phi * log_K) * _char_func(phi, j, spec) / (1j * phi)
        return value.real

    # The integrand oscillates and decays; splitting at the knee lets the
    # adaptive rule spend its subdivisions where they matter, and keeps the
    # requested accuracy inside what double precision can actually deliver.
    near, _ = quad(integrand, 1e-10, 40.0, limit=200, epsabs=1e-10, epsrel=1e-8)
    far, _ = quad(integrand, 40.0, INTEGRATION_LIMIT, limit=200, epsabs=1e-10, epsrel=1e-8)
    return 0.5 + (near + far) / math.pi


def heston_price(spec: OptionSpec) -> float:
    """European call or put under the Heston model."""
    if spec.instrument not in SUPPORTED:
        raise UnsupportedInstrument(
            f"the Heston engine prices {SUPPORTED}, not {spec.instrument!r}"
        )
    S, K, T, r, q = spec.S, spec.K, spec.T, spec.r, spec.q

    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(spec.phi * (S - K), 0.0) if T <= 0.0 else _degenerate(spec)
    if spec.v0 <= 0.0 and spec.theta_v <= 0.0:
        # No variance anywhere: the asset is a deterministic forward.
        return math.exp(-r * T) * max(spec.phi * (spec.forward - K), 0.0)
    if spec.xi <= DETERMINISTIC_VARIANCE_XI:
        return _deterministic_variance_price(spec)

    forward_value = S * math.exp(-q * T) - K * math.exp(-r * T)
    call = S * math.exp(-q * T) * _probability(1, spec) - K * math.exp(-r * T) * _probability(
        2, spec
    )

    # The quadrature is accurate to about 1e-7 in absolute price, which is
    # nothing next to an at-the-money option and everything next to a deep
    # out-of-the-money one: for a short-dated call struck far above the spot
    # the integral returns a small negative number. Clamping the call to its
    # static lower bound fixes that.
    #
    # The clamp must happen before the put is derived, not after. Clamping only
    # the call and deriving the put from the raw value breaks put-call parity
    # by exactly the amount clamped, which is the kind of defect Check Type 2
    # exists to catch in somebody else's model.
    call = max(call, max(forward_value, 0.0))

    if spec.option_type == "put":
        # Parity is model-free, so the put follows from the clamped call and
        # the two stay consistent by construction.
        return call - forward_value
    return call


def _deterministic_variance_price(spec: OptionSpec) -> float:
    """The exact ``xi -> 0`` limit of the Heston model.

    With no volatility of volatility the variance path is deterministic,
    ``v(t) = theta + (v0 - theta) exp(-kappa t)``, so the option is a
    Black-Scholes option at the root-mean-square volatility over the life of
    the contract.  Taking this branch is not only faster than the integral: at
    a tiny ``xi`` the characteristic function becomes a near-delta spike that
    adaptive quadrature cannot resolve to full double precision.
    """
    from plumbline.engines.analytic import black_scholes_price

    kappa, theta, v0, T = spec.kappa, spec.theta_v, spec.v0, spec.T
    integrated = theta * T + (v0 - theta) * (1.0 - math.exp(-kappa * T)) / kappa
    sigma = math.sqrt(max(integrated, 0.0) / T)
    return black_scholes_price(spec.with_(model="bsm", sigma=sigma))


def _degenerate(spec: OptionSpec) -> float:
    if spec.S <= 0.0:
        return 0.0 if spec.phi > 0 else spec.K * spec.discount
    return spec.S * math.exp(-spec.q * spec.T) if spec.phi > 0 else 0.0


def heston_greeks(spec: OptionSpec) -> Greeks:
    """Bump-and-reprice sensitivities on the semi-analytical price.

    ``vega`` is the derivative with respect to the *initial volatility*
    ``sqrt(v0)``, so it stays comparable with a Black-Scholes vega.
    ``theta`` is minus the derivative in time to expiry.
    """
    h_s = max(1e-4 * spec.S, 1e-6)
    h_v0 = 1e-5
    h_r = 1e-4
    h_t = 1e-4

    base = heston_price(spec)
    up = heston_price(spec.with_(S=spec.S + h_s))
    down = heston_price(spec.with_(S=spec.S - h_s))
    delta = (up - down) / (2.0 * h_s)
    gamma = (up - 2.0 * base + down) / (h_s * h_s)

    dv = (
        heston_price(spec.with_(v0=spec.v0 + h_v0))
        - heston_price(spec.with_(v0=max(spec.v0 - h_v0, 0.0)))
    ) / (2.0 * h_v0)
    vega = dv * 2.0 * math.sqrt(max(spec.v0, 1e-12))

    if spec.T - h_t <= 0.0:
        theta = -(heston_price(spec.with_(T=spec.T + h_t)) - base) / h_t
    else:
        theta = -(
            heston_price(spec.with_(T=spec.T + h_t))
            - heston_price(spec.with_(T=spec.T - h_t))
        ) / (2.0 * h_t)

    rho = (
        heston_price(spec.with_(r=spec.r + h_r)) - heston_price(spec.with_(r=spec.r - h_r))
    ) / (2.0 * h_r)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def heston_mc_price(
    spec: OptionSpec, paths: int = 200_000, steps: int = 400, seed: int = 7
) -> float:
    """Full-truncation Euler cross-check of :func:`heston_price`.

    Full truncation (Lord et al. 2010) clamps the variance at zero only where
    it enters the diffusion, which keeps the discretisation bias small without
    the reflection scheme's systematic upward drift.
    """
    S, K, T, r, q = spec.S, spec.K, spec.T, spec.r, spec.q
    kappa, theta, xi, rho = spec.kappa, spec.theta_v, spec.xi, spec.rho_sv
    dt = T / steps
    sqrt_dt = math.sqrt(dt)
    rng = np.random.default_rng(seed)
    n_pairs = max(paths // 2, 1)

    x = np.full(2 * n_pairs, math.log(S))
    v = np.full(2 * n_pairs, spec.v0)

    for _ in range(steps):
        z1 = rng.standard_normal(n_pairs)
        z2 = rng.standard_normal(n_pairs)
        z1 = np.concatenate((z1, -z1))
        z2 = np.concatenate((z2, -z2))
        w2 = rho * z1 + math.sqrt(1.0 - rho * rho) * z2
        v_plus = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_plus)
        x += (r - q - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z1
        v += kappa * (theta - v_plus) * dt + xi * sqrt_v * sqrt_dt * w2

    ST = np.exp(x)
    payoff = np.maximum(spec.phi * (ST - K), 0.0)
    payoff = 0.5 * (payoff[:n_pairs] + payoff[n_pairs:])
    return float(math.exp(-r * T) * payoff.mean())


def price(spec: OptionSpec, with_greeks: bool = True) -> PriceResult:
    return PriceResult(
        price=heston_price(spec),
        greeks=heston_greeks(spec) if with_greeks else None,
        engine="heston_cf",
    )

"""Section 8 -- Ground Truth Engine self-validation (GT-01 to GT-07, AC-02).

Plumbline must prove its own reference engines before it may judge anything
else.  Each test here pins one engine against a value that does not come from
Plumbline: a textbook figure, a published benchmark, or a second engine built
on entirely different mathematics.

NFR-01 sets the bar at 1e-4 relative error for the closed-form engines.  The
simulation and grid engines carry their own published tolerances, stated in
each test.
"""

from __future__ import annotations

import math

import pytest

from plumbline.contracts import OptionSpec
from plumbline.engines import analytic, binomial, fdm, heston, montecarlo

#: NFR-01. Closed-form engines must reach this against their reference value.
CLOSED_FORM_TOLERANCE = 1e-4


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


# ---------------------------------------------------------------------------
# GT-01
# ---------------------------------------------------------------------------


def test_gt01_black_scholes_matches_hull_textbook_case():
    """GT-01: Hull's standard case, S=100 K=100 r=5% sigma=20% T=1, is ~10.45.

    Hull, *Options, Futures, and Other Derivatives*, worked example in the
    Black-Scholes-Merton chapter.  The figure quoted to four significant
    figures is 10.45; the exact value of the formula is 10.450584.
    """
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    price = analytic.black_scholes_price(spec)

    assert relative_error(price, 10.450584) < CLOSED_FORM_TOLERANCE
    assert round(price, 2) == 10.45


def test_gt01_put_and_parity():
    """The put from the same formula satisfies put-call parity exactly."""
    call = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    put = call.with_(option_type="put")

    c = analytic.black_scholes_price(call)
    p = analytic.black_scholes_price(put)
    forward = 100.0 - 100.0 * math.exp(-0.05)

    assert abs((c - p) - forward) < 1e-12


# ---------------------------------------------------------------------------
# GT-02
# ---------------------------------------------------------------------------


def test_gt02_american_put_binomial_matches_benchmark():
    """GT-02: the American put benchmark, S=K=100 r=5% sigma=20% T=1.

    The high-precision value of this standard case is 6.0903.  It is confirmed
    below by two engines that share no code: a 4000-step binomial tree and a
    Crank-Nicolson finite difference grid.
    """
    spec = OptionSpec("american", "put", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    benchmark = 6.0903

    tree = binomial.binomial_price(spec, steps=4000)
    grid = fdm.fdm_price(spec, space_steps=1600, time_steps=1600)

    assert relative_error(tree, benchmark) < 1e-4
    assert relative_error(grid, benchmark) < 1e-3
    assert abs(tree - grid) < 1e-3


def test_gt02_american_put_is_worth_more_than_european():
    """The early exercise premium must be positive for a put with r > 0."""
    american = OptionSpec("american", "put", S=90, K=100, T=1.0, r=0.06, q=0.0, sigma=0.25)
    european = american.with_(instrument="european")

    assert binomial.binomial_price(american, 2000) > analytic.black_scholes_price(european)


def test_gt02_american_call_without_dividends_equals_european():
    """With no dividend an American call is never exercised early (Merton)."""
    american = OptionSpec("american", "call", S=100, K=95, T=1.0, r=0.05, q=0.0, sigma=0.30)
    european = american.with_(instrument="european")

    tree = binomial.binomial_price(american, 3000)
    closed = analytic.black_scholes_price(european)

    assert relative_error(tree, closed) < 1e-3


# ---------------------------------------------------------------------------
# GT-03
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gt03_monte_carlo_within_one_percent_of_black_scholes():
    """GT-03: 1,000,000 paths must land within 1 percent of the closed form."""
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    closed = analytic.black_scholes_price(spec)

    result = montecarlo.monte_carlo(spec, paths=1_000_000, seed=1)

    assert relative_error(result.price, closed) < 1e-2
    assert result.stderr > 0.0
    assert abs(result.price - closed) < 4.0 * result.stderr


def test_gt03_variance_reduction_actually_reduces_variance():
    """FR-B-03: both techniques must earn their place, not merely exist."""
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)

    plain = montecarlo.monte_carlo(
        spec, paths=100_000, seed=3, antithetic=False, control_variate=False
    )
    antithetic = montecarlo.monte_carlo(
        spec, paths=100_000, seed=3, antithetic=True, control_variate=False
    )
    both = montecarlo.monte_carlo(spec, paths=100_000, seed=3, antithetic=True, control_variate=True)

    assert antithetic.stderr < plain.stderr
    assert both.stderr < antithetic.stderr
    assert both.control_beta != 0.0


# ---------------------------------------------------------------------------
# GT-04
# ---------------------------------------------------------------------------


def test_gt04_crank_nicolson_within_one_tenth_percent_of_black_scholes():
    """GT-04: the finite difference grid must reach 0.1 percent."""
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    closed = analytic.black_scholes_price(spec)

    grid = fdm.fdm_price(spec)

    assert relative_error(grid, closed) < 1e-3


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S,K,T,sigma", [(100, 90, 0.5, 0.15), (80, 100, 2.0, 0.35)])
def test_gt04_crank_nicolson_across_the_surface(option_type, S, K, T, sigma):
    """One point proves nothing; the scheme must hold across the surface."""
    spec = OptionSpec("european", option_type, S=S, K=K, T=T, r=0.04, q=0.01, sigma=sigma)

    assert relative_error(fdm.fdm_price(spec), analytic.black_scholes_price(spec)) < 1e-3


def test_gt04_rannacher_start_up_keeps_a_digital_accurate():
    """A discontinuous payoff is where an unsmoothed scheme breaks first."""
    spec = OptionSpec("digital", "call", S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25)

    assert relative_error(fdm.fdm_price(spec), analytic.digital_price(spec)) < 1e-3


# ---------------------------------------------------------------------------
# GT-05
# ---------------------------------------------------------------------------


def test_gt05_heston_reduces_to_black_scholes_when_vol_of_vol_vanishes():
    """GT-05, part one: the limit stated in Heston (1993) section 1.

    With no volatility of volatility and the variance started at its long-run
    level, the Heston characteristic function must return the Black-Scholes
    price exactly.  This is the sharpest available test of the integral, the
    branch handling and the parameter mapping all at once.
    """
    heston_spec = OptionSpec(
        "european",
        "call",
        S=100,
        K=100,
        T=1.0,
        r=0.03,
        q=0.0,
        model="heston",
        v0=0.04,
        theta_v=0.04,
        kappa=2.0,
        xi=1e-5,
        rho_sv=0.0,
    )
    black_scholes = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2)

    assert relative_error(
        heston.heston_price(heston_spec), analytic.black_scholes_price(black_scholes)
    ) < CLOSED_FORM_TOLERANCE


def test_gt05_heston_benchmark_parameter_set():
    """GT-05, part two: the standard equity-index Heston parameter set.

    Parameters: v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711,
    S=K=100, T=1, r=q=0.  This set is the one calibrated to index options and
    reused throughout the Heston literature as a test case.  The value below is
    confirmed independently in
    ``test_gt05_heston_characteristic_function_matches_simulation``, which
    prices the same contract with a full-truncation Euler simulation that
    shares no code with the integral.
    """
    spec = OptionSpec(
        "european",
        "call",
        S=100,
        K=100,
        T=1.0,
        r=0.0,
        q=0.0,
        model="heston",
        v0=0.0175,
        theta_v=0.0398,
        kappa=1.5768,
        xi=0.5751,
        rho_sv=-0.5711,
    )

    assert relative_error(heston.heston_price(spec), 5.785155) < 1e-4


@pytest.mark.slow
def test_gt05_heston_characteristic_function_matches_simulation():
    """The Fourier integral and an Euler simulation must meet in the middle."""
    spec = OptionSpec(
        "european",
        "call",
        S=100,
        K=100,
        T=1.0,
        r=0.0,
        q=0.0,
        model="heston",
        v0=0.0175,
        theta_v=0.0398,
        kappa=1.5768,
        xi=0.5751,
        rho_sv=-0.5711,
    )

    integral = heston.heston_price(spec)
    simulated = heston.heston_mc_price(spec, paths=400_000, steps=800)

    assert relative_error(simulated, integral) < 5e-3


def test_gt05_heston_put_call_parity_holds():
    """Parity is model-free, so it must hold under Heston as well."""
    call = OptionSpec(
        "european",
        "call",
        S=100,
        K=110,
        T=0.75,
        r=0.04,
        q=0.01,
        model="heston",
        v0=0.05,
        theta_v=0.04,
        kappa=1.5,
        xi=0.4,
        rho_sv=-0.6,
    )
    put = call.with_(option_type="put")

    gap = (heston.heston_price(call) - heston.heston_price(put)) - (
        100.0 * math.exp(-0.01 * 0.75) - 110.0 * math.exp(-0.04 * 0.75)
    )

    assert abs(gap) < 1e-8


# ---------------------------------------------------------------------------
# GT-06
# ---------------------------------------------------------------------------


def test_gt06_geometric_asian_matches_kemna_vorst_closed_form():
    """GT-06: the geometric average is lognormal, so the closed form is exact.

    The check runs against an independent simulation, because the closed form
    and the simulation share nothing but the contract definition.
    """
    spec = OptionSpec("asian", "call", S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25)

    closed = analytic.geometric_asian_price(spec)
    simulated = montecarlo.monte_carlo(spec, paths=200_000, steps=4000, seed=11)

    assert relative_error(closed, simulated.price) < 2e-3


def test_gt06_geometric_asian_is_cheaper_than_the_vanilla():
    """Averaging cuts the variance of the payoff, so the Asian must cost less."""
    asian = OptionSpec("asian", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.30)
    vanilla = asian.with_(instrument="european")

    assert analytic.geometric_asian_price(asian) < analytic.black_scholes_price(vanilla)


def test_gt06_discrete_geometric_average_converges_to_the_continuous_one():
    """The discrete control's mean must approach the Kemna-Vorst value."""
    spec = OptionSpec("asian", "call", S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25)
    continuous = analytic.geometric_asian_price(spec)

    coarse = montecarlo._discrete_geometric_asian(spec, 50)
    fine = montecarlo._discrete_geometric_asian(spec, 5000)

    assert abs(fine - continuous) < abs(coarse - continuous)
    assert abs(fine - continuous) < 1e-3


# ---------------------------------------------------------------------------
# GT-07
# ---------------------------------------------------------------------------

BARRIER_KINDS = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


@pytest.mark.parametrize("kind", BARRIER_KINDS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gt07_barrier_closed_form_matches_two_other_methods(kind, option_type):
    """GT-07: the Reiner-Rubinstein table against a PDE and a simulation.

    Each of the eight standard barrier contracts is priced three ways.  The
    finite difference grid uses a Dirichlet condition on the barrier; the
    simulation uses the Brownian-bridge no-hit probability.  Neither shares any
    line of code with the closed form.
    """
    barrier = 90.0 if kind.startswith("down") else 130.0
    spec = OptionSpec(
        "barrier",
        option_type,
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        q=0.02,
        sigma=0.25,
        barrier=barrier,
        barrier_kind=kind,
    )

    closed = analytic.barrier_price(spec)
    grid = fdm.fdm_price(spec)
    simulated = montecarlo.monte_carlo(spec, paths=200_000, steps=400, seed=5)

    assert abs(closed - grid) < max(1e-3, 1e-3 * abs(closed))
    assert abs(closed - simulated.price) < 4.0 * simulated.stderr + 1e-3


@pytest.mark.parametrize("kind,barrier", [("down", 90.0), ("up", 130.0)])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gt07_in_plus_out_equals_the_vanilla(kind, barrier, option_type):
    """A contract that knocks in plus one that knocks out is the vanilla."""
    base = OptionSpec(
        "barrier",
        option_type,
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        q=0.02,
        sigma=0.25,
        barrier=barrier,
        barrier_kind=f"{kind}-and-out",
    )

    knock_out = analytic.barrier_price(base)
    knock_in = analytic.barrier_price(base.with_(barrier_kind=f"{kind}-and-in"))
    vanilla = analytic.black_scholes_price(base.with_(instrument="european"))

    assert abs(knock_out + knock_in - vanilla) < 1e-10


# ---------------------------------------------------------------------------
# NFR-02
# ---------------------------------------------------------------------------


def test_nfr02_every_engine_returns_double_precision():
    """NFR-02: all arithmetic is IEEE 754 double precision.

    Python floats and NumPy's default dtype are both binary64.  This test pins
    that, so a later change to a lower-precision dtype cannot pass unnoticed.
    """
    import numpy as np

    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2)

    assert np.array(0.0).dtype == np.float64
    assert np.finfo(float).bits == 64
    for value in (
        analytic.black_scholes_price(spec),
        binomial.binomial_price(spec, 100),
        fdm.fdm_price(spec, 200, 100),
        montecarlo.monte_carlo(spec, paths=2_000).price,
    ):
        assert isinstance(value, float)

"""Section 8 -- Ground Truth Engine self-validation (GT-01 to GT-07, AC-02).

Plumbline must prove its own reference engines before it may judge anything
else.  Each test here pins one engine against a value that does not come from
Plumbline: a textbook figure, a published benchmark, or a second engine built
on entirely different mathematics.

These are necessary and they are not sufficient. Two engines by one author
share one set of assumptions, and every defect found in these engines so far
passed the internal cross-checks while being wrong. The external validation
against QuantLib lives in tests/test_external_oracle.py, and BENCHMARKS.md
records which numbers are externally sourced and which are not.

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

    The reference value 6.0903706065 comes from QuantLib's QdFpAmericanEngine
    on its high-precision scheme, which implements the Andersen, Lake &
    Offengenden (2016) fixed-point method. It is asserted directly against
    QuantLib in tests/test_external_oracle.py; repeated here so this file
    still pins the number when QuantLib is not installed.

    Two engines that share no code confirm it below: a binomial tree and a
    Crank-Nicolson grid.
    """
    spec = OptionSpec("american", "put", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    benchmark = 6.0903706065

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


def test_gt03_monte_carlo_is_unbiased_when_dividends_and_rates_differ():
    """The simulation must price correctly with r != q in the fast lane.

    A mutation once swapped the discount rate into the terminal-spot control's
    expectation and survived every test in the inner lanes. GT-03 above, the
    only other check of the raw estimator's correctness against a closed
    form, uses q = 0 -- where r-discounting and q-discounting still differ,
    but that test is marked slow and never ran in the fast or integration
    lanes. The audit could not catch it either: its sample model delegates to
    the same engines, so both sides go wrong together.

    This test keeps a correctness pin on the raw estimator inside the fast
    lane, at parameters where confusing r with q moves the price by ~3.
    """
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.20)
    closed = analytic.black_scholes_price(spec)

    result = montecarlo.monte_carlo(spec, paths=200_000, seed=17)

    # Four standard errors of sampling room; the bias this guards against is
    # hundreds of standard errors at these parameters.
    assert abs(result.price - closed) < 4.0 * result.stderr
    assert abs(result.price - closed) / closed < 5e-3


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
    # The control variate for a geometric Asian is its own payoff, so the
    # residual variance is zero and the path count changes nothing. What is
    # being tested is the discrete average converging on the continuous one,
    # which is the step count.
    simulated = montecarlo.monte_carlo(spec, paths=20_000, steps=4000, seed=11)

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
    # The Brownian-bridge correction is exact between observations, so the
    # step count is not what makes a barrier accurate here; the tolerance
    # below scales with the standard error, so this stays a real test.
    simulated = montecarlo.monte_carlo(spec, paths=100_000, steps=100, seed=5)

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


LOOKBACK_BRANCHES = [
    ("floating call", dict(strike_type="floating"), "call"),
    ("floating put", dict(strike_type="floating"), "put"),
    ("fixed call", dict(strike_type="fixed"), "call"),
    ("fixed put", dict(strike_type="fixed"), "put"),
]


@pytest.mark.parametrize("name,extra,option_type", LOOKBACK_BRANCHES)
def test_gt07_every_lookback_branch_matches_its_closed_form(name, extra, option_type):
    """All four lookback branches, simulation against closed form.

    Each branch is pinned separately on purpose. A mutation once flipped the
    sign of the reflection term in the floating *call* and survived the whole
    suite, because the only numerical lookback comparisons covered the
    floating put and the fixed call -- two branches had no numeric check at
    all and nothing else would have caught a wrong one.

    The bridge-extreme sampling is exact between observations, so the
    simulation targets the same continuous-monitoring value the formulas
    compute. Measured agreement at these settings: 0.3 to 0.5 standard
    errors on every branch.
    """
    spec = OptionSpec(
        "lookback", option_type, S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25, **extra
    )
    reference = analytic.lookback_price(spec)
    simulated = montecarlo.monte_carlo(spec, paths=100_000, steps=252, seed=5)

    # Four standard errors of sampling room. A wrong reflection term, a
    # flipped sign on the carry tail or a broken extreme sampler moves the
    # price by tens of standard errors, not by fractions of one.
    assert abs(simulated.price - reference) < 4.0 * simulated.stderr, (
        f"{name}: MC {simulated.price:.6f} (se {simulated.stderr:.2e}) "
        f"vs closed form {reference:.6f}"
    )


# ---------------------------------------------------------------------------
# NFR-02
# ---------------------------------------------------------------------------


def test_nfr01_the_reference_engines_meet_their_accuracy_at_production_settings():
    """NFR-01 at the settings the audit actually runs, not at hand-picked ones.

    The GT- tests above pin each engine at a chosen point with a chosen step
    count. That is not the same thing as the engine being accurate at its
    default, across the grid a real audit sweeps, and the difference is not
    academic: a reference engine whose own error exceeds the tolerance it is
    used to enforce will fail a correct model.

    This is the test that was missing. Before the Broadie-Detemple corrections
    the binomial engine reached 4.0e-3 relative error here, four times the
    audit's own band, while GT-02 passed because it ran at 4000 steps.
    """
    from plumbline.audit.grid import default_grid

    worst = 0.0
    worst_case = ""
    for spec in default_grid("european"):
        exact = analytic.black_scholes_price(spec)
        tree = binomial.binomial_price(spec)  # default step count, as the audit uses
        error = relative_error(tree, exact)
        if error > worst:
            worst, worst_case = error, f"{spec.label()} tree={tree:.8f} exact={exact:.8f}"

    assert worst < 1e-4, f"worst relative error {worst:.3e} at {worst_case}"


def test_the_binomial_engine_has_no_sawtooth_in_its_step_count():
    """Adjacent step counts must not straddle the answer with a large error.

    Plain Cox-Ross-Rubinstein oscillates because the payoff kink falls between
    two terminal nodes and moves as the step count changes. The error flips
    sign from one step count to the next and does not shrink between them,
    which also makes a convergence check meaningless.
    """
    spec = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    exact = analytic.black_scholes_price(spec)

    errors = [binomial.binomial_price(spec, steps) - exact for steps in range(400, 408)]

    assert max(abs(e) for e in errors) < 1e-4, errors
    # A sawtooth shows up as the sign changing at every single step.
    sign_changes = sum(
        1 for a, b in zip(errors, errors[1:]) if a * b < 0.0
    )
    assert sign_changes < len(errors) - 1, f"error alternates at every step: {errors}"


def test_the_binomial_engine_still_converges_as_steps_rise():
    """Richardson must not flatten the convergence Check Type 4 looks for."""
    spec = OptionSpec("american", "put", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    fine = binomial.binomial_price(spec, 6000)

    coarse_error = abs(binomial.binomial_price(spec, 100) - fine)
    finer_error = abs(binomial.binomial_price(spec, 1600) - fine)

    assert finer_error < coarse_error


def test_the_heston_engine_never_returns_a_negative_price():
    """FR-C-18 applied to Plumbline's own engine.

    The Fourier integral is accurate to about 1e-7 in absolute price. For a
    short-dated call struck far above the spot the true value is smaller than
    that, and the raw integral came back negative.
    """
    worst = 0.0
    for strike in (150.0, 200.0, 250.0, 300.0):
        for maturity in (0.05, 0.1, 0.25, 0.5):
            spec = OptionSpec(
                "european",
                "call",
                S=100,
                K=strike,
                T=maturity,
                r=0.04,
                q=0.02,
                model="heston",
                v0=0.05,
                theta_v=0.04,
                kappa=1.5,
                xi=0.5,
                rho_sv=-0.9,
            )
            price = heston.heston_price(spec)
            worst = min(worst, price)
            assert price >= 0.0, f"K={strike} T={maturity}: {price:.6e}"
    assert worst >= 0.0


@pytest.mark.parametrize("strike", [60.0, 100.0, 160.0, 250.0])
@pytest.mark.parametrize("maturity", [0.05, 0.25, 2.0, 10.0])
def test_heston_put_call_parity_is_exact_including_deep_out_of_the_money(strike, maturity):
    """Parity must hold where the clamp bites, not only where it does not.

    Clamping the call at zero and then deriving the put from the unclamped
    value broke parity by exactly the amount clamped. The clamp now happens
    once, before the put is derived, so the two cannot disagree.
    """
    call = OptionSpec(
        "european",
        "call",
        S=100,
        K=strike,
        T=maturity,
        r=0.04,
        q=0.02,
        model="heston",
        v0=0.05,
        theta_v=0.04,
        kappa=1.5,
        xi=0.5,
        rho_sv=-0.9,
    )
    put = call.with_(option_type="put")

    gap = (heston.heston_price(call) - heston.heston_price(put)) - (
        100.0 * math.exp(-0.02 * maturity) - strike * math.exp(-0.04 * maturity)
    )

    assert abs(gap) < 1e-12, f"parity gap {gap:.3e}"


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


# ---------------------------------------------------------------------------
# exact degenerate corners and the helpers behind them
# ---------------------------------------------------------------------------


def test_the_degenerate_arithmetic_asian_corner_is_the_time_average_of_the_forward():
    """limits.degenerate_value on a zero-volatility arithmetic Asian.

    With sigma = 0 the path is deterministic, so the average is the time
    average of the forward and both branches of the carry handling are exact
    closed forms:

        b != 0:  S * (exp(b T) - 1) / (b T)
        b == 0:  S
    """
    from plumbline.engines.limits import degenerate_value

    # b != 0: r = 6%, q = 1%, so carry b = 5%.
    spec = OptionSpec(
        "asian", "call", S=100.0, K=101.0, T=1.0, r=0.06, q=0.01, sigma=0.0,
        averaging="arithmetic",
    )
    b, S, T, K = 0.05, 100.0, 1.0, 101.0
    average = S * math.expm1(b * T) / (b * T)
    expected = math.exp(-0.06 * T) * max(average - K, 0.0)

    assert degenerate_value(spec) == pytest.approx(expected, rel=1e-14)

    # b == 0: r = q, the flat-forward branch.
    flat = OptionSpec(
        "asian", "call", S=100.0, K=99.0, T=2.0, r=0.04, q=0.04, sigma=0.0,
        averaging="arithmetic",
    )
    expected_flat = math.exp(-0.04 * 2.0) * (100.0 - 99.0)
    assert degenerate_value(flat) == pytest.approx(expected_flat, rel=1e-14)


def test_the_lookback_carry_floor_preserves_the_sign_of_b():
    """The documented contract of _safe_carry.

    The lookback formulas are singular at exactly b = 0, so the module nudges
    b off zero by _CARRY_FLOOR while keeping its sign -- nudging a negative b
    to +floor would silently reprice short-carry contracts with the wrong
    tail. A mutation once flipped this sign choice and survived, because no
    test pinned the contract directly.
    """
    assert analytic._safe_carry(0.0) > 0.0
    assert analytic._safe_carry(+1e-12) > 0.0
    assert analytic._safe_carry(-1e-12) < 0.0
    assert analytic._safe_carry(0.03) == 0.03
    assert analytic._safe_carry(-0.02) == -0.02

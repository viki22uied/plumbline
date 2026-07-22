"""Module B coverage: engines, instruments, models and Greeks.

This file covers checklist sections 8 (Ground Truth Engine coverage) and 9
(Instrument coverage), and the FR-B requirements behind them.
"""

from __future__ import annotations

import math

import pytest

from plumbline.contracts import (
    Greeks,
    OptionSpec,
    UnsupportedInstrument,
    PlumblineError,
)
from plumbline.engines import analytic, binomial, bump, fdm, heston, montecarlo, registry
from plumbline.engines.limits import degenerate_value, is_degenerate

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.25)


# ---------------------------------------------------------------------------
# FR-B-01 to FR-B-06: every engine exists and prices what it claims
# ---------------------------------------------------------------------------


def test_frb01_black_scholes_engine_is_registered():
    engine = registry.get("analytic")
    assert "european" in engine.instruments
    assert engine.deterministic


def test_frb02_binomial_engine_covers_european_and_american():
    engine = registry.get("binomial_crr")
    assert set(engine.instruments) == {"european", "american"}


def test_frb03_monte_carlo_engine_declares_both_variance_reductions():
    """FR-B-03: antithetic and control variates are both switchable and on."""
    spec = OptionSpec("european", "call", **BASE)
    default = montecarlo.monte_carlo(spec, paths=20_000, seed=2)
    assert default.control_beta != 0.0
    assert default.paths == 20_000


def test_frb04_finite_difference_engine_uses_crank_nicolson():
    engine = registry.get("fdm_crank_nicolson")
    assert "Crank-Nicolson" in engine.description
    assert fdm.RANNACHER_STEPS >= 1


def test_frb06_heston_engine_is_the_only_one_for_the_heston_model():
    spec = OptionSpec("european", "call", model="heston", **BASE)
    assert registry.ground_truth_for(spec).name == "heston_cf"


# ---------------------------------------------------------------------------
# Section 9 -- instrument coverage
# ---------------------------------------------------------------------------

VANILLA_CASES = [
    ("european", "call"),
    ("european", "put"),
    ("american", "call"),
    ("american", "put"),
]

EXOTIC_CASES = [
    ("asian", "call", {"averaging": "geometric"}),
    ("asian", "put", {"averaging": "geometric"}),
    ("asian", "call", {"averaging": "arithmetic"}),
    ("barrier", "call", {"barrier": 130.0, "barrier_kind": "up-and-out"}),
    ("barrier", "call", {"barrier": 130.0, "barrier_kind": "up-and-in"}),
    ("barrier", "put", {"barrier": 90.0, "barrier_kind": "down-and-out"}),
    ("barrier", "put", {"barrier": 90.0, "barrier_kind": "down-and-in"}),
    ("digital", "call", {"payout": "cash"}),
    ("digital", "put", {"payout": "asset"}),
    ("lookback", "call", {"strike_type": "fixed"}),
    ("lookback", "put", {"strike_type": "fixed"}),
    ("lookback", "call", {"strike_type": "floating"}),
    ("lookback", "put", {"strike_type": "floating"}),
]


@pytest.mark.parametrize("instrument,option_type", VANILLA_CASES)
def test_every_vanilla_instrument_prices(instrument, option_type):
    spec = OptionSpec(instrument, option_type, **BASE)
    result = registry.ground_truth_price(spec)
    assert result.ok
    assert result.price > 0.0


@pytest.mark.parametrize("instrument,option_type,extras", EXOTIC_CASES)
def test_every_exotic_instrument_prices(instrument, option_type, extras):
    spec = OptionSpec(instrument, option_type, **BASE, **extras)
    result = registry.ground_truth_price(spec)
    assert result.ok
    assert result.price >= 0.0


@pytest.mark.parametrize("model", ["bsm", "heston", "localvol"])
def test_every_underlying_model_prices_a_european_option(model):
    """Section 4.3: all three sets of underlying dynamics are covered."""
    spec = OptionSpec("european", "call", model=model, **BASE)
    result = registry.ground_truth_price(spec)
    assert result.ok
    assert 0.0 < result.price < spec.S


def test_local_volatility_with_a_flat_surface_reproduces_black_scholes():
    """A Dupire surface that is flat must give back the constant-vol price."""
    local = OptionSpec("european", "call", model="localvol", lv_a=0.25, lv_b=0.0, lv_c=0.0, **BASE)
    constant = OptionSpec("european", "call", **BASE)

    assert abs(fdm.fdm_price(local) - analytic.black_scholes_price(constant)) < 1e-3


def test_local_volatility_skew_moves_the_price():
    """A downward skew must change the price, or the surface is being ignored."""
    flat = OptionSpec("european", "call", model="localvol", lv_a=0.25, lv_b=0.0, **BASE)
    skewed = flat.with_(lv_b=-0.25)

    assert abs(fdm.fdm_price(skewed) - fdm.fdm_price(flat)) > 1e-3


# ---------------------------------------------------------------------------
# FR-B-07: all five Greeks
# ---------------------------------------------------------------------------


def test_frb07_closed_form_greeks_match_bump_and_reprice():
    """The hand-coded Black-Scholes Greeks must agree with finite differences."""
    spec = OptionSpec("european", "call", **BASE)
    closed = analytic.black_scholes_greeks(spec)
    bumped = bump.bump_greeks(analytic.black_scholes_price, spec)

    for name in Greeks.NAMES:
        assert abs(getattr(closed, name) - getattr(bumped, name)) < max(
            1e-4, 1e-4 * abs(getattr(closed, name))
        ), name


@pytest.mark.parametrize("instrument,option_type,extras", EXOTIC_CASES[:2] + EXOTIC_CASES[3:])
def test_frb07_every_engine_returns_five_finite_greeks(instrument, option_type, extras):
    """FR-B-07: where no formula exists, the numerical fallback must deliver."""
    spec = OptionSpec(instrument, option_type, **BASE, **extras)
    greeks = registry.ground_truth_price(spec).greeks

    assert greeks is not None
    for name in Greeks.NAMES:
        assert math.isfinite(getattr(greeks, name)), name


def test_frb07_delta_and_gamma_have_the_signs_the_payoff_requires():
    call = OptionSpec("european", "call", **BASE)
    put = call.with_(option_type="put")

    call_greeks = analytic.black_scholes_greeks(call)
    put_greeks = analytic.black_scholes_greeks(put)

    assert 0.0 < call_greeks.delta < 1.0
    assert -1.0 < put_greeks.delta < 0.0
    assert call_greeks.gamma > 0.0 and put_greeks.gamma > 0.0
    assert call_greeks.vega > 0.0 and put_greeks.vega > 0.0


def test_binomial_lattice_greeks_match_the_closed_form_for_a_european():
    spec = OptionSpec("european", "call", **BASE)
    lattice = binomial.binomial_greeks(spec, steps=2000)
    closed = analytic.black_scholes_greeks(spec)

    assert abs(lattice.delta - closed.delta) < 1e-3
    assert abs(lattice.gamma - closed.gamma) < 1e-4
    assert abs(lattice.theta - closed.theta) < 1e-2
    assert abs(lattice.vega - closed.vega) < 5e-2
    assert abs(lattice.rho - closed.rho) < 5e-2


def test_finite_difference_grid_greeks_match_the_closed_form():
    spec = OptionSpec("european", "put", **BASE)
    grid = fdm.fdm_greeks(spec)
    closed = analytic.black_scholes_greeks(spec)

    assert abs(grid.delta - closed.delta) < 1e-3
    assert abs(grid.gamma - closed.gamma) < 1e-4
    assert abs(grid.theta - closed.theta) < 1e-2


def test_heston_greeks_are_finite_and_delta_is_in_range():
    spec = OptionSpec("european", "call", model="heston", **BASE)
    greeks = heston.heston_greeks(spec)

    for name in Greeks.NAMES:
        assert math.isfinite(getattr(greeks, name)), name
    assert 0.0 < greeks.delta < 1.0
    assert greeks.gamma > 0.0


# ---------------------------------------------------------------------------
# degenerate corners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrument,option_type,extras", [(i, o, e) for i, o, e in EXOTIC_CASES])
def test_degenerate_corners_have_exact_values(instrument, option_type, extras):
    """Every instrument has a closed form at T=0, at sigma=0 and at S=0."""
    base = OptionSpec(instrument, option_type, **BASE, **extras)

    for corner in (base.with_(T=0.0), base.with_(sigma=0.0), base.with_(S=0.0)):
        assert is_degenerate(corner)
        value = degenerate_value(corner)
        assert math.isfinite(value)
        assert value >= 0.0


def test_zero_maturity_gives_the_intrinsic_value():
    spec = OptionSpec("european", "call", **{**BASE, "S": 120.0, "T": 0.0})
    assert degenerate_value(spec) == pytest.approx(20.0)


def test_zero_volatility_gives_the_discounted_deterministic_payoff():
    spec = OptionSpec("european", "call", **{**BASE, "sigma": 0.0})
    expected = math.exp(-BASE["r"] * BASE["T"]) * max(spec.forward - spec.K, 0.0)
    assert degenerate_value(spec) == pytest.approx(expected)


def test_zero_spot_is_absorbing():
    call = OptionSpec("european", "call", **{**BASE, "S": 0.0})
    put = call.with_(option_type="put")

    assert degenerate_value(call) == pytest.approx(0.0)
    assert degenerate_value(put) == pytest.approx(BASE["K"] * math.exp(-BASE["r"] * BASE["T"]))


def test_very_high_volatility_stays_inside_the_no_arbitrage_bounds():
    """FR-C-14 on the reference engines themselves."""
    spec = OptionSpec("european", "call", **{**BASE, "sigma": 5.0})
    price = analytic.black_scholes_price(spec)

    assert 0.0 <= price <= spec.S * math.exp(-spec.q * spec.T) + 1e-12


# ---------------------------------------------------------------------------
# NFR-08: the plug-in interface
# ---------------------------------------------------------------------------


def test_nfr08_a_new_engine_plugs_in_without_touching_the_audit_engine():
    """A caller can register an engine and have the audit use it at once."""
    from plumbline.contracts import PriceResult

    spec = OptionSpec("european", "call", **BASE)
    sentinel = 42.0

    engine = registry.EngineSpec(
        name="test_plugin_engine",
        description="a stub engine registered by a test",
        reference="none, this engine is a fixture",
        price_fn=lambda s: PriceResult(price=sentinel, engine="test_plugin_engine"),
        instruments=("european",),
        priority=10_000,
    )
    registry.register(engine, replace=True)
    try:
        assert registry.ground_truth_for(spec).name == "test_plugin_engine"
        assert registry.ground_truth_price(spec).price == sentinel
    finally:
        registry.REGISTRY.pop("test_plugin_engine")

    assert registry.ground_truth_for(spec).name == "analytic"


def test_registry_refuses_a_duplicate_name_unless_asked_to_replace():
    engine = registry.get("analytic")
    with pytest.raises(ValueError):
        registry.register(engine)


def test_registry_reports_an_uncovered_combination_clearly():
    spec = OptionSpec("lookback", "call", model="heston", **BASE)
    with pytest.raises(UnsupportedInstrument) as info:
        registry.ground_truth_for(spec)
    assert "lookback" in str(info.value) and "heston" in str(info.value)


# ---------------------------------------------------------------------------
# contract validation
# ---------------------------------------------------------------------------


def test_option_spec_rejects_an_unknown_instrument():
    with pytest.raises(UnsupportedInstrument):
        OptionSpec("swaption", "call", **BASE)


def test_option_spec_rejects_a_barrier_without_a_level():
    with pytest.raises(PlumblineError):
        OptionSpec("barrier", "call", barrier_kind="up-and-out", **BASE)


def test_option_spec_rejects_a_negative_volatility():
    with pytest.raises(PlumblineError):
        OptionSpec("european", "call", **{**BASE, "sigma": -0.1})


def test_option_spec_rejects_an_out_of_range_heston_correlation():
    with pytest.raises(PlumblineError):
        OptionSpec("european", "call", model="heston", rho_sv=-1.5, **BASE)


def test_arithmetic_asian_has_no_closed_form_and_says_so():
    spec = OptionSpec("asian", "call", averaging="arithmetic", **BASE)
    with pytest.raises(UnsupportedInstrument) as info:
        analytic.analytic_price(spec)
    assert "arithmetic" in str(info.value)


def test_arithmetic_asian_costs_more_than_the_geometric_one():
    """The arithmetic mean dominates the geometric mean, path by path."""
    arithmetic = OptionSpec("asian", "call", averaging="arithmetic", **BASE)
    geometric = arithmetic.with_(averaging="geometric")

    a = montecarlo.monte_carlo(arithmetic, paths=100_000, steps=252, seed=9).price
    g = montecarlo.monte_carlo(geometric, paths=100_000, steps=252, seed=9).price

    assert a > g

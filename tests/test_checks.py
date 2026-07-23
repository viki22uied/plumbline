"""Module C -- the six check types (checklist sections 2 to 7).

Each check type is tested twice: once against a model that satisfies it, and
once against a model built to break exactly that one property.  A check that
only ever passes proves nothing.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from conftest import write_model
from plumbline.audit.checks import (
    AuditConfig,
    check_arbitrage,
    check_convergence,
    check_edge_cases,
    check_greeks,
    check_put_call_parity,
    check_reference_price,
    model_greeks,
)
from plumbline.audit.grid import ParameterGrid, default_grid
from plumbline.contracts import Greeks, Tolerance
from plumbline.ingestion import load_model

SMALL_GRID = ParameterGrid(
    spots=(95.0, 105.0), strikes=(100.0,), maturities=(0.5,), vols=(0.2,), rates=(0.03,)
)

CORRECT_SOURCE = textwrap.dedent(
    """
    import math

    def _N(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
        phi = 1.0 if option_type == "call" else -1.0
        if T <= 0.0:
            return max(phi * (S - K), 0.0)
        if S <= 0.0:
            return 0.0 if phi > 0 else K * math.exp(-r * T)
        if sigma <= 0.0:
            forward = S * math.exp((r - q) * T)
            return math.exp(-r * T) * max(phi * (forward - K), 0.0)
        v = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
        d2 = d1 - v
        return phi * (S * math.exp(-q * T) * _N(phi * d1) - K * math.exp(-r * T) * _N(phi * d2))
    """
)


@pytest.fixture
def correct_model(tmp_path):
    path = write_model(tmp_path, "correct.py", CORRECT_SOURCE)
    with load_model(path) as mut:
        yield mut


def _statuses(results):
    return [result.status for result in results]


def _flat(source: str, tmp_path, name: str):
    return load_model(write_model(tmp_path, name, textwrap.dedent(source)))


# ---------------------------------------------------------------------------
# Check Type 1 -- reference price comparison
# ---------------------------------------------------------------------------


def test_check1_passes_a_correct_model_and_reports_both_differences(correct_model):
    results = check_reference_price(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert results and all(status == "PASS" for status in _statuses(results))
    evidence = results[0].evidence
    assert "absolute_difference" in evidence  # FR-C-02
    assert "relative_difference" in evidence  # FR-C-02
    assert "reference_price" in evidence and "model_price" in evidence  # FR-C-01


def test_check1_runs_across_the_whole_grid_not_one_point(correct_model):
    """FR-C-04: one result per grid point, both option types included."""
    results = check_reference_price(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert len(results) == len(SMALL_GRID) == 4
    assert {result.spec["option_type"] for result in results} == {"call", "put"}


def test_check1_fails_a_model_that_is_off_by_a_scale_factor(tmp_path):
    source = CORRECT_SOURCE.replace("return phi * (", "return 1.02 * phi * (")
    with _flat(source, tmp_path, "scaled.py") as mut:
        results = check_reference_price(mut, SMALL_GRID, Tolerance(), AuditConfig())

    assert all(status == "FAIL" for status in _statuses(results))
    assert results[0].explanation


def test_check1_honours_a_user_set_tolerance(tmp_path):
    """FR-C-03: PASS and FAIL are decided by the Tolerance the user gives."""
    source = CORRECT_SOURCE.replace("return phi * (", "return 1.001 * phi * (")
    with _flat(source, tmp_path, "nearly.py") as mut:
        strict = check_reference_price(mut, SMALL_GRID, Tolerance(relative=1e-6), AuditConfig())
        loose = check_reference_price(mut, SMALL_GRID, Tolerance(relative=1e-2), AuditConfig())

    assert all(status == "FAIL" for status in _statuses(strict))
    assert all(status == "PASS" for status in _statuses(loose))


# ---------------------------------------------------------------------------
# Check Type 2 -- put-call parity
# ---------------------------------------------------------------------------


def test_check2_passes_a_correct_model_on_every_pair(correct_model):
    results = check_put_call_parity(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert len(results) == len(SMALL_GRID.call_put_pairs()) == 2
    assert all(status == "PASS" for status in _statuses(results))


def test_check2_reports_the_exact_numeric_gap_on_failure(tmp_path):
    """FR-C-06: the failure must carry the parity gap as a number."""
    source = CORRECT_SOURCE.replace(
        "K * math.exp(-r * T) * _N(phi * d2)", "K * _N(phi * d2)"
    )
    with _flat(source, tmp_path, "no_discount.py") as mut:
        results = check_put_call_parity(mut, SMALL_GRID, Tolerance(), AuditConfig())

    assert all(status == "FAIL" for status in _statuses(results))
    gap = results[0].evidence["parity_gap"]
    assert isinstance(gap, float) and abs(gap) > 1e-6
    assert "parity_right_hand_side" in results[0].evidence


def test_check2_is_skipped_where_the_equation_does_not_hold(correct_model):
    """Parity in this form is a European statement, and the report says so."""
    grid = default_grid("american")
    results = check_put_call_parity(correct_model, grid, Tolerance(), AuditConfig())

    assert len(results) == 1 and results[0].status == "SKIP"
    assert "European" in results[0].evidence["reason"]


# ---------------------------------------------------------------------------
# Check Type 3 -- Greek consistency
# ---------------------------------------------------------------------------


def test_check3_derives_all_five_greeks_from_the_model_alone(correct_model, vanilla):
    """FR-C-07: bump-and-reprice on the model's own price function."""
    greeks = model_greeks(correct_model, vanilla)

    for name in Greeks.NAMES:
        assert math.isfinite(getattr(greeks, name)), name
    assert greeks.delta == pytest.approx(0.6368, abs=1e-3)
    assert greeks.gamma == pytest.approx(0.01876, abs=1e-4)
    assert greeks.vega == pytest.approx(37.524, abs=1e-2)
    assert greeks.rho == pytest.approx(53.232, abs=1e-2)
    assert greeks.theta == pytest.approx(-6.414, abs=1e-2)


def test_check3_passes_a_correct_model(correct_model):
    results = check_greeks(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert results and all(status == "PASS" for status in _statuses(results))
    greeks_checked = {r.evidence.get("greek") for r in results if "greek" in r.evidence}
    assert greeks_checked == set(Greeks.NAMES)  # FR-C-08 covers all five


def test_check3_fails_a_model_whose_surface_has_the_wrong_slope(tmp_path):
    """A term linear in the spot leaves the price near enough and delta wrong."""
    source = CORRECT_SOURCE.replace(
        "return phi * (S * math.exp(-q * T)", "return 0.25 * (S - K) + phi * (S * math.exp(-q * T)"
    ).replace("_N(phi * d2))", "_N(phi * d2))")
    with _flat(source, tmp_path, "wrong_slope.py") as mut:
        results = check_greeks(mut, SMALL_GRID, Tolerance(), AuditConfig())

    delta_results = [r for r in results if r.evidence.get("greek") == "delta"]
    assert delta_results and all(r.status == "FAIL" for r in delta_results)
    assert abs(delta_results[0].evidence["absolute_difference"]) == pytest.approx(0.25, abs=1e-3)


def test_check3_flags_a_delta_outside_the_range_the_payoff_allows(tmp_path):
    """FR-C-09: call delta in [0, 1], put delta in [-1, 0]."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            # Delta is 3.0 everywhere, which no option payoff permits.
            return 3.0 * S
    """
    with _flat(source, tmp_path, "huge_delta.py") as mut:
        results = check_greeks(mut, SMALL_GRID, Tolerance(), AuditConfig())

    range_checks = [r for r in results if r.case.startswith("delta range")]
    assert range_checks and all(r.status == "FAIL" for r in range_checks)
    assert range_checks[0].evidence["allowed_low"] == 0.0
    assert range_checks[0].evidence["allowed_high"] == 1.0


def test_check3_delta_range_for_a_put_is_minus_one_to_zero(correct_model):
    grid = ParameterGrid(
        option_types=("put",), spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,)
    )
    results = check_greeks(correct_model, grid, Tolerance(), AuditConfig())
    range_check = next(r for r in results if r.case.startswith("delta range"))

    assert range_check.status == "PASS"
    assert (range_check.evidence["allowed_low"], range_check.evidence["allowed_high"]) == (-1.0, 0.0)


# ---------------------------------------------------------------------------
# Check Type 4 -- convergence and stability
# ---------------------------------------------------------------------------


def test_check4_passes_a_convergent_monte_carlo_model(mc_model_path):
    """FR-C-10, FR-C-11: more paths, smaller error."""
    config = AuditConfig(precision_levels=(2_000, 20_000, 200_000), convergence_cases=2)
    with load_model(mc_model_path) as mut:
        results = check_convergence(mut, SMALL_GRID, Tolerance(), config)

    assert results and all(status == "PASS" for status in _statuses(results))
    evidence = results[0].evidence
    assert evidence["precision_levels"] == [2_000, 20_000, 200_000]
    assert evidence["absolute_errors"][-1] < evidence["absolute_errors"][0]


def test_check4_fails_a_model_that_converges_to_the_wrong_value(biased_mc_model_path):
    """The defect this check exists to catch: a bias, not noise."""
    config = AuditConfig(precision_levels=(2_000, 20_000, 200_000), convergence_cases=2)
    with load_model(biased_mc_model_path) as mut:
        results = check_convergence(mut, SMALL_GRID, Tolerance(), config)

    assert results and all(status == "FAIL" for status in _statuses(results))
    errors = results[0].evidence["absolute_errors"]
    assert errors[-1] > 0.05  # the bias survives every path count
    assert results[0].explanation


def test_check4_records_the_series_needed_for_the_convergence_plot(mc_model_path):
    """Report section 5 needs the whole series, not only the verdict."""
    config = AuditConfig(precision_levels=(2_000, 20_000), convergence_cases=1)
    with load_model(mc_model_path) as mut:
        results = check_convergence(mut, SMALL_GRID, Tolerance(), config)

    evidence = results[0].evidence
    assert len(evidence["model_prices"]) == len(evidence["precision_levels"]) == 2
    assert evidence["reference_price"] > 0.0


# ---------------------------------------------------------------------------
# Check Type 5 -- edge cases
# ---------------------------------------------------------------------------


def test_check5_passes_a_model_that_handles_every_boundary(correct_model):
    results = check_edge_cases(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert results and all(status == "PASS" for status in _statuses(results))
    cases = {result.case.split(" | ")[0] for result in results}
    assert cases == {"zero volatility", "zero time to expiry", "high volatility", "zero spot"}


@pytest.mark.parametrize(
    "case,source_fragment",
    [
        ("zero volatility", "if sigma <= 0.0:\n            return 0.0"),
        ("zero time to expiry", "if T <= 0.0:\n            return 0.0"),
    ],
)
def test_check5_fails_a_model_that_returns_zero_at_a_boundary(tmp_path, case, source_fragment):
    """FR-C-12 and FR-C-13: a known payoff is not zero just because it is easy."""
    source = CORRECT_SOURCE
    if case == "zero volatility":
        source = source.replace(
            "if sigma <= 0.0:\n        forward = S * math.exp((r - q) * T)\n"
            "        return math.exp(-r * T) * max(phi * (forward - K), 0.0)",
            "if sigma <= 0.0:\n        return 0.0",
        )
    else:
        source = source.replace(
            "if T <= 0.0:\n        return max(phi * (S - K), 0.0)", "if T <= 0.0:\n        return 0.0"
        )

    grid = ParameterGrid(
        option_types=("call",), spots=(120.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,)
    )
    with _flat(source, tmp_path, f"edge_{case.replace(' ', '_')}.py") as mut:
        results = check_edge_cases(mut, grid, Tolerance(), AuditConfig())

    failed = [r for r in results if r.case.startswith(case)]
    assert failed and all(r.status == "FAIL" for r in failed)
    assert "expected_price" in failed[0].evidence


def test_check5_fails_a_model_that_goes_negative_at_high_volatility(tmp_path):
    """FR-C-14: extreme volatility must not produce a negative price or a fault."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            if sigma > 3.0:
                return -1.0
            return max(S - K, 0.0)
    """
    with _flat(source, tmp_path, "negative_at_high_vol.py") as mut:
        results = check_edge_cases(mut, SMALL_GRID, Tolerance(), AuditConfig(high_volatility=5.0))

    high_vol = [r for r in results if r.case.startswith("high volatility")]
    assert high_vol and all(r.status == "FAIL" for r in high_vol)


def test_check5_fails_a_model_that_prices_a_call_above_zero_at_a_zero_spot(tmp_path):
    """FR-C-15: zero is absorbing, so a call there is worthless."""
    source = """
        import math

        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            if S <= 0.0:
                return 5.0
            return max(S - K, 0.0)
    """
    grid = ParameterGrid(
        option_types=("call",), spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,)
    )
    with _flat(source, tmp_path, "zero_spot.py") as mut:
        results = check_edge_cases(mut, grid, Tolerance(), AuditConfig())

    zero_spot = [r for r in results if r.case.startswith("zero spot")]
    assert zero_spot and zero_spot[0].status == "FAIL"
    assert zero_spot[0].evidence["expected_price"] == pytest.approx(0.0)


def test_check5_records_a_fault_at_high_volatility_rather_than_stopping(tmp_path):
    """NFR-05: a raise inside the model becomes a result, not a crash."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            if sigma > 3.0:
                raise OverflowError("variance term overflowed")
            return max(S - K, 0.0)
    """
    with _flat(source, tmp_path, "raises_at_high_vol.py") as mut:
        results = check_edge_cases(mut, SMALL_GRID, Tolerance(), AuditConfig())

    high_vol = [r for r in results if r.case.startswith("high volatility")]
    assert high_vol and high_vol[0].status == "ERROR"
    assert "OverflowError" in high_vol[0].evidence["model_message"]


# ---------------------------------------------------------------------------
# Check Type 6 -- arbitrage-free sanity
# ---------------------------------------------------------------------------


def test_check6_passes_a_correct_model(correct_model):
    results = check_arbitrage(correct_model, SMALL_GRID, Tolerance(), AuditConfig())

    assert results and all(status == "PASS" for status in _statuses(results))


def test_check6_flags_a_call_price_that_rises_with_the_strike(tmp_path):
    """FR-C-16: a call must be non-increasing in the strike."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            # The strike is added instead of subtracted.
            return max(S + 0.05 * K, 0.0) * 0.1
    """
    with _flat(source, tmp_path, "rising_in_strike.py") as mut:
        results = check_arbitrage(mut, SMALL_GRID, Tolerance(), AuditConfig())

    monotone = [r for r in results if r.case.startswith("strike monotonicity")]
    assert monotone and any(r.status == "FAIL" for r in monotone)
    failed = next(r for r in monotone if r.status == "FAIL")
    assert failed.evidence["violations"]
    assert failed.evidence["strikes"] == sorted(AuditConfig().strike_ladder)


def test_check6_flags_a_price_above_the_upper_no_arbitrage_bound(tmp_path):
    """FR-C-17: the price must sit inside its static bounds."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            return 5.0 * S
    """
    with _flat(source, tmp_path, "too_expensive.py") as mut:
        results = check_arbitrage(mut, SMALL_GRID, Tolerance(), AuditConfig())

    bounds = [r for r in results if r.case.startswith("no-arbitrage bounds")]
    assert bounds and all(r.status == "FAIL" for r in bounds)
    assert bounds[0].evidence["upper_bound"] < bounds[0].evidence["model_price"]
    assert "intrinsic_value" in bounds[0].evidence
    assert "spot" in bounds[0].evidence


def test_check6_flags_a_negative_price(tmp_path):
    """FR-C-18: a negative price is flagged wherever it appears."""
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            return S - K - 20.0
    """
    with _flat(source, tmp_path, "negative.py") as mut:
        results = check_arbitrage(mut, SMALL_GRID, Tolerance(), AuditConfig())

    scan = next(r for r in results if r.case.startswith("negative price scan"))
    assert scan.status == "FAIL"
    assert scan.evidence["negative_prices"]
    assert scan.evidence["tested"] == len(SMALL_GRID)


def test_check6_negative_price_scan_passes_when_no_price_is_negative(correct_model):
    results = check_arbitrage(correct_model, SMALL_GRID, Tolerance(), AuditConfig())
    scan = next(r for r in results if r.case.startswith("negative price scan"))

    assert scan.status == "PASS"
    assert scan.evidence["tested"] == len(SMALL_GRID)


# ---------------------------------------------------------------------------
# grid behaviour
# ---------------------------------------------------------------------------


def test_the_grid_sample_is_deterministic_and_spreads_over_the_axes():
    grid = default_grid("european")
    first = grid.sample(8, seed=0)
    again = grid.sample(8, seed=0)

    assert [spec.label() for spec in first] == [spec.label() for spec in again]
    assert len({spec.T for spec in first}) > 1
    assert len({spec.option_type for spec in first}) > 1


def test_a_barrier_grid_keeps_every_spot_on_the_live_side_of_the_barrier():
    grid = default_grid("barrier", extras={"barrier": 120.0, "barrier_kind": "up-and-out"})
    assert all(spot < 120.0 for spot in grid.spots)

    grid = default_grid("barrier", extras={"barrier": 90.0, "barrier_kind": "down-and-out"})
    assert all(spot > 90.0 for spot in grid.spots)

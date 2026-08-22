"""Validation against financepy -- a SECOND oracle this author did not write.

Table 1 of BENCHMARKS.md used to rest on a single external library. Two
implementations by one author ratify each other's mistakes; the same is true
of two implementations only when they share assumptions, and QuantLib and
financepy share nothing: different codebases, different maintainers, and in
several places different numerical conventions from Plumbline as well. Where
all three agree, the answer is very likely right.

Honesty about conventions, because that is where this file could mislead:

* **Digitals** are compared like for like. Both sides price the continuous
  lognormal probability, so agreement is measured in the last few digits.
* **Barriers** are NOT like for like. financepy monitors its barrier
  discretely at ``num_obs_per_year`` dates; Plumbline's Reiner-Rubinstein
  formulas assume continuous monitoring. The gap between them is the known
  O(sqrt(dt)) discrete-monitoring bias, it shrinks as observations are added,
  and the tolerance below exists to absorb exactly that -- not to hide a
  formula error, which moves the price by far more. The test also asserts the
  convergence direction on one contract.
* **Geometric Asian**: financepy's GEOMETRIC method uses a continuous-mean /
  discrete-variance hybrid rather than either exact convention. It converges
  to the continuous Kemna-Vorst value as fixings are added; away from the
  money the residual is larger, and the tolerance states so.

financepy is an optional dependency of the ``oracle`` extra::

    pip install "plumbline[oracle]"

The tests skip without it. They carry the ``oracle`` marker and run in CI.
"""

from __future__ import annotations

import pytest

fp = pytest.importorskip("financepy", reason="financepy is not installed")

from financepy.market.curves import DiscountCurveFlat
from financepy.models.black_scholes import BlackScholes
from financepy.products.equity.equity_asian_option import (
    AsianOptionValuationMethods,
    EquityAsianOption,
)
from financepy.products.equity.equity_barrier_option import EquityBarrierOption
from financepy.products.equity.equity_digital_option import (
    EquityDigitalOption,
    FinDigitalOptionTypes,
)
from financepy.utils import Date, OptionTypes
from financepy.utils.global_types import EquityBarrierTypes

from plumbline.contracts import OptionSpec
from plumbline.engines import analytic

pytestmark = pytest.mark.oracle

TODAY = Date(15, 8, 2026)


def _curves(r: float, q: float):
    return DiscountCurveFlat(TODAY, r), DiscountCurveFlat(TODAY, q)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    "payout,fin_payout",
    [("cash", FinDigitalOptionTypes.CASH_OR_NOTHING),
     ("asset", FinDigitalOptionTypes.ASSET_OR_NOTHING)],
)
def test_digitals_match_a_second_library(option_type, payout, fin_payout):
    """Cash- and asset-or-nothing digitals against financepy.

    Like-for-like: both sides integrate the same continuous lognormal law, so
    the honest expectation is near machine precision (measured: 1.2e-8 cash,
    4.8e-6 asset).
    """
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry = TODAY.add_years(1)
    disc, div = _curves(r, q)

    if option_type == "call":
        fin_opt = OptionTypes.EUROPEAN_CALL
    else:
        fin_opt = OptionTypes.EUROPEAN_PUT

    digital = EquityDigitalOption(expiry, float(K), fin_opt, fin_payout)
    reference = digital.value(TODAY, S, disc, div, BlackScholes(sigma))

    mine = analytic.digital_price(
        OptionSpec(
            "digital", option_type, S=S, K=K, T=1.0, r=r, q=q, sigma=sigma,
            payout=payout,
        )
    )

    tolerance = 1e-7 if payout == "cash" else 2e-5
    assert abs(mine - reference) < tolerance


BARRIER_CASES = [
    ("down-and-out", EquityBarrierTypes.DOWN_AND_OUT_CALL, 90.0, "call"),
    ("up-and-out", EquityBarrierTypes.UP_AND_OUT_CALL, 130.0, "call"),
    ("down-and-in", EquityBarrierTypes.DOWN_AND_IN_CALL, 90.0, "call"),
    ("up-and-in", EquityBarrierTypes.UP_AND_IN_CALL, 130.0, "call"),
    ("down-and-out", EquityBarrierTypes.DOWN_AND_OUT_PUT, 90.0, "put"),
    ("up-and-out", EquityBarrierTypes.UP_AND_OUT_PUT, 130.0, "put"),
    ("down-and-in", EquityBarrierTypes.DOWN_AND_IN_PUT, 90.0, "put"),
    ("up-and-in", EquityBarrierTypes.UP_AND_IN_PUT, 130.0, "put"),
]


@pytest.mark.slow
@pytest.mark.parametrize("kind,fin_kind,barrier,option_type", BARRIER_CASES)
def test_barriers_match_a_second_library(kind, fin_kind, barrier, option_type):
    """All eight standard barriers against financepy's engine.

    Not like-for-like and labelled as such: financepy samples the barrier at
    200 000 dates over the year, Plumbline assumes continuous monitoring. At
    that density the residual monitoring bias is below 1.5e-2 absolute on
    every case measured (it is largest for the out-calls near the barrier);
    the assertion sits just above that ceiling. A derivation error -- a wrong
    mu, a flipped eta, a missing reflection term -- shifts these prices by
    tenths to units, far outside the band.
    """
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry = TODAY.add_years(1)
    disc, div = _curves(r, q)
    obs_per_year = 200_000

    option = EquityBarrierOption(
        expiry, float(K), fin_kind, float(barrier), num_obs_per_year=obs_per_year
    )
    reference = option.value(TODAY, S, disc, div, BlackScholes(sigma))

    mine = analytic.barrier_price(
        OptionSpec(
            "barrier", option_type, S=S, K=K, T=1.0, r=r, q=q, sigma=sigma,
            barrier=barrier, barrier_kind=kind,
        )
    )

    assert abs(mine - reference) < 2e-2


@pytest.mark.slow
def test_barrier_discrete_monitoring_converges_toward_the_continuous_value():
    """Direction check for the convention gap the barrier test absorbs.

    A discretely monitored knock-out pays off at least as often as a
    continuously monitored one, so its value must FALL toward Plumbline's
    continuous value as the observation count rises. If adding observations
    moved the value the wrong way, the tolerance above would be papering over
    something worse than a convention difference.
    """
    r, q = 0.05, 0.02
    expiry = TODAY.add_years(1)
    disc, div = _curves(r, q)

    coarse = EquityBarrierOption(
        expiry, 100.0, EquityBarrierTypes.DOWN_AND_OUT_CALL, 90.0,
        num_obs_per_year=25_000,
    ).value(TODAY, 100.0, disc, div, BlackScholes(0.25))
    fine = EquityBarrierOption(
        expiry, 100.0, EquityBarrierTypes.DOWN_AND_OUT_CALL, 90.0,
        num_obs_per_year=200_000,
    ).value(TODAY, 100.0, disc, div, BlackScholes(0.25))
    continuous = analytic.barrier_price(
        OptionSpec(
            "barrier", "call", S=100.0, K=100.0, T=1.0, r=r, q=q, sigma=0.25,
            barrier=90.0, barrier_kind="down-and-out",
        )
    )

    assert coarse > fine > continuous


@pytest.mark.slow
@pytest.mark.parametrize(
    "S,K,years,r,q,sigma,tolerance",
    [
        # ATM agrees to ~8e-5; the hybrid method drifts away from the money
        # and at long maturities, hence the per-case bands instead of one.
        (100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 3e-4),
        (90.0, 105.0, 0.75, 0.03, 0.00, 0.18, 4e-3),
        (80.0, 95.0, 2.0, 0.06, 0.03, 0.40, 8e-3),
    ],
)
def test_geometric_asian_matches_a_second_library(S, K, years, r, q, sigma, tolerance):
    """Continuous Kemna-Vorst against financepy's GEOMETRIC method.

    financepy's implementation holds the mean at the continuous midpoint
    while using a discrete-fixings variance, so it approaches the exact
    continuous value as fixings accumulate rather than matching it exactly.
    Measured residuals: 7.9e-5 ATM, 1.4e-3 OTM short, 5.0e-3 long high-vol.
    """
    expiry = TODAY.add_years(years)
    disc, div = _curves(r, q)

    option = EquityAsianOption(
        TODAY, expiry, float(K), OptionTypes.EUROPEAN_CALL, num_obs=20_000
    )
    reference = option.value(
        TODAY, float(S), disc, div, BlackScholes(sigma),
        AsianOptionValuationMethods.GEOMETRIC,
    )

    mine = analytic.geometric_asian_price(
        OptionSpec("asian", "call", S=S, K=K, T=years, r=r, q=q, sigma=sigma)
    )

    assert abs(mine - reference) < tolerance

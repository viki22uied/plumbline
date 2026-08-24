"""Validation against QuantLib, an oracle Plumbline's author did not write.

Every other test file in this suite proves *internal consistency*: Plumbline's
engines agreeing with Plumbline's other engines, or with numbers typed into
Plumbline's own test file. That is a necessary check and it is not validation.
Two implementations by one author share one set of assumptions, and correlated
errors ratify each other rather than cancelling. The seven defects found in
these engines all passed internal consistency while being wrong.

This file is the answer to that. QuantLib is a 25-year-old library maintained
by a large independent community, used in production across the industry, and
written by people who have never seen this repository. Where Plumbline and
QuantLib agree to machine precision, the shared answer is almost certainly
right. Where they disagree, one of them has a bug and it is worth finding out
which.

QuantLib is an optional dependency, installed with::

    pip install "plumbline[oracle]"

The tests skip without it, so the suite stays green on a machine that does not
have it. CI installs it.

Convention note: QuantLib works in dates, not year fractions. Every case below
takes the maturity date first and asks QuantLib's own day counter what year
fraction that is, then hands the same number to Plumbline. Comparing against
an oracle while disagreeing about what "one year" means would measure the
calendar, not the mathematics.
"""

from __future__ import annotations

import math

import pytest

ql = pytest.importorskip("QuantLib", reason="QuantLib is not installed")

from plumbline.contracts import OptionSpec
from plumbline.engines import analytic, binomial, fdm, heston

pytestmark = pytest.mark.oracle

TODAY = ql.Date(15, 8, 2026)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()


@pytest.fixture(autouse=True)
def _evaluation_date():
    ql.Settings.instance().evaluationDate = TODAY
    yield


def _maturity(years: float) -> tuple[ql.Date, float]:
    """A QuantLib date and the year fraction QuantLib assigns to it."""
    expiry = TODAY + int(round(years * 365))
    return expiry, DAY_COUNT.yearFraction(TODAY, expiry)


def _process(S: float, r: float, q: float, sigma: float):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DAY_COUNT)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, CALENDAR, sigma, DAY_COUNT)),
    )


EUROPEAN_CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
    (90.0, 105.0, 0.25, 0.03, 0.0, 0.15),
    (130.0, 90.0, 3.0, 0.08, 0.03, 0.45),
    (100.0, 100.0, 1.0, -0.01, 0.02, 0.30),
    (60.0, 100.0, 5.0, 0.02, 0.01, 0.60),
]


@pytest.mark.parametrize("S,K,years,r,q,sigma", EUROPEAN_CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_scholes_matches_quantlib_to_machine_precision(
    S, K, years, r, q, sigma, option_type
):
    """Two closed forms of the same formula should agree to the last bits.

    This is the strongest assertion in the whole suite. There is no
    discretisation and no sampling on either side, so any difference above
    rounding is a real disagreement about the mathematics.
    """
    expiry, T = _maturity(years)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(_process(S, r, q, sigma)))

    mine = analytic.black_scholes_price(
        OptionSpec("european", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    )

    assert mine == pytest.approx(option.NPV(), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("S,K,years,r,q,sigma", EUROPEAN_CASES[:3])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_scholes_greeks_match_quantlib(S, K, years, r, q, sigma, option_type):
    """A price can be right while its slope is wrong. QuantLib checks both."""
    expiry, T = _maturity(years)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(_process(S, r, q, sigma)))

    mine = analytic.black_scholes_greeks(
        OptionSpec("european", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    )

    assert mine.delta == pytest.approx(option.delta(), rel=1e-10)
    assert mine.gamma == pytest.approx(option.gamma(), rel=1e-10)
    # QuantLib quotes vega and rho per percentage point, and theta per year.
    assert mine.vega == pytest.approx(option.vega(), rel=1e-10)
    assert mine.rho == pytest.approx(option.rho(), rel=1e-10)
    assert mine.theta == pytest.approx(option.thetaPerDay() * 365.0, rel=1e-6)


def test_the_american_put_benchmark_against_quantlibs_high_precision_engine():
    """GT-02's benchmark, sourced from an engine built for exactly this.

    QuantLib's ``QdFpAmericanEngine`` implements the Andersen, Lake and
    Offengenden (2016) fixed-point method, which is the current high-accuracy
    reference for American puts. It shares no approach with a binomial tree.
    """
    S = K = 100.0
    r, q, sigma = 0.05, 0.0, 0.20
    expiry, T = _maturity(1.0)

    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, K), ql.AmericanExercise(TODAY, expiry)
    )
    option.setPricingEngine(
        ql.QdFpAmericanEngine(
            _process(S, r, q, sigma), ql.QdFpAmericanEngine.highPrecisionScheme()
        )
    )
    reference = option.NPV()

    # The value this repository quotes as the benchmark.
    assert reference == pytest.approx(6.09037, abs=5e-6)

    mine = binomial.binomial_price(
        OptionSpec("american", "put", S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    )
    assert abs(mine - reference) / reference < 1e-4  # NFR-01, at the default settings


@pytest.mark.parametrize("S,K,years,sigma", [(100, 100, 1.0, 0.2), (90, 100, 0.5, 0.3),
                                             (110, 100, 2.0, 0.25)])
@pytest.mark.parametrize("option_type", ["put", "call"])
def test_american_options_match_quantlib(S, K, years, sigma, option_type):
    r, q = 0.06, 0.02
    expiry, T = _maturity(years)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K
        ),
        ql.AmericanExercise(TODAY, expiry),
    )
    option.setPricingEngine(ql.BinomialVanillaEngine(_process(S, r, q, sigma), "crr", 8000))

    mine = binomial.binomial_price(
        OptionSpec("american", option_type, S=float(S), K=float(K), T=T, r=r, q=q, sigma=sigma)
    )

    assert abs(mine - option.NPV()) < 1e-3


HESTON_PARAMETERS = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)


@pytest.mark.parametrize("K,years", [(80, 1.0), (100, 1.0), (120, 2.0), (100, 10.0),
                                     (150, 0.5), (100, 0.25)])
def test_heston_matches_quantlibs_analytic_engine(K, years):
    """The parameter set of Albrecher, Mayer, Schoutens & Tistaert (2007).

    The long maturities matter most here. The 1993 form of the characteristic
    function crosses a branch cut of the complex logarithm and returns wrong
    prices past roughly a year; the ten-year case would expose that.
    """
    S, r, q = 100.0, 0.0, 0.0
    expiry, T = _maturity(years)
    p = HESTON_PARAMETERS

    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DAY_COUNT)),
        ql.QuoteHandle(ql.SimpleQuote(S)),
        p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(expiry)
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 192))

    mine = heston.heston_price(
        OptionSpec(
            "european", "call", S=S, K=float(K), T=T, r=r, q=q, model="heston",
            v0=p["v0"], kappa=p["kappa"], theta_v=p["theta"], xi=p["xi"], rho_sv=p["rho"],
        )
    )

    assert mine == pytest.approx(option.NPV(), abs=1e-8)


def test_the_heston_benchmark_value_this_repository_quotes():
    """GT-05's 5.785155, checked against an implementation we did not write."""
    S, K, r, q = 100.0, 100.0, 0.0, 0.0
    expiry, T = _maturity(1.0)
    p = HESTON_PARAMETERS

    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DAY_COUNT)),
        ql.QuoteHandle(ql.SimpleQuote(S)),
        p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(expiry)
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 192))

    assert option.NPV() == pytest.approx(5.785155, abs=1e-6)


BARRIER_CASES = [
    ("down-and-out", ql.Barrier.DownOut, 90.0),
    ("down-and-in", ql.Barrier.DownIn, 90.0),
    ("up-and-out", ql.Barrier.UpOut, 130.0),
    ("up-and-in", ql.Barrier.UpIn, 130.0),
]


@pytest.mark.parametrize("kind,ql_kind,barrier", BARRIER_CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_barriers_match_quantlibs_reiner_rubinstein(kind, ql_kind, barrier, option_type):
    """All eight standard barriers against QuantLib's own analytic engine."""
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry, T = _maturity(1.0)

    option = ql.BarrierOption(
        ql_kind,
        barrier,
        0.0,  # rebate
        ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K
        ),
        ql.EuropeanExercise(expiry),
    )
    option.setPricingEngine(ql.AnalyticBarrierEngine(_process(S, r, q, sigma)))

    mine = analytic.barrier_price(
        OptionSpec(
            "barrier", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            barrier=barrier, barrier_kind=kind,
        )
    )

    assert mine == pytest.approx(option.NPV(), abs=1e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_cash_or_nothing_digitals_match_quantlib(option_type):
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry, T = _maturity(1.0)

    option = ql.VanillaOption(
        ql.CashOrNothingPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K, 1.0
        ),
        ql.EuropeanExercise(expiry),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(_process(S, r, q, sigma)))

    mine = analytic.digital_price(
        OptionSpec(
            "digital", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma, payout="cash"
        )
    )

    assert mine == pytest.approx(option.NPV(), abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_asset_or_nothing_digitals_match_quantlib(option_type):
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry, T = _maturity(1.0)

    option = ql.VanillaOption(
        ql.AssetOrNothingPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K
        ),
        ql.EuropeanExercise(expiry),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(_process(S, r, q, sigma)))

    mine = analytic.digital_price(
        OptionSpec(
            "digital", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma, payout="asset"
        )
    )

    assert mine == pytest.approx(option.NPV(), abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_continuous_geometric_asian_matches_quantlib(option_type):
    """Kemna-Vorst, against QuantLib's continuous-averaging engine."""
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry, T = _maturity(1.0)

    option = ql.ContinuousAveragingAsianOption(
        ql.Average().Geometric,
        ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K
        ),
        ql.EuropeanExercise(expiry),
    )
    option.setPricingEngine(
        ql.AnalyticContinuousGeometricAveragePriceAsianEngine(_process(S, r, q, sigma))
    )

    mine = analytic.geometric_asian_price(
        OptionSpec("asian", option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    )

    assert mine == pytest.approx(option.NPV(), abs=1e-9)


def test_the_finite_difference_engine_matches_quantlibs_grid():
    """Two independent Crank-Nicolson implementations of the same PDE."""
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.02, 0.25
    expiry, T = _maturity(1.0)

    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(expiry)
    )
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(_process(S, r, q, sigma), 2000, 2000)
    )

    mine = fdm.fdm_price(
        OptionSpec("european", "call", S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    )

    assert abs(mine - option.NPV()) < 1e-3

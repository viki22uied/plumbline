"""Module C -- the six check types (FR-C-01 to FR-C-18).

Every check function has the same shape::

    check(mut, grid, tolerance, config) -> list[CheckResult]

and every one is called inside its own try/except by the audit engine, so a
check that blows up costs its own results and nothing else (NFR-05).

A result is one of:

``PASS``        the model satisfied the check on this parameter set
``FAIL``        the model violated the check, with numeric evidence
``ERROR``       the model raised, returned a non-number, or crashed
``TIMEOUT``     the model exceeded the per-call time limit (FR-A-06)
``NOT_PRICED``  a table model has no row for this parameter set
``SKIP``        the check does not apply here, with the reason recorded
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from plumbline.audit.grid import ParameterGrid
from plumbline.contracts import (
    Greeks,
    OptionSpec,
    PriceResult,
    Tolerance,
    PlumblineError,
    relative_difference,
)
from plumbline.engines.limits import degenerate_value
from plumbline.engines.registry import ground_truth_for, ground_truth_price
from plumbline.ingestion import ModelUnderTest

PASS, FAIL, ERROR, TIMEOUT, SKIP, NOT_PRICED = (
    "PASS",
    "FAIL",
    "ERROR",
    "TIMEOUT",
    "SKIP",
    "NOT_PRICED",
)

CHECK_NAMES = {
    1: "Reference Price Comparison",
    2: "Put-Call Parity",
    3: "Greek Consistency",
    4: "Convergence and Stability",
    5: "Edge Case and Boundary Behaviour",
    6: "Arbitrage-Free Sanity",
}


@dataclass
class CheckResult:
    """One row of section 3 of the Audit Report."""

    check_type: int
    case: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def check_name(self) -> str:
        return CHECK_NAMES[self.check_type]

    @property
    def counts_against(self) -> bool:
        """True when this result lowers the Audit Score."""
        return self.status in (FAIL, ERROR, TIMEOUT)

    @property
    def counts_for(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_type": self.check_type,
            "check_name": self.check_name,
            "case": self.case,
            "status": self.status,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "spec": self.spec,
        }


@dataclass
class AuditConfig:
    """Knobs that shape how hard the audit pushes."""

    #: How many grid points each costly check samples. Check Type 1 always
    #: runs on the whole grid; the others each cost several model calls per
    #: point, so they take a deterministic spread of it instead.
    greek_cases: int = 12
    edge_cases: int = 6
    convergence_cases: int = 4
    arbitrage_cases: int = 8
    #: Seed for that spread. It is written into the report, so a run repeats.
    sample_seed: int = 0
    #: Precision levels for Check Type 4, smallest first.
    precision_levels: tuple[int, ...] = (1_000, 4_000, 16_000, 64_000)
    #: The "very high volatility" of FR-C-14.
    high_volatility: float = 5.0
    #: Strike ladder for the monotonicity test of FR-C-16.
    strike_ladder: tuple[float, ...] = (60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
    #: Relative spot bump used for the model's own numerical Greeks.
    greek_bump: float = 1e-3
    #: A convergence run must shrink the error to at most this share of the
    #: error at the coarsest precision, or reach the tolerance outright.
    convergence_shrink: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "greek_cases": self.greek_cases,
            "edge_cases": self.edge_cases,
            "convergence_cases": self.convergence_cases,
            "arbitrage_cases": self.arbitrage_cases,
            "sample_seed": self.sample_seed,
            "precision_levels": list(self.precision_levels),
            "high_volatility": self.high_volatility,
            "strike_ladder": list(self.strike_ladder),
            "greek_bump": self.greek_bump,
            "convergence_shrink": self.convergence_shrink,
        }


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _mut_result(mut: ModelUnderTest, spec: OptionSpec) -> PriceResult:
    try:
        return mut.price(spec)
    except Exception as exc:  # a broken model must not stop the audit
        return PriceResult(status=ERROR, message=f"{type(exc).__name__}: {exc}", engine="mut")


def _failed_result(check_type: int, spec: OptionSpec, result: PriceResult) -> CheckResult | None:
    """Turn a non-OK model call into a CheckResult, or return None if it was OK."""
    if result.status == "OK":
        return None
    status = {
        "TIMEOUT": TIMEOUT,
        "NOT_PRICED": NOT_PRICED,
    }.get(result.status, ERROR)
    explanations = {
        TIMEOUT: (
            "The model did not return a price before the time limit. A pricing "
            "routine that cannot answer in seconds usually has a loop that does "
            "not end, or a step count far larger than the problem needs."
        ),
        NOT_PRICED: (
            "The submitted price table has no row for this parameter set, so "
            "this check could not be run against it."
        ),
        ERROR: (
            "The model raised an error instead of returning a price. A correct "
            "pricing function must return a finite number for every valid input."
        ),
    }
    return CheckResult(
        check_type=check_type,
        case=spec.label(),
        status=status,
        evidence={"model_message": result.message},
        explanation=explanations[status],
        spec=spec.to_dict(),
    )


def _mut_price(mut: ModelUnderTest, spec: OptionSpec) -> float:
    """Model price as a plain float; NaN when the call did not succeed."""
    result = _mut_result(mut, spec)
    return result.price if result.status == "OK" else math.nan


def model_greeks(
    mut: ModelUnderTest, spec: OptionSpec, bump: float = 1e-3
) -> Greeks:
    """The model's own Greeks, by bump-and-reprice on its price function only."""
    from plumbline.engines.bump import bump_greeks

    return bump_greeks(lambda s: _mut_price(mut, s), spec, bump=bump)


# ---------------------------------------------------------------------------
# Check Type 1 -- Reference Price Comparison (FR-C-01 to FR-C-04)
# ---------------------------------------------------------------------------


def check_reference_price(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in grid:
        result = _mut_result(mut, spec)
        bad = _failed_result(1, spec, result)
        if bad is not None:
            results.append(bad)
            continue

        try:
            truth = ground_truth_price(spec, with_greeks=False)
            engine_name = truth.engine
        except PlumblineError as exc:
            results.append(
                CheckResult(
                    check_type=1,
                    case=spec.label(),
                    status=SKIP,
                    evidence={"reason": str(exc)},
                    spec=spec.to_dict(),
                )
            )
            continue

        band = tolerance.relative
        if not _is_deterministic(spec):
            band = max(band, tolerance.stochastic_relative)

        absolute = result.price - truth.price
        relative = relative_difference(result.price, truth.price)
        ok = abs(absolute) <= max(tolerance.absolute, band * abs(truth.price))

        results.append(
            CheckResult(
                check_type=1,
                case=spec.label(),
                status=PASS if ok else FAIL,
                evidence={
                    "model_price": result.price,
                    "reference_price": truth.price,
                    "reference_engine": engine_name,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                    "tolerance_relative": band,
                },
                explanation=""
                if ok
                else (
                    f"The model priced this option at {result.price:.6f}. The "
                    f"{engine_name} reference engine gives {truth.price:.6f}. That is a "
                    f"relative error of {relative * 100:.3f} percent, which is larger "
                    f"than the allowed {band * 100:.3f} percent. The model and the "
                    f"reference disagree on the value of the same contract, so at least "
                    f"one of the two is wrong, and the reference engine is the one with "
                    f"a published proof."
                ),
                spec=spec.to_dict(),
            )
        )
    return results


def _is_deterministic(spec: OptionSpec) -> bool:
    try:
        return ground_truth_for(spec).deterministic
    except PlumblineError:
        return True


# ---------------------------------------------------------------------------
# Check Type 2 -- Put-Call Parity (FR-C-05, FR-C-06)
# ---------------------------------------------------------------------------


def check_put_call_parity(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    """``C - P = S exp(-qT) - K exp(-rT)`` for European exercise.

    Parity is model-free: it follows from a static hedge, not from any
    assumption about the dynamics of the underlying.  A model that breaks it
    admits an arbitrage no matter how good its volatility model is.
    """
    if grid.instrument != "european":
        return [
            CheckResult(
                check_type=2,
                case=f"instrument={grid.instrument}",
                status=SKIP,
                evidence={
                    "reason": "put-call parity in this form holds for European "
                    "exercise only; American and path-dependent contracts obey "
                    "inequalities, not this equation"
                },
            )
        ]

    results: list[CheckResult] = []
    for call_spec, put_spec in grid.call_put_pairs():
        call_result = _mut_result(mut, call_spec)
        put_result = _mut_result(mut, put_spec)
        bad = _failed_result(2, call_spec, call_result) or _failed_result(2, put_spec, put_result)
        if bad is not None:
            results.append(bad)
            continue

        expected = call_spec.S * math.exp(-call_spec.q * call_spec.T) - call_spec.K * math.exp(
            -call_spec.r * call_spec.T
        )
        gap = (call_result.price - put_result.price) - expected
        scale = max(abs(expected), call_spec.K * math.exp(-call_spec.r * call_spec.T), 1.0)
        ok = abs(gap) <= max(tolerance.absolute, tolerance.relative * scale)

        results.append(
            CheckResult(
                check_type=2,
                case=call_spec.label().replace("call", "call/put"),
                status=PASS if ok else FAIL,
                evidence={
                    "call_price": call_result.price,
                    "put_price": put_result.price,
                    "call_minus_put": call_result.price - put_result.price,
                    "parity_right_hand_side": expected,
                    "parity_gap": gap,
                },
                explanation=""
                if ok
                else (
                    f"Put-call parity requires call minus put to equal "
                    f"{expected:.6f} for these parameters. The model gives "
                    f"{call_result.price - put_result.price:.6f}, a gap of {gap:+.6f}. "
                    f"This relationship comes from a portfolio that is risk free, so "
                    f"the gap is a risk-free profit of {abs(gap):.6f} per contract. "
                    f"The usual cause is a discount factor applied to the wrong term: "
                    f"the strike must be discounted at the risk-free rate and the spot "
                    f"at the dividend yield."
                ),
                spec=call_spec.to_dict(),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Check Type 3 -- Greek Consistency (FR-C-07 to FR-C-09)
# ---------------------------------------------------------------------------


def check_greeks(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in grid.sample(config.greek_cases, config.sample_seed):
        try:
            truth_engine = ground_truth_for(spec)
        except PlumblineError as exc:
            results.append(
                CheckResult(3, spec.label(), SKIP, {"reason": str(exc)}, spec=spec.to_dict())
            )
            continue

        model_side = model_greeks(mut, spec, bump=config.greek_bump)
        truth_side = truth_engine.price_fn(spec).greeks or Greeks()

        for name in Greeks.NAMES:
            actual = getattr(model_side, name)
            expected = getattr(truth_side, name)
            if not math.isfinite(expected):
                results.append(
                    CheckResult(
                        3,
                        f"{name} | {spec.label()}",
                        SKIP,
                        {"reason": f"the reference engine has no {name} here"},
                        spec=spec.to_dict(),
                    )
                )
                continue
            if not math.isfinite(actual):
                results.append(
                    CheckResult(
                        3,
                        f"{name} | {spec.label()}",
                        ERROR,
                        {"reference_value": expected},
                        explanation=(
                            f"The model's {name} could not be computed: at least one "
                            f"of the bumped prices was not a finite number. A price "
                            f"function must stay finite in a neighbourhood of every "
                            f"valid input, or no sensitivity exists."
                        ),
                        spec=spec.to_dict(),
                    )
                )
                continue

            ok = tolerance.greek_within(actual, expected)
            results.append(
                CheckResult(
                    3,
                    f"{name} | {spec.label()}",
                    PASS if ok else FAIL,
                    {
                        "greek": name,
                        "model_value": actual,
                        "reference_value": expected,
                        "absolute_difference": actual - expected,
                        "relative_difference": relative_difference(actual, expected),
                        "reference_engine": truth_engine.name,
                    },
                    explanation=""
                    if ok
                    else (
                        f"The model's {name}, measured by bumping the model's own price "
                        f"function, is {actual:.6f}. The reference value is "
                        f"{expected:.6f}. The price surface of the model therefore has "
                        f"the wrong slope in this parameter, even where the price level "
                        f"may look acceptable. A hedge built on this {name} would be "
                        f"the wrong size by {abs(actual - expected):.6f} per unit."
                    ),
                    spec=spec.to_dict(),
                )
            )

        results.extend(_delta_range_check(spec, model_side))
    return results


def _delta_range_check(spec: OptionSpec, model_side: Greeks) -> list[CheckResult]:
    """FR-C-09: delta must lie in the range the contract's payoff allows."""
    if spec.instrument not in ("european", "american"):
        return []
    delta = model_side.delta
    if not math.isfinite(delta):
        return []

    # The payoff bound: an option can never replicate more than one share.
    low, high = (0.0, 1.0) if spec.option_type == "call" else (-1.0, 0.0)
    bound_reason = "one share, the most an option's payoff can replicate"
    # A European option is bounded more tightly still, by exp(-q T) shares.
    # That is reported as evidence, not enforced, so the pass/fail rule stays
    # the payoff bound that holds for every exercise style.
    european_cap = math.exp(-spec.q * spec.T) if spec.instrument == "european" else None

    epsilon = 1e-6
    ok = low - epsilon <= delta <= high + epsilon
    return [
        CheckResult(
            3,
            f"delta range | {spec.label()}",
            PASS if ok else FAIL,
            {
                "model_delta": delta,
                "allowed_low": low,
                "allowed_high": high,
                "european_tight_bound": european_cap,
            },
            explanation=""
            if ok
            else (
                f"The model's delta is {delta:.6f}, outside the range "
                f"[{low:.6f}, {high:.6f}] that this contract allows. The upper limit is "
                f"{bound_reason}. A delta outside this range means the model claims the "
                f"option moves more than the asset it is written on, which no payoff "
                f"function permits."
            ),
            spec=spec.to_dict(),
        )
    ]


# ---------------------------------------------------------------------------
# Check Type 4 -- Convergence and Stability (FR-C-10, FR-C-11)
# ---------------------------------------------------------------------------


def check_convergence(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    """Raise the model's precision knob and watch the error fall.

    A numerical method that is merely *close* at one setting may be close by
    luck.  A method that is correct gets closer as it is given more work; a
    method with a bias converges to the wrong number, and this check is what
    separates the two.
    """
    if not mut.supports_precision:
        return [
            CheckResult(
                check_type=4,
                case="model has no precision parameter",
                status=SKIP,
                evidence={
                    "reason": "the model exposes no precision knob, so its output "
                    "cannot be refined; FR-C-10 applies only when one exists"
                },
            )
        ]

    results: list[CheckResult] = []
    for spec in grid.sample(config.convergence_cases, config.sample_seed):
        try:
            reference = ground_truth_price(spec, with_greeks=False).price
        except PlumblineError as exc:
            results.append(
                CheckResult(4, spec.label(), SKIP, {"reason": str(exc)}, spec=spec.to_dict())
            )
            continue

        levels: list[int] = []
        prices: list[float] = []
        errors: list[float] = []
        failed: CheckResult | None = None
        for level in config.precision_levels:
            result = _mut_result(mut, spec.with_(precision=level))
            failed = _failed_result(4, spec, result)
            if failed is not None:
                break
            levels.append(level)
            prices.append(result.price)
            errors.append(abs(result.price - reference))

        if failed is not None:
            results.append(failed)
            continue

        band = max(tolerance.absolute, tolerance.stochastic_relative * abs(reference))
        first, last = errors[0], errors[-1]
        converged = last <= band or last <= config.convergence_shrink * first
        diverging = last > first + band

        if converged and not diverging:
            status, explanation = PASS, ""
        else:
            status = FAIL
            direction = "grew" if diverging else "did not shrink"
            explanation = (
                f"The model's error against the reference {direction} as its precision "
                f"rose from {levels[0]} to {levels[-1]}: the error went from "
                f"{first:.6g} to {last:.6g}, while the reference value is "
                f"{reference:.6f}. A convergent method spends more work to get closer. "
                f"An error that stalls at a fixed level is the signature of a bias in "
                f"the method itself, not of sampling noise, and no extra precision will "
                f"remove it."
            )

        results.append(
            CheckResult(
                check_type=4,
                case=spec.label(),
                status=status,
                evidence={
                    "reference_price": reference,
                    "precision_levels": levels,
                    "model_prices": prices,
                    "absolute_errors": errors,
                    "tolerance_absolute": band,
                },
                explanation=explanation,
                spec=spec.to_dict(),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Check Type 5 -- Edge Case and Boundary Behaviour (FR-C-12 to FR-C-15)
# ---------------------------------------------------------------------------


def check_edge_cases(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    base_points = grid.sample(config.edge_cases, config.sample_seed)

    for base in base_points:
        results.extend(_edge_zero_volatility(mut, base, tolerance))
        results.extend(_edge_zero_maturity(mut, base, tolerance))
        results.extend(_edge_high_volatility(mut, base, config))
        results.extend(_edge_zero_spot(mut, base, tolerance))
    return results


def _edge_case(
    check_case: str,
    spec: OptionSpec,
    mut: ModelUnderTest,
    expected: float,
    tolerance: Tolerance,
    explanation_builder: Callable[[float, float], str],
) -> list[CheckResult]:
    result = _mut_result(mut, spec)
    bad = _failed_result(5, spec, result)
    if bad is not None:
        bad.case = f"{check_case} | {spec.label()}"
        return [bad]

    ok = tolerance.within(result.price, expected)
    return [
        CheckResult(
            check_type=5,
            case=f"{check_case} | {spec.label()}",
            status=PASS if ok else FAIL,
            evidence={
                "model_price": result.price,
                "expected_price": expected,
                "absolute_difference": result.price - expected,
            },
            explanation="" if ok else explanation_builder(result.price, expected),
            spec=spec.to_dict(),
        )
    ]


def _edge_zero_volatility(
    mut: ModelUnderTest, base: OptionSpec, tolerance: Tolerance
) -> list[CheckResult]:
    """FR-C-12: at zero volatility the payoff is known with certainty."""
    spec = base.with_(sigma=0.0)
    expected = degenerate_value(spec)
    return _edge_case(
        "zero volatility",
        spec,
        mut,
        expected,
        tolerance,
        lambda got, want: (
            f"With volatility set to zero the future price of the asset is certain: "
            f"it grows at the cost of carry to {spec.forward:.6f}, so the option is "
            f"worth its discounted deterministic payoff, {want:.6f}. The model returned "
            f"{got:.6f}. A model that keeps a positive time value at zero volatility is "
            f"adding value that has no source of uncertainty behind it, which usually "
            f"means a hard-coded volatility floor or a division that never reaches "
            f"its limit."
        ),
    )


def _edge_zero_maturity(
    mut: ModelUnderTest, base: OptionSpec, tolerance: Tolerance
) -> list[CheckResult]:
    """FR-C-13: at zero time to expiry the option is worth its intrinsic value."""
    spec = base.with_(T=0.0)
    expected = degenerate_value(spec)
    return _edge_case(
        "zero time to expiry",
        spec,
        mut,
        expected,
        tolerance,
        lambda got, want: (
            f"At expiry an option is worth exactly its payoff, which is {want:.6f} "
            f"here. The model returned {got:.6f}. A non-zero difference at expiry means "
            f"the model still applies time value, discounting, or a volatility term "
            f"when there is no time left for any of them to act on."
        ),
    )


def _edge_high_volatility(
    mut: ModelUnderTest, base: OptionSpec, config: AuditConfig
) -> list[CheckResult]:
    """FR-C-14: a very high volatility must not produce a negative price or a fault."""
    spec = base.with_(sigma=config.high_volatility)
    result = _mut_result(mut, spec)
    bad = _failed_result(5, spec, result)
    if bad is not None:
        bad.case = f"high volatility | {spec.label()}"
        bad.explanation = (
            f"At a volatility of {config.high_volatility:.1f} the model did not return "
            f"a usable price. Extreme volatility is where an unstable numerical scheme "
            f"overflows or where a variance term goes negative. A pricing function must "
            f"stay finite across the whole valid domain, not only near the parameters "
            f"it was tested on. Reported problem: {result.message}"
        )
        return [bad]

    upper = _no_arbitrage_upper(spec)
    ok = result.price >= 0.0 and math.isfinite(result.price)
    bound_ok = upper is None or result.price <= upper * (1.0 + 1e-6)
    return [
        CheckResult(
            check_type=5,
            case=f"high volatility | {spec.label()}",
            status=PASS if (ok and bound_ok) else FAIL,
            evidence={
                "model_price": result.price,
                "volatility": config.high_volatility,
                "upper_bound": upper,
            },
            explanation=""
            if (ok and bound_ok)
            else (
                f"At a volatility of {config.high_volatility:.1f} the model returned "
                f"{result.price:.6f}."
                + (
                    " A price cannot be negative: the holder of an option is never "
                    "forced to exercise it."
                    if not ok
                    else f" That is above the upper no-arbitrage bound of {upper:.6f}, "
                    f"so the option would cost more than the payoff it can ever deliver."
                )
            ),
            spec=spec.to_dict(),
        )
    ]


def _edge_zero_spot(
    mut: ModelUnderTest, base: OptionSpec, tolerance: Tolerance
) -> list[CheckResult]:
    """FR-C-15: a spot of zero is absorbing, so the limit value is known."""
    spec = base.with_(S=0.0)
    expected = degenerate_value(spec)
    return _edge_case(
        "zero spot",
        spec,
        mut,
        expected,
        tolerance,
        lambda got, want: (
            f"Zero is an absorbing state for a geometric Brownian motion: once the "
            f"asset is worth nothing it stays worth nothing. The correct value here is "
            f"{want:.6f}. The model returned {got:.6f}. A model that prices a call above "
            f"zero at a zero spot is taking the logarithm of zero somewhere and "
            f"recovering from it with the wrong sign or a default value."
        ),
    )


# ---------------------------------------------------------------------------
# Check Type 6 -- Arbitrage-Free Sanity (FR-C-16 to FR-C-18)
# ---------------------------------------------------------------------------


def check_arbitrage(
    mut: ModelUnderTest,
    grid: ParameterGrid,
    tolerance: Tolerance,
    config: AuditConfig,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for base in grid.sample(config.arbitrage_cases, config.sample_seed):
        results.extend(_strike_monotonicity(mut, base, config, tolerance))
        results.extend(_no_arbitrage_bounds(mut, base, tolerance))
    results.extend(_negative_price_scan(mut, grid, config))
    return results


def _strike_monotonicity(
    mut: ModelUnderTest, base: OptionSpec, config: AuditConfig, tolerance: Tolerance
) -> list[CheckResult]:
    """FR-C-16: a call is non-increasing in the strike, a put non-decreasing."""
    if base.instrument not in ("european", "american"):
        return []
    strikes = sorted(config.strike_ladder)
    prices: list[float] = []
    for strike in strikes:
        result = _mut_result(mut, base.with_(K=strike))
        bad = _failed_result(6, base.with_(K=strike), result)
        if bad is not None:
            bad.case = f"strike monotonicity | {base.label()}"
            return [bad]
        prices.append(result.price)

    slack = max(tolerance.absolute, 1e-8)
    if base.option_type == "call":
        breaks = [
            (strikes[i], strikes[i + 1], prices[i], prices[i + 1])
            for i in range(len(strikes) - 1)
            if prices[i + 1] > prices[i] + slack
        ]
        direction, hedge = "fall", "a call spread"
    else:
        breaks = [
            (strikes[i], strikes[i + 1], prices[i], prices[i + 1])
            for i in range(len(strikes) - 1)
            if prices[i + 1] < prices[i] - slack
        ]
        direction, hedge = "rise", "a put spread"

    ok = not breaks
    return [
        CheckResult(
            check_type=6,
            case=f"strike monotonicity | {base.label()}",
            status=PASS if ok else FAIL,
            evidence={"strikes": strikes, "prices": prices, "violations": breaks},
            explanation=""
            if ok
            else (
                f"The price of this {base.option_type} must {direction} as the strike "
                f"rises, but the model breaks that at "
                + "; ".join(
                    f"K={a:g} priced {pa:.6f} against K={b:g} priced {pb:.6f}"
                    for a, b, pa, pb in breaks
                )
                + f". Buying {hedge} across those two strikes would cost a negative "
                f"amount and can never pay out a negative amount, which is a risk-free "
                f"profit."
            ),
            spec=base.to_dict(),
        )
    ]


def _no_arbitrage_upper(spec: OptionSpec) -> float | None:
    """Largest value the contract can have without admitting an arbitrage."""
    if spec.instrument == "european":
        return (
            spec.S * math.exp(-spec.q * spec.T)
            if spec.option_type == "call"
            else spec.K * math.exp(-spec.r * spec.T)
        )
    if spec.instrument == "american":
        return spec.S if spec.option_type == "call" else spec.K
    if spec.instrument == "digital":
        return (
            spec.cash_amount * math.exp(-spec.r * spec.T)
            if spec.payout == "cash"
            else spec.S * math.exp(-spec.q * spec.T)
        )
    return None


def _no_arbitrage_lower(spec: OptionSpec) -> float | None:
    if spec.instrument == "european":
        forward_value = spec.S * math.exp(-spec.q * spec.T) - spec.K * math.exp(
            -spec.r * spec.T
        )
        return max(spec.phi * forward_value, 0.0)
    if spec.instrument == "american":
        return max(spec.phi * (spec.S - spec.K), 0.0)
    if spec.instrument == "digital":
        return 0.0
    return None


def _no_arbitrage_bounds(
    mut: ModelUnderTest, base: OptionSpec, tolerance: Tolerance
) -> list[CheckResult]:
    """FR-C-17: the price must sit inside its static no-arbitrage bounds."""
    lower, upper = _no_arbitrage_lower(base), _no_arbitrage_upper(base)
    if lower is None or upper is None:
        return [
            CheckResult(
                6,
                f"no-arbitrage bounds | {base.label()}",
                SKIP,
                {"reason": f"no static bound is defined for {base.instrument!r}"},
                spec=base.to_dict(),
            )
        ]

    result = _mut_result(mut, base)
    bad = _failed_result(6, base, result)
    if bad is not None:
        bad.case = f"no-arbitrage bounds | {base.label()}"
        return [bad]

    slack = max(tolerance.absolute, tolerance.relative * max(abs(upper), 1.0))
    ok = lower - slack <= result.price <= upper + slack
    return [
        CheckResult(
            6,
            f"no-arbitrage bounds | {base.label()}",
            PASS if ok else FAIL,
            {
                "model_price": result.price,
                "lower_bound": lower,
                "upper_bound": upper,
                "intrinsic_value": base.intrinsic(),
                "spot": base.S,
            },
            explanation=""
            if ok
            else (
                f"The model priced this contract at {result.price:.6f}, outside the "
                f"range [{lower:.6f}, {upper:.6f}] that holds without any model at all. "
                + (
                    "Below the lower bound, buying the option and selling the "
                    "replicating forward locks in a profit."
                    if result.price < lower
                    else "Above the upper bound, the option costs more than the asset "
                    "it can ever deliver."
                )
            ),
            spec=base.to_dict(),
        )
    ]


def _negative_price_scan(
    mut: ModelUnderTest, grid: ParameterGrid, config: AuditConfig
) -> list[CheckResult]:
    """FR-C-18: no tested input may produce a negative price."""
    offenders: list[dict[str, Any]] = []
    tested = 0
    for spec in grid:
        result = _mut_result(mut, spec)
        if result.status != "OK":
            continue
        tested += 1
        if result.price < 0.0:
            offenders.append({"case": spec.label(), "price": result.price})

    ok = not offenders
    return [
        CheckResult(
            check_type=6,
            case=f"negative price scan over {tested} parameter sets",
            status=PASS if ok else FAIL,
            evidence={"tested": tested, "negative_prices": offenders[:20]},
            explanation=""
            if ok
            else (
                f"The model returned a negative price on {len(offenders)} of {tested} "
                f"parameter sets, the first being {offenders[0]['case']} at "
                f"{offenders[0]['price']:.6f}. An option is a right and not an "
                f"obligation, so its value can be zero but never less. A negative price "
                f"means a payoff was used without the max(., 0) that defines it, or a "
                f"numerical scheme produced an oscillation that went below zero."
            ),
        )
    ]


#: The six check types in the order the Audit Report lists them.
ALL_CHECKS: tuple[tuple[int, Callable[..., list[CheckResult]]], ...] = (
    (1, check_reference_price),
    (2, check_put_call_parity),
    (3, check_greeks),
    (4, check_convergence),
    (5, check_edge_cases),
    (6, check_arbitrage),
)

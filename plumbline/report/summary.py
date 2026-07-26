"""Plain-language summary of an audit (FR-D-05).

The numbers in section 3 of the Audit Report say *that* a model is wrong.  This
module says *what* is wrong, in words a reader without a quantitative finance
background can act on.  It works on the finished results, so it sees patterns
that no single check can: a constant relative error across the whole grid is a
different defect from an error that only appears at short maturities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from plumbline.audit.checks import CHECK_NAMES, CheckResult

#: A relative error whose spread is below this share of its mean is "constant".
CONSTANT_SPREAD = 0.05


@dataclass
class Finding:
    """One plain-language statement about the model."""

    title: str
    detail: str
    check_type: int
    severity: str = "high"  # high | medium | low

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "detail": self.detail,
            "check_type": self.check_type,
            "severity": self.severity,
        }


def diagnose(results: list[CheckResult]) -> list[Finding]:
    """Turn the failed checks into a short list of named defects."""
    findings: list[Finding] = []
    failures = [r for r in results if r.counts_against]
    if not failures:
        return findings

    findings.extend(_diagnose_price_pattern(failures))
    findings.extend(_diagnose_by_check_type(failures))
    return findings


def _diagnose_price_pattern(failures: list[CheckResult]) -> list[Finding]:
    """Look at the shape of the Check Type 1 errors, not only their size."""
    price_fails = [
        r
        for r in failures
        if r.check_type == 1 and "relative_difference" in r.evidence
    ]
    if len(price_fails) < 3:
        return []

    errors = [float(r.evidence["relative_difference"]) for r in price_fails]
    mean = sum(errors) / len(errors)
    spread = max(errors) - min(errors)

    if abs(mean) > 1e-9 and spread <= CONSTANT_SPREAD * abs(mean):
        return [
            Finding(
                title="The price is wrong by the same proportion everywhere",
                detail=(
                    f"On every parameter set that failed, the model's price is about "
                    f"{mean * 100:.2f} percent away from the reference. An error that "
                    f"does not change with the market parameters is a scale error, not "
                    f"a modelling error: a factor applied once too often, a discount "
                    f"factor left out, or a unit that is not the one the caller "
                    f"expects."
                ),
                check_type=1,
                severity="high",
            )
        ]

    by_maturity: dict[float, list[float]] = {}
    for result in price_fails:
        maturity = float(result.spec.get("T", math.nan))
        by_maturity.setdefault(maturity, []).append(
            abs(float(result.evidence["relative_difference"]))
        )
    if len(by_maturity) > 1:
        worst = max(by_maturity, key=lambda t: sum(by_maturity[t]) / len(by_maturity[t]))
        best = min(by_maturity, key=lambda t: sum(by_maturity[t]) / len(by_maturity[t]))
        worst_mean = sum(by_maturity[worst]) / len(by_maturity[worst])
        best_mean = sum(by_maturity[best]) / len(by_maturity[best])
        if worst_mean > 3.0 * max(best_mean, 1e-12):
            return [
                Finding(
                    title="The error depends strongly on the time to expiry",
                    detail=(
                        f"At {worst:g} years the average relative error is "
                        f"{worst_mean * 100:.2f} percent, but at {best:g} years it is "
                        f"only {best_mean * 100:.2f} percent. An error that grows or "
                        f"shrinks with maturity points at the time scaling of the "
                        f"volatility term. Diffusion over a period of length T scales "
                        f"with the square root of T, not with T."
                    ),
                    check_type=1,
                    severity="high",
                )
            ]
    return []


_CHECK_FINDINGS = {
    2: (
        "The model breaks put-call parity",
        "Put-call parity comes from a portfolio that needs no model to value: "
        "hold a call, sell a put, and the result is a forward. The model prices "
        "that forward wrongly, so its call and its put are not consistent with "
        "each other. Check the discounting: the strike is discounted at the "
        "risk-free rate and the spot at the dividend yield.",
    ),
    3: (
        "The price surface has the wrong shape",
        "The model's sensitivities, measured by bumping the model's own price "
        "function, disagree with the reference. A price can be near enough while "
        "its slope is not, and every hedge is built from the slope. A position "
        "hedged with these numbers would be the wrong size.",
    ),
    4: (
        "The model does not converge to the correct value",
        "Raising the model's precision did not bring it closer to the reference. "
        "This separates a noisy method from a biased one. Noise falls when the "
        "method is given more work; bias does not. A biased method is wrong at "
        "every setting, and no run time budget fixes it.",
    ),
    5: (
        "The model fails at the boundaries of its domain",
        "At zero volatility, at zero time to expiry, at a zero spot price, and at "
        "extreme volatility, the correct answer is known exactly and needs no "
        "model. A model that misses these is missing a guard clause, and the same "
        "missing guard usually shows up as instability just inside the boundary.",
    ),
    6: (
        "The model admits an arbitrage",
        "The model's prices break a relationship that holds for every "
        "arbitrage-free market, whatever the dynamics of the underlying. A trader "
        "could construct a portfolio from these prices that costs nothing and "
        "cannot lose. Any price set with this property is not usable.",
    ),
}


def _diagnose_by_check_type(failures: list[CheckResult]) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[int] = set()
    for result in failures:
        if result.check_type in seen or result.check_type not in _CHECK_FINDINGS:
            continue
        seen.add(result.check_type)
        title, detail = _CHECK_FINDINGS[result.check_type]
        count = sum(1 for r in failures if r.check_type == result.check_type)
        findings.append(
            Finding(
                title=title,
                detail=f"{detail} This check failed on {count} parameter sets.",
                check_type=result.check_type,
                severity="high",
            )
        )
    return findings


def headline(report: Any) -> str:
    """One sentence a non-expert reader can act on."""
    badge = report.score.badge
    score = report.score.total
    failures = len(report.failures)
    total = len(report.results)
    name = report.model.get("name", "the model")

    if badge == "PASS":
        return (
            f"{name} passed all {total} checks with an Audit Score of {score:.1f} "
            f"out of 100. No mathematical or numerical error was found."
        )
    if badge == "PARTIAL":
        return (
            f"{name} scored {score:.1f} out of 100. It failed {failures} of {total} "
            f"checks. Use the model only for the cases the report marks as passed, "
            f"and read section 4 before you use any price from it."
        )
    return (
        f"{name} scored {score:.1f} out of 100 and failed {failures} of {total} "
        f"checks. Do not use this model to value a position. Section 4 names each "
        f"error found."
    )


def check_type_summary(report: Any) -> list[dict[str, Any]]:
    """One row per check type, for the summary table of section 2."""
    rows = []
    for bucket in report.score.per_check:
        rows.append(
            {
                "check_type": bucket.check_type,
                "name": CHECK_NAMES[bucket.check_type],
                "passes": bucket.passes,
                "failures": bucket.failures,
                "errors": bucket.errors,
                "skipped": bucket.skipped,
                "pass_rate": bucket.pass_rate,
                "weight": bucket.weight,
                "ran": bucket.ran,
            }
        )
    return rows

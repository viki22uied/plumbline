"""The Audit Score (FR-D-03).

The formula is fixed, published here, and applied identically to every audit.

1. Each check type ``t`` that produced at least one countable result gets a
   pass rate ``p_t = passes / (passes + failures + errors + timeouts)``.
   Skipped and not-priced results are excluded: a check that could not run
   must neither reward nor punish the model.
2. Each check type carries the weight in :data:`CHECK_WEIGHTS`.
3. The Audit Score is ``100 * sum(w_t p_t) / sum(w_t)`` over the check types
   that ran.  A model audited on fewer check types is scored only on those,
   and the report states which ones were skipped.

The badge is deliberately harsher than the score, because an average hides a
single fatal defect:

``PASS``     every countable result passed
``FAIL``     the score is below :data:`FAIL_THRESHOLD`, or any check type
             failed every case it ran
``PARTIAL``  anything in between
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from plumbline.audit.checks import CHECK_NAMES, CheckResult

#: Weight of each check type in the Audit Score.
CHECK_WEIGHTS: dict[int, float] = {
    1: 0.30,  # reference price -- the model is either right or it is not
    2: 0.15,  # put-call parity -- model-free, so a failure is unambiguous
    3: 0.15,  # Greeks -- the shape of the price surface
    4: 0.10,  # convergence -- the method, not just the answer
    5: 0.15,  # edge cases -- where wrong models usually break first
    6: 0.15,  # arbitrage bounds -- the cheapest sanity a price must satisfy
}

FAIL_THRESHOLD = 70.0


@dataclass
class CheckTypeScore:
    check_type: int
    passes: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    weight: float = 0.0

    @property
    def name(self) -> str:
        return CHECK_NAMES[self.check_type]

    @property
    def countable(self) -> int:
        return self.passes + self.failures + self.errors

    @property
    def pass_rate(self) -> float:
        return self.passes / self.countable if self.countable else 0.0

    @property
    def ran(self) -> bool:
        return self.countable > 0

    def to_dict(self) -> dict:
        return {
            "check_type": self.check_type,
            "check_name": self.name,
            "passes": self.passes,
            "failures": self.failures,
            "errors": self.errors,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "weight": self.weight,
            "ran": self.ran,
        }


@dataclass
class Score:
    total: float
    badge: str
    per_check: list[CheckTypeScore] = field(default_factory=list)
    formula: str = (
        "score = 100 * sum(weight_t * pass_rate_t) / sum(weight_t), "
        "over the check types that produced at least one countable result"
    )

    def to_dict(self) -> dict:
        return {
            "audit_score": self.total,
            "badge": self.badge,
            "formula": self.formula,
            "weights": CHECK_WEIGHTS,
            "fail_threshold": FAIL_THRESHOLD,
            "per_check": [item.to_dict() for item in self.per_check],
        }


def score_results(results: Iterable[CheckResult]) -> Score:
    """Apply the published formula to a finished set of check results."""
    buckets = {
        check_type: CheckTypeScore(check_type, weight=CHECK_WEIGHTS[check_type])
        for check_type in CHECK_NAMES
    }

    for result in results:
        bucket = buckets[result.check_type]
        if result.status == "PASS":
            bucket.passes += 1
        elif result.status == "FAIL":
            bucket.failures += 1
        elif result.status in ("ERROR", "TIMEOUT"):
            bucket.errors += 1
        else:  # SKIP, NOT_PRICED
            bucket.skipped += 1

    ran = [bucket for bucket in buckets.values() if bucket.ran]
    if not ran:
        return Score(total=0.0, badge="FAIL", per_check=list(buckets.values()))

    weight_sum = sum(bucket.weight for bucket in ran)
    total = 100.0 * sum(bucket.weight * bucket.pass_rate for bucket in ran) / weight_sum

    if all(bucket.failures == 0 and bucket.errors == 0 for bucket in ran):
        badge = "PASS"
    elif total < FAIL_THRESHOLD or any(bucket.pass_rate == 0.0 for bucket in ran):
        badge = "FAIL"
    else:
        badge = "PARTIAL"

    return Score(total=total, badge=badge, per_check=list(buckets.values()))

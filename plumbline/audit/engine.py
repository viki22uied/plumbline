"""Module C orchestration -- run every check and assemble the Audit Report.

Each check type runs inside its own guard.  A check that raises costs its own
results and nothing else, which is the whole point of NFR-05: an audit of a
broken model must still produce a report about the parts that did run.
"""

from __future__ import annotations

import platform
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import plumbline
from plumbline.audit.checks import (
    ALL_CHECKS,
    CHECK_NAMES,
    ERROR,
    AuditConfig,
    CheckResult,
)
from plumbline.audit.grid import ParameterGrid, default_grid
from plumbline.audit.scoring import Score, score_results
from plumbline.contracts import Tolerance
from plumbline.engines.registry import REGISTRY
from plumbline.ingestion import ModelUnderTest


@dataclass
class AuditReport:
    """Everything section 9 of the PRD requires, in one serialisable object."""

    audit_id: str
    model: dict[str, Any]
    grid: dict[str, Any]
    tolerance: dict[str, float]
    config: dict[str, Any]
    results: list[CheckResult]
    score: Score
    started_at: str
    finished_at: str
    duration_s: float
    plumbline_version: str = plumbline.__version__
    engines: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    # -- views ---------------------------------------------------------------

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.counts_against]

    def by_check_type(self, check_type: int) -> list[CheckResult]:
        return [r for r in self.results if r.check_type == check_type]

    def convergence_series(self) -> list[dict[str, Any]]:
        """Data behind the convergence plots of report section 5."""
        series = []
        for result in self.by_check_type(4):
            evidence = result.evidence
            if "precision_levels" in evidence and evidence["precision_levels"]:
                series.append(
                    {
                        "case": result.case,
                        "status": result.status,
                        "levels": evidence["precision_levels"],
                        "prices": evidence["model_prices"],
                        "errors": evidence["absolute_errors"],
                        "reference": evidence["reference_price"],
                    }
                )
        return series

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "plumbline_version": self.plumbline_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "model": self.model,
            "score": self.score.to_dict(),
            "grid": self.grid,
            "tolerance": self.tolerance,
            "config": self.config,
            "engines": self.engines,
            "environment": self.environment,
            "check_names": CHECK_NAMES,
            "results": [r.to_dict() for r in self.results],
            "convergence": self.convergence_series(),
        }


def _environment() -> dict[str, str]:
    """What ran the audit, for section 7 of the report.

    The native backend is recorded whether or not it was built. A reader
    comparing two reports needs to know which one had it, because the two
    backends draw from different random streams and their Monte Carlo figures
    will differ inside the sampling error.
    """
    from plumbline.engines import montecarlo, native

    described = native.describe()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "monte_carlo_backend_default": montecarlo.DEFAULT_BACKEND,
        "native_backend": described["version"] or "not built",
        "native_backend_threads": str(described["hardware_threads"]),
    }


def run_audit(
    mut: ModelUnderTest,
    grid: ParameterGrid | None = None,
    tolerance: Tolerance | None = None,
    config: AuditConfig | None = None,
    check_types: Sequence[int] | None = None,
) -> AuditReport:
    """Run the Audit and return the report.

    ``check_types`` restricts the run, which is how NFR-03 is met: the five
    fast check types run without the Monte Carlo convergence sweep of Check
    Type 4, which has its own budget under NFR-04.
    """
    grid = grid or default_grid()
    tolerance = tolerance or Tolerance()
    config = config or AuditConfig()
    wanted = set(check_types) if check_types is not None else set(CHECK_NAMES)

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[CheckResult] = []

    for check_type, check_fn in ALL_CHECKS:
        if check_type not in wanted:
            continue
        try:
            results.extend(check_fn(mut, grid, tolerance, config))
        except Exception:
            results.append(
                CheckResult(
                    check_type=check_type,
                    case=f"{CHECK_NAMES[check_type]} did not complete",
                    status=ERROR,
                    evidence={"traceback": traceback.format_exc()},
                    explanation=(
                        "This check type stopped with an internal error, so its "
                        "results are missing from the score. The other check types "
                        "were not affected. The traceback is in the evidence."
                    ),
                )
            )

    duration = time.perf_counter() - started
    return AuditReport(
        audit_id=uuid.uuid4().hex[:16],
        model=mut.describe(),
        grid=grid.to_dict(),
        tolerance=tolerance.to_dict(),
        config={**config.to_dict(), "check_types": sorted(wanted)},
        results=results,
        score=score_results(results),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_s=duration,
        engines=[engine.to_dict() for engine in REGISTRY.values()],
        environment=_environment(),
    )


def audit_file(
    path: str,
    entry: str = "price",
    grid: ParameterGrid | None = None,
    tolerance: Tolerance | None = None,
    config: AuditConfig | None = None,
    check_types: Iterable[int] | None = None,
    call_timeout: float | None = None,
) -> AuditReport:
    """Load a Model Under Test from ``path``, audit it, and close it."""
    from plumbline.ingestion import load_model

    with load_model(path, entry=entry, call_timeout=call_timeout) as mut:
        return run_audit(
            mut,
            grid=grid,
            tolerance=tolerance,
            config=config,
            check_types=list(check_types) if check_types is not None else None,
        )

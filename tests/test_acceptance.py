"""Section 12 -- acceptance criteria, and the measured non-functional targets.

AC-01 and AC-02 are satisfied by the rest of the suite; the two tests here
assert that the coverage is complete rather than repeating it.  AC-03 to AC-06
are measured directly.
"""

from __future__ import annotations

import os
import platform
import time

import pytest

from plumbline.audit.checks import ALL_CHECKS, CHECK_NAMES, AuditConfig
from plumbline.audit.engine import audit_file, run_audit
from plumbline.audit.grid import ParameterGrid, default_grid
from plumbline.contracts import Tolerance
from plumbline.engines.registry import REGISTRY
from plumbline.ingestion import load_model
from plumbline.report import write_report

VANILLA_GRID = default_grid("european")


# ---------------------------------------------------------------------------
# AC-01 and AC-02
# ---------------------------------------------------------------------------


def test_ac01_every_check_type_is_implemented_and_wired_in():
    """AC-01: all six check types exist and the engine runs each of them."""
    assert {number for number, _ in ALL_CHECKS} == set(CHECK_NAMES) == {1, 2, 3, 4, 5, 6}
    for _, check_fn in ALL_CHECKS:
        assert callable(check_fn)


def test_ac02_every_ground_truth_engine_is_registered_with_its_reference():
    """AC-02: each engine names the published method it implements."""
    expected = {"analytic", "binomial_crr", "fdm_crank_nicolson", "heston_cf", "monte_carlo"}
    assert set(REGISTRY) == expected

    for engine in REGISTRY.values():
        assert engine.reference and len(engine.reference) > 10
        assert engine.version
        assert engine.instruments


# ---------------------------------------------------------------------------
# AC-04
# ---------------------------------------------------------------------------

#: The five seeded errors of ``samples/broken_model.py``, and the evidence that
#: proves each was found. See the module docstring of that file.
SEEDED_ERRORS = {
    "E1 volatility scales with T instead of sqrt(T)": (
        1,
        lambda results: any(
            r.status == "FAIL" and abs(r.spec["T"] - 0.25) < 1e-12 for r in results
        ),
    ),
    "E2 the put branch does not discount the strike": (
        2,
        lambda results: any(
            r.status == "FAIL" and abs(r.evidence.get("parity_gap", 0.0)) > 1e-6 for r in results
        ),
    ),
    "E3 zero time to expiry returns nothing": (
        5,
        lambda results: any(
            r.status == "FAIL" and r.case.startswith("zero time to expiry") for r in results
        ),
    ),
    "E4 zero volatility returns nothing": (
        5,
        lambda results: any(
            r.status == "FAIL" and r.case.startswith("zero volatility") for r in results
        ),
    ),
    "E5 a strike-proportional term makes the call rise with the strike": (
        6,
        lambda results: any(
            r.status == "FAIL" and r.case.startswith("strike monotonicity") for r in results
        ),
    ),
}


@pytest.fixture(scope="module")
def broken_audit(broken_model_path):
    return audit_file(broken_model_path, grid=VANILLA_GRID, check_types=[1, 2, 3, 5, 6])


@pytest.mark.parametrize("error_name", sorted(SEEDED_ERRORS))
def test_ac04_the_audit_flags_every_seeded_error(broken_audit, error_name):
    """AC-04: all five deliberate errors are found, each by its own check."""
    check_type, detector = SEEDED_ERRORS[error_name]
    results = broken_audit.by_check_type(check_type)

    assert detector(results), f"{error_name} was not flagged by check type {check_type}"


def test_ac04_the_broken_model_does_not_pass(broken_audit):
    assert broken_audit.score.badge == "FAIL"
    assert broken_audit.score.total < 70.0


def test_ac04_every_flagged_error_is_explained_in_plain_language(broken_audit):
    from plumbline.report.summary import diagnose

    assert all(result.explanation for result in broken_audit.failures)
    assert diagnose(broken_audit.results)


# ---------------------------------------------------------------------------
# AC-05
# ---------------------------------------------------------------------------


def test_ac05_a_correct_model_returns_a_full_pass(good_model_path):
    """AC-05: the reference implementation itself must pass every check."""
    report = audit_file(good_model_path, grid=VANILLA_GRID, check_types=[1, 2, 3, 4, 5, 6])

    assert report.score.badge == "PASS"
    assert report.score.total == pytest.approx(100.0)
    assert not report.failures
    ran = {bucket.check_type for bucket in report.score.per_check if bucket.ran}
    assert ran == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize(
    "instrument",
    ["european", "american", "asian", "barrier", "digital", "lookback"],
)
def test_ac05_the_correct_model_passes_on_every_instrument(good_model_path, instrument):
    """The reference model must pass for each covered instrument, not only one."""
    grid = default_grid(instrument)
    grid.spots = grid.spots[:1]
    grid.strikes = grid.strikes[:1]
    grid.maturities = grid.maturities[:1]
    grid.vols = grid.vols[:1]

    report = audit_file(good_model_path, grid=grid, check_types=[1, 5, 6])

    assert report.score.badge == "PASS", [
        (r.case, r.status, r.explanation) for r in report.failures
    ]


# ---------------------------------------------------------------------------
# AC-06
# ---------------------------------------------------------------------------


def test_ac06_cli_api_and_all_three_report_formats_work_end_to_end(good_model_path, tmp_path):
    """AC-06: one pass through every delivery surface Plumbline offers."""
    from fastapi.testclient import TestClient

    from plumbline.api import app
    from plumbline.cli import main

    grid = ParameterGrid(spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,))

    # 1. the library
    report = audit_file(good_model_path, grid=grid, check_types=[1, 2])
    assert report.score.badge == "PASS"

    # 2. all three report formats
    paths = write_report(report, str(tmp_path / "reports"))
    assert os.path.isfile(paths.json)
    assert os.path.isfile(paths.markdown)
    assert paths.pdf and os.path.isfile(paths.pdf)

    # 3. the command line interface
    assert main(
        [
            "audit",
            good_model_path,
            "--out",
            str(tmp_path / "cli"),
            "--checks",
            "1",
            "--spots",
            "100",
            "--strikes",
            "100",
            "--maturities",
            "1.0",
            "--vols",
            "0.2",
            "--no-history",
        ]
    ) == 0

    # 4. the REST API
    client = TestClient(app)
    assert client.get("/engines").status_code == 200
    with open(good_model_path, "rb") as handle:
        response = client.post(
            "/audit",
            files={"file": ("good_model.py", handle, "text/x-python")},
            params={"checks": "1"},
        )
    assert response.status_code == 200 and response.json()["badge"] == "PASS"


# ---------------------------------------------------------------------------
# AC-03: the non-functional targets, measured
# ---------------------------------------------------------------------------


def test_nfr03_a_vanilla_audit_finishes_in_under_five_seconds(good_model_path):
    """NFR-03: check types 1, 2, 3, 5 and 6 on a vanilla option, under 5 s."""
    started = time.perf_counter()
    report = audit_file(good_model_path, grid=VANILLA_GRID, check_types=[1, 2, 3, 5, 6])
    elapsed = time.perf_counter() - started

    assert report.score.badge == "PASS"
    assert elapsed < 5.0, f"took {elapsed:.2f} s on {platform.processor() or 'this machine'}"


@pytest.mark.slow
def test_nfr04_a_convergence_audit_finishes_in_under_sixty_seconds(mc_model_path):
    """NFR-04: the Monte Carlo convergence sweep, under 60 s."""
    config = AuditConfig(convergence_cases=4)
    started = time.perf_counter()
    report = audit_file(mc_model_path, grid=VANILLA_GRID, config=config, check_types=[4])
    elapsed = time.perf_counter() - started

    assert report.by_check_type(4)
    assert elapsed < 60.0, f"took {elapsed:.2f} s"


def test_nfr05_a_fault_in_one_check_does_not_stop_the_others(good_model_path, monkeypatch):
    """NFR-05: a check type that raises costs its own results and no more."""
    import plumbline.audit.engine as engine_module

    def exploding_check(mut, grid, tolerance, config):
        raise RuntimeError("this check type is broken")

    patched = tuple(
        (number, exploding_check if number == 2 else fn) for number, fn in ALL_CHECKS
    )
    monkeypatch.setattr(engine_module, "ALL_CHECKS", patched)

    grid = ParameterGrid(spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,))
    with load_model(good_model_path) as mut:
        report = run_audit(mut, grid=grid, check_types=[1, 2, 5])

    check_two = report.by_check_type(2)
    assert len(check_two) == 1 and check_two[0].status == "ERROR"
    assert "this check type is broken" in check_two[0].evidence["traceback"]
    assert all(r.status == "PASS" for r in report.by_check_type(1))
    assert report.by_check_type(5)


def test_nfr05_a_model_that_faults_on_some_inputs_is_still_fully_audited(tmp_path):
    """The audit must report every case, not stop at the first fault."""
    source = (
        "def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):\n"
        "    if S > 100.0:\n"
        "        raise ValueError('this model cannot price above par')\n"
        "    return max(S - K, 0.0)\n"
    )
    path = os.path.join(str(tmp_path), "partial.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)

    grid = ParameterGrid(
        spots=(90.0, 110.0), strikes=(100.0,), maturities=(1.0,), vols=(0.2,), option_types=("call",)
    )
    report = audit_file(path, grid=grid, check_types=[1])
    statuses = {result.spec["S"]: result.status for result in report.by_check_type(1)}

    assert statuses[110.0] == "ERROR"
    assert statuses[90.0] in ("PASS", "FAIL")


def test_nfr10_the_report_records_the_platform_it_ran_on(good_model_path):
    """NFR-10: the same code runs on all three platforms; the report says which."""
    grid = ParameterGrid(spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,))
    report = audit_file(good_model_path, grid=grid, check_types=[1])

    assert report.environment["python"].startswith("3.")
    assert report.environment["platform"]


def test_a_user_set_tolerance_reaches_every_check(good_model_path):
    """The Tolerance object the user passes is the one the checks apply."""
    grid = ParameterGrid(spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,))
    tolerance = Tolerance(relative=1e-9, greek_relative=1e-9)
    report = audit_file(good_model_path, grid=grid, tolerance=tolerance, check_types=[1])

    assert report.tolerance["relative"] == 1e-9
    assert report.by_check_type(1)[0].evidence["tolerance_relative"] == 1e-9

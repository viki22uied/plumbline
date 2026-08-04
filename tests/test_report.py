"""Module D -- report generation and history (FR-D-01 to FR-D-06)."""

from __future__ import annotations

import json
import os

import pytest

from plumbline.audit.checks import AuditConfig
from plumbline.audit.engine import audit_file
from plumbline.audit.grid import ParameterGrid
from plumbline.audit.history import AuditHistory
from plumbline.audit.scoring import CHECK_WEIGHTS, score_results
from plumbline.report import render_json, render_markdown, write_report
from plumbline.report.summary import diagnose, headline

SMALL_GRID = ParameterGrid(
    spots=(95.0, 105.0), strikes=(100.0,), maturities=(0.5,), vols=(0.2,), rates=(0.03,)
)
FAST_CHECKS = [1, 2, 3, 5, 6]


@pytest.fixture(scope="module")
def good_report(good_model_path):
    return audit_file(good_model_path, grid=SMALL_GRID, check_types=FAST_CHECKS)


@pytest.fixture(scope="module")
def broken_report(broken_model_path):
    return audit_file(broken_model_path, grid=SMALL_GRID, check_types=FAST_CHECKS)


# ---------------------------------------------------------------------------
# FR-D-01, FR-D-02, FR-D-03
# ---------------------------------------------------------------------------


def test_frd01_one_report_is_produced_for_one_run(good_report):
    assert good_report.audit_id
    assert good_report.started_at and good_report.finished_at
    assert good_report.duration_s > 0.0


def test_frd02_every_result_carries_a_status_and_numeric_evidence(good_report):
    assert good_report.results
    for result in good_report.results:
        assert result.status in ("PASS", "FAIL", "ERROR", "TIMEOUT", "SKIP", "NOT_PRICED")
        assert isinstance(result.evidence, dict)
        assert result.check_name


def test_frd03_the_audit_score_follows_the_documented_formula(broken_report):
    """The score must be reproducible by hand from the published weights."""
    score = broken_report.score
    ran = [bucket for bucket in score.per_check if bucket.ran]
    weight_sum = sum(bucket.weight for bucket in ran)
    expected = 100.0 * sum(b.weight * b.pass_rate for b in ran) / weight_sum

    assert score.total == pytest.approx(expected)
    assert score.formula
    assert set(CHECK_WEIGHTS) == {1, 2, 3, 4, 5, 6}
    assert sum(CHECK_WEIGHTS.values()) == pytest.approx(1.0)


def test_frd03_the_badge_is_pass_only_when_nothing_failed(good_report, broken_report):
    assert good_report.score.badge == "PASS"
    assert good_report.score.total == pytest.approx(100.0)
    assert broken_report.score.badge == "FAIL"


def test_skipped_results_neither_reward_nor_punish_the_model():
    from plumbline.audit.checks import CheckResult

    results = [
        CheckResult(1, "a", "PASS"),
        CheckResult(1, "b", "SKIP"),
        CheckResult(1, "c", "NOT_PRICED"),
    ]
    score = score_results(results)

    assert score.total == pytest.approx(100.0)
    assert score.per_check[0].skipped == 2


def test_a_run_with_no_countable_result_scores_zero_and_fails():
    from plumbline.audit.checks import CheckResult

    score = score_results([CheckResult(1, "a", "SKIP")])

    assert score.total == 0.0
    assert score.badge == "FAIL"


# ---------------------------------------------------------------------------
# FR-D-04: three formats
# ---------------------------------------------------------------------------


def test_frd04_all_three_formats_are_written(good_report, tmp_path):
    paths = write_report(good_report, str(tmp_path))

    assert os.path.isfile(paths.json)
    assert os.path.isfile(paths.markdown)
    assert paths.pdf and os.path.isfile(paths.pdf)
    assert os.path.getsize(paths.pdf) > 2_000


def test_frd04_the_json_report_is_machine_readable_and_complete(good_report):
    payload = json.loads(render_json(good_report))

    assert payload["audit_id"] == good_report.audit_id
    assert payload["score"]["badge"] == "PASS"
    assert len(payload["results"]) == len(good_report.results)
    assert payload["grid"]["size"] == len(SMALL_GRID)
    assert payload["engines"] and payload["environment"]


def test_frd04_the_markdown_report_has_all_seven_sections(good_report):
    text = render_markdown(good_report)

    for number, title in (
        (1, "Model identification"),
        (2, "Audit Score and summary badge"),
        (3, "Full results"),
        (4, "Plain-language explanation"),
        (5, "Convergence plots"),
        (6, "Parameter grid used"),
        (7, "Versions"),
    ):
        assert f"## Section {number} -- {title}" in text


def test_frd04_the_markdown_report_names_the_engine_versions(good_report):
    text = render_markdown(good_report)

    assert "analytic" in text and "1.0.0" in text
    assert "Plumbline version" in text


# ---------------------------------------------------------------------------
# FR-D-05: plain language
# ---------------------------------------------------------------------------


def test_frd05_every_failed_check_carries_a_plain_language_explanation(broken_report):
    failures = broken_report.failures

    assert failures
    for result in failures:
        assert result.explanation, result.case
        assert len(result.explanation) > 60


def test_frd05_the_summary_names_the_defect_not_only_the_numbers(broken_report):
    findings = diagnose(broken_report.results)

    assert findings
    titles = " ".join(finding.title for finding in findings).lower()
    assert "parity" in titles or "arbitrage" in titles or "wrong" in titles
    for finding in findings:
        assert len(finding.detail) > 80


def test_frd05_the_headline_tells_a_reader_what_to_do(good_report, broken_report):
    assert "passed all" in headline(good_report)
    assert "Do not use this model" in headline(broken_report)


def test_frd05_a_constant_relative_error_is_named_as_a_scale_error(tmp_path, good_model_path):
    """The diagnosis must separate a scale error from a modelling error."""
    source = (
        "from plumbline.contracts import OptionSpec\n"
        "from plumbline.engines.registry import ground_truth_price\n"
        "def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):\n"
        "    kwargs.pop('instrument', None); kwargs.pop('option_type', None)\n"
        "    spec = OptionSpec(instrument=instrument, option_type=option_type,"
        " S=S, K=K, T=T, r=r, q=q, sigma=sigma, **kwargs)\n"
        "    return 1.05 * ground_truth_price(spec).price\n"
    )
    path = os.path.join(str(tmp_path), "scaled.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)

    report = audit_file(path, grid=SMALL_GRID, check_types=[1])
    findings = diagnose(report.results)

    assert any("same proportion" in finding.title for finding in findings)


# ---------------------------------------------------------------------------
# FR-D-06: history
# ---------------------------------------------------------------------------


def test_frd06_reports_are_stored_and_can_be_compared_over_time(
    good_report, broken_report, tmp_path
):
    history = AuditHistory(str(tmp_path / "store"))
    history.save(good_report)
    history.save(broken_report)

    entries = history.entries()
    assert len(entries) == 2
    assert {entry.badge for entry in entries} == {"PASS", "FAIL"}

    one_model = history.for_model(good_report.model["name"])
    assert len(one_model) == 1
    assert one_model[0].audit_id == good_report.audit_id


def test_frd06_a_stored_report_round_trips(good_report, tmp_path):
    history = AuditHistory(str(tmp_path / "store"))
    history.save(good_report)

    loaded = history.load(good_report.audit_id)

    assert loaded["audit_id"] == good_report.audit_id
    assert len(loaded["results"]) == len(good_report.results)


def test_frd06_an_unknown_audit_id_is_reported_clearly(tmp_path):
    history = AuditHistory(str(tmp_path / "store"))
    with pytest.raises(KeyError):
        history.load("deadbeef")


# ---------------------------------------------------------------------------
# convergence plots (report section 5)
# ---------------------------------------------------------------------------


def test_convergence_plots_are_written_when_check_four_ran(mc_model_path, tmp_path):
    config = AuditConfig(precision_levels=(2_000, 20_000), convergence_cases=1)
    report = audit_file(mc_model_path, grid=SMALL_GRID, config=config, check_types=[4])

    paths = write_report(report, str(tmp_path))

    assert report.convergence_series()
    assert paths.plots and os.path.isfile(paths.plots[0])
    assert "![convergence plot]" in open(paths.markdown, encoding="utf-8").read()


def test_a_report_without_check_four_says_there_is_no_plot(good_report):
    text = render_markdown(good_report)
    assert "no convergence series" in text

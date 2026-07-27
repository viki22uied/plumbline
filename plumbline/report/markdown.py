"""Markdown rendering of the Audit Report (FR-D-04).

The seven sections are exactly the ones fixed by section 9 of the PRD, in the
same order, with the same content, as the JSON and the PDF.  A reader must be
able to move between the three formats without looking for anything.
"""

from __future__ import annotations

import math
import os
from typing import Any

from plumbline.audit.checks import CHECK_NAMES
from plumbline.report.summary import check_type_summary, diagnose, headline

#: Rows of the full results table before it is truncated in Markdown.
MAX_TABLE_ROWS = 400


def render_markdown(report: Any, plot_paths: list[str] | None = None) -> str:
    """Render the whole Audit Report as one Markdown document."""
    lines: list[str] = []
    add = lines.append

    add(f"# Plumbline Audit Report")
    add("")
    add(f"Audit ID `{report.audit_id}` -- {report.finished_at}")
    add("")
    add(f"> {headline(report)}")
    add("")

    _section_1_model(report, add)
    _section_2_score(report, add)
    _section_3_results(report, add)
    _section_4_explanations(report, add)
    _section_5_convergence(report, add, plot_paths or [])
    _section_6_grid(report, add)
    _section_7_versions(report, add)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def _section_1_model(report: Any, add) -> None:
    add("## Section 1 -- Model identification")
    add("")
    model = report.model
    rows = [
        ("Name", model.get("name", "unknown")),
        ("Kind", model.get("kind", "unknown")),
        ("Input signature", f"`{model.get('signature', 'not reported')}`"),
        ("Submitted at", report.started_at),
        ("Precision knob", "yes" if model.get("supports_precision") else "no"),
    ]
    for key in ("path", "entry", "call_timeout_s", "timeouts", "sandbox_restarts", "rows"):
        if key in model:
            rows.append((key.replace("_", " ").capitalize(), model[key]))
    add("| Field | Value |")
    add("| --- | --- |")
    for key, value in rows:
        add(f"| {key} | {value} |")
    add("")


def _section_2_score(report: Any, add) -> None:
    badge = report.score.badge
    add("## Section 2 -- Audit Score and summary badge")
    add("")
    add(f"**Badge: {badge}**  |  **Audit Score: {report.score.total:.2f} / 100**")
    add("")
    add(f"Formula: {report.score.formula}")
    add("")
    add("| # | Check type | Pass | Fail | Error | Skip | Pass rate | Weight |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in check_type_summary(report):
        rate = f"{row['pass_rate'] * 100:.1f}%" if row["ran"] else "not run"
        add(
            f"| {row['check_type']} | {row['name']} | {row['passes']} | "
            f"{row['failures']} | {row['errors']} | {row['skipped']} | {rate} | "
            f"{row['weight']:.2f} |"
        )
    add("")


def _section_3_results(report: Any, add) -> None:
    add("## Section 3 -- Full results")
    add("")
    for check_type in sorted(CHECK_NAMES):
        results = report.by_check_type(check_type)
        if not results:
            continue
        add(f"### Check Type {check_type} -- {CHECK_NAMES[check_type]}")
        add("")
        add("| Status | Case | Numeric evidence |")
        add("| --- | --- | --- |")
        for result in results[:MAX_TABLE_ROWS]:
            add(
                f"| {result.status} | {_escape(result.case)} | "
                f"{_escape(_evidence_text(result.evidence))} |"
            )
        if len(results) > MAX_TABLE_ROWS:
            add(
                f"| ... | {len(results) - MAX_TABLE_ROWS} further rows are in the "
                f"JSON report | |"
            )
        add("")


def _section_4_explanations(report: Any, add) -> None:
    add("## Section 4 -- Plain-language explanation of each failed check")
    add("")
    failures = report.failures
    if not failures:
        add("No check failed. There is nothing to explain in this section.")
        add("")
        return

    findings = diagnose(report.results)
    if findings:
        add("### What is wrong with this model")
        add("")
        for finding in findings:
            add(f"**{finding.title}**")
            add("")
            add(finding.detail)
            add("")

    add("### Every failed case")
    add("")
    for result in failures:
        add(f"- **[Check Type {result.check_type}] {_escape(result.case)}** -- {result.status}")
        if result.explanation:
            add(f"  - {result.explanation}")
        add(f"  - Evidence: `{_evidence_text(result.evidence)}`")
    add("")


def _section_5_convergence(report: Any, add, plot_paths: list[str]) -> None:
    add("## Section 5 -- Convergence plots")
    add("")
    series = report.convergence_series()
    if not series:
        add("Check Type 4 produced no convergence series, so there is no plot.")
        add("")
        return

    for index, item in enumerate(series):
        add(f"### {item['case']} -- {item['status']}")
        add("")
        add(f"Reference value: {item['reference']:.8f}")
        add("")
        add("| Precision | Model price | Absolute error |")
        add("| --- | --- | --- |")
        for level, price, error in zip(item["levels"], item["prices"], item["errors"]):
            add(f"| {level} | {price:.8f} | {error:.3e} |")
        add("")
        if index < len(plot_paths):
            add(f"![convergence plot]({os.path.basename(plot_paths[index])})")
            add("")


def _section_6_grid(report: Any, add) -> None:
    add("## Section 6 -- Parameter grid used, for reproducibility")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    for key, value in report.grid.items():
        add(f"| {key} | {_escape(str(value))} |")
    add("")
    add("### Tolerance")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    for key, value in report.tolerance.items():
        add(f"| {key} | {value} |")
    add("")
    add("### Audit configuration")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    for key, value in report.config.items():
        add(f"| {key} | {_escape(str(value))} |")
    add("")


def _section_7_versions(report: Any, add) -> None:
    add("## Section 7 -- Versions")
    add("")
    add(f"Plumbline version: **{report.plumbline_version}**")
    add("")
    add("| Ground Truth Engine | Version | Instruments | Models | Reference |")
    add("| --- | --- | --- | --- | --- |")
    for engine in report.engines:
        add(
            f"| {engine['name']} | {engine['version']} | "
            f"{', '.join(engine['instruments'])} | {', '.join(engine['models'])} | "
            f"{_escape(engine['reference'])} |"
        )
    add("")
    add("| Environment | Value |")
    add("| --- | --- |")
    for key, value in report.environment.items():
        add(f"| {key} | {value} |")
    add("")
    add(f"Audit duration: {report.duration_s:.2f} seconds.")
    add("")


# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _evidence_text(evidence: dict[str, Any]) -> str:
    parts = []
    for key, value in evidence.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}" if math.isfinite(value) else f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            shown = ", ".join(
                f"{v:.6g}" if isinstance(v, float) else str(v) for v in list(value)[:6]
            )
            more = "" if len(value) <= 6 else f", +{len(value) - 6} more"
            parts.append(f"{key}=[{shown}{more}]")
        elif isinstance(value, str) and len(value) > 160:
            parts.append(f"{key}={value[:157]}...")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "-"

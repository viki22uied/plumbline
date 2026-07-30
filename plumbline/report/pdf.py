"""PDF rendering of the Audit Report (FR-D-04).

Built with ReportLab's platypus flowables, from the same report object the
Markdown and JSON writers use, so the three formats cannot drift apart.  The
seven sections of PRD section 9 map one to one onto the seven builders below.
"""

from __future__ import annotations

import math
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from plumbline.audit.checks import CHECK_NAMES
from plumbline.report.summary import check_type_summary, diagnose, headline

BADGE_COLOURS = {
    "PASS": colors.HexColor("#1b7f3b"),
    "PARTIAL": colors.HexColor("#b8860b"),
    "FAIL": colors.HexColor("#a11f1f"),
}
STATUS_COLOURS = {
    "PASS": colors.HexColor("#1b7f3b"),
    "FAIL": colors.HexColor("#a11f1f"),
    "ERROR": colors.HexColor("#a11f1f"),
    "TIMEOUT": colors.HexColor("#a11f1f"),
    "SKIP": colors.HexColor("#666666"),
    "NOT_PRICED": colors.HexColor("#666666"),
}
#: Result rows printed per check type. The JSON report always holds them all.
MAX_PDF_ROWS = 60


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PlumblineTitle", parent=base["Title"], fontSize=20, spaceAfter=4
        ),
        "h2": ParagraphStyle(
            "PlumblineH2", parent=base["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4
        ),
        "h3": ParagraphStyle(
            "PlumblineH3", parent=base["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=3
        ),
        "body": ParagraphStyle(
            "PlumblineBody",
            parent=base["BodyText"],
            fontSize=8.8,
            leading=12,
            alignment=TA_LEFT,
        ),
        "cell": ParagraphStyle(
            "PlumblineCell", parent=base["BodyText"], fontSize=7, leading=8.6
        ),
        "mono": ParagraphStyle(
            "PlumblineMono", parent=base["BodyText"], fontSize=6.8, leading=8.2, fontName="Courier"
        ),
    }


def render_pdf(report: Any, path: str, plot_paths: list[str] | None = None) -> str:
    """Write the Audit Report to ``path`` and return the path."""
    style = _styles()
    document = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Plumbline Audit Report {report.audit_id}",
        author="Plumbline",
    )

    story: list[Any] = []
    story.append(Paragraph("Plumbline Audit Report", style["title"]))
    story.append(
        Paragraph(
            f"Audit ID {report.audit_id} &mdash; {report.finished_at}", style["body"]
        )
    )
    story.append(Spacer(1, 6))
    story.append(Paragraph(headline(report), style["body"]))
    story.append(Spacer(1, 6))

    _pdf_section_1(report, story, style)
    _pdf_section_2(report, story, style)
    _pdf_section_3(report, story, style)
    _pdf_section_4(report, story, style)
    _pdf_section_5(report, story, style, plot_paths or [])
    _pdf_section_6(report, story, style)
    _pdf_section_7(report, story, style)

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(
        16 * mm,
        10 * mm,
        "Plumbline -- independent verification of derivative pricing models. "
        "Not investment advice.",
    )
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"page {document.page}")
    canvas.restoreState()


def _table(rows: list[list[Any]], widths: list[float], style: dict, header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")))
        commands.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    table.setStyle(TableStyle(commands))
    return table


def _pdf_section_1(report: Any, story: list, style: dict) -> None:
    story.append(Paragraph("Section 1 &mdash; Model identification", style["h2"]))
    model = report.model
    rows = [["Field", "Value"]]
    for key in (
        "name",
        "kind",
        "signature",
        "path",
        "entry",
        "supports_precision",
        "call_timeout_s",
        "timeouts",
        "sandbox_restarts",
        "rows",
    ):
        if key in model:
            rows.append(
                [
                    Paragraph(key.replace("_", " ").capitalize(), style["cell"]),
                    Paragraph(str(model[key]), style["cell"]),
                ]
            )
    rows.append([Paragraph("Submitted at", style["cell"]), Paragraph(report.started_at, style["cell"])])
    story.append(_table(rows, [45 * mm, 133 * mm], style))


def _pdf_section_2(report: Any, story: list, style: dict) -> None:
    story.append(Paragraph("Section 2 &mdash; Audit Score and summary badge", style["h2"]))
    badge = report.score.badge
    badge_table = Table(
        [[f"{badge}", f"Audit Score {report.score.total:.2f} / 100"]],
        colWidths=[40 * mm, 138 * mm],
    )
    badge_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), BADGE_COLOURS.get(badge, colors.grey)),
                ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
            ]
        )
    )
    story.append(badge_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Formula: {report.score.formula}", style["body"]))
    story.append(Spacer(1, 4))

    rows = [["#", "Check type", "Pass", "Fail", "Error", "Skip", "Pass rate", "Weight"]]
    for row in check_type_summary(report):
        rows.append(
            [
                str(row["check_type"]),
                Paragraph(row["name"], style["cell"]),
                str(row["passes"]),
                str(row["failures"]),
                str(row["errors"]),
                str(row["skipped"]),
                f"{row['pass_rate'] * 100:.1f}%" if row["ran"] else "not run",
                f"{row['weight']:.2f}",
            ]
        )
    story.append(
        _table(
            rows,
            [8 * mm, 58 * mm, 14 * mm, 14 * mm, 15 * mm, 14 * mm, 25 * mm, 18 * mm],
            style,
        )
    )


def _pdf_section_3(report: Any, story: list, style: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Section 3 &mdash; Full results", style["h2"]))
    for check_type in sorted(CHECK_NAMES):
        results = report.by_check_type(check_type)
        if not results:
            continue
        story.append(
            Paragraph(
                f"Check Type {check_type} &mdash; {CHECK_NAMES[check_type]} "
                f"({len(results)} results)",
                style["h3"],
            )
        )
        rows = [["Status", "Case", "Numeric evidence"]]
        shown = results[:MAX_PDF_ROWS]
        for result in shown:
            rows.append(
                [
                    Paragraph(
                        f'<font color="{STATUS_COLOURS.get(result.status, colors.black)}">'
                        f"<b>{result.status}</b></font>",
                        style["cell"],
                    ),
                    Paragraph(_xml(result.case), style["cell"]),
                    Paragraph(_xml(_evidence_text(result.evidence)), style["mono"]),
                ]
            )
        story.append(_table(rows, [16 * mm, 62 * mm, 100 * mm], style))
        if len(results) > MAX_PDF_ROWS:
            story.append(
                Paragraph(
                    f"{len(results) - MAX_PDF_ROWS} further rows of this check type "
                    f"are in the JSON report.",
                    style["body"],
                )
            )


def _pdf_section_4(report: Any, story: list, style: dict) -> None:
    story.append(PageBreak())
    story.append(
        Paragraph("Section 4 &mdash; Plain-language explanation of each failed check", style["h2"])
    )
    if not report.failures:
        story.append(
            Paragraph("No check failed. There is nothing to explain in this section.", style["body"])
        )
        return

    findings = diagnose(report.results)
    if findings:
        story.append(Paragraph("What is wrong with this model", style["h3"]))
        for finding in findings:
            story.append(Paragraph(f"<b>{_xml(finding.title)}</b>", style["body"]))
            story.append(Paragraph(_xml(finding.detail), style["body"]))
            story.append(Spacer(1, 4))

    story.append(Paragraph("Every failed case", style["h3"]))
    for result in report.failures:
        story.append(
            Paragraph(
                f"<b>[Check Type {result.check_type}] {_xml(result.case)}</b> "
                f"&mdash; {result.status}",
                style["body"],
            )
        )
        if result.explanation:
            story.append(Paragraph(_xml(result.explanation), style["body"]))
        story.append(Paragraph(_xml(_evidence_text(result.evidence)), style["mono"]))
        story.append(Spacer(1, 4))


def _pdf_section_5(report: Any, story: list, style: dict, plot_paths: list[str]) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Section 5 &mdash; Convergence plots", style["h2"]))
    series = report.convergence_series()
    if not series:
        story.append(
            Paragraph(
                "Check Type 4 produced no convergence series, so there is no plot.",
                style["body"],
            )
        )
        return

    for index, item in enumerate(series):
        story.append(Paragraph(f"{_xml(item['case'])} &mdash; {item['status']}", style["h3"]))
        rows = [["Precision", "Model price", "Absolute error"]]
        for level, price, error in zip(item["levels"], item["prices"], item["errors"]):
            rows.append([str(level), f"{price:.8f}", f"{error:.3e}"])
        story.append(_table(rows, [30 * mm, 40 * mm, 40 * mm], style))
        if index < len(plot_paths):
            story.append(Spacer(1, 4))
            story.append(Image(plot_paths[index], width=150 * mm, height=84 * mm))
        story.append(Spacer(1, 6))


def _pdf_section_6(report: Any, story: list, style: dict) -> None:
    story.append(PageBreak())
    story.append(
        Paragraph("Section 6 &mdash; Parameter grid used, for reproducibility", style["h2"])
    )
    for title, mapping in (
        ("Parameter grid", report.grid),
        ("Tolerance", report.tolerance),
        ("Audit configuration", report.config),
    ):
        story.append(Paragraph(title, style["h3"]))
        rows = [["Field", "Value"]]
        for key, value in mapping.items():
            rows.append(
                [
                    Paragraph(_xml(str(key)), style["cell"]),
                    Paragraph(_xml(str(value)), style["cell"]),
                ]
            )
        story.append(_table(rows, [45 * mm, 133 * mm], style))
        story.append(Spacer(1, 4))


def _pdf_section_7(report: Any, story: list, style: dict) -> None:
    story.append(Paragraph("Section 7 &mdash; Versions", style["h2"]))
    story.append(
        Paragraph(f"Plumbline version: <b>{report.plumbline_version}</b>", style["body"])
    )
    rows = [["Ground Truth Engine", "Version", "Instruments", "Models", "Reference"]]
    for engine in report.engines:
        rows.append(
            [
                Paragraph(_xml(engine["name"]), style["cell"]),
                Paragraph(engine["version"], style["cell"]),
                Paragraph(", ".join(engine["instruments"]), style["cell"]),
                Paragraph(", ".join(engine["models"]), style["cell"]),
                Paragraph(_xml(engine["reference"]), style["cell"]),
            ]
        )
    story.append(_table(rows, [30 * mm, 15 * mm, 42 * mm, 24 * mm, 67 * mm], style))
    story.append(Spacer(1, 4))

    rows = [["Environment", "Value"]]
    for key, value in report.environment.items():
        rows.append([Paragraph(key, style["cell"]), Paragraph(_xml(str(value)), style["cell"])])
    story.append(_table(rows, [45 * mm, 133 * mm], style))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(f"Audit duration: {report.duration_s:.2f} seconds.", style["body"])
    )


# ---------------------------------------------------------------------------


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _evidence_text(evidence: dict[str, Any]) -> str:
    parts = []
    for key, value in evidence.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}" if math.isfinite(value) else f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            shown = ", ".join(
                f"{v:.6g}" if isinstance(v, float) else str(v) for v in list(value)[:5]
            )
            more = "" if len(value) <= 5 else f", +{len(value) - 5} more"
            parts.append(f"{key}=[{shown}{more}]")
        elif isinstance(value, str) and len(value) > 200:
            parts.append(f"{key}={value[:197]}...")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "-"

"""Module D -- Report Generation (FR-D-01 to FR-D-06).

One call writes all three formats fixed by FR-D-04, from one report object::

    paths = write_report(report, "out/")

The JSON file is the machine-readable record and holds every result row.  The
Markdown and PDF files hold the same seven sections, with the longest result
tables truncated and a pointer to the JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from plumbline.report.markdown import render_markdown
from plumbline.report.plots import convergence_plots
from plumbline.report.summary import Finding, diagnose, headline

__all__ = [
    "Finding",
    "ReportPaths",
    "convergence_plots",
    "diagnose",
    "headline",
    "render_json",
    "render_markdown",
    "write_report",
]


@dataclass
class ReportPaths:
    """Where each rendered format was written."""

    json: str
    markdown: str
    pdf: str | None = None
    plots: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "json": self.json,
            "markdown": self.markdown,
            "pdf": self.pdf,
            "plots": list(self.plots or []),
        }


def render_json(report: Any, indent: int = 2) -> str:
    """The machine-readable Audit Report."""
    return json.dumps(report.to_dict(), indent=indent, default=str)


def write_report(
    report: Any,
    directory: str,
    basename: str | None = None,
    formats: tuple[str, ...] = ("json", "markdown", "pdf"),
) -> ReportPaths:
    """Write the Audit Report in every requested format (FR-D-04).

    The PDF is the only format that needs a third-party renderer.  If ReportLab
    is missing, the other formats are still written and the PDF path comes back
    as ``None``, because a missing renderer must not cost the whole report.
    """
    os.makedirs(directory, exist_ok=True)
    basename = basename or f"plumbline-audit-{report.audit_id}"

    json_path = os.path.join(directory, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(render_json(report))

    plot_paths: list[str] = []
    if "pdf" in formats or "markdown" in formats:
        plot_paths = convergence_plots(report, directory, prefix=f"{basename}-convergence")

    markdown_path = os.path.join(directory, f"{basename}.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report, plot_paths))

    pdf_path: str | None = None
    if "pdf" in formats:
        try:
            from plumbline.report.pdf import render_pdf

            pdf_path = render_pdf(report, os.path.join(directory, f"{basename}.pdf"), plot_paths)
        except ImportError:
            pdf_path = None

    return ReportPaths(
        json=json_path, markdown=markdown_path, pdf=pdf_path, plots=plot_paths
    )

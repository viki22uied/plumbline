"""Convergence plots for section 5 of the Audit Report.

Matplotlib is used through its non-interactive Agg backend, so the plots build
the same way on a workstation and on a headless continuous integration
machine.  If matplotlib is not installed the report still builds; the plot
section says so instead of failing the run.
"""

from __future__ import annotations

import os
from typing import Any

PLOT_DPI = 130


def convergence_plots(report: Any, directory: str, prefix: str = "convergence") -> list[str]:
    """Write one error-against-precision plot per convergence case.

    Returns the paths written, oldest style first. An empty list means either
    that Check Type 4 did not run or that matplotlib is not available.
    """
    series = report.convergence_series()
    if not series:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(directory, exist_ok=True)
    written: list[str] = []

    for index, item in enumerate(series):
        levels = item["levels"]
        errors = item["errors"]
        if len(levels) < 2:
            continue

        figure, axes = plt.subplots(figsize=(6.4, 3.6))
        axes.plot(levels, errors, marker="o", linewidth=1.6, color="#1f3b73")
        axes.set_xscale("log")
        if min(errors) > 0.0:
            axes.set_yscale("log")
        axes.set_xlabel("precision setting given to the model")
        axes.set_ylabel("absolute error against the reference")
        axes.set_title(
            f"Check Type 4 -- {item['status']}\n{item['case']}", fontsize=9, loc="left"
        )
        axes.grid(True, which="both", linewidth=0.4, alpha=0.5)
        figure.tight_layout()

        path = os.path.join(directory, f"{prefix}-{index + 1}.png")
        figure.savefig(path, dpi=PLOT_DPI)
        plt.close(figure)
        written.append(path)

    return written

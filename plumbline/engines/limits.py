"""Exact values at the degenerate corners of parameter space.

Check Type 5 (FR-C-12 to FR-C-15) drives every model to ``T = 0``,
``sigma = 0``, ``S = 0`` and a very large ``sigma``.  At those points the
option value is not a limit that a numerical scheme should be asked to find;
it is a closed form.  This module is the single place that states it, so the
audit's expectation and the Ground Truth Engines cannot drift apart.
"""

from __future__ import annotations

import math

from plumbline.contracts import OptionSpec


def is_degenerate(spec: OptionSpec) -> bool:
    """True when no stochastic model is needed to value ``spec`` exactly."""
    return spec.T <= 0.0 or spec.sigma <= 0.0 or spec.S <= 0.0 or spec.K <= 0.0


def degenerate_value(spec: OptionSpec) -> float:
    """Exact value of ``spec`` at a degenerate corner.

    Every instrument routes to its own closed form.  The one case the analytic
    module cannot serve is the arithmetic-average Asian, which has no closed
    form in general but does have one when the path is deterministic.
    """
    from plumbline.engines.analytic import analytic_price
    from plumbline.engines.binomial import _degenerate_value as american_corner

    if spec.instrument == "american":
        value = american_corner(spec)
        if value is not None:
            return value
        raise ValueError("degenerate_value called on a non-degenerate American spec")

    if spec.instrument == "asian" and spec.averaging == "arithmetic":
        return _arithmetic_asian_corner(spec)

    return analytic_price(spec)


def _arithmetic_asian_corner(spec: OptionSpec) -> float:
    if spec.T <= 0.0 or spec.S <= 0.0:
        return max(spec.phi * (spec.S - spec.K), 0.0)
    if spec.sigma > 0.0:
        raise ValueError("arithmetic Asian corner needs sigma == 0")
    b, T = spec.carry, spec.T
    # Time average of the deterministic forward path S exp(b t) over [0, T].
    average = spec.S * (math.expm1(b * T) / (b * T)) if abs(b) > 1e-12 else spec.S
    return spec.discount * max(spec.phi * (average - spec.K), 0.0)

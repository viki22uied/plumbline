"""Parameter grids for the audit (FR-C-04).

A single parameter set proves nothing: a model can be right at the money and
wrong in the wings, right at one year and wrong at one week.  Every price
check therefore runs over a cartesian grid, and the grid used is written into
the Audit Report so the run can be reproduced (report section 6).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Any, Iterator

from plumbline.contracts import OptionSpec


@dataclass
class ParameterGrid:
    """The cartesian product of market parameters used by an audit."""

    instrument: str = "european"
    option_types: tuple[str, ...] = ("call", "put")
    model: str = "bsm"

    spots: tuple[float, ...] = (90.0, 100.0, 110.0)
    strikes: tuple[float, ...] = (95.0, 100.0, 105.0)
    maturities: tuple[float, ...] = (0.25, 1.0)
    rates: tuple[float, ...] = (0.03,)
    dividends: tuple[float, ...] = (0.0,)
    vols: tuple[float, ...] = (0.15, 0.30)

    #: Instrument extras held fixed across the grid.
    extras: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[OptionSpec]:
        for option_type, S, K, T, r, q, sigma in product(
            self.option_types,
            self.spots,
            self.strikes,
            self.maturities,
            self.rates,
            self.dividends,
            self.vols,
        ):
            yield OptionSpec(
                instrument=self.instrument,
                option_type=option_type,
                model=self.model,
                S=S,
                K=K,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                **self.extras,
            )

    def __len__(self) -> int:
        return (
            len(self.option_types)
            * len(self.spots)
            * len(self.strikes)
            * len(self.maturities)
            * len(self.rates)
            * len(self.dividends)
            * len(self.vols)
        )

    def points(self) -> list[OptionSpec]:
        return list(self)

    def call_put_pairs(self) -> list[tuple[OptionSpec, OptionSpec]]:
        """Matched call/put pairs on identical parameters, for Check Type 2."""
        seen: dict[tuple, OptionSpec] = {}
        pairs = []
        for spec in self:
            key = (spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma)
            if spec.option_type == "call":
                seen[key] = spec
        for spec in self:
            if spec.option_type == "put" and (
                key := (spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma)
            ) in seen:
                pairs.append((seen[key], spec))
        return pairs

    def sample(self, count: int, seed: int = 0) -> list[OptionSpec]:
        """A deterministic spread of ``count`` points, for the costly checks.

        A fixed stride would be worse than it looks: the grid is a cartesian
        product, so any stride that divides one of the axis lengths keeps
        returning the same value on that axis and the check never sees the rest
        of it.  A seeded random sample has no such alignment, and the seed
        lives in the report, so the selection is still reproducible.
        """
        points = self.points()
        if count >= len(points):
            return points
        chosen = random.Random(seed).sample(range(len(points)), max(count, 1))
        return [points[index] for index in sorted(chosen)]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["size"] = len(self)
        return data


def default_grid(instrument: str = "european", **overrides: Any) -> ParameterGrid:
    """The grid an audit uses when the caller does not supply one.

    The instrument extras are filled with a sensible in-range default so a
    barrier or a lookback audit works out of the box.
    """
    extras: dict[str, Any] = dict(overrides.pop("extras", {}))
    if instrument == "barrier":
        extras.setdefault("barrier", 120.0)
        extras.setdefault("barrier_kind", "up-and-out")
    if instrument == "asian":
        extras.setdefault("averaging", "geometric")
    if instrument == "digital":
        extras.setdefault("payout", "cash")
        extras.setdefault("cash_amount", 1.0)
    if instrument == "lookback":
        extras.setdefault("strike_type", "fixed")

    grid = ParameterGrid(instrument=instrument, extras=extras)
    for key, value in overrides.items():
        if not hasattr(grid, key):
            raise AttributeError(f"ParameterGrid has no field {key!r}")
        setattr(grid, key, tuple(value) if isinstance(value, (list, tuple)) else value)

    if instrument == "barrier":
        barrier = float(extras["barrier"])
        is_up = str(extras["barrier_kind"]).startswith("up")
        # Keep every grid spot strictly on the live side of the barrier, or the
        # contract is already settled and the check measures nothing.
        grid.spots = tuple(
            s for s in grid.spots if (s < barrier if is_up else s > barrier)
        ) or ((barrier * 0.9,) if is_up else (barrier * 1.1,))
    return grid

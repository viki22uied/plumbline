"""Ground Truth Engine registry and plug-in interface (NFR-08, FR-E-03).

A new instrument or a new engine is added by calling :func:`register` with a
callable of type ``(OptionSpec) -> PriceResult``.  Nothing in the Validation
and Audit Engine needs to change: the checks ask this registry which engine is
authoritative for a given problem and compare against whatever comes back.

Selection is by explicit priority per instrument, highest first, restricted to
engines that declare support for the spec's underlying model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from plumbline.contracts import (
    OptionSpec,
    PriceResult,
    UnsupportedInstrument,
)
from plumbline.engines.limits import degenerate_value, is_degenerate

PriceFn = Callable[[OptionSpec], PriceResult]


@dataclass(frozen=True)
class EngineSpec:
    """One registered Ground Truth Engine."""

    name: str
    description: str
    reference: str
    price_fn: PriceFn
    instruments: tuple[str, ...]
    models: tuple[str, ...] = ("bsm",)
    #: Bumped by hand when an engine's numerical output changes, so an Audit
    #: Report always records which build produced its reference values.
    version: str = "1.0.0"
    #: Higher wins when several engines cover the same problem.
    priority: int = 0
    #: False for simulation engines, whose output carries sampling noise.
    deterministic: bool = True
    #: Extra predicate for engines that cover only part of an instrument.
    applies: Callable[[OptionSpec], bool] | None = field(default=None, compare=False)

    def supports(self, spec: OptionSpec) -> bool:
        if spec.instrument not in self.instruments or spec.model not in self.models:
            return False
        return self.applies is None or self.applies(spec)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "reference": self.reference,
            "version": self.version,
            "instruments": list(self.instruments),
            "models": list(self.models),
            "priority": self.priority,
            "deterministic": self.deterministic,
        }


REGISTRY: dict[str, EngineSpec] = {}


def register(engine: EngineSpec, *, replace: bool = False) -> EngineSpec:
    """Add an engine to the registry. This is the whole plug-in interface."""
    if engine.name in REGISTRY and not replace:
        raise ValueError(f"engine {engine.name!r} is already registered")
    REGISTRY[engine.name] = engine
    return engine


def deterministic_fallback(spec: OptionSpec) -> float:
    """Exact value at a degenerate corner. Re-exported for engine internals."""
    return degenerate_value(spec)


def candidates(spec: OptionSpec) -> list[EngineSpec]:
    """Every registered engine that can price ``spec``, best first."""
    found = [e for e in REGISTRY.values() if e.supports(spec)]
    return sorted(found, key=lambda e: e.priority, reverse=True)


def ground_truth_for(spec: OptionSpec) -> EngineSpec:
    """The authoritative engine for ``spec``."""
    found = candidates(spec)
    if not found:
        raise UnsupportedInstrument(
            f"no Ground Truth Engine covers instrument={spec.instrument!r} "
            f"model={spec.model!r}"
        )
    return found[0]


def ground_truth_price(spec: OptionSpec) -> PriceResult:
    """Price ``spec`` with its authoritative engine, corners handled first."""
    if is_degenerate(spec):
        return PriceResult(
            price=degenerate_value(spec),
            engine="closed_form_limit",
            diagnostics={"reason": "degenerate parameters"},
        )
    return ground_truth_for(spec).price_fn(spec)


def get(name: str) -> EngineSpec:
    if name not in REGISTRY:
        raise KeyError(f"no engine named {name!r}; known engines: {sorted(REGISTRY)}")
    return REGISTRY[name]


def _install_builtin_engines() -> None:
    """Register the engines shipped with Plumbline."""
    from plumbline.engines import analytic, binomial, fdm, heston, montecarlo

    register(
        EngineSpec(
            name="analytic",
            description="Closed-form Black-Scholes-Merton and exotic formulas",
            reference="Merton (1973); Reiner-Rubinstein (1991); Kemna-Vorst (1990); "
            "Goldman-Sosin-Gatto (1979); Conze-Viswanathan (1991)",
            price_fn=analytic.price,
            instruments=analytic.SUPPORTED,
            priority=100,
            applies=lambda s: not (s.instrument == "asian" and s.averaging == "arithmetic"),
        )
    )
    register(
        EngineSpec(
            name="binomial_crr",
            description="Cox-Ross-Rubinstein binomial tree, European and American",
            reference="Cox, Ross & Rubinstein (1979); Hull ch. 21",
            price_fn=binomial.price,
            instruments=binomial.SUPPORTED,
            priority=90,
        )
    )
    register(
        EngineSpec(
            name="fdm_crank_nicolson",
            description="Crank-Nicolson finite differences with Rannacher start-up",
            reference="Crank & Nicolson (1947); Rannacher (1984); Wilmott ch. 78",
            price_fn=fdm.price,
            instruments=fdm.SUPPORTED,
            models=("bsm", "localvol"),
            priority=80,
        )
    )
    register(
        EngineSpec(
            name="heston_cf",
            description="Heston stochastic volatility, characteristic function integral",
            reference="Heston (1993); Albrecher et al. (2007) little-trap form",
            price_fn=heston.price,
            instruments=heston.SUPPORTED,
            models=("heston",),
            priority=100,
        )
    )
    register(
        EngineSpec(
            name="monte_carlo",
            description="Monte Carlo with antithetic and control variates",
            reference="Glasserman (2003) ch. 4; Kemna & Vorst (1990) control",
            price_fn=montecarlo.price,
            instruments=montecarlo.SUPPORTED,
            priority=50,
            deterministic=False,
        )
    )


_install_builtin_engines()

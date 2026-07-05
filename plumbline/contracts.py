"""Canonical data contracts shared by every Plumbline module.

The Model Under Test (MUT) contract (FR-A-01) is one callable::

    def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs) -> float

``instrument``   one of :data:`INSTRUMENTS`
``option_type``  ``"call"`` or ``"put"``
``S``            spot price of the underlying asset
``K``            strike price
``T``            time to expiry in years
``r``            continuously compounded risk-free rate
``q``            continuously compounded dividend yield
``sigma``        Black-Scholes volatility (annualised)
``kwargs``       instrument extras (``barrier``, ``rebate``, ``averaging``,
                 ``strike_type``, ``payout``, ``running_min``, ``running_max``)
                 and the optional precision knob ``precision`` (Check Type 4).

The callable returns one float: the present value of one contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from typing import Any

# --- vocabulary ------------------------------------------------------------

INSTRUMENTS = (
    "european",
    "american",
    "asian",
    "barrier",
    "digital",
    "lookback",
)

OPTION_TYPES = ("call", "put")

#: Positional names of the required MUT signature, in order (FR-A-03).
REQUIRED_SIGNATURE = (
    "instrument",
    "option_type",
    "S",
    "K",
    "T",
    "r",
    "q",
    "sigma",
)

#: Underlying asset dynamics (PRD section 4.3).
MODELS = ("bsm", "heston", "localvol")

BARRIER_KINDS = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
AVERAGING_KINDS = ("geometric", "arithmetic")
DIGITAL_PAYOUTS = ("cash", "asset")
LOOKBACK_STRIKES = ("fixed", "floating")


class PlumblineError(Exception):
    """Base class for every error raised on purpose by Plumbline."""


class IngestionError(PlumblineError):
    """The Model Under Test does not satisfy the ingestion contract."""


class UnsupportedInstrument(PlumblineError):
    """No Ground Truth Engine covers the requested instrument."""


# --- parameter set ---------------------------------------------------------


@dataclass(frozen=True)
class OptionSpec:
    """One fully specified pricing problem.

    This object is the single currency between the Validation Engine, the
    Ground Truth Engines and the Model Under Test.  It is frozen so a check
    cannot mutate the grid point it was handed.
    """

    instrument: str
    option_type: str
    S: float
    K: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2

    # exotic extras (only the ones relevant to ``instrument`` are read)
    barrier: float | None = None
    barrier_kind: str | None = None
    rebate: float = 0.0
    averaging: str = "geometric"
    payout: str = "cash"
    cash_amount: float = 1.0
    strike_type: str = "fixed"
    running_min: float | None = None
    running_max: float | None = None

    # --- underlying dynamics (section 4.3) ---------------------------------
    #: ``"bsm"`` (constant volatility), ``"heston"`` or ``"localvol"``.
    model: str = "bsm"

    # Heston parameters. ``sigma`` is ignored when ``model == "heston"``.
    v0: float = 0.04  # initial variance
    kappa: float = 2.0  # mean reversion speed
    theta_v: float = 0.04  # long-run variance
    xi: float = 0.3  # volatility of variance
    rho_sv: float = -0.7  # spot / variance correlation

    # Local volatility surface, in the parametric form
    # ``sigma_loc(S, t) = clip(lv_a + lv_b * log(S / lv_ref) + lv_c * t, 0.01, 3.0)``.
    lv_a: float = 0.2
    lv_b: float = 0.0
    lv_c: float = 0.0
    lv_ref: float = 100.0

    #: Optional numerical-precision knob forwarded to engines and to the MUT.
    precision: int | None = None

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise PlumblineError(f"model {self.model!r} is not one of {MODELS}")
        if self.instrument not in INSTRUMENTS:
            raise UnsupportedInstrument(
                f"instrument {self.instrument!r} is not one of {INSTRUMENTS}"
            )
        if self.option_type not in OPTION_TYPES:
            raise PlumblineError(
                f"option_type {self.option_type!r} is not one of {OPTION_TYPES}"
            )
        for name in ("S", "K", "T", "sigma"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value < 0.0:
                raise PlumblineError(f"{name}={value!r} must be finite and >= 0")
        if self.instrument == "barrier":
            if self.barrier is None or self.barrier <= 0.0:
                raise PlumblineError("a barrier option needs barrier > 0")
            if self.barrier_kind not in BARRIER_KINDS:
                raise PlumblineError(
                    f"barrier_kind {self.barrier_kind!r} is not one of {BARRIER_KINDS}"
                )
        if self.instrument == "asian" and self.averaging not in AVERAGING_KINDS:
            raise PlumblineError(f"averaging must be one of {AVERAGING_KINDS}")
        if self.instrument == "digital" and self.payout not in DIGITAL_PAYOUTS:
            raise PlumblineError(f"payout must be one of {DIGITAL_PAYOUTS}")
        if self.instrument == "lookback" and self.strike_type not in LOOKBACK_STRIKES:
            raise PlumblineError(f"strike_type must be one of {LOOKBACK_STRIKES}")
        if self.model == "heston":
            if self.v0 < 0.0 or self.theta_v < 0.0:
                raise PlumblineError("Heston variances v0 and theta_v must be >= 0")
            if self.kappa <= 0.0 or self.xi <= 0.0:
                raise PlumblineError("Heston kappa and xi must be > 0")
            if not -1.0 <= self.rho_sv <= 1.0:
                raise PlumblineError("Heston rho_sv must be in [-1, 1]")

    # -- derived quantities -------------------------------------------------

    @property
    def phi(self) -> float:
        """+1 for a call, -1 for a put."""
        return 1.0 if self.option_type == "call" else -1.0

    @property
    def carry(self) -> float:
        """Cost of carry ``b = r - q``."""
        return self.r - self.q

    @property
    def discount(self) -> float:
        return math.exp(-self.r * self.T)

    @property
    def forward(self) -> float:
        return self.S * math.exp(self.carry * self.T)

    def intrinsic(self) -> float:
        """Payoff if the option expired now at spot ``S``."""
        if self.instrument == "digital":
            in_money = (
                self.S > self.K if self.option_type == "call" else self.S < self.K
            )
            if not in_money:
                return 0.0
            return self.cash_amount if self.payout == "cash" else self.S
        return max(self.phi * (self.S - self.K), 0.0)

    # -- transport ----------------------------------------------------------

    def with_(self, **changes: Any) -> "OptionSpec":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_mut_kwargs(self) -> dict[str, Any]:
        """The exact keyword arguments handed to a Model Under Test."""
        kwargs = asdict(self)
        if kwargs.get("precision") is None:
            kwargs.pop("precision")
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        bits = [self.instrument, self.option_type]
        if self.instrument == "barrier":
            bits.append(f"{self.barrier_kind}@{self.barrier:g}")
        if self.instrument == "asian":
            bits.append(self.averaging)
        if self.instrument == "digital":
            bits.append(self.payout)
        if self.instrument == "lookback":
            bits.append(self.strike_type)
        bits.append(f"S={self.S:g} K={self.K:g} T={self.T:g} r={self.r:g} sig={self.sigma:g}")
        return " ".join(bits)


@dataclass(frozen=True)
class Greeks:
    """The five core sensitivities (FR-B-07).

    Conventions: ``vega`` is per 1.00 of volatility, ``theta`` is per year
    (negative for a long option that loses time value), ``rho`` is per 1.00
    of interest rate.
    """

    delta: float = math.nan
    gamma: float = math.nan
    vega: float = math.nan
    theta: float = math.nan
    rho: float = math.nan

    NAMES = ("delta", "gamma", "vega", "theta", "rho")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.NAMES}


@dataclass
class PriceResult:
    """What an engine (or a sandboxed MUT call) returns."""

    price: float = math.nan
    greeks: Greeks | None = None
    engine: str = ""
    status: str = "OK"  # OK | ERROR | TIMEOUT
    message: str = ""
    elapsed_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "OK" and math.isfinite(self.price)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["greeks"] = self.greeks.to_dict() if self.greeks else None
        return out


@dataclass
class Tolerance:
    """Allowed difference between a MUT result and the Ground Truth value."""

    relative: float = 1e-3
    absolute: float = 1e-6
    #: Monte Carlo style engines get their own, looser band.
    stochastic_relative: float = 1e-2
    #: Greeks are computed by bump-and-reprice on both sides; looser still.
    greek_relative: float = 5e-2
    greek_absolute: float = 1e-4

    def within(self, actual: float, expected: float) -> bool:
        return abs(actual - expected) <= max(
            self.absolute, self.relative * abs(expected)
        )

    def greek_within(self, actual: float, expected: float) -> bool:
        return abs(actual - expected) <= max(
            self.greek_absolute, self.greek_relative * abs(expected)
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def relative_difference(actual: float, expected: float) -> float:
    """Relative difference that stays finite when ``expected`` is near zero."""
    denominator = max(abs(expected), 1e-12)
    return (actual - expected) / denominator

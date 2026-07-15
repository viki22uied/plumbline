"""Module A -- Model Ingestion (FR-A-01 to FR-A-06).

Two shapes of Model Under Test are accepted:

* a Python source file exporting a callable (default name ``price``) with the
  signature documented in :mod:`plumbline.contracts`;
* a CSV or JSON file of already-computed prices, for a model whose source is
  not available.

Both are wrapped in the same :class:`ModelUnderTest` interface, so the
Validation and Audit Engine never learns which one it is auditing.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Iterable

from plumbline.contracts import (
    REQUIRED_SIGNATURE,
    IngestionError,
    OptionSpec,
    PriceResult,
)
from plumbline.sandbox import Sandbox, SandboxConfig

#: Fields a table model may key on, in the order they are matched.
TABLE_KEY_FIELDS = (
    "instrument",
    "option_type",
    "S",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "barrier",
    "barrier_kind",
    "averaging",
    "payout",
    "strike_type",
)

#: A table lookup matches a float column when it agrees to this many places.
TABLE_MATCH_TOLERANCE = 1e-9


class ModelUnderTest:
    """The interface every Model Under Test presents to the audit."""

    #: Human-readable identity, printed in section 1 of the Audit Report.
    name: str = "unnamed model"
    #: Description of the accepted input signature, also for the report.
    signature_description: str = ""
    #: True when the model exposes a numerical precision knob (FR-C-10).
    supports_precision: bool = False

    def price(self, spec: OptionSpec) -> PriceResult:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "ModelUnderTest":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": type(self).__name__,
            "signature": self.signature_description,
            "supports_precision": self.supports_precision,
        }


# ---------------------------------------------------------------------------
# Python source models
# ---------------------------------------------------------------------------


def check_signature(names: Iterable[str], var_keyword: bool) -> None:
    """Raise :class:`IngestionError` naming exactly what is wrong (FR-A-03/04)."""
    names = list(names)
    missing = [required for required in REQUIRED_SIGNATURE if required not in names]
    problems: list[str] = []
    if missing:
        problems.append(
            "the entry point does not accept these required parameters: "
            + ", ".join(missing)
        )
    if not var_keyword:
        problems.append(
            "the entry point has no **kwargs parameter, so it cannot receive the "
            "instrument extras (barrier, averaging, payout, strike_type, precision)"
        )
    if problems:
        raise IngestionError(
            "the Model Under Test does not match the required signature.\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\n  required parameters: ("
            + ", ".join(REQUIRED_SIGNATURE)
            + ", **kwargs) -> float"
            + f"\n  found parameters:    ({', '.join(names) or 'none'})"
        )


class PythonModel(ModelUnderTest):
    """A Model Under Test given as a Python source file (FR-A-01)."""

    def __init__(
        self,
        path: str,
        entry: str = "price",
        call_timeout: float | None = None,
        precision_argument: str = "precision",
    ):
        self.path = os.path.abspath(path)
        self.entry = entry
        self.name = f"{os.path.basename(self.path)}:{entry}"
        self.precision_argument = precision_argument
        config = SandboxConfig()
        if call_timeout is not None:
            config.call_timeout = call_timeout
        self._sandbox = Sandbox(self.path, entry, config)
        self._sandbox.start()

        # The sandbox reports the signature; the parent decides whether it is
        # acceptable, so a bad model cannot vouch for itself (FR-A-03/04).
        try:
            check_signature(self._sandbox.signature, self._sandbox.var_keyword)
        except IngestionError:
            self._sandbox.close()
            raise
        self.signature_description = f"def {entry}({', '.join(self._sandbox.signature)})"
        self.supports_precision = precision_argument in self._sandbox.signature or (
            self._sandbox.var_keyword
        )

    def price(self, spec: OptionSpec) -> PriceResult:
        return self._sandbox.call(spec.to_mut_kwargs())

    def close(self) -> None:
        self._sandbox.close()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "path": self.path,
                "entry": self.entry,
                "call_timeout_s": self._sandbox.config.call_timeout,
                "timeouts": self._sandbox.timeouts,
                "sandbox_restarts": self._sandbox.restarts,
            }
        )
        return info


# ---------------------------------------------------------------------------
# Table models
# ---------------------------------------------------------------------------


class TableModel(ModelUnderTest):
    """A Model Under Test given as pre-computed prices (FR-A-02).

    The file supplies one row per priced parameter set.  A row must carry a
    ``price`` column and any subset of :data:`TABLE_KEY_FIELDS`; the columns
    actually present become the lookup key.  A parameter set with no matching
    row is reported as ``NOT_PRICED`` rather than as a failure, because a
    missing row is a gap in the submission, not a mathematical error.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        rows = _read_rows(self.path)
        if not rows:
            raise IngestionError(f"{self.path} contains no rows")
        if not any("price" in row for row in rows):
            raise IngestionError(f"{self.path} has no 'price' column")

        self.key_fields = tuple(
            field for field in TABLE_KEY_FIELDS if field in rows[0]
        )
        if not self.key_fields:
            raise IngestionError(
                f"{self.path} has none of the key columns {TABLE_KEY_FIELDS}"
            )
        self._rows = [(_key_of(row, self.key_fields), _as_float(row["price"])) for row in rows]
        self.signature_description = (
            f"table of {len(self._rows)} prices keyed on {', '.join(self.key_fields)}"
        )
        self.supports_precision = False

    def price(self, spec: OptionSpec) -> PriceResult:
        wanted = _key_of(spec.to_dict(), self.key_fields)
        for key, value in self._rows:
            if _keys_match(key, wanted):
                return PriceResult(price=value, engine="mut-table", status="OK")
        return PriceResult(
            status="NOT_PRICED",
            engine="mut-table",
            message=f"the table has no row for {spec.label()}",
            price=math.nan,
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"path": self.path, "rows": len(self._rows), "key": list(self.key_fields)})
        return info


def _read_rows(path: str) -> list[dict[str, Any]]:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("prices", payload.get("rows", []))
        if not isinstance(payload, list):
            raise IngestionError(f"{path} must hold a JSON list of rows")
        return [dict(row) for row in payload]
    if extension in (".csv", ".txt"):
        with open(path, newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise IngestionError(f"{path}: a table model must be .csv or .json")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _key_of(row: dict[str, Any], fields: tuple[str, ...]) -> tuple:
    key = []
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            key.append(None)
        elif field in ("instrument", "option_type", "barrier_kind", "averaging", "payout", "strike_type"):
            key.append(str(value))
        else:
            key.append(_as_float(value))
    return tuple(key)


def _keys_match(left: tuple, right: tuple) -> bool:
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                return False
        elif not (abs(a - b) <= TABLE_MATCH_TOLERANCE * max(1.0, abs(b))):
            return False
    return True


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def load_model(
    path: str,
    entry: str = "price",
    call_timeout: float | None = None,
) -> ModelUnderTest:
    """Load a Model Under Test from ``path`` (FR-A-01, FR-A-02)."""
    if not os.path.isfile(path):
        raise IngestionError(f"no such Model Under Test file: {path}")
    extension = os.path.splitext(path)[1].lower()
    if extension == ".py":
        return PythonModel(path, entry=entry, call_timeout=call_timeout)
    if extension in (".csv", ".json", ".txt"):
        return TableModel(path)
    raise IngestionError(
        f"{path}: a Model Under Test must be a .py source file or a .csv/.json price table"
    )

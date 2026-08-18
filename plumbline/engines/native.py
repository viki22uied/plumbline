"""Loader for the optional native Monte Carlo backend.

The backend is a plain C++ shared library with a C ABI, loaded through
``ctypes``.  It is optional in the strict sense: if it is missing, if it fails
to load, or if it refuses a request, every caller falls back to the NumPy
engine and the audit is unaffected.  Nothing in Plumbline requires it.

Build it with::

    python native/build.py --check

The loader searches, in order:

1. ``$PLUMBLINE_NATIVE_LIB``, a full path to the library;
2. ``plumbline/engines/_native/``, where ``native/build.py`` writes it;
3. the platform's ordinary library search path.

Two safety rails sit between Python and the library.  The struct sizes are
checked against the ones the library was compiled with before the first call,
so a stale build is caught at load time rather than as unexplained numbers
later.  And the library refuses a degenerate contract outright rather than
returning a number, because the exact value in that case belongs to
:mod:`plumbline.engines.limits` and there must be one source for it.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
from typing import Any

from plumbline.contracts import OptionSpec

#: Instrument codes, matching the PLUMBLINE_* defines in native/plumbline_mc.h.
INSTRUMENT_CODES = {
    "european": 0,
    "asian": 1,
    "barrier": 2,
    "digital": 3,
    "lookback": 4,
}

#: Status codes returned by ``plumbline_mc_price``.
STATUS_OK = 0
STATUS_MESSAGES = {
    0: "ok",
    1: "the library was built against a different struct layout",
    2: "the native backend does not price this instrument",
    3: "the native backend rejected a parameter",
    4: "the contract is degenerate and has an exact closed form",
}

SUPPORTED = tuple(INSTRUMENT_CODES)

_ENVIRONMENT_VARIABLE = "PLUMBLINE_NATIVE_LIB"
_PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_native")

_lock = threading.Lock()
_library: ctypes.CDLL | None = None
_load_error: str = ""
_loaded = False


class _Request(ctypes.Structure):
    """Mirror of ``PlumblineMCRequest``. Every field is eight bytes wide."""

    _fields_ = [
        ("struct_size", ctypes.c_int64),
        ("instrument", ctypes.c_int64),
        ("is_call", ctypes.c_int64),
        ("spot", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("rate", ctypes.c_double),
        ("dividend", ctypes.c_double),
        ("volatility", ctypes.c_double),
        ("barrier", ctypes.c_double),
        ("barrier_is_down", ctypes.c_int64),
        ("barrier_is_out", ctypes.c_int64),
        ("averaging_arithmetic", ctypes.c_int64),
        ("payout_asset", ctypes.c_int64),
        ("cash_amount", ctypes.c_double),
        ("strike_fixed", ctypes.c_int64),
        ("paths", ctypes.c_int64),
        ("steps", ctypes.c_int64),
        ("seed", ctypes.c_uint64),
        ("antithetic", ctypes.c_int64),
        ("control_variate", ctypes.c_int64),
        ("control_mean", ctypes.c_double),
        ("threads", ctypes.c_int64),
        ("block_pairs", ctypes.c_int64),
    ]


class _Result(ctypes.Structure):
    """Mirror of ``PlumblineMCResult``."""

    _fields_ = [
        ("struct_size", ctypes.c_int64),
        ("price", ctypes.c_double),
        ("standard_error", ctypes.c_double),
        ("control_beta", ctypes.c_double),
        ("paths", ctypes.c_int64),
        ("threads", ctypes.c_int64),
        ("blocks", ctypes.c_int64),
    ]


class NativeBackendError(RuntimeError):
    """The native backend was asked for something it could not do."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _library_filename() -> str:
    system = platform.system()
    if system == "Windows":
        return "plumbline_mc.dll"
    if system == "Darwin":
        return "libplumbline_mc.dylib"
    return "libplumbline_mc.so"


def _candidate_paths() -> list[str]:
    filename = _library_filename()
    candidates = []
    override = os.environ.get(_ENVIRONMENT_VARIABLE)
    if override:
        candidates.append(override)
    candidates.append(os.path.join(_PACKAGE_DIRECTORY, filename))
    candidates.append(filename)  # let the platform's loader search
    return candidates


def _bind(library: ctypes.CDLL) -> None:
    library.plumbline_mc_price.argtypes = [
        ctypes.POINTER(_Request),
        ctypes.POINTER(_Result),
    ]
    library.plumbline_mc_price.restype = ctypes.c_int32
    library.plumbline_backend_version.argtypes = []
    library.plumbline_backend_version.restype = ctypes.c_char_p
    library.plumbline_backend_threads.argtypes = []
    library.plumbline_backend_threads.restype = ctypes.c_int64
    library.plumbline_request_size.argtypes = []
    library.plumbline_request_size.restype = ctypes.c_int64
    library.plumbline_result_size.argtypes = []
    library.plumbline_result_size.restype = ctypes.c_int64


def _load() -> ctypes.CDLL | None:
    global _library, _load_error, _loaded
    with _lock:
        if _loaded:
            return _library
        _loaded = True

        errors = []
        for path in _candidate_paths():
            try:
                library = ctypes.CDLL(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            try:
                _bind(library)
                # The ABI check. A library compiled from a different header is
                # worse than a missing one, because it would still return
                # numbers.
                if library.plumbline_request_size() != ctypes.sizeof(_Request):
                    raise NativeBackendError(
                        f"{path}: the library's request struct is "
                        f"{library.plumbline_request_size()} bytes, this build expects "
                        f"{ctypes.sizeof(_Request)}"
                    )
                if library.plumbline_result_size() != ctypes.sizeof(_Result):
                    raise NativeBackendError(
                        f"{path}: the library's result struct is "
                        f"{library.plumbline_result_size()} bytes, this build expects "
                        f"{ctypes.sizeof(_Result)}"
                    )
            except (AttributeError, NativeBackendError) as exc:
                errors.append(str(exc))
                continue

            _library = library
            _load_error = ""
            return _library

        _load_error = "; ".join(errors) or "no candidate path was tried"
        return None


def reset() -> None:
    """Forget the cached load. Used by the build script and by the tests."""
    global _library, _loaded, _load_error
    with _lock:
        _library = None
        _loaded = False
        _load_error = ""


def available() -> bool:
    """True when the native backend is loaded and usable."""
    return _load() is not None


def load_error() -> str:
    """Why the backend is not available, for the report and the CLI."""
    _load()
    return _load_error


def backend_version() -> str | None:
    library = _load()
    if library is None:
        return None
    return library.plumbline_backend_version().decode("utf-8")


def backend_threads() -> int:
    library = _load()
    return int(library.plumbline_backend_threads()) if library else 0


def describe() -> dict[str, Any]:
    """What the Audit Report records about the backend."""
    return {
        "available": available(),
        "version": backend_version(),
        "hardware_threads": backend_threads(),
        "library": _resolved_path(),
        "error": load_error() or None,
    }


def _resolved_path() -> str | None:
    if not available():
        return None
    for path in _candidate_paths():
        if os.path.isfile(path):
            return os.path.abspath(path)
    return _library_filename()


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------


def supports(spec: OptionSpec) -> bool:
    """True when the native backend covers this contract."""
    if spec.model != "bsm" or spec.instrument not in INSTRUMENT_CODES:
        return False
    if spec.instrument == "barrier" and spec.rebate != 0.0:
        return False
    # A degenerate contract has an exact closed form; the backend refuses it
    # on purpose, so do not offer it one.
    return not (spec.T <= 0.0 or spec.sigma <= 0.0 or spec.S <= 0.0)


def price(
    spec: OptionSpec,
    control_mean: float,
    paths: int,
    steps: int,
    seed: int,
    antithetic: bool = True,
    control_variate: bool = True,
    threads: int = 0,
    block_pairs: int = 0,
) -> tuple[float, float, float, int, int]:
    """Price ``spec`` natively.

    ``control_mean`` is supplied by the caller rather than recomputed here, so
    the closed forms behind the control variate have exactly one implementation
    and the two backends cannot disagree about them.

    Returns ``(price, standard_error, control_beta, paths, threads)``.
    Raises :class:`NativeBackendError` if the backend is missing or refuses.
    """
    library = _load()
    if library is None:
        raise NativeBackendError(f"the native backend is not available: {load_error()}")
    if not supports(spec):
        raise NativeBackendError(
            f"the native backend does not cover {spec.label()!r}"
        )

    request = _Request(
        struct_size=ctypes.sizeof(_Request),
        instrument=INSTRUMENT_CODES[spec.instrument],
        is_call=1 if spec.option_type == "call" else 0,
        spot=spec.S,
        strike=spec.K,
        maturity=spec.T,
        rate=spec.r,
        dividend=spec.q,
        volatility=spec.sigma,
        barrier=float(spec.barrier or 0.0),
        barrier_is_down=1 if (spec.barrier_kind or "").startswith("down") else 0,
        barrier_is_out=1 if (spec.barrier_kind or "").endswith("out") else 0,
        averaging_arithmetic=1 if spec.averaging == "arithmetic" else 0,
        payout_asset=1 if spec.payout == "asset" else 0,
        cash_amount=spec.cash_amount,
        strike_fixed=1 if spec.strike_type == "fixed" else 0,
        paths=int(paths),
        steps=int(steps),
        seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
        antithetic=1 if antithetic else 0,
        control_variate=1 if control_variate else 0,
        control_mean=float(control_mean),
        threads=int(threads),
        block_pairs=int(block_pairs),
    )
    result = _Result(struct_size=ctypes.sizeof(_Result))

    status = library.plumbline_mc_price(ctypes.byref(request), ctypes.byref(result))
    if status != STATUS_OK:
        raise NativeBackendError(
            STATUS_MESSAGES.get(status, f"unknown status {status}")
            + f" ({spec.label()})"
        )

    return (
        float(result.price),
        float(result.standard_error),
        float(result.control_beta),
        int(result.paths),
        int(result.threads),
    )

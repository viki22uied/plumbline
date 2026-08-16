"""Child-process entry point for the Model Under Test sandbox.

Started by :mod:`plumbline.sandbox` as ``python -m plumbline._worker``.  Speaks
newline-delimited JSON on stdin and stdout.  Commands:

``{"cmd": "load", "path": ..., "entry": ...}``
    Import the model and report its signature.
``{"cmd": "price", "kwargs": {...}}``
    Call the model and report one price.

The real stdout file descriptor is claimed for the protocol before the model is
imported, and ``sys.stdout`` is redirected to the null device, so anything the
model prints cannot corrupt the channel.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import math
import os
import sys
import traceback
from typing import Any, Callable

_CHANNEL: io.TextIOWrapper | None = None
_MODEL: Callable[..., Any] | None = None


def _claim_stdout() -> io.TextIOWrapper:
    """Take the protocol channel and point ``sys.stdout`` at the null device."""
    fd = os.dup(1)
    channel = io.TextIOWrapper(os.fdopen(fd, "wb", 0), encoding="utf-8", write_through=True)
    devnull = open(os.devnull, "w", encoding="utf-8")
    os.dup2(devnull.fileno(), 1)
    sys.stdout = devnull
    return channel


def _apply_restrictions() -> None:
    """Best-effort in-process restrictions (NFR-09).

    This blocks accidents, not attacks: a determined model can undo every one
    of these.  The container image is the real isolation boundary.
    """
    # realpath, not abspath: on macOS the temporary directory is handed over as
    # /var/folders/... while the child's own getcwd() reports the resolved
    # /private/var/folders/... . Comparing the two unresolved would make the
    # sandbox refuse writes to the very directory it just granted.
    workdir = os.path.realpath(os.environ.get("PLUMBLINE_SANDBOX_WORKDIR", os.getcwd()))

    # -- no network ---------------------------------------------------------
    try:
        import socket

        def _blocked(*args: Any, **kwargs: Any):
            raise PermissionError("the Plumbline sandbox blocks network access")

        socket.socket = _blocked  # type: ignore[assignment]
        socket.create_connection = _blocked  # type: ignore[assignment]
    except Exception:  # pragma: no cover - socket is always importable
        pass

    # -- no writes outside the workdir --------------------------------------
    import builtins

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            try:
                target = os.path.realpath(os.fspath(file))
                inside = os.path.commonpath([target, workdir]) == workdir
            except (TypeError, ValueError):
                # A file descriptor, or a path on another drive letter.
                inside = not isinstance(file, (str, bytes, os.PathLike))
            if not inside:
                raise PermissionError(
                    f"the Plumbline sandbox allows writes only under {workdir}"
                )
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]

    # -- CPU and memory caps (POSIX only; Windows has no rlimit) ------------
    try:
        import resource

        limit = int(os.environ.get("PLUMBLINE_SANDBOX_MEMORY", 0))
        if limit > 0:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _load(path: str, entry: str) -> dict[str, Any]:
    global _MODEL

    if not os.path.isfile(path):
        return {"status": "ERROR", "message": f"no such model file: {path}"}

    spec = importlib.util.spec_from_file_location("plumbline_mut", path)
    if spec is None or spec.loader is None:
        return {"status": "ERROR", "message": f"{path} is not an importable Python module"}
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["plumbline_mut"] = module
        spec.loader.exec_module(module)
    except BaseException:
        return {
            "status": "ERROR",
            "message": f"the model raised while being imported:\n{traceback.format_exc()}",
        }

    candidate = getattr(module, entry, None)
    if not callable(candidate):
        exported = sorted(n for n in vars(module) if callable(getattr(module, n)) and not n.startswith("_"))
        return {
            "status": "ERROR",
            "message": (
                f"the model file defines no callable named {entry!r}; "
                f"callables found: {exported or 'none'}"
            ),
        }

    _MODEL = candidate
    try:
        signature = inspect.signature(candidate)
        names = list(signature.parameters)
        var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        var_positional = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()
        )
    except (TypeError, ValueError):
        names, var_keyword, var_positional = [], True, True

    return {
        "status": "OK",
        "signature": names,
        "var_keyword": var_keyword,
        "var_positional": var_positional,
    }


def _price(kwargs: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        return {"status": "ERROR", "message": "no model is loaded"}
    try:
        value = _MODEL(**kwargs)
    except BaseException as exc:
        return {
            "status": "ERROR",
            "message": f"{type(exc).__name__}: {exc}",
        }
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {
            "status": "ERROR",
            "message": f"the model returned {value!r}, which is not a number",
        }
    if not math.isfinite(value):
        return {"status": "ERROR", "message": f"the model returned {value}"}
    return {"status": "OK", "price": value}


def _send(payload: dict[str, Any]) -> None:
    assert _CHANNEL is not None
    _CHANNEL.write(json.dumps(payload) + "\n")
    _CHANNEL.flush()


def main() -> None:
    global _CHANNEL
    _CHANNEL = _claim_stdout()
    _apply_restrictions()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"status": "ERROR", "message": f"bad request: {exc}"})
            continue

        command = message.get("cmd")
        if command == "load":
            _send(_load(message.get("path", ""), message.get("entry", "price")))
        elif command == "price":
            _send(_price(message.get("kwargs", {})))
        elif command == "stop":
            break
        else:
            _send({"status": "ERROR", "message": f"unknown command {command!r}"})


if __name__ == "__main__":
    main()

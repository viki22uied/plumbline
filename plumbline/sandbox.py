"""Isolated execution of a Model Under Test (FR-A-05, FR-A-06, NFR-05, NFR-09).

The Model Under Test runs in a child process that speaks newline-delimited
JSON on its standard streams.  The parent never imports the model, so a
segfault, an infinite loop, a ``sys.exit`` or a memory blow-up in the model
cannot take the audit down: the parent kills the child, records the fault and
carries on with the next check.

Timeouts are enforced with a reader thread and a queue rather than with
``signal.alarm`` or ``select`` on a pipe, because neither works on Windows and
NFR-10 requires all three platforms.

**What the sandbox does and does not guarantee.**  Inside the child, the
worker blocks the socket module, blocks writes outside a private temporary
directory, and on POSIX applies CPU and address-space rlimits.  That stops an
honest model from misbehaving by accident, which is what NFR-09 is for.  It is
not an OS-level jail and it does not stop deliberately hostile code; run
untrusted models in the bundled container, which is.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

from plumbline.contracts import IngestionError, PriceResult

#: Seconds allowed for one call into the Model Under Test (FR-A-06).
DEFAULT_CALL_TIMEOUT = 10.0
#: Seconds allowed for the child to import the model and report its signature.
STARTUP_TIMEOUT = 30.0
#: Address-space cap for the child, in bytes (POSIX only).
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3


@dataclass
class SandboxConfig:
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES
    #: Directory the child may write to. A private temp dir is made if None.
    workdir: str | None = None


class SandboxError(IngestionError):
    """The sandbox could not start or could not be spoken to."""


class Sandbox:
    """A restartable child process that prices with the Model Under Test."""

    def __init__(self, model_path: str, entry: str = "price", config: SandboxConfig | None = None):
        self.model_path = os.path.abspath(model_path)
        self.entry = entry
        self.config = config or SandboxConfig()
        self._owns_workdir = self.config.workdir is None
        self.workdir = self.config.workdir or tempfile.mkdtemp(prefix="plumbline-mut-")
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._stderr_lines: list[str] = []
        self.signature: list[str] = []
        self.var_keyword = False
        self.timeouts = 0
        self.restarts = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self.restarts += 1

        self._lines = queue.Queue()
        # Remove sensitive variables from the child environment while keeping
        # the rest.  A whitelist approach breaks Python's own initialization on
        # some platforms (e.g. Windows needs variables that are hard to
        # enumerate exhaustively).  Blacklisting known-dangerous prefixes is
        # safer: it blocks the most common secret-bearing variables while
        # letting the interpreter start correctly.
        _SENSITIVE_PREFIXES = (
            "AWS_",
            "AZURE_",
            "GCP_",
            "GOOGLE_",
            "OPENAI_",
            "ANTHROPIC_",
            "GITHUB_",
            "GITLAB_",
            "HEROKU_",
            "SLACK_",
            "STRIPE_",
            "SENDGRID_",
            "Twilio",
            "MONGO",
            "DATABASE_",
            "DB_",
            "REDIS_",
            "SECRET",
            "TOKEN",
            "CREDENTIAL",
            "PASSWORD",
            "API_KEY",
            "PRIVATE",
        )
        _SENSITIVE_EXACT = frozenset({
            "HOMEPATH", "HOMEDRIVE",
        })
        env = {
            k: v for k, v in os.environ.items()
            if k not in _SENSITIVE_EXACT
            and not any(k.startswith(p) for p in _SENSITIVE_PREFIXES)
        }
        env["PYTHONUNBUFFERED"] = "1"
        env["PLUMBLINE_SANDBOX_WORKDIR"] = self.workdir
        env["PLUMBLINE_SANDBOX_MEMORY"] = str(self.config.memory_limit_bytes)
        # The child runs with its cwd inside the sandbox directory, so it needs
        # the parent's import path handed to it explicitly -- both to find
        # plumbline itself and to let the model import what the parent can.
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in sys.path if p and os.path.isdir(p)]
            + [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
        )
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "plumbline._worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=self.workdir,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._stderr_lines = []
        self._stderr_reader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_reader.start()

        reply = self._exchange(
            {"cmd": "load", "path": self.model_path, "entry": self.entry},
            timeout=STARTUP_TIMEOUT,
        )
        if reply.get("status") != "OK":
            self.stop()
            raise SandboxError(reply.get("message", "the sandbox could not load the model"))
        self.signature = list(reply.get("signature", []))
        self.var_keyword = bool(reply.get("var_keyword", False))

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        # The reader thread holds the stdout pipe until EOF reaches it, which
        # lands shortly after -- but not necessarily before -- wait() returns.
        # Joining keeps the process's handle state deterministic for whatever
        # the caller does next (the tests delete the work directory here).
        for thread in (self._reader,):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        if self._owns_workdir:
            # On Windows a just-terminated process's directory handle can
            # outlive wait() by a tick; deleting over it would fail silently.
            # A short bounded retry turns that race into a certainty without
            # ever making a live sandbox wait.
            for delay in (0.0, 0.05, 0.1, 0.2, 0.4):
                if delay:
                    time.sleep(delay)
                shutil.rmtree(self.workdir, ignore_errors=True)
                if not os.path.isdir(self.workdir):
                    return

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- calling ------------------------------------------------------------

    def call(self, kwargs: dict[str, Any]) -> PriceResult:
        """Price one parameter set. Never raises for a fault in the model."""
        started = time.perf_counter()
        try:
            self.start()
        except SandboxError as exc:
            return PriceResult(
                status="ERROR", message=str(exc), engine="mut", elapsed_s=0.0
            )

        reply = self._exchange(
            {"cmd": "price", "kwargs": kwargs}, timeout=self.config.call_timeout
        )
        elapsed = time.perf_counter() - started

        status = reply.get("status", "ERROR")
        if status == "TIMEOUT":
            self.timeouts += 1
        return PriceResult(
            price=float(reply["price"]) if status == "OK" else float("nan"),
            engine="mut",
            status=status,
            message=reply.get("message", ""),
            elapsed_s=elapsed,
        )

    # -- transport ----------------------------------------------------------

    def _pump(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                self._lines.put(line)
        except (ValueError, OSError):
            pass
        finally:
            self._lines.put(None)  # sentinel: the child's stream ended

    def _pump_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                self._stderr_lines.append(line)
        except (ValueError, OSError):
            pass

    def _exchange(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            return {"status": "ERROR", "message": "the sandbox process is not running"}
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            return {"status": "ERROR", "message": f"the sandbox stopped accepting input: {exc}"}

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.stop()
                return {
                    "status": "TIMEOUT",
                    "message": f"the Model Under Test did not answer within {timeout:g} s",
                }
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr_text = "".join(self._stderr_lines[-20:]) if self._stderr_lines else ""
                self.stop()
                return {
                    "status": "ERROR",
                    "message": "the Model Under Test crashed the sandbox process"
                    + (f"\nstderr:\n{stderr_text}" if stderr_text.strip() else ""),
                }
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # The model printed to stdout. Ignore the noise and keep reading.
                continue

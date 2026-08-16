"""Module E -- REST API (FR-E-02, FR-E-03, FR-E-04).

FastAPI generates the OpenAPI document from the endpoint signatures and the
response models below, so the documentation at ``/docs`` cannot fall behind the
code (FR-E-04).

Run it with::

    uvicorn plumbline.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

import plumbline
from plumbline.audit.checks import AuditConfig
from plumbline.audit.engine import audit_file
from plumbline.audit.grid import default_grid
from plumbline.audit.history import AuditHistory
from plumbline.contracts import INSTRUMENTS, MODELS, OptionSpec, Tolerance, PlumblineError
from plumbline.engines.registry import REGISTRY, ground_truth_price
from plumbline.report import render_markdown, write_report

#: Where uploaded models and rendered reports are kept between requests.
WORK_ROOT = os.environ.get(
    "PLUMBLINE_API_WORKDIR", os.path.join(tempfile.gettempdir(), "plumbline-api")
)

app = FastAPI(
    title="Plumbline",
    version=plumbline.__version__,
    description=(
        "Plumbline checks a derivative pricing model against known, correct "
        "mathematics. Submit a model, receive an Audit Report. Plumbline does "
        "not manage money, does not place trades, and does not give investment "
        "advice."
    ),
    contact={"name": "Plumbline", "url": "https://github.com/viki22uied/plumbline"},
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)


# ---------------------------------------------------------------------------
# response models
# ---------------------------------------------------------------------------


class EngineInfo(BaseModel):
    """One Ground Truth Engine and what it covers (FR-E-03)."""

    name: str
    version: str
    description: str
    reference: str
    instruments: list[str]
    models: list[str]
    priority: int
    deterministic: bool


class PriceRequest(BaseModel):
    # The vocabulary is validated by OptionSpec itself, which raises a message
    # naming the allowed values; repeating it here would be a second source of
    # truth that can fall out of step with the contract.
    instrument: str = Field("european", description=f"one of {', '.join(INSTRUMENTS)}")
    option_type: Literal["call", "put"] = "call"
    model: str = Field("bsm", description=f"one of {', '.join(MODELS)}")
    S: float = Field(100.0, gt=0, description="spot price of the underlying asset")
    K: float = Field(100.0, gt=0, description="strike price")
    T: float = Field(1.0, ge=0, description="time to expiry, in years")
    r: float = Field(0.05, description="continuously compounded risk-free rate")
    q: float = Field(0.0, description="continuously compounded dividend yield")
    sigma: float = Field(0.2, ge=0, description="volatility, annualised")
    barrier: float | None = None
    barrier_kind: str | None = None
    averaging: str = "geometric"
    payout: str = "cash"
    strike_type: str = "fixed"


class PriceResponse(BaseModel):
    price: float
    engine: str
    greeks: dict[str, float] | None = None
    case: str


class AuditSummary(BaseModel):
    audit_id: str
    badge: str
    audit_score: float
    headline: str
    duration_s: float
    report_json_url: str
    report_markdown_url: str
    report_pdf_url: str | None = None


class HistoryRow(BaseModel):
    audit_id: str
    model_name: str
    finished_at: str
    badge: str
    audit_score: float


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": plumbline.__version__}


@app.get("/engines", response_model=list[EngineInfo], tags=["engines"], summary="List the Ground Truth Engines")
def list_engines() -> list[dict[str, Any]]:
    """Every registered Ground Truth Engine and the instruments it covers."""
    return [engine.to_dict() for engine in REGISTRY.values()]


@app.post("/price", response_model=PriceResponse, tags=["engines"], summary="Price one contract")
def price(request: PriceRequest) -> dict[str, Any]:
    """Price one contract with the authoritative reference engine."""
    try:
        spec = OptionSpec(**request.model_dump())
        result = ground_truth_price(spec)
    except PlumblineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "price": result.price,
        "engine": result.engine,
        "greeks": result.greeks.to_dict() if result.greeks else None,
        "case": spec.label(),
    }


@app.post(
    "/audit",
    response_model=AuditSummary,
    tags=["audit"],
    summary="Submit a Model Under Test and receive an Audit Report",
)
async def submit_audit(
    file: UploadFile = File(..., description="a .py model file, or a .csv/.json price table"),
    entry: str = Query("price", description="name of the callable in a .py model"),
    instrument: str = Query("european"),
    checks: str = Query("1,2,3,4,5,6", description="comma separated check types"),
    tolerance: float = Query(1e-3, gt=0, description="relative tolerance"),
    timeout: float = Query(10.0, gt=0, description="seconds allowed per model call"),
) -> dict[str, Any]:
    """Run a full Audit on an uploaded model and return where the report is."""
    if instrument not in INSTRUMENTS:
        raise HTTPException(status_code=422, detail=f"instrument must be one of {INSTRUMENTS}")

    upload_dir = tempfile.mkdtemp(prefix="upload-", dir=_ensure_root())
    filename = os.path.basename(file.filename or "model.py")
    model_path = os.path.join(upload_dir, filename)
    with open(model_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        report = audit_file(
            model_path,
            entry=entry,
            grid=default_grid(instrument),
            tolerance=Tolerance(relative=tolerance),
            config=AuditConfig(),
            check_types=[int(part) for part in checks.split(",") if part.strip()],
            call_timeout=timeout,
        )
    except PlumblineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    out_dir = os.path.join(_ensure_root(), "reports", report.audit_id)
    paths = write_report(report, out_dir)
    AuditHistory(os.path.join(_ensure_root(), "history")).save(report)

    from plumbline.report.summary import headline

    return {
        "audit_id": report.audit_id,
        "badge": report.score.badge,
        "audit_score": report.score.total,
        "headline": headline(report),
        "duration_s": report.duration_s,
        "report_json_url": f"/audit/{report.audit_id}/report.json",
        "report_markdown_url": f"/audit/{report.audit_id}/report.md",
        "report_pdf_url": f"/audit/{report.audit_id}/report.pdf" if paths.pdf else None,
    }


@app.get("/audit/{audit_id}/report.json", tags=["audit"], summary="Machine-readable Audit Report")
def get_report_json(audit_id: str) -> dict[str, Any]:
    return AuditHistory(os.path.join(_ensure_root(), "history")).load(_safe_id(audit_id))


@app.get(
    "/audit/{audit_id}/report.md",
    response_class=PlainTextResponse,
    tags=["audit"],
    summary="Human-readable Audit Report",
)
def get_report_markdown(audit_id: str) -> str:
    path = _report_path(audit_id, ".md")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


@app.get("/audit/{audit_id}/report.pdf", tags=["audit"], summary="Audit Report as a PDF")
def get_report_pdf(audit_id: str) -> FileResponse:
    return FileResponse(_report_path(audit_id, ".pdf"), media_type="application/pdf")


@app.get("/history", response_model=list[HistoryRow], tags=["audit"], summary="Past audits")
def history(model_name: str | None = Query(None, description="filter by model name")) -> list[dict[str, Any]]:
    """Every stored Audit Report, so one model can be compared over time."""
    store = AuditHistory(os.path.join(_ensure_root(), "history"))
    entries = store.for_model(model_name) if model_name else store.entries()
    return [entry.to_dict() for entry in entries]


# ---------------------------------------------------------------------------


def _ensure_root() -> str:
    os.makedirs(WORK_ROOT, exist_ok=True)
    return WORK_ROOT


def _safe_id(audit_id: str) -> str:
    if not audit_id.isalnum():
        raise HTTPException(status_code=422, detail="an audit id is alphanumeric")
    return audit_id


def _report_path(audit_id: str, extension: str) -> str:
    audit_id = _safe_id(audit_id)
    directory = os.path.join(_ensure_root(), "reports", audit_id)
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith(extension):
                return os.path.join(directory, name)
    raise HTTPException(status_code=404, detail=f"no {extension} report for audit {audit_id}")


__all__ = ["app", "render_markdown"]

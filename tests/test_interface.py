"""Module E -- the CLI and the REST API (FR-E-01 to FR-E-04, AC-06)."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from plumbline.api import app
from plumbline.cli import main

client = TestClient(app)


# ---------------------------------------------------------------------------
# FR-E-01: the command line interface
# ---------------------------------------------------------------------------


def test_fre01_one_command_runs_a_full_audit(good_model_path, tmp_path, capsys):
    exit_code = main(
        [
            "audit",
            good_model_path,
            "--out",
            str(tmp_path),
            "--checks",
            "1,2,5,6",
            "--spots",
            "100",
            "--strikes",
            "100",
            "--maturities",
            "1.0",
            "--vols",
            "0.2",
            "--no-history",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "BADGE PASS" in output
    assert "SCORE 100.00" in output
    assert any(name.endswith(".json") for name in os.listdir(tmp_path))
    assert any(name.endswith(".md") for name in os.listdir(tmp_path))
    assert any(name.endswith(".pdf") for name in os.listdir(tmp_path))


def test_fre01_a_failing_audit_returns_a_non_zero_exit_code(broken_model_path, tmp_path, capsys):
    """A build pipeline must be able to gate on the result."""
    exit_code = main(
        [
            "audit",
            broken_model_path,
            "--out",
            str(tmp_path),
            "--checks",
            "1",
            "--spots",
            "100",
            "--strikes",
            "100",
            "--maturities",
            "0.25",
            "--vols",
            "0.2",
            "--no-history",
        ]
    )

    assert exit_code == 1
    assert "BADGE FAIL" in capsys.readouterr().out


def test_fre01_the_cli_reports_a_bad_model_file_without_a_traceback(tmp_path, capsys):
    exit_code = main(["audit", str(tmp_path / "does_not_exist.py"), "--out", str(tmp_path)])

    assert exit_code == 2
    assert "plumbline:" in capsys.readouterr().err


def test_the_cli_lists_the_engines(capsys):
    assert main(["engines"]) == 0
    output = capsys.readouterr().out
    for name in ("analytic", "binomial_crr", "fdm_crank_nicolson", "heston_cf", "monte_carlo"):
        assert name in output


def test_the_cli_lists_the_engines_as_json(capsys):
    assert main(["engines", "--json"]) == 0
    engines = json.loads(capsys.readouterr().out)

    assert len(engines) == 5
    assert all({"name", "version", "instruments", "models"} <= set(e) for e in engines)


def test_the_cli_prices_one_contract(capsys):
    assert main(
        ["price", "--spot", "100", "--strike", "100", "--maturity", "1", "--rate", "0.05", "--vol", "0.2"]
    ) == 0
    output = capsys.readouterr().out

    assert "10.45058" in output
    assert "delta" in output and "rho" in output


def test_the_cli_history_command_reads_the_store(good_model_path, tmp_path, capsys):
    from plumbline.audit.engine import audit_file
    from plumbline.audit.grid import ParameterGrid
    from plumbline.audit.history import AuditHistory

    grid = ParameterGrid(spots=(100.0,), strikes=(100.0,), maturities=(1.0,), vols=(0.2,))
    report = audit_file(good_model_path, grid=grid, check_types=[1])
    AuditHistory(str(tmp_path)).save(report)

    assert main(["history", "--store", str(tmp_path)]) == 0
    assert report.model["name"] in capsys.readouterr().out


# ---------------------------------------------------------------------------
# FR-E-02, FR-E-03, FR-E-04: the REST API
# ---------------------------------------------------------------------------


def test_fre03_the_api_lists_every_engine_and_what_it_covers():
    response = client.get("/engines")

    assert response.status_code == 200
    engines = response.json()
    assert {engine["name"] for engine in engines} == {
        "analytic",
        "binomial_crr",
        "fdm_crank_nicolson",
        "heston_cf",
        "monte_carlo",
    }
    analytic = next(e for e in engines if e["name"] == "analytic")
    assert "european" in analytic["instruments"]
    assert analytic["version"] == "1.0.0"


def test_the_api_prices_one_contract():
    response = client.post(
        "/price",
        json={"instrument": "european", "option_type": "call", "S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["price"] == pytest.approx(10.450584, abs=1e-5)
    assert body["engine"] == "analytic"
    assert set(body["greeks"]) == {"delta", "gamma", "vega", "theta", "rho"}


def test_the_api_rejects_an_unknown_instrument_with_a_clear_message():
    response = client.post("/price", json={"instrument": "swaption"})

    assert response.status_code == 422
    assert "swaption" in response.json()["detail"]


def test_fre02_the_api_audits_an_uploaded_model(good_model_path):
    with open(good_model_path, "rb") as handle:
        response = client.post(
            "/audit",
            files={"file": ("good_model.py", handle, "text/x-python")},
            params={"checks": "1,2", "instrument": "european"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["badge"] == "PASS"
    assert body["audit_score"] == pytest.approx(100.0)
    assert body["report_json_url"].endswith("report.json")

    json_report = client.get(body["report_json_url"])
    assert json_report.status_code == 200
    assert json_report.json()["audit_id"] == body["audit_id"]

    markdown = client.get(body["report_markdown_url"])
    assert markdown.status_code == 200
    assert "Section 1" in markdown.text

    pdf = client.get(body["report_pdf_url"])
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"


def test_fre02_the_api_reports_an_invalid_model_as_a_client_error(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("def price(S):\n    return 1.0\n", encoding="utf-8")

    with open(path, "rb") as handle:
        response = client.post("/audit", files={"file": ("bad.py", handle, "text/x-python")})

    assert response.status_code == 422
    assert "signature" in response.json()["detail"]


def test_the_api_history_endpoint_lists_past_audits(good_model_path):
    with open(good_model_path, "rb") as handle:
        client.post(
            "/audit",
            files={"file": ("good_model.py", handle, "text/x-python")},
            params={"checks": "1"},
        )

    response = client.get("/history")

    assert response.status_code == 200
    assert response.json()


def test_fre04_the_openapi_document_is_generated_from_the_code():
    """FR-E-04: the documentation cannot fall behind the endpoints."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Plumbline"
    for path in ("/engines", "/price", "/audit", "/history"):
        assert path in document["paths"]
    assert document["info"]["license"]["name"] == "Apache-2.0"


def test_the_api_has_a_health_probe():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_api_rejects_a_path_traversal_attempt_in_an_audit_id():
    response = client.get("/audit/..%2F..%2Fetc/report.md")
    assert response.status_code in (404, 422)

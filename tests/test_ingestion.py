"""Module A -- ingestion and the sandbox (FR-A-01 to FR-A-06, NFR-05, NFR-09)."""

from __future__ import annotations

import json
import math
import os
import textwrap

import pytest

from conftest import write_model
from plumbline.contracts import IngestionError, OptionSpec
from plumbline.ingestion import PythonModel, TableModel, check_signature, load_model
from plumbline.sandbox import Sandbox, SandboxConfig

SPEC = OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2)

MINIMAL = textwrap.dedent(
    """
    def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
        return max(S - K, 0.0)
    """
)


def _model(tmp_path, name: str, source: str, **kwargs) -> PythonModel:
    return load_model(write_model(tmp_path, name, textwrap.dedent(source)), **kwargs)


# ---------------------------------------------------------------------------
# FR-A-01, FR-A-03, FR-A-04: the signature contract
# ---------------------------------------------------------------------------


def test_fra01_a_python_model_with_the_right_signature_loads_and_prices(tmp_path):
    with _model(tmp_path, "ok.py", MINIMAL) as mut:
        assert mut.price(SPEC).price == pytest.approx(0.0)
        assert mut.price(SPEC.with_(S=120)).price == pytest.approx(20.0)
        assert "price(" in mut.signature_description


def test_fra03_a_missing_required_parameter_is_named(tmp_path):
    """FR-A-04: the error must say which parameter is missing."""
    source = "def price(instrument, option_type, S, K, T, r, **kwargs):\n    return 1.0\n"
    with pytest.raises(IngestionError) as info:
        _model(tmp_path, "missing.py", source)

    message = str(info.value)
    assert "q" in message and "sigma" in message
    assert "required parameters" in message


def test_fra03_a_missing_kwargs_parameter_is_named(tmp_path):
    source = "def price(instrument, option_type, S, K, T, r, q, sigma):\n    return 1.0\n"
    with pytest.raises(IngestionError) as info:
        _model(tmp_path, "no_kwargs.py", source)

    assert "**kwargs" in str(info.value)


def test_fra04_a_file_without_the_entry_point_lists_what_it_did_find(tmp_path):
    source = "def valuation(**kwargs):\n    return 1.0\n"
    with pytest.raises(IngestionError) as info:
        _model(tmp_path, "wrong_name.py", source)

    assert "price" in str(info.value) and "valuation" in str(info.value)


def test_a_model_that_raises_on_import_is_reported_not_propagated(tmp_path):
    source = "raise RuntimeError('this model is broken at import time')\n"
    with pytest.raises(IngestionError) as info:
        _model(tmp_path, "bad_import.py", source)

    assert "broken at import time" in str(info.value)


def test_check_signature_accepts_the_documented_contract():
    check_signature(
        ["instrument", "option_type", "S", "K", "T", "r", "q", "sigma", "kwargs"],
        var_keyword=True,
    )


def test_load_model_rejects_an_unknown_file_type(tmp_path):
    path = write_model(tmp_path, "model.xlsx", "not a model")
    with pytest.raises(IngestionError):
        load_model(path)


def test_load_model_reports_a_missing_file():
    with pytest.raises(IngestionError):
        load_model("no-such-model-file.py")


# ---------------------------------------------------------------------------
# FR-A-02: table models
# ---------------------------------------------------------------------------


def test_fra02_a_csv_price_table_is_accepted(tmp_path):
    csv = "instrument,option_type,S,K,T,r,q,sigma,price\neuropean,call,100,100,1.0,0.05,0.0,0.2,10.450584\n"
    path = write_model(tmp_path, "prices.csv", csv)

    with load_model(path) as mut:
        assert isinstance(mut, TableModel)
        assert mut.price(SPEC).price == pytest.approx(10.450584)


def test_fra02_a_json_price_table_is_accepted(tmp_path):
    rows = [
        {
            "instrument": "european",
            "option_type": "call",
            "S": 100,
            "K": 100,
            "T": 1.0,
            "r": 0.05,
            "q": 0.0,
            "sigma": 0.2,
            "price": 10.450584,
        }
    ]
    path = write_model(tmp_path, "prices.json", json.dumps(rows))

    with load_model(path) as mut:
        assert mut.price(SPEC).price == pytest.approx(10.450584)


def test_fra02_a_parameter_set_with_no_row_is_reported_as_not_priced(tmp_path):
    csv = "instrument,option_type,S,K,T,r,q,sigma,price\neuropean,call,100,100,1.0,0.05,0.0,0.2,10.45\n"
    path = write_model(tmp_path, "sparse.csv", csv)

    with load_model(path) as mut:
        result = mut.price(SPEC.with_(S=123.0))

    assert result.status == "NOT_PRICED"
    assert math.isnan(result.price)


def test_fra02_a_table_without_a_price_column_is_rejected(tmp_path):
    path = write_model(tmp_path, "no_price.csv", "instrument,S\neuropean,100\n")
    with pytest.raises(IngestionError):
        load_model(path)


def test_fra02_an_empty_table_is_rejected(tmp_path):
    path = write_model(tmp_path, "empty.json", "[]")
    with pytest.raises(IngestionError):
        load_model(path)


# ---------------------------------------------------------------------------
# FR-A-05, FR-A-06, NFR-05: isolation, timeouts and faults
# ---------------------------------------------------------------------------


def test_fra05_a_model_that_kills_its_process_does_not_kill_the_audit(tmp_path):
    source = """
        import os
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            if S > 100.0:
                os._exit(1)
            return 1.0
    """
    with _model(tmp_path, "suicidal.py", source) as mut:
        assert mut.price(SPEC).status == "OK"
        crashed = mut.price(SPEC.with_(S=120.0))
        assert crashed.status == "ERROR"
        # The sandbox restarts, so the audit carries on with the next case.
        assert mut.price(SPEC).status == "OK"


def test_fra06_a_slow_model_is_recorded_as_a_timeout(tmp_path):
    source = """
        import time
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            time.sleep(30.0)
            return 1.0
    """
    with _model(tmp_path, "slow.py", source, call_timeout=1.0) as mut:
        result = mut.price(SPEC)

    assert result.status == "TIMEOUT"
    assert "did not answer within" in result.message
    assert result.elapsed_s >= 1.0


def test_a_model_that_raises_is_reported_with_its_exception_type(tmp_path):
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            return 1.0 / 0.0
    """
    with _model(tmp_path, "raises.py", source) as mut:
        result = mut.price(SPEC)

    assert result.status == "ERROR"
    assert "ZeroDivisionError" in result.message


def test_a_model_that_returns_a_non_number_is_rejected(tmp_path):
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            return "about ten dollars"
    """
    with _model(tmp_path, "not_a_number.py", source) as mut:
        result = mut.price(SPEC)

    assert result.status == "ERROR"
    assert "not a number" in result.message


def test_a_model_that_returns_nan_is_rejected(tmp_path):
    source = """
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            return float("nan")
    """
    with _model(tmp_path, "nan.py", source) as mut:
        assert mut.price(SPEC).status == "ERROR"


def test_a_model_that_prints_does_not_corrupt_the_protocol(tmp_path):
    """The child claims the real stdout, so model output cannot break parsing."""
    source = """
        import sys
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            print("progress: pricing", S)
            sys.stdout.write("more noise\\n")
            return 7.5
    """
    with _model(tmp_path, "chatty.py", source) as mut:
        assert mut.price(SPEC).price == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# NFR-09: the sandbox restrictions
# ---------------------------------------------------------------------------


def test_nfr09_the_sandbox_blocks_network_access(tmp_path):
    source = """
        import socket
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            socket.socket()
            return 1.0
    """
    with _model(tmp_path, "networked.py", source) as mut:
        result = mut.price(SPEC)

    assert result.status == "ERROR"
    assert "network" in result.message


def test_nfr09_the_sandbox_blocks_writes_outside_its_working_directory(tmp_path):
    escape = os.path.join(str(tmp_path), "escaped.txt").replace("\\", "\\\\")
    source = f"""
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            with open(r"{escape}", "w") as handle:
                handle.write("this must not be written")
            return 1.0
    """
    with _model(tmp_path, "writer.py", source) as mut:
        result = mut.price(SPEC)

    assert result.status == "ERROR"
    assert "writes only under" in result.message
    assert not os.path.exists(os.path.join(str(tmp_path), "escaped.txt"))


def test_nfr09_the_sandbox_allows_writes_inside_its_working_directory(tmp_path):
    source = """
        import os
        def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
            with open(os.path.join(os.getcwd(), "scratch.txt"), "w") as handle:
                handle.write("a model may use scratch space")
            return 3.0
    """
    with _model(tmp_path, "scratch.py", source) as mut:
        assert mut.price(SPEC).price == pytest.approx(3.0)


def test_the_sandbox_cleans_up_its_working_directory(tmp_path):
    path = write_model(tmp_path, "tidy.py", MINIMAL)
    sandbox = Sandbox(path, config=SandboxConfig(call_timeout=5.0))
    sandbox.start()
    workdir = sandbox.workdir
    assert os.path.isdir(workdir)

    sandbox.close()
    assert not os.path.isdir(workdir)


def test_describe_reports_what_the_report_needs(tmp_path):
    with _model(tmp_path, "describe.py", MINIMAL) as mut:
        described = mut.describe()

    assert described["kind"] == "PythonModel"
    assert described["name"].endswith("describe.py:price")
    assert "call_timeout_s" in described
    assert described["supports_precision"] is True

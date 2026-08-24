"""Shared fixtures and test lanes for the Plumbline suite.

The suite is tiered, because the cost of a test is not the arithmetic in it.
A pure pricing test runs in single-digit milliseconds. A test that audits a
model has to start a sandbox, which is a Python interpreter that imports the
engine stack, and that costs about a second before any pricing happens. There
are roughly sixty-five such spawns here, so they, and not the mathematics,
set the wall time.

Three lanes, marked automatically by file so nobody has to remember:

``fast``         pure numerics and the external oracle. No subprocess, no
                 HTTP, no PDF. This is the lane to run while editing.
``integration``  ingestion, the sandbox, the audit engine, reports, the CLI
                 and the API. Correct to run before committing.
``native``       the optional C++ backend, skipped when it is not built.

    pytest -m fast                     # the inner loop
    pytest -m "not slow"               # everything but the long simulations
    pytest                             # the lot, as CI runs it
"""

from __future__ import annotations

import os

import pytest

from plumbline.contracts import OptionSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO_ROOT, "samples")

#: Which lane each test file belongs to. A file not listed here gets no lane
#: marker and therefore runs only in an unfiltered run, which is a loud enough
#: signal that a new file needs classifying.
LANES = {
    "test_ground_truth.py": "fast",
    "test_engines.py": "fast",
    "test_external_oracle.py": "fast",
    "test_checks.py": "integration",
    "test_ingestion.py": "integration",
    "test_report.py": "integration",
    "test_interface.py": "integration",
    "test_acceptance.py": "integration",
    "test_native_backend.py": "native",
}


def pytest_collection_modifyitems(config, items):
    """Tag every test with its lane, derived from the file it lives in.

    A test already marked ``slow`` gets no lane. Otherwise a single long
    simulation living in a fast-lane file would be pulled into the inner loop
    by ``-m fast`` and dominate it -- which is exactly what one 400,000-path
    Heston cross-check was doing, at twenty of the lane's twenty-nine seconds.
    Slow tests stay reachable through an unfiltered run or ``-m slow``.
    """
    for item in items:
        if item.get_closest_marker("slow"):
            continue
        lane = LANES.get(os.path.basename(str(item.fspath)))
        if lane:
            item.add_marker(getattr(pytest.mark, lane))


@pytest.fixture(scope="session")
def samples_dir() -> str:
    return SAMPLES


@pytest.fixture(scope="session")
def good_model_path() -> str:
    return os.path.join(SAMPLES, "good_model.py")


@pytest.fixture(scope="session")
def broken_model_path() -> str:
    return os.path.join(SAMPLES, "broken_model.py")


@pytest.fixture(scope="session")
def mc_model_path() -> str:
    return os.path.join(SAMPLES, "mc_model.py")


@pytest.fixture(scope="session")
def biased_mc_model_path() -> str:
    return os.path.join(SAMPLES, "biased_mc_model.py")


@pytest.fixture
def vanilla() -> OptionSpec:
    """The reference contract used across the suite."""
    return OptionSpec("european", "call", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)


def write_model(directory: str, name: str, source: str) -> str:
    """Write a throwaway Model Under Test and return its path."""
    path = os.path.join(str(directory), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return path

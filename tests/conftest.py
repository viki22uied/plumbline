"""Shared fixtures for the Plumbline test suite."""

from __future__ import annotations

import os

import pytest

from plumbline.contracts import OptionSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO_ROOT, "samples")


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

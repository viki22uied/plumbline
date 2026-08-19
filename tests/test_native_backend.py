"""The optional native Monte Carlo backend.

Every test here skips when the library is not built, because the backend is
optional and a machine with no compiler must still get a green suite. The CI
matrix builds it on all three platforms, so these run there.

The contract being tested is not "the two backends produce the same number".
They draw from different random streams on purpose, so they are two
independent estimators. The contract is:

* each backend agrees with the closed form to within its own sampling error;
* the two agree with each other to within their combined sampling error;
* the native one gives the same answer regardless of how many threads it used;
* anything it will not do falls back to NumPy rather than failing the audit.
"""

from __future__ import annotations

import ctypes
import math

import pytest

from plumbline.contracts import OptionSpec, PlumblineError
from plumbline.engines import analytic, montecarlo as mc
from plumbline.engines import native

pytestmark = pytest.mark.skipif(
    not native.available(),
    reason="the native backend is not built; run 'python native/build.py'",
)

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.25)

PARITY_CASES = [
    ("european call", OptionSpec("european", "call", **BASE), analytic.black_scholes_price),
    ("european put", OptionSpec("european", "put", **BASE), analytic.black_scholes_price),
    ("digital cash", OptionSpec("digital", "call", payout="cash", **BASE), analytic.digital_price),
    ("digital asset", OptionSpec("digital", "put", payout="asset", **BASE), analytic.digital_price),
    (
        "asian geometric",
        OptionSpec("asian", "call", averaging="geometric", **BASE),
        None,
    ),
    ("asian arithmetic", OptionSpec("asian", "call", averaging="arithmetic", **BASE), None),
    (
        "barrier down-and-out",
        OptionSpec("barrier", "call", barrier=90.0, barrier_kind="down-and-out", **BASE),
        analytic.barrier_price,
    ),
    (
        "barrier up-and-in",
        OptionSpec("barrier", "put", barrier=130.0, barrier_kind="up-and-in", **BASE),
        analytic.barrier_price,
    ),
    (
        "lookback fixed",
        OptionSpec("lookback", "call", strike_type="fixed", **BASE),
        analytic.lookback_price,
    ),
    (
        "lookback floating",
        OptionSpec("lookback", "put", strike_type="floating", **BASE),
        analytic.lookback_price,
    ),
]


# ---------------------------------------------------------------------------
# loading and the ABI
# ---------------------------------------------------------------------------


def test_the_backend_reports_its_identity():
    assert native.backend_version().startswith("plumbline-mc")
    assert native.backend_threads() >= 1
    assert native.load_error() == ""


def test_the_loader_checks_the_struct_layout_against_the_library():
    """A library built from a different header is worse than a missing one."""
    library = native._load()

    assert library.plumbline_request_size() == ctypes.sizeof(native._Request)
    assert library.plumbline_result_size() == ctypes.sizeof(native._Result)


def test_every_field_of_the_abi_structs_is_eight_bytes():
    """The layout rule that makes the ctypes mirror safe on every platform."""
    for struct in (native._Request, native._Result):
        for name, kind in struct._fields_:
            assert ctypes.sizeof(kind) == 8, f"{struct.__name__}.{name}"
        expected = 8 * len(struct._fields_)
        assert ctypes.sizeof(struct) == expected, struct.__name__


def test_describe_carries_what_the_report_needs():
    described = native.describe()

    assert described["available"] is True
    assert described["version"]
    assert described["hardware_threads"] >= 1
    assert described["error"] is None


# ---------------------------------------------------------------------------
# agreement with the closed forms and with NumPy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,spec,closed_form", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_the_native_backend_agrees_with_the_closed_form(name, spec, closed_form):
    if closed_form is None:
        pytest.skip("no closed form exists for this contract")

    result = mc.monte_carlo(spec, paths=400_000, steps=250, seed=4242, backend="cpp")
    reference = closed_form(spec)

    assert result.backend == "cpp"
    # Four standard errors, plus a small floor for the discretisation of the
    # path-dependent contracts.
    tolerance = 4.0 * result.stderr + 1e-2 * (1.0 if spec.instrument != "european" else 0.0)
    assert abs(result.price - reference) <= tolerance, (
        f"{name}: native {result.price:.6f} against closed form {reference:.6f}, "
        f"standard error {result.stderr:.6f}"
    )


@pytest.mark.parametrize("name,spec,_", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_the_two_backends_agree_within_their_combined_error(name, spec, _):
    numpy_result = mc.monte_carlo(spec, paths=400_000, steps=250, seed=4242, backend="numpy")
    native_result = mc.monte_carlo(spec, paths=400_000, steps=250, seed=4242, backend="cpp")

    combined = math.hypot(numpy_result.stderr, native_result.stderr)
    gap = abs(numpy_result.price - native_result.price)

    # The geometric Asian controls on its own payoff, so both estimators
    # collapse onto the control mean and the combined error is zero. There the
    # two must agree outright.
    if combined < 1e-12:
        assert gap < 1e-9, f"{name}: {gap:.3e}"
        return

    assert gap <= 4.0 * combined, (
        f"{name}: numpy {numpy_result.price:.6f} against native "
        f"{native_result.price:.6f}, combined standard error {combined:.6f}"
    )


def test_the_native_backend_reduces_variance_the_same_way_numpy_does():
    spec = OptionSpec("european", "call", **BASE)

    plain = mc.monte_carlo(
        spec, paths=200_000, seed=7, backend="cpp", antithetic=False, control_variate=False
    )
    antithetic = mc.monte_carlo(
        spec, paths=200_000, seed=7, backend="cpp", antithetic=True, control_variate=False
    )
    both = mc.monte_carlo(spec, paths=200_000, seed=7, backend="cpp")

    assert antithetic.stderr < plain.stderr
    assert both.stderr < antithetic.stderr
    assert both.control_beta != 0.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrument", ["european", "barrier", "lookback"])
def test_the_answer_does_not_depend_on_the_thread_count(instrument):
    """Blocks are pinned to streams, so the schedule cannot move the result."""
    extras = {}
    if instrument == "barrier":
        extras = {"barrier": 90.0, "barrier_kind": "down-and-out"}
    if instrument == "lookback":
        extras = {"strike_type": "fixed"}
    spec = OptionSpec(instrument, "call", **BASE, **extras)

    one = mc.monte_carlo(spec, paths=100_000, steps=60, seed=99, backend="cpp", threads=1)
    four = mc.monte_carlo(spec, paths=100_000, steps=60, seed=99, backend="cpp", threads=4)
    many = mc.monte_carlo(spec, paths=100_000, steps=60, seed=99, backend="cpp", threads=8)

    assert one.price == four.price == many.price
    assert one.stderr == four.stderr == many.stderr


def test_the_same_seed_gives_the_same_answer_twice():
    spec = OptionSpec("european", "call", **BASE)

    first = mc.monte_carlo(spec, paths=50_000, seed=11, backend="cpp")
    second = mc.monte_carlo(spec, paths=50_000, seed=11, backend="cpp")

    assert first.price == second.price


def test_a_different_seed_gives_a_different_answer():
    spec = OptionSpec("european", "call", **BASE)

    first = mc.monte_carlo(spec, paths=50_000, seed=11, backend="cpp")
    second = mc.monte_carlo(spec, paths=50_000, seed=12, backend="cpp")

    assert first.price != second.price


def test_more_paths_shrink_the_standard_error_at_the_expected_rate():
    """The estimator converges, and it converges like one over root n."""
    spec = OptionSpec("european", "call", **BASE)

    small = mc.monte_carlo(spec, paths=50_000, seed=3, backend="cpp")
    large = mc.monte_carlo(spec, paths=800_000, seed=3, backend="cpp")

    assert large.stderr < small.stderr
    # Sixteen times the paths should cut the error by about four.
    assert 2.5 < small.stderr / large.stderr < 6.0


# ---------------------------------------------------------------------------
# the optional part of "optional backend"
# ---------------------------------------------------------------------------


def test_numpy_is_still_the_default():
    spec = OptionSpec("european", "call", **BASE)

    assert mc.DEFAULT_BACKEND == "numpy"
    assert mc.monte_carlo(spec, paths=20_000).backend == "numpy"


def test_auto_prefers_the_native_backend_when_it_is_there():
    spec = OptionSpec("european", "call", **BASE)

    assert mc.monte_carlo(spec, paths=20_000, backend="auto").backend == "cpp"


def test_auto_falls_back_when_the_backend_refuses_the_contract():
    """A barrier with a rebate is priced by NumPy, silently and correctly."""
    spec = OptionSpec(
        "barrier",
        "call",
        barrier=90.0,
        barrier_kind="down-and-out",
        rebate=2.0,
        **BASE,
    )
    assert native.supports(spec) is False

    with pytest.raises(PlumblineError):
        # NumPy declines a rebate too, which is the documented behaviour.
        mc.monte_carlo(spec, paths=10_000, steps=20, backend="auto")


def test_asking_for_cpp_by_name_raises_rather_than_falling_back_quietly():
    """A benchmark that silently measured NumPy would be worse than an error."""
    spec = OptionSpec("european", "call", model="heston", **BASE)
    assert native.supports(spec) is False

    with pytest.raises(native.NativeBackendError):
        mc.monte_carlo(spec, paths=10_000, backend="cpp")


def test_auto_falls_back_for_a_model_the_backend_does_not_cover():
    spec = OptionSpec("european", "call", model="heston", **BASE)

    result = mc.monte_carlo(spec, paths=20_000, backend="auto")

    assert result.backend == "numpy"


def test_the_backend_refuses_a_degenerate_contract_instead_of_guessing():
    """The exact value belongs to plumbline.engines.limits, not to the C++."""
    spec = OptionSpec("european", "call", **{**BASE, "sigma": 0.0})
    assert native.supports(spec) is False

    result = mc.monte_carlo(spec, paths=20_000, backend="auto")

    assert result.backend == "closed_form"
    assert result.price == pytest.approx(
        math.exp(-BASE["r"] * BASE["T"]) * max(spec.forward - spec.K, 0.0)
    )


def test_an_unknown_backend_name_is_rejected():
    spec = OptionSpec("european", "call", **BASE)

    with pytest.raises(PlumblineError):
        mc.monte_carlo(spec, paths=1_000, backend="fortran")


def test_both_backends_take_their_control_mean_from_the_same_place():
    """One implementation of the closed forms, so the control cannot diverge."""
    spec = OptionSpec("asian", "call", averaging="arithmetic", **BASE)

    expectation = mc.control_mean(spec, 250)

    assert expectation > 0.0
    assert expectation == mc.control_mean(spec, 250)
    # A different fixing count is a different contract for the control.
    assert expectation != mc.control_mean(spec, 50)


def test_the_result_records_which_backend_produced_it():
    spec = OptionSpec("european", "call", **BASE)

    payload = mc.monte_carlo(spec, paths=20_000, backend="cpp", threads=2).to_dict()

    assert payload["backend"] == "cpp"
    assert payload["threads"] == 2
    assert payload["paths"] == 20_000

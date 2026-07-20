"""A correct Model Under Test, built from the Plumbline Ground Truth engines.

This is the AC-05 sample: submitted to Plumbline, it must return a full PASS on
every check.  It exists to prove the audit does not flag a correct model, which
is exactly as important as flagging a broken one -- an auditor that fails
everything is as useless as one that passes everything.
"""

from plumbline.contracts import OptionSpec
from plumbline.engines.registry import ground_truth_price


def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs):
    """Price one contract. See plumbline.contracts for the full signature."""
    kwargs.pop("instrument", None)
    kwargs.pop("option_type", None)
    spec = OptionSpec(
        instrument=instrument,
        option_type=option_type,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        **kwargs,
    )
    return ground_truth_price(spec).price

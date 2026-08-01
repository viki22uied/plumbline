"""Module E -- Command Line Interface (FR-E-01).

One command runs a full Audit::

    plumbline audit path/to/model.py

Other subcommands exist for the pieces a user needs around that: listing the
Ground Truth Engines, pricing one contract with a reference engine, and reading
the audit history of a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

import plumbline
from plumbline.audit.checks import AuditConfig
from plumbline.audit.engine import audit_file
from plumbline.audit.grid import default_grid
from plumbline.audit.history import AuditHistory
from plumbline.contracts import INSTRUMENTS, MODELS, OptionSpec, Tolerance, PlumblineError
from plumbline.engines.registry import REGISTRY, ground_truth_price
from plumbline.report import write_report

EXIT_OK = 0
EXIT_AUDIT_FAILED = 1
EXIT_USAGE = 2

BADGE_EXIT = {"PASS": EXIT_OK, "PARTIAL": EXIT_AUDIT_FAILED, "FAIL": EXIT_AUDIT_FAILED}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plumbline",
        description=(
            "Plumbline checks a derivative pricing model against known, correct "
            "mathematics, and writes an Audit Report."
        ),
    )
    parser.add_argument("--version", action="version", version=f"plumbline {plumbline.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="run a full Audit on a Model Under Test")
    audit.add_argument("model", help="path to a .py model file, or a .csv/.json price table")
    audit.add_argument("--entry", default="price", help="name of the callable in a .py model")
    audit.add_argument("--instrument", default="european", choices=list(INSTRUMENTS))
    audit.add_argument("--model-type", default="bsm", choices=list(MODELS), dest="model_type")
    audit.add_argument(
        "--checks",
        default="1,2,3,4,5,6",
        help="comma separated check types to run, for example 1,2,5",
    )
    audit.add_argument("--tolerance", type=float, default=1e-3, help="relative tolerance")
    audit.add_argument(
        "--mc-tolerance",
        type=float,
        default=1e-2,
        dest="mc_tolerance",
        help="relative tolerance where the reference engine is a simulation",
    )
    audit.add_argument("--timeout", type=float, default=10.0, help="seconds per model call")
    audit.add_argument("--out", default="reports", help="directory for the Audit Report")
    audit.add_argument(
        "--formats",
        default="json,markdown,pdf",
        help="report formats to write, from json, markdown, pdf",
    )
    audit.add_argument("--spots", help="comma separated spot prices for the grid")
    audit.add_argument("--strikes", help="comma separated strikes for the grid")
    audit.add_argument("--maturities", help="comma separated times to expiry, in years")
    audit.add_argument("--vols", help="comma separated volatilities")
    audit.add_argument("--rates", help="comma separated risk-free rates")
    audit.add_argument("--dividends", help="comma separated dividend yields")
    audit.add_argument("--barrier", type=float, help="barrier level, for a barrier audit")
    audit.add_argument("--barrier-kind", dest="barrier_kind", help="for example up-and-out")
    audit.add_argument("--no-history", action="store_true", help="do not store this audit")
    audit.add_argument("--quiet", action="store_true", help="print the summary only")

    engines = subparsers.add_parser("engines", help="list the Ground Truth Engines")
    engines.add_argument("--json", action="store_true", help="print machine-readable output")

    price = subparsers.add_parser("price", help="price one contract with a reference engine")
    price.add_argument("--instrument", default="european", choices=list(INSTRUMENTS))
    price.add_argument("--type", default="call", choices=("call", "put"), dest="option_type")
    price.add_argument("--spot", type=float, required=True)
    price.add_argument("--strike", type=float, required=True)
    price.add_argument("--maturity", type=float, required=True)
    price.add_argument("--rate", type=float, default=0.0)
    price.add_argument("--dividend", type=float, default=0.0)
    price.add_argument("--vol", type=float, default=0.2)
    price.add_argument("--barrier", type=float)
    price.add_argument("--barrier-kind", dest="barrier_kind")
    price.add_argument("--model-type", default="bsm", choices=list(MODELS), dest="model_type")
    price.add_argument("--json", action="store_true")

    history = subparsers.add_parser("history", help="list stored Audit Reports")
    history.add_argument("--model", help="show only the audits of this model name")
    history.add_argument("--store", help="path to the history directory")
    history.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "engines":
            return _cmd_engines(args)
        if args.command == "price":
            return _cmd_price(args)
        if args.command == "history":
            return _cmd_history(args)
    except PlumblineError as exc:
        print(f"plumbline: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"plumbline: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.print_help()
    return EXIT_USAGE


# ---------------------------------------------------------------------------


def _floats(text: str | None) -> tuple[float, ...] | None:
    if not text:
        return None
    return tuple(float(part) for part in text.split(",") if part.strip())


def _cmd_audit(args: Any) -> int:
    overrides: dict[str, Any] = {}
    for name, value in (
        ("spots", _floats(args.spots)),
        ("strikes", _floats(args.strikes)),
        ("maturities", _floats(args.maturities)),
        ("vols", _floats(args.vols)),
        ("rates", _floats(args.rates)),
        ("dividends", _floats(args.dividends)),
    ):
        if value:
            overrides[name] = value
    if args.model_type != "bsm":
        overrides["model"] = args.model_type

    extras: dict[str, Any] = {}
    if args.barrier is not None:
        extras["barrier"] = args.barrier
    if args.barrier_kind:
        extras["barrier_kind"] = args.barrier_kind
    if extras:
        overrides["extras"] = extras

    grid = default_grid(args.instrument, **overrides)
    tolerance = Tolerance(relative=args.tolerance, stochastic_relative=args.mc_tolerance)
    check_types = [int(part) for part in args.checks.split(",") if part.strip()]

    report = audit_file(
        args.model,
        entry=args.entry,
        grid=grid,
        tolerance=tolerance,
        config=AuditConfig(),
        check_types=check_types,
        call_timeout=args.timeout,
    )

    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
    paths = write_report(report, args.out, formats=formats)

    if not args.no_history:
        AuditHistory().save(report)

    _print_audit_summary(report, paths, quiet=args.quiet)
    return BADGE_EXIT.get(report.score.badge, EXIT_AUDIT_FAILED)


def _print_audit_summary(report: Any, paths: Any, quiet: bool) -> None:
    from plumbline.report.summary import headline

    print()
    print(f"  Plumbline {report.plumbline_version}   audit {report.audit_id}")
    print(f"  model: {report.model.get('name')}")
    print()
    print(f"  BADGE {report.score.badge}    SCORE {report.score.total:.2f} / 100")
    print()
    if not quiet:
        print(f"  {'#':>2}  {'check type':38s} {'pass':>6} {'fail':>6} {'err':>5} {'skip':>5}")
        for bucket in report.score.per_check:
            if not bucket.ran and bucket.skipped == 0:
                continue
            print(
                f"  {bucket.check_type:>2}  {bucket.name:38s} {bucket.passes:>6} "
                f"{bucket.failures:>6} {bucket.errors:>5} {bucket.skipped:>5}"
            )
        print()
    print(f"  {headline(report)}")
    print()
    for label, path in (("json", paths.json), ("markdown", paths.markdown), ("pdf", paths.pdf)):
        if path:
            print(f"  report ({label}): {_display_path(path)}")
    print()


def _display_path(path: str) -> str:
    """A path relative to the working directory, where one exists.

    On Windows a report can land on a different drive letter from the working
    directory, and there is no relative path between two drives.
    """
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def _cmd_engines(args: Any) -> int:
    engines = [engine.to_dict() for engine in REGISTRY.values()]
    if args.json:
        print(json.dumps(engines, indent=2))
        return EXIT_OK
    print()
    print(f"  {'engine':22s} {'ver':6s} {'instruments':46s} models")
    for engine in engines:
        print(
            f"  {engine['name']:22s} {engine['version']:6s} "
            f"{', '.join(engine['instruments']):46s} {', '.join(engine['models'])}"
        )
    print()
    for engine in engines:
        print(f"  {engine['name']}: {engine['description']}")
        print(f"      reference: {engine['reference']}")
    print()
    return EXIT_OK


def _cmd_price(args: Any) -> int:
    spec = OptionSpec(
        instrument=args.instrument,
        option_type=args.option_type,
        model=args.model_type,
        S=args.spot,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        q=args.dividend,
        sigma=args.vol,
        barrier=args.barrier,
        barrier_kind=args.barrier_kind,
    )
    result = ground_truth_price(spec)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return EXIT_OK
    print()
    print(f"  {spec.label()}")
    print(f"  engine: {result.engine}")
    print(f"  price:  {result.price:.8f}")
    if result.greeks:
        for name, value in result.greeks.to_dict().items():
            print(f"  {name:6s}: {value: .8f}")
    print()
    return EXIT_OK


def _cmd_history(args: Any) -> int:
    history = AuditHistory(args.store)
    entries = history.for_model(args.model) if args.model else history.entries()
    if args.json:
        print(json.dumps([entry.to_dict() for entry in entries], indent=2))
        return EXIT_OK
    if not entries:
        print(f"  no stored audits under {history.root}")
        return EXIT_OK
    print()
    print(f"  {'finished':22s} {'badge':8s} {'score':>7}  model")
    for entry in sorted(entries, key=lambda e: e.finished_at):
        print(
            f"  {entry.finished_at:22s} {entry.badge:8s} {entry.audit_score:>7.2f}  "
            f"{entry.model_name}"
        )
    print()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

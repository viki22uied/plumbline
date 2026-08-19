"""Measure the native Monte Carlo backend against the NumPy one.

    python benchmarks/bench_backends.py
    python benchmarks/bench_backends.py --paths 400000 --markdown

Two speedups are reported for every contract, and both are needed to read the
result honestly:

``x1``  the native backend held to a single thread. This is the part that comes
        from the code itself: one pass per path instead of one pass per array
        per step, no temporaries, and no interpreter between the steps.
``xN``  the native backend on every core. NumPy's inner loops here are single
        threaded, so this number includes the thread count and is not a
        like-for-like comparison of the two implementations.

Quoting ``xN`` alone would flatter the C++. Quoting ``x1`` alone would hide
what the backend is actually for. The README carries both.

Each timing is the best of several repeats, not the mean: the best run is the
one least disturbed by whatever else the machine was doing.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from plumbline.contracts import OptionSpec
from plumbline.engines import montecarlo as mc
from plumbline.engines import native

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.25)

CASES: list[tuple[str, OptionSpec]] = [
    ("european call", OptionSpec("european", "call", **BASE)),
    ("digital cash-or-nothing", OptionSpec("digital", "call", payout="cash", **BASE)),
    ("asian arithmetic", OptionSpec("asian", "call", averaging="arithmetic", **BASE)),
    ("asian geometric", OptionSpec("asian", "put", averaging="geometric", **BASE)),
    (
        "barrier down-and-out",
        OptionSpec("barrier", "call", barrier=90.0, barrier_kind="down-and-out", **BASE),
    ),
    (
        "barrier up-and-in",
        OptionSpec("barrier", "put", barrier=130.0, barrier_kind="up-and-in", **BASE),
    ),
    ("lookback fixed strike", OptionSpec("lookback", "call", strike_type="fixed", **BASE)),
    ("lookback floating strike", OptionSpec("lookback", "put", strike_type="floating", **BASE)),
]


def timed(call, repeats: int) -> tuple[float, float]:
    """Return (best seconds, returned price). One warm-up run is discarded."""
    price = call()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        price = call()
        timings.append(time.perf_counter() - started)
    return min(timings), price


def run(paths: int, steps: int, repeats: int, threads: int) -> list[dict]:
    rows = []
    for name, spec in CASES:
        stepped = spec.instrument in ("asian", "barrier", "lookback")
        used_steps = steps if stepped else 1

        numpy_time, numpy_price = timed(
            lambda s=spec, k=used_steps: mc.monte_carlo(
                s, paths=paths, steps=k, seed=20260816, backend="numpy"
            ).price,
            repeats,
        )
        single_time, single_price = timed(
            lambda s=spec, k=used_steps: mc.monte_carlo(
                s, paths=paths, steps=k, seed=20260816, backend="cpp", threads=1
            ).price,
            repeats,
        )
        many_time, many_price = timed(
            lambda s=spec, k=used_steps: mc.monte_carlo(
                s, paths=paths, steps=k, seed=20260816, backend="cpp", threads=threads
            ).price,
            repeats,
        )

        rows.append(
            {
                "case": name,
                "steps": used_steps,
                "numpy_s": numpy_time,
                "cpp_1_s": single_time,
                "cpp_n_s": many_time,
                "speedup_1": numpy_time / single_time,
                "speedup_n": numpy_time / many_time,
                "numpy_price": numpy_price,
                "cpp_price": many_price,
                "price_gap": abs(numpy_price - many_price),
            }
        )
    return rows


def print_table(rows: list[dict], markdown: bool) -> None:
    if markdown:
        print("| Contract | Steps | NumPy | C++ 1 thread | C++ all cores | x1 | xN |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            print(
                f"| {row['case']} | {row['steps']} | {row['numpy_s'] * 1000:.0f} ms | "
                f"{row['cpp_1_s'] * 1000:.0f} ms | {row['cpp_n_s'] * 1000:.0f} ms | "
                f"**{row['speedup_1']:.1f}x** | **{row['speedup_n']:.1f}x** |"
            )
        return

    header = f"{'contract':26s} {'steps':>6s} {'numpy':>9s} {'cpp x1':>9s} {'cpp xN':>9s} {'x1':>6s} {'xN':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:26s} {row['steps']:>6d} "
            f"{row['numpy_s'] * 1000:>8.0f}m {row['cpp_1_s'] * 1000:>8.0f}m "
            f"{row['cpp_n_s'] * 1000:>8.0f}m {row['speedup_1']:>5.1f}x {row['speedup_n']:>5.1f}x"
        )
    print("-" * len(header))
    print(
        f"{'median':26s} {'':>6s} {'':>9s} {'':>9s} {'':>9s} "
        f"{statistics.median(r['speedup_1'] for r in rows):>5.1f}x "
        f"{statistics.median(r['speedup_n'] for r in rows):>5.1f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=400_000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0, help="0 uses every core")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", help="also write the raw numbers here")
    args = parser.parse_args()

    if not native.available():
        print("the native backend is not built, so there is nothing to compare against.")
        print(f"reason: {native.load_error()}")
        print("build it with: python native/build.py --check")
        return 1

    threads = args.threads or native.backend_threads()

    print(f"machine:  {platform.processor() or platform.machine()}")
    print(f"platform: {platform.platform()}")
    print(f"python:   {platform.python_version()}")
    print(f"backend:  {native.backend_version()}")
    print(f"paths:    {args.paths:,}   steps: {args.steps}   repeats: {args.repeats}")
    print(f"threads:  {threads} of {native.backend_threads()} reported by the library")
    print()

    rows = run(args.paths, args.steps, args.repeats, threads)
    print_table(rows, args.markdown)

    worst = max(rows, key=lambda r: r["price_gap"])
    print()
    print(
        f"largest price gap between backends: {worst['price_gap']:.6f} "
        f"on {worst['case']} (they draw from different streams, so this is "
        f"sampling noise and not disagreement)"
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    "backend": native.backend_version(),
                    "paths": args.paths,
                    "steps": args.steps,
                    "threads": threads,
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"raw numbers written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

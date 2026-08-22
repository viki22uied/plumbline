"""Reproducible mutation testing for the Ground Truth Engines and audit logic.

Why this exists
---------------
Coverage says a line ran. It does not say an assertion would have failed had
that line been wrong. This harness injects plausible quant bugs into the
engines and the audit logic one at a time, runs the test suite against each,
and reports which mutants were killed.

Provenance, stated plainly: an earlier 27-bug run documented in BENCHMARKS.md
was performed by hand-editing files, and no script for it was committed. That
run is history; it cannot be re-executed from this repository. Everything in
``MUTANTS`` below IS committed, so the score it produces can be reproduced
with one command::

    python tests/mutation/run_mutation.py            # full set
    python tests/mutation/run_mutation.py --only M01,M14
    python tests/mutation/run_mutation.py --strict   # exit 1 on any survivor

Honesty policy
--------------
A surviving mutant is never deleted and its test is never weakened to force a
kill. A survivor means either (a) the suite has a real gap -- fix it by adding
a check that pins the behaviour, exactly as was done for the floating-strike
lookback call sign flip -- or (b) the mutant is genuinely equivalent to the
original -- document it in BENCHMARKS.md Table 4 with reasoning.

Lane: each mutant runs ``pytest -m "fast or integration"``. The fast lane has
the closed forms, both external oracle libraries and every engine-vs-engine
cross-check; integration adds sandbox, audit, CLI and API. Pure ``fast``
would let audit-logic mutations survive silently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

LANE_ARGS = ["-m", "fast or integration", "-x", "-q", "--no-header", "-p", "no:cacheprovider"]


@dataclass(frozen=True)
class Mutant:
    mid: str
    description: str
    file: str  # relative to repo root
    old: str
    new: str


# Each entry must occur EXACTLY ONCE in its file; the harness refuses to run
# otherwise rather than patching the wrong line. New entries should target
# functions with no existing mutant first; see BENCHMARKS.md Table 4.
MUTANTS = [
    # --- analytic.py -------------------------------------------------------
    Mutant(
        "M01",
        "lookback carry floor nudges b away from zero instead of toward it",
        "plumbline/engines/analytic.py",
        "return _CARRY_FLOOR if b >= 0.0 else -_CARRY_FLOOR",
        "return -_CARRY_FLOOR if b >= 0.0 else _CARRY_FLOOR",
    ),
    Mutant(
        "M02",
        "asset digital d1 reconstructed with the wrong sign",
        "plumbline/engines/analytic.py",
        "d1 = d2 + sigma * math.sqrt(T)",
        "d1 = d2 - sigma * math.sqrt(T)",
    ),
    Mutant(
        "M03",
        "barrier rebate E term drops the discount factor",
        "plumbline/engines/analytic.py",
        "E = R * disc_r * (",
        "E = R * (",
    ),
    Mutant(
        "M04",
        "Kemna-Vorst adjusted carry uses +sigma^2/6 instead of -",
        "plumbline/engines/analytic.py",
        "b_a = 0.5 * (b - sigma * sigma / 6.0)",
        "b_a = 0.5 * (b + sigma * sigma / 6.0)",
    ),
    Mutant(
        "M05",
        "Black-Scholes delta reads N(d2) instead of N(d1)",
        "plumbline/engines/analytic.py",
        "delta = phi * disc_q * _N(phi * d1)",
        "delta = phi * disc_q * _N(phi * d2)",
    ),
    # --- binomial.py ---------------------------------------------------------
    Mutant(
        "M06",
        "Richardson extrapolation replaced by the fine tree alone",
        "plumbline/engines/binomial.py",
        "return 2.0 * fine - coarse",
        "return fine",
    ),
    Mutant(
        "M07",
        "lattice theta sign flipped",
        "plumbline/engines/binomial.py",
        "theta = (v2[1] - value) / (2.0 * dt)",
        "theta = (value - v2[1]) / (2.0 * dt)",
    ),
    Mutant(
        "M08",
        "CRR up-factor drops the sqrt on dt",
        "plumbline/engines/binomial.py",
        "u = math.exp(sigma * math.sqrt(dt))",
        "u = math.exp(sigma * dt)",
    ),
    # --- fdm.py --------------------------------------------------------------
    Mutant(
        "M09",
        "Rannacher start-up removed: pure Crank-Nicolson from step zero",
        "plumbline/engines/fdm.py",
        "theta = 1.0 if step < RANNACHER_STEPS else 0.5",
        "theta = 0.5",
    ),
    Mutant(
        "M10",
        "digital payoff cell averaging disabled (pointwise sampling always)",
        "plumbline/engines/fdm.py",
        'if spec.instrument != "digital" or dx <= 0.0:',
        "if True:",
    ),
    Mutant(
        "M11",
        "spatial domain narrowed from 7 to 2.5 standard deviations",
        "plumbline/engines/fdm.py",
        "DOMAIN_WIDTH_SDS = 7.0",
        "DOMAIN_WIDTH_SDS = 2.5",
    ),
    # --- heston.py -----------------------------------------------------------
    Mutant(
        "M12",
        "little-trap substitution inverted back to the naive 1993 g",
        "plumbline/engines/heston.py",
        "c = numer / denom",
        "c = denom / numer",
    ),
    Mutant(
        "M13",
        "negative-price clamp applied after parity derivation",
        "plumbline/engines/heston.py",
        "call = max(call, max(forward_value, 0.0))",
        "call = call",
    ),
    # --- montecarlo.py -------------------------------------------------------
    Mutant(
        "M14",
        "Brownian-bridge extreme span added instead of subtracted",
        "plumbline/engines/montecarlo.py",
        "span = np.sqrt((x - x_prev) ** 2 - 2.0 * var_step * np.log(u))",
        "span = np.sqrt((x - x_prev) ** 2 + 2.0 * var_step * np.log(u))",
    ),
    Mutant(
        "M15",
        "terminal-spot control expectation discounts at r instead of q",
        "plumbline/engines/montecarlo.py",
        "return spec.S * math.exp(-spec.q * spec.T)",
        "return spec.S * math.exp(-spec.r * spec.T)",
    ),
    Mutant(
        "M16",
        "barrier survival weight inverted: out priced as in",
        "plumbline/engines/montecarlo.py",
        'weight = survival if spec.barrier_kind.endswith("out") else (1.0 - survival)',
        'weight = survival if spec.barrier_kind.endswith("in") else (1.0 - survival)',
    ),
    # --- bump.py -------------------------------------------------------------
    Mutant(
        "M17",
        "central-difference delta divided by h instead of 2h",
        "plumbline/engines/bump.py",
        "delta = (up - down) / (2.0 * h_s)",
        "delta = (up - down) / h_s",
    ),
    # --- limits.py -----------------------------------------------------------
    Mutant(
        "M18",
        "degenerate arithmetic Asian corner takes the b==0 branch when b!=0",
        "plumbline/engines/limits.py",
        "if abs(b) > 1e-12 else spec.S",
        "if abs(b) <= 1e-12 else spec.S",
    ),
    # --- registry.py ---------------------------------------------------------
    Mutant(
        "M19",
        "engine selection sorts priorities ascending: lowest wins",
        "plumbline/engines/registry.py",
        "reverse=True",
        "reverse=False",
    ),
    # --- audit -----------------------------------------------------------------
    Mutant(
        "M20",
        "delta range bounds swapped between calls and puts",
        "plumbline/audit/checks.py",
        'low, high = (0.0, 1.0) if spec.option_type == "call" else (-1.0, 0.0)',
        'low, high = (-1.0, 0.0) if spec.option_type == "call" else (0.0, 1.0)',
    ),
    Mutant(
        "M21",
        "put-call parity weight dropped from 0.15 to 0.05",
        "plumbline/audit/scoring.py",
        "2: 0.15,",
        "2: 0.05,",
    ),
]


def _git_is_clean_for_targets() -> None:
    import shutil

    git = shutil.which("git")
    if git is None:
        print(
            "NOTE: git not found on PATH; falling back to content checks.\n"
            "The harness verifies every target still contains its expected\n"
            "original text before patching, and restores it afterwards.\n"
        )
        _verify_originals_present()
        return
    result = subprocess.run(
        [git, "status", "--porcelain", "--"]
        + [mutant.file for mutant in MUTANTS],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git status failed:\n{result.stderr}")
    if result.stdout.strip():
        raise SystemExit(
            "refusing to run: target engine files have uncommitted changes.\n"
            "The harness patches and restores them; commit or stash first.\n"
            + result.stdout
        )


def _verify_originals_present() -> None:
    """Abort loudly if a target file no longer holds its expected original text.

    A previous harness process killed between patch and restore would leave a
    mutant applied. Patching on top of that would corrupt the engine silently;
    refusing to start turns it into a repairable, visible failure instead.
    """
    problems = []
    seen_files: set[str] = set()
    for mutant in MUTANTS:
        if mutant.file in seen_files:
            continue
        seen_files.add(mutant.file)
        source = (REPO / mutant.file).read_text(encoding="utf-8")
        missing = [m for m in MUTANTS if m.file == mutant.file and m.old not in source]
        if missing:
            problems.append((mutant.file, [m.mid for m in missing]))
    if problems:
        lines = [
            "refusing to run: these files no longer contain the original text"
            " the mutants expect, which usually means a previous run was killed"
            " between patching and restoring:",
        ]
        for path, ids in problems:
            lines.append(f"  {path}: mutants {', '.join(ids)}")
        lines.append("restore the files from version control, then re-run.")
        raise SystemExit("\n".join(lines))


def run_one(mutant: Mutant, verbose: bool = True) -> tuple[bool, float]:
    """Apply the mutant, run the test lane, restore. Returns (killed, seconds)."""
    path = REPO / mutant.file
    source = path.read_text(encoding="utf-8")
    count = source.count(mutant.old)
    if count != 1:
        raise SystemExit(
            f"{mutant.mid}: expected exactly one occurrence of "
            f"{mutant.old!r} in {mutant.file}, found {count}"
        )

    started = time.perf_counter()
    try:
        path.write_text(source.replace(mutant.old, mutant.new), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *LANE_ARGS],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        killed = proc.returncode != 0
    finally:
        path.write_text(source, encoding="utf-8")
    elapsed = time.perf_counter() - started

    if verbose:
        state = "KILLED " if killed else "SURVIVED"
        print(f"  {mutant.mid} {state} {elapsed:6.1f}s  {mutant.description}")
    return killed, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma separated mutant ids, e.g. M01,M14")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any mutant survives")
    parser.add_argument("--json", dest="json_path", help="also write results as JSON")
    args = parser.parse_args()

    selected = list(MUTANTS)
    if args.only:
        wanted = {part.strip().upper() for part in args.only.split(",")}
        selected = [m for m in MUTANTS if m.mid.upper() in wanted]
        missing = wanted - {m.mid.upper() for m in selected}
        if missing:
            raise SystemExit(f"unknown mutant ids: {sorted(missing)}")

    _git_is_clean_for_targets()

    print(f"running {len(selected)} mutants against: pytest {' '.join(LANE_ARGS)}\n")
    results = []
    for mutant in selected:
        killed, elapsed = run_one(mutant)
        results.append({"id": mutant.mid, "description": mutant.description,
                        "killed": killed, "seconds": round(elapsed, 1)})

    killed_count = sum(1 for r in results if r["killed"])
    total_time = sum(r["seconds"] for r in results)
    score = 100.0 * killed_count / len(results)

    print()
    print(f"killed {killed_count} of {len(results)}. Mutation score {score:.0f}%.")
    print(f"wall time {total_time:.0f}s.")

    survivors = [r["id"] for r in results if not r["killed"]]
    if survivors:
        print("survivors:", ", ".join(survivors))
        print("investigate; do NOT delete the mutant or weaken a test to kill it.")
    else:
        print("no survivors.")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps({"score_percent": score, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")

    return 1 if args.strict and survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())

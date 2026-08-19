# Changelog

All notable changes to Plumbline are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Plumbline uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-08-16

### Added

- **An optional C++ backend for the Monte Carlo engine.** Plain C++ with a C
  ABI, loaded through `ctypes`, built by one command. It is not a Python
  extension module, so no Python headers are needed and one build serves every
  interpreter on the machine.
- `python native/build.py --check` builds it and confirms it loads.
- `plumbline backend` reports whether it is present and why not when it is not.
- `plumbline audit --mc-backend {numpy,cpp,auto}` selects the backend for a run.
- `benchmarks/bench_backends.py` measures the two backends against each other.
- The Audit Report records which backend was available, in section 7.

### Measured

- Median 3.0x faster on a single thread, 12.0x on twelve cores, at 400,000
  paths and 250 steps. Barriers and lookbacks gain most, at 3.5x to 4.0x
  single threaded, because those are the contracts where the vectorised
  version makes the most passes over memory per time step.

### Notes

- NumPy remains the default and the reference. Nothing requires the backend.
- The two backends use different random generators on purpose, so they are
  independent estimators. They agree within their combined standard error,
  not to the last bit.
- The native result is bit-identical across thread counts. Path-pairs are cut
  into fixed blocks pinned to fixed random streams, each with its own
  accumulator slot, merged in index order.

## [1.0.0] — 2026-08-16

First production release. The full system, not a minimum version.

### Added

**Module A — Model Ingestion**

- Load a Model Under Test from a Python source file, with a fixed and
  documented input signature.
- Load a Model Under Test from a CSV or JSON table of pre-computed prices.
- Validate the signature before the audit starts, and name exactly what is
  missing when it does not match.
- Run every model in an isolated child process, with the network blocked and
  writes confined to one temporary directory.
- Enforce a time limit on every call, and record a Timeout when it is passed.

**Module B — Ground Truth Engine Suite**

- Closed-form Black-Scholes-Merton engine, with all five Greeks in closed form.
- Cox-Ross-Rubinstein binomial tree, for European and American exercise, with
  lattice Greeks read off the first two tree levels.
- Monte Carlo engine with antithetic variates and control variates, using
  Brownian-bridge corrections for barrier survival and lookback extremes.
- Crank-Nicolson finite difference engine with Rannacher start-up and
  cell-averaged payoffs.
- Heston stochastic volatility engine, using the little-trap form of the
  characteristic function.
- Local volatility support on the finite difference grid.
- Closed-form engines for geometric Asian, barrier, digital and lookback
  contracts.

**Module C — Validation and Audit Engine**

- Six independent check types, each isolated so one failure costs only its own
  results.
- Parameter grids with a deterministic, reproducible sample for the checks that
  cost several model calls per point.
- A published, fixed Audit Score formula, and a badge that is deliberately
  harsher than the score.

**Module D — Report Generation**

- Audit Reports in JSON, Markdown and PDF, with the same seven sections in
  each.
- A plain-language diagnosis that names the defect, not only the numbers.
- Convergence plots for Check Type 4.
- An audit history store, so one model can be compared against its own past.

**Module E — Interface Layer**

- A command line interface that runs a full audit from one command, and returns
  a non-zero exit code when the audit does not pass.
- A REST API with endpoints for audits, prices, engines and history.
- OpenAPI documentation generated from the code.

### Verified

- All seven Ground Truth self-validation cases pass.
- The correct sample model returns a full PASS on every check type and every
  covered instrument.
- The broken sample model, with five distinct seeded errors, is flagged on all
  five.
- A vanilla audit finishes in about 1.3 seconds. The requirement is 5 seconds.
- Test coverage of the Ground Truth Engine Suite and the Validation and Audit
  Engine is 92 percent. The requirement is 85 percent.

[1.1.0]: https://github.com/viki22uied/plumbline/releases/tag/v1.1.0
[1.0.0]: https://github.com/viki22uied/plumbline/releases/tag/v1.0.0

# Plumbline

**An independent verification engine for derivative pricing models.**

[![CI](https://github.com/viki22uied/plumbline/actions/workflows/ci.yml/badge.svg)](https://github.com/viki22uied/plumbline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](#tests)
[![Platforms](https://img.shields.io/badge/platforms-linux%20%7C%20macos%20%7C%20windows-lightgrey)](#installation)

> This document uses ASD-STE100 Simplified Technical English. Sentences are
> short. Each sentence gives one fact or one instruction. The voice is active.

---

## 1. What Plumbline does

Plumbline checks a derivative pricing model. The model can come from a person.
The model can come from an AI tool. Plumbline does not care about the source.

Plumbline tests the model against known, correct mathematics. Plumbline gives a
clear PASS or FAIL result for each test. Plumbline writes a full report at the
end.

Plumbline does not manage money. Plumbline does not place a trade. Plumbline
does not connect to a broker. Read [NOTICE](NOTICE) before you use it.

## 2. Why Plumbline exists

An AI tool can write a derivative pricing model in a short time. The model does
not always come out correct. A wrong model can look correct. A wrong model can
run without an error message. A wrong model can give a wrong price.

A person who uses the wrong price can lose money. A firm that uses the wrong
model can break a rule set by a regulator. Banks employ people whose full job is
to check a model before the firm uses it. That job is called Model Validation.

Plumbline does that check, and it does it in the open.

## 3. Install

Plumbline needs Python 3.10 or higher. Plumbline runs on Linux, on macOS, and
on Windows.

### From source

```bash
git clone https://github.com/viki22uied/plumbline.git
```

```bash
cd plumbline && python -m venv .venv
```

Activate the environment. On Linux and macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the package with its development extras:

```bash
pip install -e ".[dev]"
```

### With Docker

The container is the recommended way to audit a model you do not trust.

```bash
docker build -t plumbline .
```

```bash
docker run --rm -v "$PWD:/work" plumbline audit /work/model.py --out /work/reports
```

## 4. Write a Model Under Test

Your model is one Python function. The function has this signature:

```python
def price(instrument, option_type, S, K, T, r, q, sigma, **kwargs) -> float:
    ...
```

| Parameter | Meaning |
| --- | --- |
| `instrument` | `european`, `american`, `asian`, `barrier`, `digital`, or `lookback` |
| `option_type` | `call` or `put` |
| `S` | spot price of the underlying asset |
| `K` | strike price |
| `T` | time to expiry, in years |
| `r` | risk-free rate, continuously compounded |
| `q` | dividend yield, continuously compounded |
| `sigma` | volatility, annualised |
| `**kwargs` | instrument extras, and the optional `precision` knob |

The function returns one number. The number is the present value of one
contract. The function must accept `**kwargs`. Plumbline sends the instrument
extras through it.

Plumbline also accepts a model as a table of prices. Use a `.csv` or a `.json`
file when you do not have the source code.

## 5. Run an audit

Run the full audit with one command:

```bash
plumbline audit samples/good_model.py
```

Plumbline prints a summary. Plumbline writes the report in three formats.

```
  Plumbline 1.0.0   audit a8d29c098fe8439b
  model: good_model.py:price

  BADGE PASS    SCORE 100.00 / 100

   #  check type                               pass   fail   err  skip
   1  Reference Price Comparison                 72      0     0     0
   2  Put-Call Parity                            36      0     0     0
   3  Greek Consistency                          72      0     0     0
   4  Convergence and Stability                   4      0     0     0
   5  Edge Case and Boundary Behaviour           24      0     0     0
   6  Arbitrage-Free Sanity                      17      0     0     0
```

Now run the same command on the broken sample:

```bash
plumbline audit samples/broken_model.py
```

That file holds five deliberate errors. Plumbline finds all five. Read
`samples/broken_model.py` to see what they are.

The exit code is `0` when the badge is PASS. The exit code is `1` when the badge
is PARTIAL or FAIL. Use the exit code to gate a build pipeline.

### Other commands

List the Ground Truth Engines:

```bash
plumbline engines
```

Price one contract with a reference engine:

```bash
plumbline price --spot 100 --strike 100 --maturity 1 --rate 0.05 --vol 0.2
```

Read the audit history of your models:

```bash
plumbline history
```

## 6. Run the REST API

```bash
uvicorn plumbline.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` in a browser. FastAPI builds that page from
the code, so the page cannot fall behind the endpoints.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/engines` | List the Ground Truth Engines and what they cover |
| `POST` | `/price` | Price one contract with a reference engine |
| `POST` | `/audit` | Submit a Model Under Test and receive an Audit Report |
| `GET` | `/audit/{id}/report.json` | The machine-readable report |
| `GET` | `/audit/{id}/report.md` | The human-readable report |
| `GET` | `/audit/{id}/report.pdf` | The report as a PDF |
| `GET` | `/history` | Past audits, for comparison over time |

## 7. The six check types

Plumbline runs six independent check types. A failure in one check does not stop
the others.

| # | Check type | What it proves | Weight |
| --- | --- | --- | --- |
| 1 | Reference Price Comparison | The price agrees with a published method, across a grid | 0.30 |
| 2 | Put-Call Parity | The call and the put are consistent with each other | 0.15 |
| 3 | Greek Consistency | The slope of the price surface is right, not only its level | 0.15 |
| 4 | Convergence and Stability | More precision brings the model closer, so the error is noise and not bias | 0.10 |
| 5 | Edge Case and Boundary Behaviour | The model is right where the answer is known exactly | 0.15 |
| 6 | Arbitrage-Free Sanity | The prices admit no risk-free profit | 0.15 |

The Audit Score uses this fixed formula:

```
score = 100 * sum(weight * pass_rate) / sum(weight)
```

The sum runs over the check types that produced at least one countable result. A
check that was skipped does not reward the model and does not punish it.

The badge is harsher than the score, because an average hides one fatal defect:

- **PASS** means every countable result passed.
- **FAIL** means the score is below 70, or one check type failed every case.
- **PARTIAL** means anything between those two.

### Why Check Type 4 matters most

A model can be close to correct at one setting by luck. Raise its precision, and
the truth comes out. Noise falls when a method gets more work. Bias does not.

Compare `samples/mc_model.py` with `samples/biased_mc_model.py`. Both are Monte
Carlo. Both look noisy at a low path count. One converges to the right number.
The other converges to the wrong one, and no path count fixes it.

## 8. The Ground Truth Engine Suite

Each engine uses a published method. Each engine is itself checked against a
textbook value or against a second engine that shares no code.

| Engine | Method | Covers | Reference |
| --- | --- | --- | --- |
| `analytic` | Closed-form formulas | European, digital, geometric Asian, barrier, lookback | Merton (1973); Reiner-Rubinstein (1991); Kemna-Vorst (1990); Goldman-Sosin-Gatto (1979); Conze-Viswanathan (1991) |
| `binomial_crr` | Cox-Ross-Rubinstein tree | European, American | Cox, Ross and Rubinstein (1979) |
| `fdm_crank_nicolson` | Crank-Nicolson PDE grid with Rannacher start-up | European, American, digital, barrier | Crank-Nicolson (1947); Rannacher (1984) |
| `heston_cf` | Characteristic function integral, little-trap form | European under Heston | Heston (1993); Albrecher et al. (2007) |
| `monte_carlo` | Simulation with antithetic and control variates | European, Asian, barrier, digital, lookback | Glasserman (2003) |

### Numerical decisions that are not optional

- **Rannacher start-up.** Crank-Nicolson is A-stable but not L-stable. A payoff
  with a kink or a jump makes it oscillate, and the oscillation never damps out.
  Two fully implicit steps at the start remove it.
- **Cell-averaged payoffs.** A digital payoff jumps inside one grid cell.
  Sampling it at the node makes the answer depend on which side of the jump the
  node falls. The engine averages the payoff over the cell instead.
- **Brownian-bridge corrections.** A path sampled at discrete times can step over
  a barrier and back without being seen. The simulation uses the bridge no-hit
  probability for barriers, and samples lookback extremes from the exact bridge
  law. Without them, no path count removes the bias.
- **The Heston little trap.** The 1993 form of the characteristic function
  crosses a branch cut of the complex logarithm past about one year. The little
  trap form of Albrecher et al. (2007) does not. Plumbline uses the second form.

## 9. The optional C++ backend

The Monte Carlo engine has two backends. NumPy is the default and stays the
reference for what the estimator means. A C++ library does the same arithmetic
faster, and it is optional in the strict sense: Plumbline runs, and every check
works, when it is not there.

### Build it

```bash
python native/build.py --check
```

The script finds `g++`, `clang++` or `cl` on the PATH and writes one shared
library. It is not a Python extension module. It exports plain C and is loaded
with `ctypes`, so there are no Python headers to find, no interpreter version
to match, and one build serves every Python on the machine.

Check what Plumbline sees:

```bash
plumbline backend
```

Use it for an audit:

```bash
plumbline audit model.py --mc-backend auto
```

`auto` uses the library when it is built and falls back to NumPy when it is
not. `cpp` demands the library and raises if it is missing, which is what a
benchmark needs: a run that silently measured NumPy would be worse than an
error.

### Measured speedup

Measured on four machines, three of them GitHub runners so anyone can check
them. Full tables in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

| Machine | Toolchain | Cores | x1 median | xN median |
| --- | --- | ---: | ---: | ---: |
| GitHub runner, Ubuntu | GCC | 4 | **1.2x** | **3.0x** |
| GitHub runner, macOS arm64 | Apple Clang | 3 | **1.2x** | **3.2x** |
| GitHub runner, Windows | MSVC | 4 | **2.1x** | **4.5x** |
| Developer laptop, Windows | MinGW GCC 16.1 | 12 | **3.0x** | **12.0x** |

**The headline is a range, not a number: about 1x to 3x on one thread, and 3x
to 12x across cores.** The 12x belongs to a twelve-core laptop whose NumPy is
also unusually slow, so weight the three runner rows above it. On the Ubuntu
runner one contract is 0.8x, meaning the single-threaded C++ *loses* to NumPy
there. That number is in the table too.

Reproduce it on your own machine:

```bash
python benchmarks/bench_backends.py --markdown
```

Read both columns. **x1** holds the C++ to a single thread, so it measures the
code. **xN** lets it use every core, and NumPy's inner loops here are single
threaded, so that column includes the core count and is not a like-for-like
comparison. Quoting **xN** alone would flatter the C++; quoting **x1** alone
would hide what the backend is for.

### Where the gain comes from, and where it does not

The speedup varies a lot by machine. The *ordering across contracts* barely
varies at all, and that ordering is the part with a mechanism behind it.

Barriers and lookbacks gain most on three of the four machines. The Asian
gains least, and on two machines it loses. That tracks memory traffic:

- A NumPy barrier step computes two gap arrays, a product, an exponential, a
  comparison mask and a multiply: six passes over N doubles, each one a fresh
  allocation. At scale that working set does not fit in cache, so every pass
  goes to memory. The C++ carries one path-pair in registers from `t=0` to
  expiry and allocates nothing in the loop.
- A NumPy Asian step is a single `np.exp` over a contiguous array plus two
  adds. There is almost no traffic to save, and a scalar `std::exp` loop
  cannot beat a vectorised one.

So the backend wins where the vectorised version makes the most passes, and
loses where NumPy already does one well-vectorised pass. On Linux, glibc's
`exp` is fast enough that the memory saving does not always cover the scalar
transcendental cost, which is exactly where the 0.8x comes from.

The multi-core column is the more dependable win, and it is the honest reason
to build the backend: the step loop cannot be threaded from Python.

If the goal were maximum speed rather than a demonstration of the technique,
the next move would be a vectorised `exp` in the inner loop, not more threads.
That is where the remaining gap is.

### What the backend deliberately does not do

**It does not reproduce NumPy's random stream.** It uses xoshiro256++ seeded
through splitmix64; NumPy uses PCG64 with a ziggurat normal. That makes the two
backends independent estimators of the same expectation, which is the more
useful thing to have in a validation tool: two bit-identical numbers would only
prove that one copied the other. The tests assert that each backend agrees with
the closed form within its own sampling error, and that the two agree with each
other within their combined sampling error.

**It does not price a degenerate contract.** At zero volatility, zero time to
expiry or a zero spot, the value is a closed form that lives in
`plumbline/engines/limits.py`. The library returns a refusal code and the
caller falls back to that one implementation. A second copy of those formulas
in C++ would be a second thing to keep right.

**It does not compute the control variate's expectation.** Python passes it in.
The control is only unbiased if the mean subtracted is the exact one, so the
closed forms behind it have a single implementation and the two backends cannot
drift apart on them.

### Determinism

The same seed gives the same answer, on one thread and on twelve, run after
run. That is not free: merging Welford accumulators is not associative in
floating point, so if each thread merged whatever work it happened to win from
the scheduler, the last bits would move between runs. Path-pairs are therefore
cut into fixed blocks, block *k* always draws from stream *k*, each block gets
its own accumulator slot, and the merge walks the slots in index order.

The first version of this backend did not do that, and the test that compares
one thread against eight is what caught it.

## 10. Instruments and models

**Vanilla:** European call, European put, American call, American put.

**Exotic:** Asian (arithmetic and geometric average), barrier (up-and-out,
up-and-in, down-and-out, down-and-in), digital (cash-or-nothing and
asset-or-nothing), lookback (fixed strike and floating strike).

**Underlying dynamics:** geometric Brownian motion (Black-Scholes-Merton),
Heston stochastic volatility, local volatility (Dupire).

## 11. The sandbox

Plumbline runs your model in a separate process. The parent never imports it.

The child process blocks network access. The child process blocks writes outside
one private temporary directory. On Linux and macOS the child also runs under a
memory limit. Every call has a time limit, and a call that passes it is recorded
as a Timeout.

This stops an honest model from causing damage by accident. This is not an
operating-system jail, and it does not stop code written to be hostile. Run an
untrusted model in the container.

## 12. Add a new engine

Plumbline has one plug-in interface. Nothing in the audit engine changes.

```python
from plumbline.contracts import PriceResult
from plumbline.engines.registry import EngineSpec, register

def my_price(spec):
    return PriceResult(price=..., engine="my_engine")

register(EngineSpec(
    name="my_engine",
    description="what this engine does",
    reference="the paper it comes from",
    price_fn=my_price,
    instruments=("european",),
    priority=200,
))
```

The registry picks the engine with the highest priority that covers the
instrument and the underlying model.

## 13. Tests

Run the full suite:

```bash
pytest
```

Run it without the long simulations:

```bash
pytest -m "not slow"
```

Measure the coverage:

```bash
pytest --cov=plumbline/engines --cov=plumbline/audit --cov-report=term-missing
```

The suite has 254 tests. Coverage of the Ground Truth Engine Suite and the
Validation and Audit Engine is 92 percent. The target set by the requirements is
85 percent.

See [CHECKLIST.md](CHECKLIST.md) for every requirement and the test that proves
it.

## 14. Repository layout

```
plumbline/
    contracts.py        the data contracts every module shares
    ingestion.py        Module A -- read and validate a Model Under Test
    sandbox.py          the isolated child process
    _worker.py          the child process entry point
    engines/            Module B -- the Ground Truth Engine Suite
        analytic.py         closed-form formulas
        binomial.py         Cox-Ross-Rubinstein tree
        fdm.py              Crank-Nicolson PDE grid
        heston.py           Heston characteristic function
        montecarlo.py       simulation with variance reduction
        bump.py             bump-and-reprice Greeks
        limits.py           exact values at degenerate corners
        registry.py         the plug-in interface
    audit/              Module C -- the Validation and Audit Engine
        checks.py           the six check types
        grid.py             parameter grids
        scoring.py          the Audit Score formula
        engine.py           orchestration
        history.py          past reports
    report/             Module D -- report generation
        markdown.py         the Markdown renderer
        pdf.py              the PDF renderer
        plots.py            convergence plots
        summary.py          the plain-language diagnosis
        native.py           ctypes loader for the optional C++ backend
    cli.py              Module E -- the command line interface
    api.py              Module E -- the REST API
native/                 the optional C++ backend
    plumbline_mc.h          the C ABI, shared with the ctypes loader
    plumbline_mc.cpp        the engine
    build.py                one command to build it
benchmarks/             the NumPy against C++ measurement
samples/                models to audit, correct and broken
tests/                  the test suite
```

## 15. Licence

Plumbline is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Apache 2.0 was chosen for three reasons. It is permissive, so a firm can use
Plumbline inside a commercial process. It grants patent rights explicitly, which
matters in quantitative finance. It disclaims warranty in clear terms, which
matters for any tool that touches valuation.

Read [NOTICE](NOTICE) as well. It states what Plumbline is not.

## 16. Author

Vignesh Kumar U.

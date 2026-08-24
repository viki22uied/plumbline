# Benchmarks and where they come from

A model-validation tool that validates itself against itself has proved
nothing. Two implementations by one author share one set of assumptions, and
correlated errors ratify each other instead of cancelling.

This is not a hypothetical. Seven defects have been found in Plumbline's own
engines, and **every one of them passed the internal cross-checks while being
wrong** — including a Cox-Ross-Rubinstein tree 4x less accurate than the
tolerance it was enforcing, and a Heston engine returning negative prices.
Internal consistency did not catch them. That is the argument for this file.

Every number below is therefore labelled with what actually backs it.

---

## Table 1 — Externally validated

Checked against [QuantLib](https://www.quantlib.org/) 1.43, a library this
author did not write, maintained independently for over twenty years and used
in production across the industry. The tests are in
[`tests/test_external_oracle.py`](tests/test_external_oracle.py) and run in CI.

```bash
pip install "plumbline[oracle]"
pytest tests/test_external_oracle.py
```

| What | Plumbline | QuantLib engine | Agreement |
| --- | --- | --- | --- |
| European price, 10 cases incl. negative rates | `analytic` | `AnalyticEuropeanEngine` | **1e-12 relative** |
| European Greeks (all five), 6 cases | `analytic` | `AnalyticEuropeanEngine` | **1e-10 relative** |
| American put, the GT-02 benchmark | `binomial_crr` | `QdFpAmericanEngine`, high-precision scheme | 1.5e-5 relative at default settings |
| American call and put, 6 cases | `binomial_crr` | `BinomialVanillaEngine` CRR, 8000 steps | < 1e-3 absolute |
| Heston, 6 strike/maturity pairs incl. T=10 | `heston_cf` | `AnalyticHestonEngine` | **1e-8 absolute** |
| All 8 standard barriers | `analytic` | `AnalyticBarrierEngine` | **1e-9 absolute** |
| Cash-or-nothing and asset-or-nothing digitals | `analytic` | `AnalyticEuropeanEngine` | **1e-12 absolute** |
| Continuous geometric Asian | `analytic` | `AnalyticContinuousGeometricAveragePriceAsianEngine` | **1e-9 absolute** |
| Crank-Nicolson grid | `fdm_crank_nicolson` | `FdBlackScholesVanillaEngine` | < 1e-3 absolute |

### The two numbers this repository quotes by value

**American put — 6.09037.** S = K = 100, r = 5%, q = 0, σ = 20%, T = 1 year.

Obtained from QuantLib's `QdFpAmericanEngine` on its high-precision scheme,
which implements the Andersen, Lake & Offengenden (2016) fixed-point method —
the current high-accuracy reference for American puts, and an approach with
nothing in common with a binomial tree. The exact figure is 6.0903706065.

Plumbline's tree converges on it from above, which is what a correct
Richardson-extrapolated scheme should do:

| Steps | Plumbline | Error against 6.0903706065 |
| ---: | --- | ---: |
| 200 | 6.0907536720 | 3.8e-4 |
| 800 (default) | 6.0904648982 | 9.4e-5 |
| 3200 | 6.0903866809 | 1.6e-5 |
| 12800 | 6.0903733947 | 2.8e-6 |

The older literature value of 6.0903 is correct to four decimal places.

**Heston — 5.785155.** S = K = 100, r = q = 0, T = 1 year, with
v₀ = 0.0175, κ = 1.5768, θ = 0.0398, ξ = 0.5751, ρ = −0.5711.

That parameter set is the one used in the numerical examples of Albrecher,
Mayer, Schoutens & Tistaert, *The Little Heston Trap*, Wilmott Magazine No. 1
(2007), pp. 83–92 — the paper whose characteristic-function formulation
Plumbline implements. QuantLib's `AnalyticHestonEngine` gives 5.7851554344 for
it; Plumbline gives 5.7851554343.

---

## Table 2 — Cross-engine consistency only, not externally sourced

These are real checks and they catch real bugs. They are **not** validation,
because both sides come from this repository. They are listed separately so
nobody has to guess which is which.

| Check | What it compares | Why it is still worth running |
| --- | --- | --- |
| Monte Carlo against closed forms | `monte_carlo` vs `analytic` | Catches payoff and discounting errors; the tolerance scales with the simulation's own standard error |
| Native C++ backend against NumPy | two backends, deliberately different RNGs | Independent estimators of one expectation; agreement within combined standard error |
| Heston characteristic function against Euler simulation | `heston_cf` vs `heston_mc_price` | Different mathematics, same author |
| In-out barrier parity | knock-in + knock-out = vanilla | Model-free identity, holds to 1e-10 |
| Local-vol in-out parity | same, on the local volatility grid | Caught a defect where the vanilla leg used a flat volatility |

---

## Table 3 — Model-free identities

These need no oracle at all. They follow from static replication, so they hold
whatever the model, and a violation is unambiguous. Swept across thousands of
parameter combinations in the stress suite.

| Identity | Holds to |
| --- | --- |
| Put-call parity, all engines | 1e-12 |
| Cash digital call + put = discounted cash | 1e-10 |
| Asset digital call + put = S·e^(−qT) | 1e-9 |
| Knock-in + knock-out = vanilla | 1e-10 |
| Δcall − Δput = e^(−qT) | 1e-8 |
| Γcall = Γput, νcall = νput | 1e-8 |
| ρcall − ρput = K·T·e^(−rT) | 1e-8 |
| American ≥ European | always |
| Lookback ≥ vanilla | always |
| Call price convex and non-increasing in strike | always |
| Price bounded by static no-arbitrage bounds | always |

---

## A note on the Hull figure

GT-01 uses S = K = 100, r = 5%, σ = 20%, T = 1 year, and Plumbline returns
10.450584. That parameter set is the standard worked example that appears in
essentially every derivatives textbook, Hull's included, where it is usually
quoted to two decimals as 10.45.

Plumbline does not cite a specific edition and page for it, because the author
has not verified one. What it does have is agreement with QuantLib's
independent implementation to 1e-12. **The value is externally validated; the
attribution is not.** That distinction is the whole point of this document.

---

## What is still missing

Honesty about the gaps, since the gaps are what a reviewer will look for.

- **No market data.** Every reference here is another model, not a traded
  price. Agreement with QuantLib proves both implement the same mathematics
  correctly; it says nothing about whether the mathematics describes a market.
- **No independent implementation of the exotics beyond QuantLib.** Barriers,
  digitals and geometric Asians are checked against one external library. Two
  would be better.
- **The arithmetic Asian has no external oracle.** It has no closed form, so
  its reference is Plumbline's own simulation with a control variate. It sits
  in Table 2 for that reason.
- **Lookbacks are not in Table 1.** QuantLib's continuous lookback engines use
  a different running-extremum convention, and reconciling that properly is
  outstanding work rather than something to fudge.

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

Checked against **two independent libraries this author did not write**:
[QuantLib](https://www.quantlib.org/) 1.43, maintained for over twenty years
and used in production across the industry, and, since this revision,
[financepy](https://github.com/domokane/FinancePy) 1.0.1 — a different
codebase, with different maintainers and, where noted below, different
numerical conventions. Where Plumbline agrees with both, the shared answer is
very unlikely to be wrong in the same way.

The QuantLib tests are in [`tests/test_external_oracle.py`](tests/test_external_oracle.py)
and the financepy tests in
[`tests/test_external_oracle_financepy.py`](tests/test_external_oracle_financepy.py);
both run in CI.

```bash
pip install "plumbline[oracle]"
pytest -m oracle
```

| What | Plumbline | Oracle engine | Agreement |
| --- | --- | --- | --- |
| European price, 10 cases incl. negative rates | `analytic` | QL `AnalyticEuropeanEngine` | **1e-12 relative** |
| European Greeks (all five), 6 cases | `analytic` | QL `AnalyticEuropeanEngine` | **1e-10 relative** |
| American put, the GT-02 benchmark | `binomial_crr` | QL `QdFpAmericanEngine`, high-precision scheme | 1.5e-5 relative at default settings |
| American call and put, 6 cases | `binomial_crr` | QL `BinomialVanillaEngine` CRR, 8000 steps | < 1e-3 absolute |
| Heston, 6 strike/maturity pairs incl. T=10 | `heston_cf` | QL `AnalyticHestonEngine` | **1e-8 absolute** |
| All 8 standard barriers | `analytic` | QL `AnalyticBarrierEngine` | **1e-9 absolute** |
| All 8 standard barriers **with a nonzero rebate** | `analytic` | QL `AnalyticBarrierEngine` | **~1e-14 absolute** |
| Cash-or-nothing and asset-or-nothing digitals | `analytic` | QL `AnalyticEuropeanEngine` | **1e-12 absolute** |
| Continuous geometric Asian | `analytic` | QL `AnalyticContinuousGeometricAveragePriceAsianEngine` | **1e-9 absolute** |
| Floating-strike lookback, call and put | `analytic` | QL `AnalyticContinuousFloatingLookbackEngine` | **1e-14 absolute** |
| Fixed-strike lookback, call and put | `analytic` | QL `AnalyticContinuousFixedLookbackEngine` | **1e-14 absolute** |
| Crank-Nicolson grid | `fdm_crank_nicolson` | QL `FdBlackScholesVanillaEngine` | < 1e-3 absolute |
| **Arithmetic Asian**, discrete n-fixing, call & put, 4 cases | `monte_carlo` | QL `FdBlackScholesAsianEngine` PDE | within 6 Monte Carlo standard errors; measured 0.2–5.9 se, the residual being the FD grid's own error |
| Digitals, cash & asset | `analytic` | fin `EquityDigitalOption` | **1.2e-8 cash / 4.8e-6 asset absolute** |
| All 8 barriers, continuous formulas | `analytic` | fin `EquityBarrierOption`, discrete monitoring at 2×10⁵ obs/yr | ≤ 1.5e-2 absolute — the residual is financepy's known discrete-monitoring gap, which shrinks as observations are added; the test also asserts the convergence direction |
| Geometric Asian, continuous Kemna-Vorst | `analytic` | fin GEOMETRIC method (continuous-mean / discrete-variance hybrid), 20 000 fixings | ≤ 8e-3 per-case bands; 7.9e-5 ATM, degrading away from the money as the hybrid's approximation does |

Two conventions are stated rather than hidden. The arithmetic-Asian comparison
pins fixing dates to whole days so both sides average over exactly the same
schedule — a schedule that rounds to sub-day increments measures the calendar,
not the mathematics. And the financepy barrier and geometric rows are *not*
like-for-like at machine precision: their tolerances absorb documented
convention differences, sized from measurement, far below what any derivation
error would produce.

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
| Lookback branches against closed forms, all four | `monte_carlo` (bridge extremes) vs `analytic` | Each branch pinned separately within 4 standard errors; a sign flip in one branch once passed every other check |
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

## Table 4 — Mutation testing

Coverage says a line ran. It does not say an assertion would have failed had
that line been wrong. Plausible quant bugs are injected into the engines and
the audit logic one at a time, and the suite is run against each.

**Provenance, stated plainly.** An earlier run of 27 hand-injected bugs (26
killed, score 96%) was performed by editing files manually, and no script for
it was committed. That run is history; it cannot be re-executed from this
repository. The harness is now committed and reproducible:
[`tests/mutation/run_mutation.py`](tests/mutation/run_mutation.py), with 21
targeted mutants whose target strings are verified unique before patching and
whose rules -- survivors are investigated, never deleted, never papered over
-- are written into its README.

**Current committed result: 19 of 21 killed. Mutation score 90%.**

The two survivors were investigated rather than deleted, with measurements:

- **M09, Rannacher start-up removed.** Survives because the cell-averaged
  initial condition has already removed the payoff kink that pure
  Crank-Nicolson oscillates on: with Rannacher off, the digital's error
  changes by 1.8e-7 at a 100-step grid and less at finer ones. The start-up
  stays as defence in depth (it is what protects against a future payoff type
  that cell averaging does not smooth), but no honest tolerance currently
  separates it from no-Rannacher on these contracts.
- **M11, spatial domain narrowed from 7 to 2.5 standard deviations.** With
  correct Dirichlet boundary values, truncation error is dominated by node
  spacing everywhere tested -- narrowing the domain shrinks dx and *improves*
  accuracy slightly (4.9e-5 vs 2.9e-4 absolute on the vanilla reference
  case). The mutant is benign under any defensible tolerance; a width floor
  as an assertion would be testing a constant, not behaviour.

Four mutants that originally survived exposed real gaps, each fixed by adding
a check rather than by touching the mutant:

- **M01** (lookback carry floor sign) survived because no test priced a
  lookback near zero carry; `_safe_carry`'s documented sign-preserving
  contract is now pinned directly.
- **M03** (barrier rebate discount dropped) survived because every barrier
  test used rebate = 0, so the entire rebate terms E and F had never been
  numerically checked by anything. Barriers with nonzero rebates now sit in
  Table 1 against QuantLib, agreeing to ~1e-14.
- **M14** (Brownian-bridge extreme span sign flipped) initially survived for
  the same class of reason as the historical lookback survivor below: the
  exotic-instrument smoke tests assert only `price >= 0`. All four lookback
  branches are now pinned simulation-against-closed-form within four standard
  errors.
- **M15** (control-variate expectation discounting r instead of q) survived
  because GT-03 -- then the only raw-estimator correctness check -- was marked
  slow and therefore absent from every lane the mutation lane runs, and the
  audit cannot catch engine defects (its sample model delegates to the same
  engines). A fast-lane test now pins the estimator with q != r, where the
  induced bias is hundreds of standard errors wide.
- **M18** (degenerate arithmetic-Asian corner branch swapped) survived because
  nothing called the corner directly; exact closed-form pins for both carry
  branches are now in the fast lane.

One thing the mutation run shows that is worth stating plainly: **the
integration tests cannot catch a broken reference engine.** They audit
`samples/good_model.py`, which delegates to the same engines the audit compares
it against, so a wrong engine makes the model wrong in exactly the same way and
the audit still passes. That is the self-validation problem in miniature, and
it is the reason Table 1 exists.

---

## Historical note on the earlier manual run

The 27-bug manual run's one survivor was a sign flip on the reflection term of
the floating-strike lookback *call*. It survived because the only numerical
lookback comparisons in the suite covered the floating put and the fixed call
-- two of the four branches had no check at all, and the formula itself turned
out to be correct. That closure (all four branches against QuantLib, Table 1)
is what made M14's later detection by a *different* mechanism possible.

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
  price. Agreement with two libraries proves all three implement the same
  mathematics correctly; it says nothing about whether the mathematics
  describes a market.
- ~~**The arithmetic Asian has no external oracle.**~~ **Closed.** It is now
  checked against QuantLib's `FdBlackScholesAsianEngine` -- a PDE in the
  running-average variable, a genuinely different numerical method from
  Plumbline's simulation -- with fixing schedules aligned to whole days.
  Measured agreement: 0.2 to 5.9 Monte Carlo standard errors, the residual
  being the FD grid's own discretisation. See Table 1.
- ~~**No independent implementation of the exotics beyond QuantLib.**~~
  **Closed for barriers, digitals and the geometric Asian**, against
  financepy as a second external library (Table 1). Two honest caveats: the
  financepy barrier comparison absorbs its discrete-monitoring convention
  rather than matching machine precision, and its geometric method is a
  hybrid approximation -- both sized from measurement and stated in Table 1.
  The remaining exotics (lookbacks) still have only QuantLib as an external
  source; they do have four internal cross-checks each (closed form vs
  simulation per branch, plus the model-free identities of Table 3).
- **The mutation guarantee is only as large as the committed set.** The
  reproducible harness runs 21 targeted mutants at 90% killed, with both
  survivors investigated above; the earlier manual run's 27 bugs are history,
  not evidence. A larger set would find more.

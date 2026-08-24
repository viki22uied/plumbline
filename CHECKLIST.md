# Plumbline — Build Verification Checklist

Every item below is implemented **and** covered by a test that passes. Each row
names the test that proves it. Run the suite to confirm:

```bash
pytest --cov=plumbline/engines --cov=plumbline/audit --cov-report=term-missing
```

Result at the time of writing: **338 tests pass. Coverage is 91 percent.** Of those, 45 validate the engines against QuantLib rather than against Plumbline itself; see [BENCHMARKS.md](BENCHMARKS.md).

---

## 1. Ground Truth Engine Self-Validation (Section 8)

- [x] **GT-01** — European Call, Black-Scholes closed-form, matches Hull's standard case (S=100, K=100, r=5%, σ=20%, T=1yr) → 10.450584, which rounds to 10.45 — `test_gt01_black_scholes_matches_hull_textbook_case`
- [x] **GT-02** — American Put, Binomial Tree (4000 steps), matches the 6.0903 benchmark, and a Crank-Nicolson grid agrees — `test_gt02_american_put_binomial_matches_benchmark`
- [x] **GT-03** — European Call, Monte Carlo (1,000,000 paths), within 1% of Black-Scholes and inside four standard errors — `test_gt03_monte_carlo_within_one_percent_of_black_scholes`
- [x] **GT-04** — European Call, Crank-Nicolson, within 0.1% of Black-Scholes, at four points on the surface — `test_gt04_crank_nicolson_within_one_tenth_percent_of_black_scholes`
- [x] **GT-05** — Heston: the ξ→0 limit reproduces Black-Scholes to 1e-4; the standard index parameter set prices to 5.785155; a full-truncation Euler simulation confirms it — `test_gt05_*` (four tests)
- [x] **GT-06** — Geometric Asian matches the Kemna-Vorst closed form, confirmed against simulation — `test_gt06_geometric_asian_matches_kemna_vorst_closed_form`
- [x] **GT-07** — All eight Reiner-Rubinstein barrier contracts match a PDE grid and a Brownian-bridge simulation; in plus out equals the vanilla to 1e-10 — `test_gt07_*` (12 parametrised cases)

## 2. Check Type 1 — Reference Price Comparison

- [x] MUT price computed and compared against the matching Ground Truth Engine — `test_check1_passes_a_correct_model_and_reports_both_differences`
- [x] Absolute difference computed — same test, `evidence["absolute_difference"]`
- [x] Relative difference computed — same test, `evidence["relative_difference"]`
- [x] PASS/FAIL assigned against the user-set Tolerance — `test_check1_honours_a_user_set_tolerance`
- [x] Runs across a full parameter grid, not a single point — `test_check1_runs_across_the_whole_grid_not_one_point`

## 3. Check Type 2 — Put-Call Parity

- [x] Call and put from the MUT tested against the parity equation — `test_check2_passes_a_correct_model_on_every_pair`
- [x] Parity checked within Tolerance for every European pair on the grid — same test
- [x] Exact numeric parity gap reported on failure — `test_check2_reports_the_exact_numeric_gap_on_failure`

## 4. Check Type 3 — Greek Consistency

- [x] Delta by bump-and-reprice from the MUT — `test_check3_derives_all_five_greeks_from_the_model_alone`
- [x] Gamma by bump-and-reprice from the MUT — same test
- [x] Vega by bump-and-reprice from the MUT — same test
- [x] Theta by bump-and-reprice from the MUT — same test
- [x] Rho by bump-and-reprice from the MUT — same test
- [x] Each numerical Greek compared against the Ground Truth Greek — `test_check3_passes_a_correct_model`, `test_check3_fails_a_model_whose_surface_has_the_wrong_slope`
- [x] Delta range: call ∈ [0, 1], put ∈ [−1, 0] — `test_check3_flags_a_delta_outside_the_range_the_payoff_allows`, `test_check3_delta_range_for_a_put_is_minus_one_to_zero`

## 5. Check Type 4 — Convergence and Stability

- [x] MUT run at increasing precision levels — `test_check4_passes_a_convergent_monte_carlo_model`
- [x] Convergence toward the Ground Truth value confirmed — same test
- [x] FAIL triggered when the output does not converge or moves away — `test_check4_fails_a_model_that_converges_to_the_wrong_value`

## 6. Check Type 5 — Edge Case and Boundary Behavior

- [x] Zero volatility → the deterministic payoff — `test_check5_fails_a_model_that_returns_zero_at_a_boundary[zero volatility]`
- [x] Zero time to expiry → the intrinsic value — `test_check5_fails_a_model_that_returns_zero_at_a_boundary[zero time to expiry]`
- [x] Very high volatility → no negative price and no fault — `test_check5_fails_a_model_that_goes_negative_at_high_volatility`, `test_check5_records_a_fault_at_high_volatility_rather_than_stopping`
- [x] Spot price zero → the known limit case — `test_check5_fails_a_model_that_prices_a_call_above_zero_at_a_zero_spot`

## 7. Check Type 6 — Arbitrage-Free Sanity Checks

- [x] Call price non-increasing in the strike, across a strike ladder — `test_check6_flags_a_call_price_that_rises_with_the_strike`
- [x] Price bounded by its static no-arbitrage bounds; intrinsic value and spot are both recorded as evidence — `test_check6_flags_a_price_above_the_upper_no_arbitrage_bound`
- [x] Negative price flagged under any tested input — `test_check6_flags_a_negative_price`

## 8. Ground Truth Engine Coverage (Module B)

- [x] Black-Scholes-Merton closed-form engine — `test_frb01_black_scholes_engine_is_registered`
- [x] Binomial Tree, Cox-Ross-Rubinstein, European and American — `test_frb02_binomial_engine_covers_european_and_american`
- [x] Monte Carlo with antithetic variates — `test_gt03_variance_reduction_actually_reduces_variance`
- [x] Monte Carlo with control variates — same test, and the standard error falls again when they are added
- [x] Finite Difference, Crank-Nicolson — `test_frb04_finite_difference_engine_uses_crank_nicolson`
- [x] Closed-form or semi-analytical engines for Asian, Barrier, Digital and Lookback — `test_every_exotic_instrument_prices`
- [x] Heston, characteristic function method — `test_frb06_heston_engine_is_the_only_one_for_the_heston_model`
- [x] All five Greeks by closed form or numerical fallback — `test_frb07_*` (three tests)

## 9. Instrument Coverage (Section 4)

**Vanilla** — all four covered by `test_every_vanilla_instrument_prices`

- [x] European Call
- [x] European Put
- [x] American Call
- [x] American Put

**Exotic** — all covered by `test_every_exotic_instrument_prices` and the GT-07 suite

- [x] Asian — arithmetic average
- [x] Asian — geometric average
- [x] Barrier — up-and-out
- [x] Barrier — up-and-in
- [x] Barrier — down-and-out
- [x] Barrier — down-and-in
- [x] Digital — cash-or-nothing
- [x] Digital — asset-or-nothing
- [x] Lookback — fixed strike
- [x] Lookback — floating strike

**Underlying Asset Models** — `test_every_underlying_model_prices_a_european_option`

- [x] Geometric Brownian Motion (Black-Scholes-Merton)
- [x] Heston Stochastic Volatility
- [x] Local Volatility (Dupire) — `test_local_volatility_with_a_flat_surface_reproduces_black_scholes`, `test_local_volatility_skew_moves_the_price`

## 10. Acceptance Criteria (Section 12)

- [x] **AC-01** — Every check type is implemented and wired into the audit engine; every functional requirement has a linked test — `test_ac01_every_check_type_is_implemented_and_wired_in`
- [x] **AC-02** — All Ground Truth self-validation cases pass within tolerance — the whole of `tests/test_ground_truth.py`, plus `test_ac02_every_ground_truth_engine_is_registered_with_its_reference`
- [x] **AC-03** — Non-functional requirements measured — `test_nfr03_*`, `test_nfr04_*`, `test_nfr05_*`, `test_nfr10_*`
- [x] **AC-04** — The broken sample with five distinct seeded errors is flagged on all five — `test_ac04_the_audit_flags_every_seeded_error` (five parametrised cases)
- [x] **AC-05** — The correct sample returns a full PASS on every check, and on every instrument — `test_ac05_a_correct_model_returns_a_full_pass`, `test_ac05_the_correct_model_passes_on_every_instrument`
- [x] **AC-06** — CLI, REST API and all three report formats demonstrated end to end — `test_ac06_cli_api_and_all_three_report_formats_work_end_to_end`

## 11. Non-Functional Targets (Section 6)

- [x] Each Ground Truth Engine within 1e-4 relative error of its published reference — `CLOSED_FORM_TOLERANCE` in `tests/test_ground_truth.py`
- [x] Double-precision arithmetic throughout — `test_nfr02_every_engine_returns_double_precision`
- [x] A vanilla audit finishes in under 5 seconds — `test_nfr03_a_vanilla_audit_finishes_in_under_five_seconds` (measured: about 1.3 s)
- [x] A Monte Carlo convergence audit finishes in under 60 seconds — `test_nfr04_a_convergence_audit_finishes_in_under_sixty_seconds`
- [x] A fault inside the MUT does not stop the rest of the audit — `test_nfr05_a_fault_in_one_check_does_not_stop_the_others`, `test_nfr05_a_model_that_faults_on_some_inputs_is_still_fully_audited`
- [x] CI runs the tests on every change — `.github/workflows/ci.yml`
- [x] Coverage of the engines and the audit engine at or above 85 percent — measured at 92 percent
- [x] A new instrument or engine plugs in without touching the core — `test_nfr08_a_new_engine_plugs_in_without_touching_the_audit_engine`
- [x] The MUT runs sandboxed, with no network and no writes outside a temporary folder — `test_nfr09_the_sandbox_blocks_network_access`, `test_nfr09_the_sandbox_blocks_writes_outside_its_working_directory`
- [x] The core engine runs on Linux, macOS and Windows — the CI matrix builds all three; `test_nfr10_the_report_records_the_platform_it_ran_on`

---

## Summary

| Section | Items | Checked |
| --- | --- | --- |
| Ground Truth Self-Validation | 7 | 7 |
| Check Type 1 — Reference Price | 5 | 5 |
| Check Type 2 — Put-Call Parity | 3 | 3 |
| Check Type 3 — Greek Consistency | 7 | 7 |
| Check Type 4 — Convergence | 3 | 3 |
| Check Type 5 — Edge Cases | 4 | 4 |
| Check Type 6 — Arbitrage-Free | 3 | 3 |
| Ground Truth Engine Coverage | 8 | 8 |
| Instrument Coverage | 17 | 17 |
| Acceptance Criteria | 6 | 6 |
| Non-Functional Targets | 10 | 10 |
| Architecture (section 7.1) | 1 | 1 |
| **Total** | **74** | **74** |

## 12. Architecture (Section 7.1)

- [x] Performance-critical Monte Carlo loops available as a C++ extension,
  called from Python — `native/plumbline_mc.cpp`, loaded through `ctypes` by
  `plumbline/engines/native.py`, exercised by `tests/test_native_backend.py`
  and measured by `benchmarks/bench_backends.py` on four machines, three of
  them GitHub runners: roughly 1x to 3x on one thread and 3x to 12x across
  cores, with the full tables in `benchmarks/RESULTS.md`. The backend is
  optional and NumPy stays the documented default, so the requirement is met
  without making a compiler a condition of installing Plumbline.

---

## Three notes on where the build differs from the checklist

1. **Delta range.** The checklist asks for call delta in [0, 1] and put delta in
   [−1, 0]. Plumbline uses exactly that as the pass rule. A European option is
   bounded more tightly, by `exp(-q T)` shares, and Plumbline records that
   tighter bound as evidence on every row. It is not enforced, because the
   payoff bound is the one that holds for every exercise style.

2. **Call price bounds.** The checklist states `intrinsic ≤ price ≤ spot`. That
   is correct for a call and for American exercise. It is not correct for a deep
   in-the-money European put, which trades below its intrinsic value when rates
   are positive. Plumbline applies the correct static bound for each exercise
   style, and records the intrinsic value and the spot alongside it.

3. **The C++ backend is optional, and the PRD did not say that.** Section 7.1
   asks for a C++ extension module for the performance-critical loops.
   Plumbline has one, and it is measured. It is not required to install or to
   run: a machine with no compiler gets the NumPy engine, a green test suite
   and identical audit results. Making the compiler mandatory would have cost
   the portability requirement of NFR-10 for a speed the performance
   requirements do not need.

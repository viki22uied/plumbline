# Mutation testing

`run_mutation.py` injects one plausible quant bug at a time into the engines
or the audit logic, runs `pytest -m "fast or integration"` against the
patched tree, restores the file, and records whether the suite caught it.

```bash
python tests/mutation/run_mutation.py             # all mutants
python tests/mutation/run_mutation.py --only M01  # one mutant
python tests/mutation/run_mutation.py --strict    # nonzero exit on a survivor
python tests/mutation/run_mutation.py --json out.json
```

## Rules this harness enforces on itself

1. **Targets are unique strings.** The harness refuses to patch if the
   `old` text appears zero or several times in the file.
2. **Clean tree required.** It refuses to start when any target file has
   uncommitted changes, because it rewrites those files in place.
3. **Survivors are evidence, not noise.** A mutant that survives is either a
   missing check in the suite -- fix that by pinning the behaviour with a new
   test, as was done for the lookback branches -- or an equivalent mutant,
   which gets documented with reasoning in BENCHMARKS.md Table 4. Deleting
   the mutant or loosening an assertion to force a kill is forbidden.

## Provenance note

An earlier mutation run (27 hand-injected bugs, 26 killed) was performed by
editing files manually and was never committed as a script, so it cannot be
re-run from this repository. The mutant set here is committed and reproducible;
BENCHMARKS.md reports both runs separately and labels each accordingly.

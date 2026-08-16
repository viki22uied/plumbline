# Repository metadata

Paste the values below into the GitHub repository settings. GitHub does not read
them from a file.

---

## Repository name

```
plumbline
```

## Description (the "About" field, 350 character limit)

```
Independent verification engine for derivative pricing models. Audits any pricing model against closed-form, lattice, PDE and Monte Carlo ground truth, then reports what is wrong in plain language. Vanilla and exotic options, Black-Scholes, Heston and local volatility.
```

## Website

```
https://viki22uied.github.io/plumbline
```

Leave this empty until the documentation site exists.

## Topics

GitHub allows twenty topics. These are the twenty, in order of relevance.

```
quantitative-finance
option-pricing
model-validation
model-risk-management
derivatives
black-scholes
monte-carlo
finite-difference
heston-model
local-volatility
binomial-tree
exotic-options
greeks
numerical-methods
financial-engineering
stochastic-calculus
arbitrage-free
quant
python
fastapi
```

Further topics that describe the project, for a search or for a release note,
if the limit is ever raised:

```
barrier-options  asian-options  lookback-options  digital-options
put-call-parity  crank-nicolson  variance-reduction  control-variates
antithetic-variates  brownian-bridge  risk-management  audit
validation-framework  numpy  scipy  pytest  rest-api  cli
```

## Settings to enable

- Issues: on
- Discussions: on
- Wiki: off. The documentation lives in the repository.
- Projects: off
- Preserve this repository: on
- Include in the home page: on

## Branch protection for `main`

- Require a pull request before merging.
- Require the `CI` status check to pass.
- Require branches to be up to date before merging.
- Do not allow force pushes.
- Do not allow deletions.

## Social preview

A social preview image is not committed. Generate one from the badge row of the
README if a preview is wanted.

## Release checklist

1. Update the version in `pyproject.toml` and in `plumbline/__init__.py`.
2. Add the release section to `CHANGELOG.md`.
3. Confirm the CI matrix is green on all three platforms.
4. Tag the release: `git tag -a v1.0.0 -m "Plumbline 1.0.0"`.
5. Attach the audit reports of both sample models to the release.

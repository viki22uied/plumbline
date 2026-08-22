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
Checks a derivative pricing model against closed-form, lattice, PDE and Monte Carlo reference engines, then reports what is wrong in plain language. Vanilla and exotic options; Black-Scholes, Heston and local volatility.
```

## Website

```
https://viki22uied.github.io/plumbline
```

Leave this empty until the documentation site exists.

## Topics

Fifteen, all of them things a person would actually search for. Buzzword tags
that describe the stack rather than the subject were dropped.

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
python
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

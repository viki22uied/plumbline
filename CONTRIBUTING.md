# Contributing to Plumbline

Thank you for your interest. This document tells you how to make a change that
can be merged.

This document uses ASD-STE100 Simplified Technical English.

## 1. The rule that matters most

Plumbline judges other people's mathematics. Plumbline must therefore be right.

Every change to a Ground Truth Engine needs a test that pins the new behaviour
against something outside Plumbline. Use one of these:

1. a value published in a textbook or a paper, with the citation in the test;
2. a second engine that shares no code with the one you changed;
3. an identity that holds without a model, such as put-call parity or the
   equality of a knock-in plus a knock-out and the vanilla.

A test that compares an engine only against itself proves nothing. It will not
be merged.

## 2. Set up

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -e ".[dev]"
```

Run the suite before you change anything, so you know it was green when you
started:

```bash
pytest
```

## 3. Make the change

- Keep the diff small. One change per pull request.
- Match the style of the file you are editing.
- Write the docstring first. If you cannot say what a function does in one
  sentence, the function does more than one thing.
- Name the reference. Every numerical method in `plumbline/engines/` states
  where it comes from, in the module docstring and in the engine registry.
- Record a deliberate shortcut. Write a comment that starts with `ponytail:`,
  name the ceiling, and name the upgrade path.

## 4. Add an engine

Do not edit the Validation and Audit Engine to add an instrument or a method.
Use the plug-in interface:

```python
from plumbline.contracts import PriceResult
from plumbline.engines.registry import EngineSpec, register

register(EngineSpec(
    name="my_engine",
    description="one sentence on what this engine does",
    reference="the paper it implements",
    price_fn=my_price_function,
    instruments=("european",),
    priority=200,
))
```

Then add a self-validation test in `tests/test_ground_truth.py`. Follow the
pattern of the `GT-` tests already there.

## 5. Add a check type

A new check type needs four things:

1. a function in `plumbline/audit/checks.py` with the shared signature;
2. an entry in `ALL_CHECKS` and in `CHECK_NAMES`;
3. a weight in `CHECK_WEIGHTS`, and the other weights adjusted to sum to 1.0;
4. two tests: one model that passes the check, and one model built to break
   exactly that property.

Write a plain-language explanation for every failure path. A reader without a
quantitative finance background must be able to act on it. That is a
requirement, not a nicety.

## 6. Before you open a pull request

Run the full suite:

```bash
pytest
```

Confirm the coverage floor:

```bash
pytest --cov=plumbline/engines --cov=plumbline/audit --cov-report=term-missing
```

Confirm the two samples still behave:

```bash
plumbline audit samples/good_model.py --no-history
```

```bash
plumbline audit samples/broken_model.py --no-history
```

The first must give badge PASS. The second must give badge FAIL, and must flag
all five of its seeded errors.

Do not edit `samples/broken_model.py` or `samples/biased_mc_model.py`. Their
defects are the test fixtures.

## 7. Commits

Write the subject in the imperative, under 72 characters. Say what the commit
does and why, not how.

```
add Rannacher start-up to the Crank-Nicolson scheme
```

Do not add trailers to commit messages.

## 8. Reporting a defect

Open an issue. Include:

- the exact parameters that reproduce it;
- what you expected, and where that expectation comes from;
- what Plumbline returned;
- the version, from `plumbline --version`.

A defect in a Ground Truth Engine is the most serious kind of defect this
project can have. Report it as such and it will be treated that way.

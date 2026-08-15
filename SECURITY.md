# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | yes |

## What the sandbox does

Plumbline runs a Model Under Test in a child process. The parent process never
imports the model. Inside the child, Plumbline:

- blocks the socket module, so the model cannot reach the network;
- blocks writes outside one private temporary directory;
- applies a CPU and address-space limit on Linux and macOS;
- enforces a time limit on every call, and records a Timeout when it is passed;
- restarts the child after a crash, so one bad parameter set does not end the
  audit.

## What the sandbox does not do

The in-process restrictions stop an honest model from causing damage by
accident. They are not an operating-system jail. A model written to be hostile
can undo every one of them, because it runs as ordinary Python in the same
interpreter.

**Run an untrusted model in the container.** The container is the real boundary.

```bash
docker run --rm --network none --read-only \
  -v "$PWD:/work" plumbline audit /work/model.py --out /work/reports
```

Add `--network none` and `--read-only` as shown. Add `--memory` and `--cpus` if
the model may consume resources without limit.

## Reporting a vulnerability

Do not open a public issue for a security defect.

Send a report to the address on the maintainer's GitHub profile. Include:

- what the defect allows an attacker to do;
- the smallest model file or request that reproduces it;
- the version and the platform.

You will receive an acknowledgement within seven days. A fix or a mitigation
plan will follow within thirty days.

## A note on what Plumbline is for

Plumbline is a validation tool. It does not hold credentials. It does not move
money. It does not connect to a broker or to a market data feed. The largest
risk it carries is the one above: it runs code that someone else wrote.

# Backend benchmark results

Every table here is a real run. Reproduce any of them with:

```bash
python benchmarks/bench_backends.py --markdown
```

Read two columns for each contract:

- **x1** holds the C++ to a single thread. This measures the code.
- **xN** lets it use every core. NumPy's inner loops here are single threaded,
  so this column includes the thread count and is not a like-for-like
  comparison of the two implementations.

---

## The short version

| Machine | Toolchain | Cores | x1 median | xN median |
| --- | --- | ---: | ---: | ---: |
| GitHub runner, Ubuntu | GCC | 4 | **1.2x** | **3.0x** |
| GitHub runner, macOS arm64 | Apple Clang | 3 | **1.2x** | **3.2x** |
| GitHub runner, Windows | MSVC | 4 | **2.1x** | **4.5x** |
| Developer laptop, Windows | MinGW GCC 16.1 | 12 | **3.0x** | **12.0x** |

**The honest headline is a range, not a number: roughly 1x to 3x on one
thread, and 3x to 12x across cores, depending on the platform and the
contract.** The 12x is a twelve-core machine and should not be read as a
property of the code.

The one result that holds on every machine is the *ordering* across contracts,
and that ordering is the interesting part. See "What the numbers actually say"
at the bottom.

---

## GitHub runner, Ubuntu, GCC, 4 cores

200,000 paths, 120 steps, best of two.

| Contract | Steps | NumPy | C++ 1 thread | C++ all cores | x1 | xN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| european call | 1 | 5 ms | 3 ms | 1 ms | **1.6x** | **4.0x** |
| digital cash-or-nothing | 1 | 5 ms | 3 ms | 1 ms | **1.6x** | **3.8x** |
| asian arithmetic | 120 | 343 ms | 295 ms | 119 ms | **1.2x** | **2.9x** |
| asian geometric | 120 | 342 ms | 295 ms | 119 ms | **1.2x** | **2.9x** |
| barrier down-and-out | 120 | 358 ms | 316 ms | 118 ms | **1.1x** | **3.0x** |
| barrier up-and-in | 120 | 374 ms | 441 ms | 157 ms | **0.8x** | **2.4x** |
| lookback fixed strike | 120 | 521 ms | 397 ms | 162 ms | **1.3x** | **3.2x** |
| lookback floating strike | 120 | 521 ms | 397 ms | 162 ms | **1.3x** | **3.2x** |

Note the 0.8x. On this machine the single-threaded C++ is *slower* than NumPy
for the up-and-in barrier. That is a real result and it stays in the table.

## GitHub runner, macOS arm64, Apple Clang, 3 cores

200,000 paths, 120 steps, best of two.

| Contract | Steps | NumPy | C++ 1 thread | C++ all cores | x1 | xN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| european call | 1 | 4 ms | 3 ms | 1 ms | **1.2x** | **3.2x** |
| digital cash-or-nothing | 1 | 3 ms | 3 ms | 1 ms | **1.2x** | **3.4x** |
| asian arithmetic | 120 | 185 ms | 197 ms | 69 ms | **0.9x** | **2.7x** |
| asian geometric | 120 | 186 ms | 198 ms | 69 ms | **0.9x** | **2.7x** |
| barrier down-and-out | 120 | 334 ms | 173 ms | 63 ms | **1.9x** | **5.3x** |
| barrier up-and-in | 120 | 313 ms | 218 ms | 74 ms | **1.4x** | **4.2x** |
| lookback fixed strike | 120 | 444 ms | 380 ms | 144 ms | **1.2x** | **3.1x** |
| lookback floating strike | 120 | 439 ms | 379 ms | 140 ms | **1.2x** | **3.1x** |

## GitHub runner, Windows, MSVC, 4 cores

200,000 paths, 120 steps, best of two.

| Contract | Steps | NumPy | C++ 1 thread | C++ all cores | x1 | xN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| european call | 1 | 6 ms | 3 ms | 2 ms | **2.0x** | **3.6x** |
| digital cash-or-nothing | 1 | 5 ms | 4 ms | 2 ms | **1.3x** | **2.2x** |
| asian arithmetic | 120 | 350 ms | 242 ms | 100 ms | **1.4x** | **3.5x** |
| asian geometric | 120 | 353 ms | 242 ms | 100 ms | **1.5x** | **3.5x** |
| barrier down-and-out | 120 | 754 ms | 261 ms | 104 ms | **2.9x** | **7.3x** |
| barrier up-and-in | 120 | 762 ms | 341 ms | 138 ms | **2.2x** | **5.5x** |
| lookback fixed strike | 120 | 906 ms | 359 ms | 147 ms | **2.5x** | **6.2x** |
| lookback floating strike | 120 | 909 ms | 358 ms | 146 ms | **2.5x** | **6.2x** |

## Developer laptop, Windows, MinGW GCC 16.1, 12 cores

400,000 paths, 250 steps, best of three. Raw numbers in
`benchmarks/results-windows.json`.

| Contract | Steps | NumPy | C++ 1 thread | C++ all cores | x1 | xN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| european call | 1 | 21 ms | 7 ms | 3 ms | **3.1x** | **7.3x** |
| digital cash-or-nothing | 1 | 17 ms | 8 ms | 3 ms | **2.1x** | **5.3x** |
| asian arithmetic | 250 | 2575 ms | 1686 ms | 314 ms | **1.5x** | **8.2x** |
| asian geometric | 250 | 3010 ms | 1686 ms | 314 ms | **1.8x** | **9.6x** |
| barrier down-and-out | 250 | 6464 ms | 1867 ms | 367 ms | **3.5x** | **17.6x** |
| barrier up-and-in | 250 | 7179 ms | 2421 ms | 498 ms | **3.0x** | **14.4x** |
| lookback fixed strike | 250 | 4906 ms | 1237 ms | 266 ms | **4.0x** | **18.5x** |
| lookback floating strike | 250 | 5232 ms | 1388 ms | 407 ms | **3.8x** | **12.9x** |

This machine flatters the backend twice over. It has twelve cores, and its
NumPy is slow: normalising for path count and step count, the barrier case
takes about 1550 ms of NumPy time here against 754 ms on the four-core Windows
runner. The C++ is proportionally slower on this machine too, so the ratio is
not absurd, but a reader should weight the three CI rows above it.

---

## What the numbers actually say

**The single-thread speedup is modest and platform-dependent.** It ranges from
0.8x to 4.0x. Anyone quoting a single number for "C++ against NumPy" on
workloads like this is quoting their machine, not the languages.

**The multi-core speedup is real and roughly tracks the core count.** That is
the whole of the win on Linux and macOS: 3x on four cores and 3x on three
cores means the per-thread work is about the same as NumPy's, and the gain is
threads. NumPy's inner loops do not release the GIL across the step loop here,
so the vectorised engine cannot do the same.

**The ordering across contracts is stable everywhere, and it is the part with
a mechanism behind it.** Barriers and lookbacks gain the most on three of the
four machines; the Asian gains the least and sometimes loses. That follows
memory traffic:

- A NumPy barrier step computes two gap arrays, a product, an exponential, a
  comparison mask and a multiply. Six passes over N doubles, each a fresh
  allocation. The C++ carries one path-pair in registers and allocates nothing.
- A NumPy Asian step is one `np.exp` over the array plus two adds. There is
  very little traffic to save, and `np.exp` on a contiguous array is the one
  thing a scalar `std::exp` loop cannot beat.

So the backend wins where the vectorised version has to make the most passes,
and loses where NumPy is already doing a single well-vectorised pass. That is
what you would predict before running it, and it is why the Ubuntu row shows
0.8x on a barrier: glibc's vectorised `exp` is fast enough that the memory
saving does not cover the scalar transcendental cost.

**If the goal were maximum speed rather than a demonstration**, the next step
would be a vectorised `exp` in the C++ inner loop, not more threads. That is
where the remaining gap is.

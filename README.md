# Big Data (ULPGC 2026-2027) – Individual Assignment 1
## Basic dense matrix multiplication in Python, Java and C

Author: Hugo González Fernández · Bachelor’s Degree in Data Science and Engineering, ULPGC

Research question: *How do runtime and memory use of the textbook O(n³) triple-loop multiplication of
n × n `float64` matrices grow with n in C, Java and Python, under one documented environment?*

The report (`Assignment01_GonzalezFernandez_Report.pdf`) describes the design, results and limits.
This README gives everything needed to rebuild and re-run the study.

---------------------------------------------------------------------------------------------------

## 1. Repository layout

```
src/c/matmul.c            C implementation (contiguous row-major double[n*n])
src/java/MatMul.java      Java implementation (double[n][n])
src/java/MatMulFlat.java  SUPPLEMENTARY Java variant with one flat double[n*n] (layout experiment)
src/python/matmul.py      pure-Python implementation (list of lists of float)
scripts/gen_inputs.py     shared input matrices (binary files, seeded, SHA-256 manifest)
scripts/test_correctness.py   known-answer + reference + cross-language tests
scripts/run_benchmarks.py     measurement driver (repetitions, memory, budgets, validation)
scripts/env_info.py       records hardware/software environment -> results/environment.json
scripts/analyze.py        tables (CSV + LaTeX) and figures from the raw data
results/                  raw_measurements.csv, runs.csv, summary_*.csv, environment.json, logs
figures/                  figures used in the report (PDF + PNG)
report/                   report.tex (+ generated tables/) -> report.pdf
data/manifest.json        SHA-256 of every generated input (the .bin files are NOT committed)
Makefile                  build / inputs / test / bench / cflags / analyze / report / all
```

## 2. Requirements (versions used in the reported run)

| Tool | Version |
|---|---|
| Linux x86-64 | Ubuntu 24.04, kernel 6.x |
| gcc | 13.3.0 |
| OpenJDK (javac, java) | 21.0.12 |
| Python | 3.13 (CPython) |
| Python packages | `numpy 2.5.3`, `pandas 3.0.5`, `matplotlib 3.11.2`, `scipy` (optional) – see `requirements.txt` |
| LaTeX (only to rebuild the PDF) | `latexmk`, `pdflatex` |

NumPy is used **only** for input generation, for the correctness reference and for the analysis. It is never
used inside the timed multiplication kernels.

## 3. Exact build and run commands

```bash
make build        # gcc -O2 -std=c11 -ffp-contract=off  (+ the -O0/-O1/-O3/-O3 -march=native variants), javac
make inputs       # python3 scripts/gen_inputs.py <all sizes of the main and supplementary campaigns>
make test         # 52 correctness checks, writes results/correctness_tests.csv
make bench        # main campaign (about 2 h of wall time on the reported machine; resumable with --resume)
make cflags       # SUPPLEMENTARY, exploratory, kept separate from the main comparison: C compiler flags,
                  # sizes around powers of two, Java flat-array layout, between-process variability (n=1024, 6 processes)
make analyze      # tables + figures
make report       # builds report/report.pdf
make all          # everything above, in order
```

Single run by hand (same command-line interface in all three languages):

```bash
# <n> <repetitions> <warm-up samples> <batch> <A.bin> <B.bin> [dump C to file]
build/matmul_c                                   1000 5 1 1 data/A_n1000.bin data/B_n1000.bin
java -XX:+UseSerialGC -Xms256m -Xmx4g -cp build/java MatMul 1000 5 1 1 data/A_n1000.bin data/B_n1000.bin
python3 src/python/matmul.py                     1000 5 1 1 data/A_n1000.bin data/B_n1000.bin
```

Output lines: `WARMUP/REP <index> <seconds per multiplication> <checksum>`, `RSS_AFTER_LOAD_KB`, `VMHWM_KB`.

> If you run on a different machine the absolute numbers change; delete `results/*.csv` first (or use a new
> folder) so that the old measurements are not mixed with the new ones, and re-run `make bench cflags analyze`.

## 4. Inputs

* Files `data/A_n<N>.bin` / `data/B_n<N>.bin`: raw little-endian IEEE-754 binary64, row-major, `n·n` values.
* Generator: `numpy.random.default_rng([seed, n, which])`, `seed = 20261012`, `which = 0` (A) or `1` (B);
  values uniform in [0, 1). All three languages read **the same files**, so the logical inputs are identical.
* `data/manifest.json` stores the SHA-256 of each file – compare after regenerating to confirm identical inputs.

## 5. Fixed settings (identical in every comparison)

| Aspect | Setting |
|---|---|
| Algorithm | triple loop i-j-k, `s += A[i][k] * B[k][j]`, `C[i][j] = s` (scalar accumulator), O(n³) |
| Numeric type | IEEE-754 binary64 (`double`, Java `double`, Python `float`) |
| Threads | one (single-threaded kernels). JVM JIT/GC helper threads may use the second vCPU. No CPU pinning |
| C | `gcc -O2 -std=c11 -ffp-contract=off` (no `-march=native`, no `-ffast-math`) → no FMA, no re-association |
| Java | HotSpot default tiered JIT; `-XX:+UseSerialGC -Xms256m -Xmx4g` |
| Python | CPython, default build, cyclic GC disabled inside the timed region |
| Timer | monotonic: `clock_gettime(CLOCK_MONOTONIC)`, `System.nanoTime()`, `time.perf_counter_ns()` |
| Timed region | **kernel only**. File loading, allocation, zero-initialisation of C, checksum and dump are outside |
| Warm-up | 3 samples if the estimated multiplication ≤ 1 s, 1 sample if ≤ 30 s, none above (always recorded, never in statistics) |
| Repetitions | 10 samples (≤ 1 s), 5 (≤ 30 s), 3 (≤ 120 s), 2 (> 120 s) |
| Tiny n | for very short multiplications a batch of `m` products is timed and divided by `m` (`batch_size` column) |
| Memory metric | peak resident set size `VmHWM` of the process (`/proc/self/status`), read just before exit |

Storage layouts differ unavoidably: C – one contiguous block; Java – an array of separate row arrays;
Python – lists of separately boxed `float` objects (≈ 40 bytes per element incl. pointer). See the report.

## 6. Budgets

* **Time:** a single timed sample (one multiplication, or one batch for tiny n) must finish within **300 s**,
  otherwise the run is killed and recorded as `timeout` (no time is invented).
* **Memory:** peak resident memory ≤ **4 GiB** per run (polled every 0.5 s; JVM `-Xmx4g`).
* The driver stops increasing n for an implementation after its first timeout / memory failure, and does not
  attempt sizes whose predicted time exceeds 3× the limit. These cases are listed in `results/runs.csv`
  (`status = timeout | memory_limit | not_attempted`) together with the reason.

## 7. Correctness

* `scripts/test_correctness.py`: scalar, hand-computed 3×3, 2×2 non-commutative, zero, identity (left and
  right), all-ones, integer matrices (exact), random matrices vs NumPy, and **bit-for-bit equality across the
  three languages**.
* Every benchmark run additionally dumps its final product and compares it element-wise with NumPy's
  `A @ B`. Accepted relative error: `2·γₙ`, `γₙ = n·u/(1−n·u)`, `u = 2⁻⁵³` (valid because all entries are
  non-negative, so the error of any summation order is ≤ γₙ·|c|). Checksum consistency is also verified.

## 8. Raw data and column dictionary

`results/raw_measurements.csv` – one row per sample: `campaign, run_id, language, config, method,
numeric_type, n, thread_count, phase (warmup|measured), repetition, batch_size, time_per_mult_s, checksum`.

`results/runs.csv` – one row per process: status, reason, planned/completed repetitions, limits, wall time,
`vmhwm_kib` (peak RSS, primary metric), `rss_after_load_kib`, `ru_maxrss_kib` (kernel accounting, cross-check),
correctness fields (`max_rel_err`, `rtol_used`, `sha256_C`), return code, stderr tail.

`results/summary_*.csv` and `report/tables/*.tex` are generated by `scripts/analyze.py`.

## 9. Use of generative AI (course policy)

Tool: Claude (Anthropic). Purpose: planning of the experimental design, drafting of the benchmark harness,
test scripts, analysis scripts and the first draft of the report text. Parts affected: `scripts/`, `src/`,
`report/`. The author remains responsible for understanding the code, validating the measurements, verifying
the references and defending the conclusions. No result or reference was taken from AI output without
being produced by the included code or checked against its source.

## 10. Important notes on the reported results

- **Execution environment:** The reported measurements were produced on a dedicated virtual machine (2 vCPU Intel Xeon @ 2.8 GHz, 7.8 GiB RAM; see `results/environment.json`). To obtain measurements for another machine, move or delete `results/*.csv`, execute `make bench cflags analyze report`, and update the narrative interpretations in the report accordingly (figures and tables are regenerated automatically from the raw data).
- **Limits and budgets:** A strict time budget of 300 s per multiplication sample was enforced; runs exceeding this limit are recorded as timeouts (treated as valid outcome limits). The supplementary experiments (Section 3.10 of the report) were designed after observing the initial benchmark results and serve an exploratory purpose.

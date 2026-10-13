#!/usr/bin/env python3
"""
Correctness tests for the three implementations (C, Java, Python), run BEFORE any benchmark.

Exact tests (expected result is exactly representable -> bit-for-bit equality required):
    n=1 scalar, hand-computed 3x3, 2x2 non-commutative case, zero matrix, identity (left and
    right), all-ones, small-integer random matrices (every partial sum < 2**53).
Floating-point tests (random uniform [0,1)): compared with NumPy's float64 A @ B using the
    a-priori relative bound 2*gamma_n (gamma_n = n*u/(1-n*u), u = 2**-53). Valid because all
    entries are >= 0, so |c_hat - c| <= gamma_n * c for ANY order of summation; the naive
    triple loop and NumPy/BLAS therefore differ by at most 2*gamma_n relative.
Cross-language test: the three implementations use the same operation order and no fused
    multiply-add, so their results must be BIT-FOR-BIT identical (compared via SHA-256).

Writes results/correctness_tests.csv. Exit status 1 if any test fails.
"""
import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
U = 2.0 ** -53


def gamma(k):
    return k * U / (1.0 - k * U)


IMPLS = {
    "c": [str(ROOT / "build" / "matmul_c")],
    "java": ["java", "-XX:+UseSerialGC", "-Xmx1g", "-cp", str(ROOT / "build" / "java"), "MatMul"],
    "python": [sys.executable, str(ROOT / "src" / "python" / "matmul.py")],
}


def run(lang, n, A, B, tmp):
    pa, pb, pc = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
    np.ascontiguousarray(A, dtype="<f8").tofile(pa)
    np.ascontiguousarray(B, dtype="<f8").tofile(pb)
    if pc.exists():
        pc.unlink()
    env = {k: v for k, v in os.environ.items() if k != "JAVA_TOOL_OPTIONS"}
    r = subprocess.run(IMPLS[lang] + [str(n), "1", "0", "1", str(pa), str(pb), str(pc)],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0 or not pc.exists():
        raise RuntimeError("%s failed: %s" % (lang, r.stderr[-300:]))
    return np.fromfile(pc, dtype="<f8").reshape(n, n)


def main():
    rng = np.random.default_rng(12345)
    cases = []   # (name, n, A, B, expected_or_None, mode)
    cases.append(("scalar 1x1 (3*4)", 1, np.array([[3.0]]), np.array([[4.0]]), np.array([[12.0]]), "exact"))
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    B3 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
    cases.append(("hand-computed 3x3", 3, A3, B3,
                  np.array([[30, 24, 18], [84, 69, 54], [138, 114, 90]], dtype=float), "exact"))
    cases.append(("2x2 non-commutative", 2, np.array([[1.0, 2.0], [3.0, 4.0]]),
                  np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[2.0, 1.0], [4.0, 3.0]]), "exact"))
    R7 = rng.random((7, 7))
    cases.append(("zero * random (n=7)", 7, np.zeros((7, 7)), R7, np.zeros((7, 7)), "exact"))
    R16 = rng.random((16, 16))
    cases.append(("identity * B (n=16)", 16, np.eye(16), R16, R16, "exact"))
    cases.append(("A * identity (n=16)", 16, R16, np.eye(16), R16, "exact"))
    cases.append(("all-ones (n=32) -> every entry 32", 32, np.ones((32, 32)), np.ones((32, 32)),
                  np.full((32, 32), 32.0), "exact"))
    I1 = rng.integers(0, 100, size=(64, 64)).astype(float)
    I2 = rng.integers(0, 100, size=(64, 64)).astype(float)
    cases.append(("random integers 0..99 (n=64), exact", 64, I1, I2,
                  (I1.astype(np.int64) @ I2.astype(np.int64)).astype(float), "exact"))
    for n in (5, 17, 33, 64, 100):
        a, b = rng.random((n, n)), rng.random((n, n))
        cases.append(("random [0,1) n=%d vs NumPy" % n, n, a, b, a @ b, "tol"))

    rows, failures = [], 0
    hashes = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, n, A, B, expected, mode in cases:
            for lang in IMPLS:
                C = run(lang, n, A, B, tmp)
                if mode == "exact":
                    ok = np.array_equal(C, expected)
                    err = float(np.max(np.abs(C - expected)))
                    tol = 0.0
                else:
                    err = float(np.max(np.abs(C - expected) / np.abs(expected)))
                    tol = 2 * gamma(n)
                    ok = err <= tol
                hashes.setdefault(name, {})[lang] = hashlib.sha256(C.tobytes()).hexdigest()
                failures += (not ok)
                rows.append(dict(test=name, n=n, language=lang, mode=mode, max_error=err,
                                 tolerance=tol, result="PASS" if ok else "FAIL"))
                print("%-42s %-7s %-5s max_err=%.3g tol=%.3g" %
                      (name, lang, "PASS" if ok else "FAIL", err, tol), flush=True)
        # cross-language bitwise identity
        for name, h in hashes.items():
            same = len(set(h.values())) == 1
            failures += (not same)
            rows.append(dict(test=name + " [bitwise identical across C/Java/Python]", n="",
                             language="all", mode="bitwise", max_error="", tolerance=0,
                             result="PASS" if same else "FAIL"))
            print("%-42s cross-language bitwise: %s" % (name, "PASS" if same else "FAIL"))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "correctness_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d checks, %d failures" % (len(rows), failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

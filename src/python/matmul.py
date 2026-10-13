#!/usr/bin/env python3
"""
Big Data (ULPGC, 2026-2027) - Individual Assignment 1
Basic dense matrix multiplication, pure-Python implementation (CPython).

Algorithm : textbook triple loop, order i-j-k, O(n^3), double precision
            (Python float = IEEE-754 binary64). No NumPy in the kernel.
Layout    : list of n lists, each holding n separately boxed float objects.
Timer     : time.perf_counter_ns() (monotonic); only the kernel is timed.
            Loading, allocation of C's row lists, checksum and dump are outside.
            The cyclic garbage collector is disabled during the timed region.

Usage: matmul.py <n> <reps> <warmup> <batch> <A.bin> <B.bin> [dumpC.bin]
batch = multiplications per timed sample; reported time is PER MULTIPLICATION.
Output format is identical to the C and Java programs.
"""
import array
import gc
import sys
import time


def read_matrix(path, n):
    rows = []
    with open(path, "rb") as f:
        for _ in range(n):
            buf = f.read(8 * n)
            if len(buf) != 8 * n:
                sys.exit("short read on %s" % path)
            a = array.array("d")
            a.frombytes(buf)
            if sys.byteorder != "little":
                a.byteswap()
            rows.append(a.tolist())  # creates one float object per element
        if f.read(1):
            sys.exit("unexpected extra data in %s" % path)
    return rows


def multiply(A, B, C, n):
    """Kernel: C = A * B, i-j-k order with a scalar accumulator."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def checksum(C, n):
    """Plain left-to-right sum, row-major. (Deliberately not built-in sum():
    since Python 3.12 sum() on floats uses compensated summation.)"""
    s = 0.0
    for i in range(n):
        row = C[i]
        for j in range(n):
            s += row[j]
    return s


def proc_status_kb(key):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def dump(C, n, path):
    with open(path, "wb") as f:
        for i in range(n):
            a = array.array("d", C[i])
            if sys.byteorder != "little":
                a.byteswap()
            a.tofile(f)


def main():
    if len(sys.argv) < 7:
        sys.exit("usage: matmul.py n reps warmup batch A.bin B.bin [dumpC.bin]")
    n = int(sys.argv[1])
    reps = int(sys.argv[2])
    warmup = int(sys.argv[3])
    batch = max(1, int(sys.argv[4]))

    impl = sys.implementation
    print("META lang=python n=%d reps=%d warmup=%d batch=%d impl=%s version=%s"
          % (n, reps, warmup, batch, impl.name, sys.version.split()[0]), flush=True)

    A = read_matrix(sys.argv[5], n)
    B = read_matrix(sys.argv[6], n)
    C = [[0.0] * n for _ in range(n)]

    print("RSS_AFTER_LOAD_KB %d" % proc_status_kb("VmRSS"), flush=True)

    gc.collect()
    gc.disable()
    for w in range(warmup):
        t0 = time.perf_counter_ns()
        for _ in range(batch):
            multiply(A, B, C, n)
        t1 = time.perf_counter_ns()
        print("WARMUP %d %.9e %.17g" % (w, (t1 - t0) * 1e-9 / batch, checksum(C, n)), flush=True)
    for r in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(batch):
            multiply(A, B, C, n)
        t1 = time.perf_counter_ns()
        print("REP %d %.9e %.17g" % (r, (t1 - t0) * 1e-9 / batch, checksum(C, n)), flush=True)
    gc.enable()

    print("VMHWM_KB %d" % proc_status_kb("VmHWM"), flush=True)
    if len(sys.argv) >= 8:
        dump(C, n, sys.argv[7])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

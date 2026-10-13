#!/usr/bin/env python3
"""
Benchmark driver for Individual Assignment 1 (basic dense matrix multiplication).

One *run* = one operating-system process executing one (language, n) configuration:
    warm-up samples (discarded from statistics, kept in the CSV) followed by
    measured samples. Every sample is the time of ONE multiplication (for tiny n
    a batch of multiplications is timed and divided by the batch size).

Declared budgets (see README / report):
    T_MULT   : maximum wall time of a single timed sample (one multiplication, or
               one batch for tiny n). A sample that does not finish within T_MULT
               seconds kills the run -> status "timeout" (a result, not a time).
    MEM_KIB  : maximum resident memory of a run (polled every 0.5 s from
               /proc/<pid>/status). Exceeding it kills the run -> "memory_limit".

Outputs (appended, flushed after every run):
    results/raw_measurements.csv : one row per sample (warm-up + measured)
    results/runs.csv             : one row per run (status, memory, correctness)

Correctness (every run): the last product is dumped to a binary file and compared
element-wise against NumPy's float64 A @ B. The accepted relative error is the
a-priori bound 2*gamma_n, gamma_n = n*u/(1-n*u), u = 2**-53 (all entries are
non-negative, so |c_hat - c| <= gamma_n * c for any summation order).
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
U = 2.0 ** -53

SIZES = [10, 50, 100, 200, 256, 400, 500, 512, 800, 1000, 1024, 1250, 1500,
         2000, 2048, 3000, 4096]
CFLAGS_SIZES = [512, 1000, 1024]
STRIDE_SIZES = [496, 504, 512, 520, 1016, 1024, 1032]   # around two powers of two
LAYOUT_SIZES = [500, 1000, 1250, 1500, 2000]
T_MULT = 300.0               # seconds per timed sample
MEM_KIB = 4 * 1024 * 1024    # 4 GiB
PRIOR_NS_PER_ITER = {"c": 2.5, "java": 2.5, "python": 60.0}  # pilot estimates (n^3 inner iterations)
JAVA_FLAGS = ["-XX:+UseSerialGC", "-Xms256m", "-Xmx4g"]
C_VARIANTS = {"O0": "-O0", "O1": "-O1", "O2": "-O2", "O3": "-O3", "O3-native": "-O3 -march=native"}

RAW_FIELDS = ["campaign", "run_id", "language", "config", "method", "numeric_type", "n",
              "thread_count", "phase", "repetition", "batch_size", "time_per_mult_s", "checksum"]
RUN_FIELDS = ["campaign", "run_id", "language", "config", "n", "numeric_type", "method",
              "thread_count", "status", "reason", "reps_planned", "warmup_planned",
              "reps_completed", "warmup_completed", "batch_size", "t_est_s", "limit_sample_s",
              "limit_mem_kib", "wall_s", "rss_after_load_kib", "vmhwm_kib", "ru_maxrss_kib",
              "predicted_data_bytes", "checksum_rel_err", "checksum_tol", "max_abs_err",
              "max_rel_err", "rtol_used", "correctness", "sha256_C", "start_utc", "returncode",
              "stderr_tail"]


# ----------------------------------------------------------------------------
def gamma(k):
    return k * U / (1.0 - k * U)


def predicted_data_bytes(lang, n):
    """Planning estimate of the three n x n matrices (not a measurement)."""
    if lang == "python":          # 8-byte pointer + 32-byte (pymalloc-rounded) float object
        return 3 * n * n * 40
    return 3 * 8 * n * n


def plan_for(t_est):
    """(reps, warmup) as a function of the estimated time of one multiplication."""
    if t_est <= 1.0:
        return 10, 3
    if t_est <= 30.0:
        return 5, 1
    if t_est <= 120.0:
        return 3, 0
    return 2, 0


def batch_for(t_est, target=0.1):
    if t_est >= 0.05:
        return 1
    return int(min(1_000_000, math.ceil(target / t_est)))


def command(lang, config, n, reps, warm, batch, a, b, dump):
    tail = [str(n), str(reps), str(warm), str(batch), str(a), str(b), str(dump)]
    if lang == "c":
        exe = ROOT / "build" / ("matmul_c" if config == "c-O2" else "cflags/matmul_c_" + config[2:])
        return [str(exe)] + tail
    if lang == "java":
        cls = "MatMulFlat" if config == "java-flat" else "MatMul"
        return ["java"] + JAVA_FLAGS + ["-cp", str(ROOT / "build" / "java"), cls] + tail
    return [sys.executable, str(ROOT / "src" / "python" / "matmul.py")] + tail


def clean_env():
    env = dict(os.environ)
    env.pop("JAVA_TOOL_OPTIONS", None)   # the sandbox injects proxy options; keep the JVM clean
    env["LC_ALL"] = "C"
    return env


def read_rss_kib(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def execute(cmd, t_limit, mem_kib):
    """Run one process; return (lines, status, reason, wall_s, ru_maxrss_kib, returncode, stderr_tail)."""
    errf = tempfile.TemporaryFile()
    t_start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, env=clean_env(),
                            start_new_session=True)
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    buf = b""
    lines = []
    last_progress = time.monotonic()
    status, reason = "ok", ""
    eof = False
    while not eof:
        for _ in sel.select(timeout=0.5):
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                eof = True
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines.append(line.decode(errors="replace").strip())
                last_progress = time.monotonic()
        if eof:
            break
        now = time.monotonic()
        if now - last_progress > t_limit:
            status, reason = "timeout", "no sample completed within %g s" % t_limit
        elif read_rss_kib(proc.pid) > mem_kib:
            status, reason = "memory_limit", "resident memory exceeded %d KiB" % mem_kib
        if status != "ok":
            os.killpg(proc.pid, signal.SIGKILL)
            break
    # drain whatever is left in the pipe, then reap the child with its resource usage
    try:
        rest = os.read(fd, 1 << 20)
        buf += rest
    except (BlockingIOError, OSError):
        pass
    for line in buf.decode(errors="replace").splitlines():
        if line.strip():
            lines.append(line.strip())
    _, wstatus, ru = os.wait4(proc.pid, 0)
    proc.returncode = 0
    wall = time.monotonic() - t_start
    rc = os.waitstatus_to_exitcode(wstatus)
    errf.seek(0)
    err_tail = errf.read()[-300:].decode(errors="replace").replace("\n", " | ")
    errf.close()
    if status == "ok" and (rc != 0 or "DONE" not in lines):
        status, reason = "error", "exit code %s" % rc
        if rc == -9:
            status, reason = "memory_limit", "killed by SIGKILL (probable out-of-memory)"
    return lines, status, reason, wall, ru.ru_maxrss, rc, err_tail


def parse(lines):
    out = {"warmup": [], "rep": [], "rss_after_load": None, "vmhwm": None, "done": False}
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if p[0] in ("WARMUP", "REP") and len(p) >= 4:
            out["warmup" if p[0] == "WARMUP" else "rep"].append((int(p[1]), float(p[2]), float(p[3])))
        elif p[0] == "RSS_AFTER_LOAD_KB":
            out["rss_after_load"] = int(p[1])
        elif p[0] == "VMHWM_KB":
            out["vmhwm"] = int(p[1])
        elif p[0] == "DONE":
            out["done"] = True
    return out


class Reference:
    """Caches the NumPy reference product for the size currently being validated."""
    def __init__(self):
        self.n = None

    def get(self, n):
        if self.n != n:
            A = np.fromfile(DATA / ("A_n%d.bin" % n), dtype="<f8").reshape(n, n)
            B = np.fromfile(DATA / ("B_n%d.bin" % n), dtype="<f8").reshape(n, n)
            self.R = A @ B
            self.sum = float(np.sum(self.R, dtype=np.longdouble))
            self.n = n
            del A, B
        return self.R, self.sum


def validate(ref, n, dump_path, checksums):
    R, Rsum = ref.get(n)
    C = np.fromfile(dump_path, dtype="<f8")
    if C.size != n * n:
        return dict(correctness="fail", reason="dump has wrong size")
    C = C.reshape(n, n)
    diff = np.abs(C - R)
    rtol = 2 * gamma(n)
    max_abs = float(diff.max())
    max_rel = float((diff / np.abs(R)).max())
    ck_tol = gamma(n * n + 2 * n)
    ck_err = max(abs(c - Rsum) for c in checksums) / Rsum if checksums else float("nan")
    ok = max_rel <= rtol and ck_err <= ck_tol and len(set(checksums)) == 1
    h = hashlib.sha256(C.tobytes()).hexdigest()
    return dict(correctness="pass" if ok else "fail", max_abs_err=max_abs, max_rel_err=max_rel,
                rtol_used=rtol, checksum_rel_err=ck_err, checksum_tol=ck_tol, sha256_C=h)


# ----------------------------------------------------------------------------
def append_csv(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(path):
    keys = set()
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                keys.add((r["campaign"], r["config"], int(r["n"])))
    return keys


def run_campaign(name, configs, sizes, args):
    RESULTS.mkdir(exist_ok=True)
    runs_csv, raw_csv = RESULTS / "runs.csv", RESULTS / "raw_measurements.csv"
    finished = done_keys(runs_csv) if args.resume else set()
    ref = Reference()
    tmp = Path(tempfile.mkdtemp(prefix="matmul_dump_"))
    # Size-major order (for each n: every language) so that slow drift of the machine
    # affects all languages alike instead of being confounded with one language.
    state = {cfg: [None, None, None] for _, cfg in configs}   # prev_n, prev_t, stop_reason
    for n in sizes:
        for lang, config in configs:
            prev_n, prev_t, stop_reason = state[config]
            key = (name, config, n)
            if key in finished:
                # recover previous median for the estimator
                with open(raw_csv, newline="") as f:
                    ts = [float(r["time_per_mult_s"]) for r in csv.DictReader(f)
                          if r["campaign"] == name and r["config"] == config
                          and int(r["n"]) == n and r["phase"] == "measured"]
                if ts:
                    prev_n, prev_t = n, statistics.median(ts)
                state[config] = [prev_n, prev_t, stop_reason]
                continue
            t_est = (prev_t * (n / prev_n) ** 3 * 1.3 if prev_t
                     else PRIOR_NS_PER_ITER[lang] * 1e-9 * n ** 3)
            reps, warm = plan_for(t_est)
            batch = batch_for(t_est)
            run_id = "%s-%s-n%d" % (name, config, n)
            base = dict(campaign=name, run_id=run_id, language=lang, config=config, n=n,
                        numeric_type="float64", method="triple-loop-ijk", thread_count=1,
                        reps_planned=reps, warmup_planned=warm, batch_size=batch,
                        t_est_s="%.6g" % t_est, limit_sample_s=T_MULT, limit_mem_kib=MEM_KIB,
                        predicted_data_bytes=predicted_data_bytes(lang, n),
                        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            if stop_reason or t_est > 3 * T_MULT or predicted_data_bytes(lang, n) > MEM_KIB * 1024:
                why = stop_reason or ("predicted %.0f s per multiplication exceeds 3x the %g s limit"
                                      % (t_est, T_MULT) if t_est > 3 * T_MULT
                                      else "predicted data size exceeds the memory budget")
                append_csv(runs_csv, RUN_FIELDS, [dict(base, status="not_attempted", reason=why,
                                                       reps_completed=0, warmup_completed=0)])
                print("[%s] %-10s n=%-5d not attempted (%s)" % (name, config, n, why), flush=True)
                continue
            dump = tmp / "C.bin"
            if dump.exists():
                dump.unlink()
            cmd = command(lang, config, n, reps, warm, batch,
                          DATA / ("A_n%d.bin" % n), DATA / ("B_n%d.bin" % n), dump)
            print("[%s] %-10s n=%-5d reps=%d warmup=%d batch=%d t_est=%.3g s ..." %
                  (name, config, n, reps, warm, batch, t_est), end=" ", flush=True)
            lines, status, reason, wall, ru_kib, rc, err_tail = execute(cmd, T_MULT, MEM_KIB)
            res = parse(lines)
            row = dict(base, status=status, reason=reason, reps_completed=len(res["rep"]),
                       warmup_completed=len(res["warmup"]), wall_s="%.3f" % wall,
                       rss_after_load_kib=res["rss_after_load"], vmhwm_kib=res["vmhwm"],
                       ru_maxrss_kib=ru_kib, returncode=rc, stderr_tail=err_tail)
            if status == "ok" and dump.exists():
                row.update(validate(ref, n, dump, [c for _, _, c in res["rep"]]))
                if row["correctness"] != "pass":
                    row["reason"] = "validation failed"
            else:
                row["correctness"] = "not_checked"
            raw = []
            for phase, items in (("warmup", res["warmup"]), ("measured", res["rep"])):
                for idx, t, chk in items:
                    raw.append(dict(campaign=name, run_id=run_id, language=lang, config=config,
                                    method="triple-loop-ijk", numeric_type="float64", n=n,
                                    thread_count=1, phase=phase, repetition=idx,
                                    batch_size=batch, time_per_mult_s=repr(t), checksum=repr(chk)))
            append_csv(raw_csv, RAW_FIELDS, raw)
            append_csv(runs_csv, RUN_FIELDS, [row])
            if res["rep"]:
                prev_n, prev_t = n, statistics.median(t for _, t, _ in res["rep"])
            msg = status if status == "ok" else "%s (%s)" % (status, reason)
            print("%s, median=%s s, peak RSS=%s MiB, check=%s" %
                  (msg, "%.4g" % prev_t if res["rep"] else "n/a",
                   "%.1f" % (res["vmhwm"] / 1024) if res["vmhwm"] else "n/a", row["correctness"]),
                  flush=True)
            if status in ("timeout", "memory_limit"):
                stop_reason = "an earlier, smaller size already hit the %s limit" % status.replace("_", " ")
            if status == "error":
                print("   stderr:", err_tail, flush=True)
            state[config] = [prev_n, prev_t, stop_reason]
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", choices=["main", "cflags", "stride", "layout", "procvar"], default="main")
    ap.add_argument("--langs", nargs="+", default=["c", "java", "python"], choices=["c", "java", "python"])
    ap.add_argument("--sizes", nargs="+", type=int)
    ap.add_argument("--resume", action="store_true", help="skip runs already present in results/runs.csv")
    ap.add_argument("--tag", default="", help="suffix for the campaign name (independent repeats of a campaign)")
    ap.add_argument("--tmult", type=float, help="override the per-sample time limit (testing only)")
    ap.add_argument("--results-dir", help="write CSVs here instead of results/ (testing only)")
    args = ap.parse_args()
    global T_MULT, RESULTS
    if args.tmult:
        T_MULT = args.tmult
    if args.results_dir:
        RESULTS = Path(args.results_dir)
        RESULTS.mkdir(parents=True, exist_ok=True)

    if args.campaign == "main":
        configs = []
        for l in args.langs:
            configs.append((l, {"c": "c-O2", "java": "java", "python": "python"}[l]))
        sizes = args.sizes or SIZES
    elif args.campaign == "cflags":
        configs = [("c", "c-" + t) for t in C_VARIANTS]
        sizes = args.sizes or CFLAGS_SIZES
    elif args.campaign == "stride":      # supplementary: sizes just below/at/above powers of two
        configs = [("c", "c-O2"), ("java", "java"), ("java", "java-flat")]
        sizes = args.sizes or STRIDE_SIZES
    elif args.campaign == "procvar":     # supplementary: between-process variability (run with --tag 1..k)
        configs = [("c", "c-O2"), ("java", "java")]
        sizes = args.sizes or [1024]
    else:                                # supplementary: Java flat-array layout vs jagged arrays
        configs = [("java", "java-flat")]
        sizes = args.sizes or LAYOUT_SIZES
    for n in sizes:
        for nm in ("A", "B"):
            if not (DATA / ("%s_n%d.bin" % (nm, n))).exists():
                sys.exit("missing input %s_n%d.bin - run scripts/gen_inputs.py first" % (nm, n))
    with open(RESULTS / ("campaign_%s.json" % (args.campaign + args.tag)), "w") as f:
        json.dump(dict(campaign=args.campaign + args.tag, sizes=sizes, configs=configs, t_mult_s=T_MULT,
                       mem_budget_kib=MEM_KIB, java_flags=JAVA_FLAGS, c_variants=C_VARIANTS,
                       rep_policy="t_est<=1s:10 reps+3 warmup; <=30s:5+1; <=120s:3+0; else 2+0",
                       batch_policy="batch=ceil(0.1 s / t_est) when t_est<0.05 s, else 1"), f, indent=2)
    run_campaign(args.campaign + args.tag, configs, sizes, args)


if __name__ == "__main__":
    main()

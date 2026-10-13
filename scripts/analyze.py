#!/usr/bin/env python3
"""
Analysis of the raw measurements: summary tables (CSV + LaTeX snippets) and figures.

Reads : results/raw_measurements.csv, results/runs.csv, results/environment.json,
        results/correctness_tests.csv
Writes: results/summary_*.csv, report/tables/*.tex, figures/*.pdf|png

Statistics: median and inter-quartile range (IQR, linear-interpolation percentiles) of the
measured samples (warm-up samples are excluded). Empirical complexity exponent: least-squares
slope of log(median time) against log(n) with a 95 % confidence interval (t distribution).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "report" / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-3 of the validated reference palette (valid for <=3 series, all pairs).
COL = {"c-O2": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARK = {"c-O2": "o", "java": "s", "python": "^"}
LABEL = {"c-O2": "C (gcc -O2)", "java": "Java (OpenJDK 21)", "python": "Python (CPython 3.13)"}
SHORT = {"c-O2": "C", "java": "Java", "python": "Python"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONFIGS = ["c-O2", "java", "python"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def fmt_t(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "--"
    if s < 1e-3:
        return "%.3g\\,$\\mu$s" % (s * 1e6)
    if s < 1:
        return "%.3g\\,ms" % (s * 1e3)
    return "%.3g\\,s" % s


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=200)
    plt.close(fig)


def write_tex(name, header, rows, colspec, caption_note=None):
    lines = [r"\begin{tabular}{%s}" % colspec, r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / (name + ".tex")).write_text("\n".join(lines) + "\n")


def t_crit(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:  # noqa: BLE001
        table = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}
        return table.get(df, 2.0)


# ----------------------------------------------------------------------------- load
raw = pd.read_csv(RES / "raw_measurements.csv")
runs = pd.read_csv(RES / "runs.csv")
env = json.loads((RES / "environment.json").read_text())
main_runs = runs[runs.campaign == "main"].copy()
main_raw = raw[raw.campaign == "main"]
meas = main_raw[main_raw.phase == "measured"]
warm = main_raw[main_raw.phase == "warmup"]

# ----------------------------------------------------------------------------- time summary
rows = []
for (cfg, n), g in meas.groupby(["config", "n"]):
    t = g.time_per_mult_s.values
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    w = warm[(warm.config == cfg) & (warm.n == n)].sort_values("repetition")
    r = main_runs[(main_runs.config == cfg) & (main_runs.n == n)].iloc[0]
    rows.append(dict(config=cfg, n=n, reps=len(t), warmups=len(w), batch=int(r.batch_size), median_s=med,
                     q1_s=q1, q3_s=q3, iqr_s=q3 - q1, rel_iqr=(q3 - q1) / med, min_s=t.min(), max_s=t.max(),
                     ns_per_inner_iter=med / n ** 3 * 1e9,
                     first_warmup_over_median=(w.time_per_mult_s.iloc[0] / med) if len(w) else np.nan,
                     first_sample_over_median=t[0] / med))
S = pd.DataFrame(rows).sort_values(["config", "n"])
S = S.merge(main_runs[["config", "n", "status", "correctness", "vmhwm_kib", "rss_after_load_kib",
                       "ru_maxrss_kib", "max_rel_err", "rtol_used", "sha256_C", "wall_s"]],
            on=["config", "n"], how="left")
S.to_csv(RES / "summary_main.csv", index=False)

# ----------------------------------------------------------------------------- scaling fits
WINDOWS = {"50-400": (50, 400), "200+": (200, 10 ** 9)}
fit_rows = []
for cfg in CONFIGS:
    for wname, (lo, hi) in WINDOWS.items():
        g = S[(S.config == cfg) & (S.n >= lo) & (S.n <= hi)]
        if len(g) < 3:
            continue
        x, y = np.log(g.n.values.astype(float)), np.log(g.median_s.values)
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, icpt), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        df = len(x) - 2
        s2 = float(np.sum((y - A @ [slope, icpt]) ** 2) / df)
        se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
        ci = t_crit(df) * se
        fit_rows.append(dict(config=cfg, window=wname, n_min=int(g.n.min()), n_max=int(g.n.max()), points=len(g),
                             exponent=slope, ci95=ci, exp_ci_lo=slope - ci, exp_ci_hi=slope + ci))
FIT = pd.DataFrame(fit_rows)
FIT.to_csv(RES / "summary_scaling_fit.csv", index=False)

# local exponents between consecutive sizes
loc = []
for cfg in CONFIGS:
    g = S[S.config == cfg].sort_values("n")
    for (n0, t0), (n1, t1) in zip(zip(g.n, g.median_s), list(zip(g.n, g.median_s))[1:]):
        loc.append(dict(config=cfg, n_from=n0, n_to=n1, local_exponent=math.log(t1 / t0) / math.log(n1 / n0)))
LOC = pd.DataFrame(loc)
LOC.to_csv(RES / "summary_local_exponent.csv", index=False)

# ----------------------------------------------------------------------------- ratios
piv = S.pivot(index="n", columns="config", values="median_s")
ratio = pd.DataFrame({"Java/C": piv["java"] / piv["c-O2"], "Python/C": piv["python"] / piv["c-O2"],
                      "Python/Java": piv["python"] / piv["java"]})
ratio.to_csv(RES / "summary_ratios.csv")

# ----------------------------------------------------------------------------- memory
mem_rows = []
for cfg in CONFIGS:
    g = S[S.config == cfg].sort_values("n")
    base = float(g[g.n == g.n.min()].vmhwm_kib.iloc[0])
    for _, r in g.iterrows():
        data_kib = r.vmhwm_kib - base
        mem_rows.append(dict(config=cfg, n=int(r.n), peak_rss_mib=r.vmhwm_kib / 1024,
                             rss_after_load_mib=r.rss_after_load_kib / 1024,
                             ru_maxrss_mib=r.ru_maxrss_kib / 1024,
                             above_baseline_mib=data_kib / 1024,
                             bytes_per_n2_above_baseline=data_kib * 1024 / r.n ** 2,
                             model_bytes_per_n2=(120 if cfg == "python" else 24)))
M = pd.DataFrame(mem_rows)
M.to_csv(RES / "summary_memory.csv", index=False)

# ----------------------------------------------------------------------------- limits
lim_rows = []
for cfg in CONFIGS:
    g = main_runs[main_runs.config == cfg].sort_values("n")
    ok = g[(g.status == "ok") & (g.correctness == "pass")]
    att = g[g.status != "not_attempted"]
    bad = g[g.status.isin(["timeout", "memory_limit", "error"])]
    lim_rows.append(dict(config=cfg,
                         largest_feasible=int(ok.n.max()) if len(ok) else None,
                         largest_attempted=int(att.n.max()) if len(att) else None,
                         outcome_at_largest_attempted=att.sort_values("n").iloc[-1].status if len(att) else None,
                         reason=(bad.iloc[0].reason if len(bad) else ""),
                         sizes_not_attempted=" ".join(str(int(v)) for v in g[g.status == "not_attempted"].n)))
L = pd.DataFrame(lim_rows)
L.to_csv(RES / "summary_limits.csv", index=False)

# ----------------------------------------------------------------------------- cross-language bitwise identity
ok_runs = main_runs[(main_runs.status == "ok")]
bit = ok_runs.groupby("n").sha256_C.nunique().rename("distinct_hashes").to_frame()
bit["languages"] = ok_runs.groupby("n").size()
bit.to_csv(RES / "summary_bitwise.csv")

# ----------------------------------------------------------------------------- cflags
cf = runs[(runs.campaign == "cflags")]
cf_meas = raw[(raw.campaign == "cflags") & (raw.phase == "measured")]
cf_rows = []
for (cfg, n), g in cf_meas.groupby(["config", "n"]):
    q1, med, q3 = np.percentile(g.time_per_mult_s, [25, 50, 75])
    r = cf[(cf.config == cfg) & (cf.n == n)].iloc[0]
    cf_rows.append(dict(config=cfg, n=n, reps=len(g), median_s=med, q1_s=q1, q3_s=q3,
                        correctness=r.correctness, sha256_C=r.sha256_C,
                        bitwise_equal_to_O2=None))
CF = pd.DataFrame(cf_rows)
if len(CF):
    for i, r in CF.iterrows():
        ref = CF[(CF.config == "c-O2") & (CF.n == r.n)]
        CF.loc[i, "bitwise_equal_to_O2"] = bool(len(ref) and ref.sha256_C.iloc[0] == r.sha256_C)
    CF.to_csv(RES / "summary_cflags.csv", index=False)

# ============================================================================= FIGURES
# Fig 1: runtime vs n (log-log) with IQR bands, cubic reference
fig, ax = plt.subplots(figsize=(6.4, 4.1))
for cfg in CONFIGS:
    g = S[S.config == cfg]
    pts = meas[meas.config == cfg]
    ax.scatter(pts.n, pts.time_per_mult_s, s=7, color=COL[cfg], alpha=0.25, linewidths=0)
    ax.fill_between(g.n, g.q1_s, g.q3_s, color=COL[cfg], alpha=0.25, linewidth=0)
    ax.plot(g.n, g.median_s, color=COL[cfg], marker=MARK[cfg], ms=4.5, lw=1.4, label=LABEL[cfg])
    ref_n = 400 if (g.n == 400).any() else g.n.iloc[len(g) // 2]
    t0 = float(g[g.n == ref_n].median_s.iloc[0])
    xs = np.array([ref_n, g.n.max()])
    ax.plot(xs, t0 * (xs / ref_n) ** 3, color=COL[cfg], lw=0.8, ls=(0, (3, 3)), alpha=0.9)
ax.plot([], [], color=INK2, lw=0.8, ls=(0, (3, 3)), label=r"$\propto n^3$ (anchored at $n=400$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("matrix size n (n × n, float64)"); ax.set_ylabel("time per multiplication (s)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig1_runtime_loglog")

# Fig 2: normalised time per inner iteration
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for cfg in CONFIGS:
    g = S[S.config == cfg]
    ax.plot(g.n, g.ns_per_inner_iter, color=COL[cfg], marker=MARK[cfg], ms=4.5, lw=1.4, label=LABEL[cfg])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("matrix size n"); ax.set_ylabel(r"median time / $n^3$  (ns per multiply-add)")
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.47), fontsize=8)
save(fig, "fig2_normalised_time")

# Fig 3: memory
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for cfg in CONFIGS:
    g = M[M.config == cfg]
    ax.plot(g.n, g.peak_rss_mib, color=COL[cfg], marker=MARK[cfg], ms=4.5, lw=1.4, label=LABEL[cfg] + " – measured peak RSS")
    base = float(g.peak_rss_mib.iloc[0])
    xs = np.logspace(np.log10(g.n.min()), np.log10(g.n.max()), 60)
    ax.plot(xs, base + g.model_bytes_per_n2.iloc[0] * xs ** 2 / 2 ** 20, color=COL[cfg], lw=0.8, ls=(0, (3, 3)))
ax.plot([], [], color=INK2, lw=0.8, ls=(0, (3, 3)), label="baseline + array-storage model")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("matrix size n"); ax.set_ylabel("peak resident memory (MiB)")
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "fig3_memory")

# Fig 4: variability (relative IQR) and warm-up effect
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for cfg in CONFIGS:
    g = S[S.config == cfg]
    axs[0].plot(g.n, g.rel_iqr * 100, color=COL[cfg], marker=MARK[cfg], ms=4, lw=1.2, label=SHORT[cfg])
    gw = g.dropna(subset=["first_warmup_over_median"])
    axs[1].plot(gw.n, gw.first_warmup_over_median, color=COL[cfg], marker=MARK[cfg], ms=4, lw=1.2, label=SHORT[cfg])
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("matrix size n"); axs[0].set_ylabel("IQR / median (%)")
axs[0].set_title("Variability between repetitions", fontsize=9, color=INK)
axs[1].axhline(1, color=INK2, lw=0.8)
axs[1].set_xscale("log"); axs[1].set_xlabel("matrix size n")
axs[1].set_ylabel("first warm-up sample / median")
axs[1].set_title("Warm-up effect", fontsize=9, color=INK)
axs[0].legend(fontsize=8)
fig.tight_layout()
save(fig, "fig4_variability_warmup")

# Fig 5: slowdown ratios
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.plot(ratio.index, ratio["Python/C"], color=COL["python"], marker=MARK["python"], ms=4.5, lw=1.4, label="Python / C")
ax.plot(ratio.index, ratio["Java/C"], color=COL["java"], marker=MARK["java"], ms=4.5, lw=1.4, label="Java / C")
ax.axhline(1, color=INK2, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("matrix size n"); ax.set_ylabel("ratio of median times")
ax.legend(fontsize=8)
save(fig, "fig5_ratios")

# Fig 6: compiler flags (separate supplementary experiment) - one linear panel per size, bars start at 0
if len(CF):
    order = ["c-O0", "c-O1", "c-O2", "c-O3", "c-O3-native"]
    names = ["-O0", "-O1", "-O2", "-O3", "-O3\nnative"]
    sizes = sorted(CF.n.unique())
    fig, axs = plt.subplots(1, len(sizes), figsize=(7.2, 2.9))
    for ax, n in zip(np.atleast_1d(axs), sizes):
        vals, lo, hi = [], [], []
        for o in order:
            r = CF[(CF.config == o) & (CF.n == n)].iloc[0]
            vals.append(r.median_s); lo.append(r.median_s - r.q1_s); hi.append(r.q3_s - r.median_s)
        cols = [COL["c-O2"] if o == "c-O2" else "#9ec5f0" for o in order]
        ax.bar(range(len(order)), vals, color=cols, width=0.7, yerr=[lo, hi], ecolor=INK2,
               error_kw=dict(lw=0.8, capsize=2))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(names, fontsize=7)
        ax.set_title("n = %d" % n, fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axs)[0].set_ylabel("median time (s), bars: IQR")
    fig.tight_layout()
    save(fig, "fig6_cflags")

# ============================================================================= LaTeX tables
# Time table
hdr = ["$n$", "C", "Java", "Python", "reps (C/J/P)"]
rws = []
for n in sorted(S.n.unique()):
    cells = [str(n)]
    reps = []
    for cfg in CONFIGS:
        r = S[(S.config == cfg) & (S.n == n)]
        if len(r):
            r = r.iloc[0]
            cells.append("%s (%s)" % (fmt_t(r.median_s), fmt_t(r.iqr_s)))
            reps.append(str(int(r.reps)))
        else:
            st = main_runs[(main_runs.config == cfg) & (main_runs.n == n)]
            tag = st.status.iloc[0].replace("_", " ") if len(st) else "--"
            cells.append("\\textit{%s}" % tag)
            reps.append("--")
    cells.append("/".join(reps))
    rws.append(cells)
write_tex("time", hdr, rws, "rllll")

# Memory table
rws = []
for n in sorted(M.n.unique()):
    cells = [str(n)]
    for cfg in CONFIGS:
        r = M[(M.config == cfg) & (M.n == n)]
        cells.append("%.1f" % r.peak_rss_mib.iloc[0] if len(r) else "--")
    rws.append(cells)
write_tex("memory", ["$n$", "C", "Java", "Python"], rws, "rrrr")

# bytes per n^2
rws = []
for cfg in CONFIGS:
    g = M[(M.config == cfg) & (M.n >= 500)]
    base = M[M.config == cfg].peak_rss_mib.iloc[0]
    rws.append([SHORT[cfg], "%.1f" % base, "%.1f" % g.bytes_per_n2_above_baseline.median(),
                "%d" % g.model_bytes_per_n2.iloc[0]])
write_tex("membytes", ["Impl.", "Baseline (MiB, $n{=}10$)", "Measured bytes per $n^2$", "Model bytes per $n^2$"],
          rws, "lrrr")

# scaling-fit table
rws = [[SHORT[r.config], "%d--%d" % (r.n_min, r.n_max), r.points, "%.2f" % r.exponent,
        "[%.2f, %.2f]" % (r.exp_ci_lo, r.exp_ci_hi)] for r in FIT.itertuples()]
write_tex("fit", ["Impl.", "Sizes used ($n$)", "Points", "Exponent $k$", "95\\,\\% CI"], rws, "llrrl")

# limits table
rws = []
for r in L.itertuples():
    rws.append([SHORT[r.config], r.largest_feasible, r.largest_attempted,
                ("%s ($>$300\\,s per sample)" % str(r.outcome_at_largest_attempted).replace("_", " "))
                if str(r.outcome_at_largest_attempted) == "timeout"
                else tex(str(r.outcome_at_largest_attempted).replace("_", " "))])
write_tex("limits", ["Impl.", "Largest feasible $n$", "Largest attempted $n$", "Outcome at largest attempted"],
          rws, "lrrl")

# cflags table
if len(CF):
    sizes = sorted(CF.n.unique())
    rws = []
    for o in ["c-O0", "c-O1", "c-O2", "c-O3", "c-O3-native"]:
        cells = [tex(o.replace("c-", "gcc -").replace("-native", " -march=native"))]
        for n in sizes:
            r = CF[(CF.config == o) & (CF.n == n)]
            cells.append(fmt_t(float(r.median_s.iloc[0])) if len(r) else "--")
        same = CF[CF.config == o].bitwise_equal_to_O2.all() if len(CF[CF.config == o]) else False
        cells.append("yes" if same else "no")
        rws.append(cells)
    write_tex("cflags", ["Flags"] + ["$n{=}%d$" % n for n in sizes] + ["same bits as -O2"], rws,
              "l" + "r" * len(sizes) + "c")

# correctness-test table
ct = pd.read_csv(RES / "correctness_tests.csv")
rws = []
for name, g in ct[ct.language != "all"].groupby("test", sort=False):
    mode = g["mode"].iloc[0]
    res = "pass" if (g.result == "PASS").all() else "FAIL"
    err = g.max_error.astype(float).max()
    tol = g.tolerance.astype(float).max()
    rws.append([tex(name), mode, "%.2g" % err, "%.2g" % tol if tol else "0 (exact)", res])
write_tex("correctness", ["Test (all three languages)", "Check", "Max. rel./abs. error", "Tolerance", "Result"],
          rws, "llrrl")

# environment table
rows_env = [("CPU", env["cpu_model"]),
            ("Cores / logical CPUs", "%s physical (vCPU), %s logical; %s thread(s)/core" %
             (env["physical_cores"], env["logical_processors"], env["threads_per_core"])),
            ("Caches", "L1d %s, L2 %s, L3 %s" % (env["cache_L1d"], env["cache_L2"], env["cache_L3"])),
            ("RAM", "%d MiB, swap %d MiB" % (env["ram_total_MiB"], env["swap_total_MiB"])),
            ("Virtualisation", "%s (%s)" % (env["hypervisor"], env["virtualization"])),
            ("OS / kernel", "%s, Linux %s" % (env["os"], env["kernel"])),
            ("Python", "%s %s (%s)" % (env["python"]["implementation"], env["python"]["version"],
                                         env["python"]["compiler"])),
            ("Java", "OpenJDK 21.0.12.1 (HotSpot 64-bit server VM); flags %s" % env["jvm_flags_used"]),
            ("C compiler", "%s; flags %s" % (env["gcc"], env["c_flags_primary"])),
            ("NumPy (reference only)", env["python"]["numpy_version_(reference_and_input_generation_only)"]),
            ("Threads in the kernel", "1 (single-threaded); no CPU pinning; frequency governor: %s" %
             env["cpu_frequency_governor"])]
write_tex("env", ["Item", "Value"], [[tex(a), tex(b)] for a, b in rows_env], "lp{10.5cm}")


# ============================================================================= SUPPLEMENTARY EXPERIMENTS
xr = runs[runs.status == "ok"]
def run_median(campaign, config, n):
    g = raw[(raw.campaign == campaign) & (raw.config == config) & (raw.n == n) & (raw.phase == "measured")]
    return float(np.median(g.time_per_mult_s)) if len(g) else np.nan

# (a) between-process variability at n = 1024
procs = [("main", "main"), ("stride", "stride")] + [("procvar%d" % k, "procvar%d" % k) for k in range(1, 5)]
pv = []
for i, (camp, _) in enumerate(procs, 1):
    pv.append(dict(process=i, campaign=camp, c=run_median(camp, "c-O2", 1024), java=run_median(camp, "java", 1024),
                   java_flat=run_median("stride", "java-flat", 1024) if camp == "stride" else np.nan))
PV = pd.DataFrame(pv)
PV.to_csv(RES / "summary_process_variability.csv", index=False)
rws = [[r.process, fmt_t(r.c), fmt_t(r.java)] for r in PV.itertuples()]
for lab, f in (("min", np.nanmin), ("median", np.nanmedian), ("max", np.nanmax)):
    rws.append(["\\textit{%s}" % lab, fmt_t(float(f(PV.c))), fmt_t(float(f(PV.java)))])
rws.append(["range / median", "%.0f\\,\\%%" % (100 * (PV.c.max() - PV.c.min()) / PV.c.median()),
            "%.0f\\,\\%%" % (100 * (PV.java.max() - PV.java.min()) / PV.java.median())])
write_tex("procvar", ["Independent process", "C", "Java"], rws, "lrr")

# (b) sizes around powers of two
stride_sizes = [496, 504, 512, 520, 1016, 1024, 1032]
st = []
for n in stride_sizes:
    st.append(dict(n=n, c=run_median("stride", "c-O2", n), java=run_median("stride", "java", n),
                   java_flat=run_median("stride", "java-flat", n)))
ST = pd.DataFrame(st)
for k in ("c", "java", "java_flat"):
    ST[k + "_ns_per_iter"] = ST[k] / ST.n ** 3 * 1e9
ST.to_csv(RES / "summary_stride.csv", index=False)
rws = [[r.n, "%.3f" % r.c_ns_per_iter, "%.3f" % r.java_ns_per_iter, "%.3f" % r.java_flat_ns_per_iter] for r in ST.itertuples()]
write_tex("stride", ["$n$", "C", "Java \\texttt{double[n][n]}", "Java flat \\texttt{double[]}"], rws, "rrrr")

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, sub in zip(axs, (ST[ST.n < 600], ST[ST.n > 600])):
    for key, lab, col, mk in (("c", "C", COL["c-O2"], "o"), ("java", "Java jagged", COL["java"], "s"),
                              ("java_flat", "Java flat (supplementary)", "#7a3b10", "D")):
        ax.plot(sub.n, sub[key + "_ns_per_iter"], color=col, marker=mk, ms=4.5, lw=1.3, label=lab)
    pw = int(round(math.log2(sub.n.median())))
    ax.axvline(2 ** pw, color=INK2, lw=0.7, ls=":")
    ax.set_xticks(sub.n); ax.set_xticklabels([str(v) for v in sub.n], fontsize=7.5)
    ax.set_xlabel("matrix size n (dotted: $2^{%d}$)" % pw)
axs[0].set_ylabel("median time / $n^3$ (ns)")
axs[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
save(fig, "fig7_powers_of_two")

# (c) Java layout: jagged vs flat vs C
lay_sizes = [500, 1000, 1250, 1500, 2000]
lay = []
for n in lay_sizes:
    lay.append(dict(n=n, c=run_median("main", "c-O2", n), java=run_median("main", "java", n),
                    java_flat=run_median("layout", "java-flat", n)))
LAY = pd.DataFrame(lay)
LAY["java_over_c"] = LAY.java / LAY.c
LAY["flat_over_c"] = LAY.java_flat / LAY.c
LAY["jagged_over_flat"] = LAY.java / LAY.java_flat
LAY.to_csv(RES / "summary_layout.csv", index=False)
rws = [[r.n, fmt_t(r.c), fmt_t(r.java), fmt_t(r.java_flat), "%.2f" % r.java_over_c, "%.2f" % r.flat_over_c]
       for r in LAY.itertuples()]
write_tex("layout", ["$n$", "C", "Java jagged", "Java flat", "jagged/C", "flat/C"], rws, "rrrrrr")

fig, ax = plt.subplots(figsize=(6.4, 3.5))
for key, lab, col, mk in (("c", "C (contiguous)", COL["c-O2"], "o"), ("java", "Java double[n][n]", COL["java"], "s"),
                          ("java_flat", "Java flat double[] (supplementary)", "#7a3b10", "D")):
    ax.plot(LAY.n, LAY[key] / LAY.n ** 3 * 1e9, color=col, marker=mk, ms=4.5, lw=1.4, label=lab)
ax.set_xlabel("matrix size n"); ax.set_ylabel("median time / $n^3$ (ns)")
ax.legend(fontsize=8, loc="upper left")
save(fig, "fig8_java_layout")

print("analysis complete")
print(S[["config", "n", "reps", "median_s", "rel_iqr", "vmhwm_kib", "correctness"]].to_string())
print(FIT.to_string())
print(L.to_string())

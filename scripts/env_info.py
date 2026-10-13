#!/usr/bin/env python3
"""Record the hardware / software environment of the experiment in results/environment.json."""
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    env = {k: v for k, v in os.environ.items() if k != "JAVA_TOOL_OPTIONS"}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return "unavailable: %s" % e


def lscpu():
    out = {}
    for line in sh("lscpu").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    cpu = lscpu()
    meminfo = dict(re.findall(r"(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text()))
    import numpy
    info = {
        "cpu_model": cpu.get("Model name"),
        "logical_processors": os.cpu_count(),
        "sockets": cpu.get("Socket(s)"),
        "cores_per_socket": cpu.get("Core(s) per socket"),
        "threads_per_core": cpu.get("Thread(s) per core"),
        "physical_cores": (int(cpu.get("Socket(s)", 1)) * int(cpu.get("Core(s) per socket", 1))),
        "hypervisor": cpu.get("Hypervisor vendor"),
        "virtualization": cpu.get("Virtualization type"),
        "cache_L1d": cpu.get("L1d cache"), "cache_L2": cpu.get("L2 cache"), "cache_L3": cpu.get("L3 cache"),
        "cpu_flags_of_interest": [f for f in ("avx", "avx2", "avx512f", "fma", "sse4_2")
                                  if f in cpu.get("Flags", "").split()],
        "ram_total_MiB": round(int(meminfo["MemTotal"]) / 1024),
        "swap_total_MiB": round(int(meminfo["SwapTotal"]) / 1024),
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'"),
        "kernel": platform.release(),
        "cpu_frequency_governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")
                                  or "not exposed (virtual machine)",
        "cpu_affinity_used": "none (processes not pinned)",
        "python": {"version": sys.version.split()[0], "implementation": platform.python_implementation(),
                   "build": platform.python_build(), "compiler": platform.python_compiler(),
                   "numpy_version_(reference_and_input_generation_only)": numpy.__version__},
        "java": sh("java -version 2>&1 | grep -v 'Picked up'"),
        "jvm_flags_used": "-XX:+UseSerialGC -Xms256m -Xmx4g (default tiered JIT: C1+C2)",
        "gcc": sh("gcc --version | head -1"),
        "c_flags_primary": "-O2 -std=c11 -ffp-contract=off  (no -march=native; no -ffast-math)",
        "threads_used_by_each_implementation": 1,
        "note_parallelism": ("The JVM may use its additional JIT-compiler / GC threads on the second "
                             "logical CPU; the multiplication itself is single-threaded."),
    }
    out = ROOT / "results" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

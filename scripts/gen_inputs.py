#!/usr/bin/env python3
"""
Generate the shared input matrices used by ALL implementations.

Each matrix is stored as a raw binary file: n*n IEEE-754 binary64 values,
little-endian, row-major, no header. Files: data/A_n<N>.bin, data/B_n<N>.bin.

Generator : NumPy Generator(PCG64), seeded with the integer list [SEED, n, which]
            where which = 0 for A and 1 for B (so every (n, matrix) has its own
            independent, reproducible stream).
Values    : uniform in [0, 1), float64.

A manifest (data/manifest.json) stores the SHA-256 of every file so that the
inputs can be verified after regeneration (the binary files are not committed).

Usage: gen_inputs.py [--seed 20261012] [--outdir data] n1 n2 n3 ...
"""
import argparse
import hashlib
import json
import os

import numpy as np

SEED_DEFAULT = 20261012


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mpath = os.path.join(args.outdir, "manifest.json")
    manifest = {"generator": "numpy.random.default_rng(PCG64), seed list [seed, n, which]",
                "numpy_version": np.__version__, "dtype": "float64 little-endian row-major",
                "value_range": "[0, 1)", "seed": args.seed, "files": {}}
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        if old.get("seed") == args.seed:
            manifest["files"] = old.get("files", {})

    for n in args.sizes:
        for which, name in ((0, "A"), (1, "B")):
            path = os.path.join(args.outdir, "%s_n%d.bin" % (name, n))
            rng = np.random.default_rng([args.seed, n, which])
            rng.random(n * n, dtype=np.float64).tofile(path)
            manifest["files"][os.path.basename(path)] = {
                "n": n, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
        print("generated n=%d" % n, flush=True)

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

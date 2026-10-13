# Big Data (ULPGC 2026-2027) - Individual Assignment 1: basic matrix multiplication
# Usage:  make all        (build, inputs, tests, benchmarks, analysis)
#         make build | inputs | test | bench | cflags | analyze | report

PY      ?= python3
CC       = gcc
# Primary C configuration: -O2, no -march=native, FMA contraction disabled so that the
# arithmetic is bit-for-bit the same as in Java/Python (see README, "Fixed settings").
CFLAGS  ?= -O2 -std=c11 -ffp-contract=off -Wall -Wextra
SIZES_ALL := 10 50 100 200 256 400 500 512 800 1000 1024 1250 1500 2000 2048 3000 4096 496 504 520 1016 1032

.PHONY: all build inputs test bench cflags analyze report clean

all: build inputs test bench cflags analyze report

build:
	mkdir -p build/java build/cflags
	$(CC) $(CFLAGS) -o build/matmul_c src/c/matmul.c
	javac -d build/java src/java/MatMul.java src/java/MatMulFlat.java
	for t in O0:-O0 O1:-O1 O2:-O2 O3:-O3 O3-native:"-O3 -march=native"; do \
	  tag=$${t%%:*}; fl=$${t#*:}; \
	  $(CC) $$fl -std=c11 -ffp-contract=off -o build/cflags/matmul_c_$$tag src/c/matmul.c; \
	done

inputs:
	$(PY) scripts/gen_inputs.py $(SIZES_ALL)

test: build
	$(PY) scripts/test_correctness.py

bench: build inputs
	$(PY) scripts/env_info.py
	$(PY) scripts/run_benchmarks.py --campaign main --resume

cflags: build inputs
	$(PY) scripts/run_benchmarks.py --campaign cflags --resume
	$(PY) scripts/run_benchmarks.py --campaign stride --resume
	$(PY) scripts/run_benchmarks.py --campaign layout --resume
	for k in 1 2 3 4; do $(PY) scripts/run_benchmarks.py --campaign procvar --tag $$k --resume; done

analyze:
	$(PY) scripts/analyze.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode report.tex

clean:
	rm -rf build

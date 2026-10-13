/*
 * Big Data (ULPGC, 2026-2027) - Individual Assignment 1
 * Basic dense matrix multiplication, C implementation.
 *
 * Algorithm : textbook triple loop, order i-j-k, O(n^3), double precision.
 * Layout    : one contiguous row-major block of n*n doubles per matrix.
 * Timer     : clock_gettime(CLOCK_MONOTONIC); only the multiplication kernel
 *             is timed. Loading inputs, allocation, zero-initialisation of C,
 *             checksum and optional dump are all OUTSIDE the timed region.
 *
 * Usage: matmul <n> <reps> <warmup> <batch> <A.bin> <B.bin> [dumpC.bin]
 *   batch = multiplications per timed sample; the reported time is the
 *   time PER MULTIPLICATION (batch time / batch). Used only for tiny n.
 *
 * Output (stdout, one record per line, flushed immediately):
 *   META lang=c ...
 *   WARMUP <idx> <seconds_per_multiplication> <checksum>
 *   REP    <idx> <seconds_per_multiplication> <checksum>
 *   RSS_AFTER_LOAD_KB <kB>
 *   VMHWM_KB <kB>
 *   DONE
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static double *read_matrix(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(2); }
    double *m = malloc(n * n * sizeof(double));
    if (!m) { fprintf(stderr, "out of memory (read)\n"); exit(3); }
    if (fread(m, sizeof(double), n * n, f) != n * n) {
        fprintf(stderr, "short read on %s (expected %zu doubles)\n", path, n * n);
        exit(2);
    }
    fclose(f);
    return m;
}

/* Kernel: C = A * B. noinline keeps it a real call so the compiler cannot
 * merge repetitions. The scalar accumulator is written back once per (i,j). */
__attribute__((noinline))
static void multiply(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/* Plain left-to-right sum of all elements in row-major order. */
static double checksum(const double *c, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n * n; i++) s += c[i];
    return s;
}

static long proc_status_kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long val = -1;
    size_t kl = strlen(key);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, kl) == 0 && line[kl] == ':') {
            val = strtol(line + kl + 1, NULL, 10);
            break;
        }
    }
    fclose(f);
    return val;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s n reps warmup batch A.bin B.bin [dumpC.bin]\n", argv[0]);
        return 1;
    }
    size_t n = strtoull(argv[1], NULL, 10);
    int reps = atoi(argv[2]);
    int warmup = atoi(argv[3]);
    long batch = atol(argv[4]);
    if (batch < 1) batch = 1;

    printf("META lang=c n=%zu reps=%d warmup=%d batch=%ld\n", n, reps, warmup, batch);
    fflush(stdout);

    double *a = read_matrix(argv[5], n);
    double *b = read_matrix(argv[6], n);
    double *c = malloc(n * n * sizeof(double));
    if (!c) { fprintf(stderr, "out of memory (C)\n"); return 3; }
    memset(c, 0, n * n * sizeof(double)); /* touch pages before timing */

    printf("RSS_AFTER_LOAD_KB %ld\n", proc_status_kb("VmRSS"));
    fflush(stdout);

    for (int w = 0; w < warmup; w++) {
        int64_t t0 = now_ns();
        for (long q = 0; q < batch; q++) multiply(a, b, c, n);
        int64_t t1 = now_ns();
        printf("WARMUP %d %.9e %.17g\n", w, (double)(t1 - t0) * 1e-9 / (double)batch, checksum(c, n));
        fflush(stdout);
    }
    for (int r = 0; r < reps; r++) {
        int64_t t0 = now_ns();
        for (long q = 0; q < batch; q++) multiply(a, b, c, n);
        int64_t t1 = now_ns();
        printf("REP %d %.9e %.17g\n", r, (double)(t1 - t0) * 1e-9 / (double)batch, checksum(c, n));
        fflush(stdout);
    }

    printf("VMHWM_KB %ld\n", proc_status_kb("VmHWM"));
    fflush(stdout);

    if (argc >= 8) {
        FILE *f = fopen(argv[7], "wb");
        if (!f || fwrite(c, sizeof(double), n * n, f) != n * n) {
            fprintf(stderr, "cannot write dump %s\n", argv[7]);
            return 4;
        }
        fclose(f);
    }
    printf("DONE\n");
    fflush(stdout);
    free(a); free(b); free(c);
    return 0;
}

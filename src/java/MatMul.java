/*
 * Big Data (ULPGC, 2026-2027) - Individual Assignment 1
 * Basic dense matrix multiplication, Java implementation.
 *
 * Algorithm : textbook triple loop, order i-j-k, O(n^3), double precision.
 * Layout    : double[n][n] (array of n separately allocated row arrays).
 * Timer     : System.nanoTime() (monotonic); only the kernel is timed.
 *             Loading, allocation, checksum and dump are outside the timer.
 *
 * Usage: java MatMul <n> <reps> <warmup> <batch> <A.bin> <B.bin> [dumpC.bin]
 * batch = multiplications per timed sample; reported time is PER MULTIPLICATION.
 * Output format is identical to the C and Python programs.
 */
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

public final class MatMul {

    static double[][] readMatrix(String path, int n) throws IOException {
        double[][] m = new double[n][n];
        byte[] buf = new byte[8 * n];
        DoubleBuffer db = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        try (FileInputStream in = new FileInputStream(path)) {
            for (int i = 0; i < n; i++) {
                int got = in.readNBytes(buf, 0, buf.length);
                if (got != buf.length) throw new IOException("short read on " + path);
                db.rewind();
                db.get(m[i]);
            }
            if (in.read() != -1) throw new IOException("unexpected extra data in " + path);
        }
        return m;
    }

    /** Kernel: C = A * B, i-j-k order with a scalar accumulator. */
    static void multiply(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
    }

    /** Plain left-to-right sum of all elements in row-major order. */
    static double checksum(double[][] c, int n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) s += c[i][j];
        }
        return s;
    }

    static long procStatusKb(String key) {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith(key + ":")) {
                    return Long.parseLong(line.substring(key.length() + 1).trim().split("\\s+")[0]);
                }
            }
        } catch (IOException | RuntimeException e) {
            // fall through
        }
        return -1;
    }

    static void dump(double[][] c, int n, String path) throws IOException {
        byte[] buf = new byte[8 * n];
        DoubleBuffer db = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path), 1 << 20)) {
            for (int i = 0; i < n; i++) {
                db.rewind();
                db.put(c[i]);
                out.write(buf);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 6) {
            System.err.println("usage: java MatMul n reps warmup batch A.bin B.bin [dumpC.bin]");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);
        int reps = Integer.parseInt(args[1]);
        int warmup = Integer.parseInt(args[2]);
        long batch = Math.max(1L, Long.parseLong(args[3]));

        System.out.printf("META lang=java n=%d reps=%d warmup=%d batch=%d jvm=%s%n", n, reps, warmup,
                batch, System.getProperty("java.vm.version"));
        System.out.flush();

        double[][] a = readMatrix(args[4], n);
        double[][] b = readMatrix(args[5], n);
        double[][] c = new double[n][n]; // zero-initialised and touched before timing

        System.out.println("RSS_AFTER_LOAD_KB " + procStatusKb("VmRSS"));
        System.out.flush();

        for (int w = 0; w < warmup; w++) {
            long t0 = System.nanoTime();
            for (long q = 0; q < batch; q++) multiply(a, b, c, n);
            long t1 = System.nanoTime();
            System.out.println("WARMUP " + w + " " + Double.toString((t1 - t0) * 1e-9 / batch) + " "
                    + Double.toString(checksum(c, n)));
            System.out.flush();
        }
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (long q = 0; q < batch; q++) multiply(a, b, c, n);
            long t1 = System.nanoTime();
            System.out.println("REP " + r + " " + Double.toString((t1 - t0) * 1e-9 / batch) + " "
                    + Double.toString(checksum(c, n)));
            System.out.flush();
        }

        System.out.println("VMHWM_KB " + procStatusKb("VmHWM"));
        System.out.flush();
        if (args.length >= 7) dump(c, n, args[6]);
        System.out.println("DONE");
        System.out.flush();
    }
}

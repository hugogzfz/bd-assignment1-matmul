/*
 * SUPPLEMENTARY experiment (not part of the controlled basic comparison).
 * Same algorithm and CLI as MatMul.java, but the matrices are stored as ONE flat
 * row-major double[n*n] (the same layout as the C program) instead of double[n][n].
 * Purpose: test whether the jagged-array layout explains the gap between Java and C.
 *
 * Usage: java MatMulFlat <n> <reps> <warmup> <batch> <A.bin> <B.bin> [dumpC.bin]
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

public final class MatMulFlat {

    static double[] readMatrix(String path, int n) throws IOException {
        double[] m = new double[n * n];
        byte[] buf = new byte[8 * n];
        DoubleBuffer db = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        try (FileInputStream in = new FileInputStream(path)) {
            for (int i = 0; i < n; i++) {
                int got = in.readNBytes(buf, 0, buf.length);
                if (got != buf.length) throw new IOException("short read on " + path);
                db.rewind();
                db.get(m, i * n, n);
            }
            if (in.read() != -1) throw new IOException("unexpected extra data in " + path);
        }
        return m;
    }

    static void multiply(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    static double checksum(double[] c) {
        double s = 0.0;
        for (int i = 0; i < c.length; i++) s += c[i];
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

    static void dump(double[] c, int n, String path) throws IOException {
        byte[] buf = new byte[8 * n];
        DoubleBuffer db = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path), 1 << 20)) {
            for (int i = 0; i < n; i++) {
                db.rewind();
                db.put(c, i * n, n);
                out.write(buf);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 6) {
            System.err.println("usage: java MatMulFlat n reps warmup batch A.bin B.bin [dumpC.bin]");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);
        int reps = Integer.parseInt(args[1]);
        int warmup = Integer.parseInt(args[2]);
        long batch = Math.max(1L, Long.parseLong(args[3]));
        System.out.printf("META lang=java-flat n=%d reps=%d warmup=%d batch=%d jvm=%s%n", n, reps, warmup,
                batch, System.getProperty("java.vm.version"));
        System.out.flush();

        double[] a = readMatrix(args[4], n);
        double[] b = readMatrix(args[5], n);
        double[] c = new double[n * n];
        System.out.println("RSS_AFTER_LOAD_KB " + procStatusKb("VmRSS"));
        System.out.flush();

        for (int w = 0; w < warmup; w++) {
            long t0 = System.nanoTime();
            for (long q = 0; q < batch; q++) multiply(a, b, c, n);
            long t1 = System.nanoTime();
            System.out.println("WARMUP " + w + " " + Double.toString((t1 - t0) * 1e-9 / batch) + " "
                    + Double.toString(checksum(c)));
            System.out.flush();
        }
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (long q = 0; q < batch; q++) multiply(a, b, c, n);
            long t1 = System.nanoTime();
            System.out.println("REP " + r + " " + Double.toString((t1 - t0) * 1e-9 / batch) + " "
                    + Double.toString(checksum(c)));
            System.out.flush();
        }
        System.out.println("VMHWM_KB " + procStatusKb("VmHWM"));
        System.out.flush();
        if (args.length >= 7) dump(c, n, args[6]);
        System.out.println("DONE");
        System.out.flush();
    }
}

// SPDX-License-Identifier: MIT

package org.blockflow.bench;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import javax.imageio.ImageIO;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

public final class PipelineBench {
    private record Config(Path input, Path out, double sigma, long minSize) {}
    private record ObjectRow(int label, long count, double centroidY, double centroidX) {}
    private record RunResult(int objects, long totalArea, double threshold, double loadSeconds,
                             double pipelineSeconds) {}

    public static void main(String[] args) throws Exception {
        Config config = parse(args);
        Files.createDirectories(config.out());
        long started = System.nanoTime();
        ArrayImg<DoubleType, DoubleArray> input = loadLuma(config.input());
        double loadSeconds = secondsSince(started);

        long pipelineStarted = System.nanoTime();
        ArrayImg<DoubleType, DoubleArray> smoothed = smooth(input, config.sigma());
        double threshold = otsuThreshold(smoothed);
        boolean[] rawMask = thresholdMask(smoothed, threshold);
        boolean[] mask = removeSmallObjects(rawMask, width(smoothed), height(smoothed), config.minSize());
        int[] labels = label(mask, width(smoothed), height(smoothed));
        List<ObjectRow> rows = measure(labels, width(smoothed), height(smoothed));
        double pipelineSeconds = secondsSince(pipelineStarted);

        writeObjects(rows, config.out().resolve("objects.csv"));
        long totalArea = rows.stream().mapToLong(ObjectRow::count).sum();
        writeSummary(config, new RunResult(rows.size(), totalArea, threshold, loadSeconds, pipelineSeconds),
                config.out().resolve("summary.json"));
        System.out.printf("objects=%d threshold=%.6f output=%s%n", rows.size(), threshold, config.out());
    }

    private static Config parse(String[] args) {
        Path input = null;
        Path out = Path.of(".tmp/imglib2-pipeline/imglib2");
        double sigma = 1.5;
        long minSize = 20;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--input" -> input = Path.of(value(args, ++i, "--input"));
                case "--out" -> out = Path.of(value(args, ++i, "--out"));
                case "--sigma" -> sigma = Double.parseDouble(value(args, ++i, "--sigma"));
                case "--min-size" -> minSize = Long.parseLong(value(args, ++i, "--min-size"));
                case "--help", "-h" -> {
                    System.out.println("PipelineBench --input IMAGE --out DIR [--sigma 1.5] [--min-size 20]");
                    System.exit(0);
                }
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        if (input == null) {
            throw new IllegalArgumentException("missing required --input");
        }
        if (!Double.isFinite(sigma) || sigma < 0.0) {
            throw new IllegalArgumentException("--sigma must be finite and non-negative");
        }
        if (minSize < 1) {
            throw new IllegalArgumentException("--min-size must be at least 1");
        }
        return new Config(input, out, sigma, minSize);
    }

    private static String value(String[] args, int index, String name) {
        if (index >= args.length) {
            throw new IllegalArgumentException(name + " needs a value");
        }
        return args[index];
    }

    private static ArrayImg<DoubleType, DoubleArray> loadLuma(Path path) throws IOException {
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IOException("could not decode " + path);
        }
        int width = image.getWidth();
        int height = image.getHeight();
        ArrayImg<DoubleType, DoubleArray> out = ArrayImgs.doubles(width, height);
        RandomAccess<DoubleType> access = out.randomAccess();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = image.getRaster().getSample(x, y, 0);
                access.setPosition(x, 0);
                access.setPosition(y, 1);
                access.get().set(value);
            }
        }
        return out;
    }

    private static ArrayImg<DoubleType, DoubleArray> smooth(ArrayImg<DoubleType, DoubleArray> input,
                                                            double sigma) {
        if (sigma == 0.0) {
            return copy(input);
        }
        double[] kernel = gaussianKernel(sigma, 3.0);
        ArrayImg<DoubleType, DoubleArray> tmp = ArrayImgs.doubles(width(input), height(input));
        ArrayImg<DoubleType, DoubleArray> out = ArrayImgs.doubles(width(input), height(input));
        convolveX(input, tmp, kernel);
        convolveY(tmp, out, kernel);
        return out;
    }

    private static ArrayImg<DoubleType, DoubleArray> copy(ArrayImg<DoubleType, DoubleArray> input) {
        ArrayImg<DoubleType, DoubleArray> out = ArrayImgs.doubles(width(input), height(input));
        Cursor<DoubleType> src = Views.iterable(input).cursor();
        Cursor<DoubleType> dst = Views.iterable(out).cursor();
        while (src.hasNext()) {
            dst.next().set(src.next());
        }
        return out;
    }

    private static double[] gaussianKernel(double sigma, double truncate) {
        int radius = Math.max(1, (int) Math.ceil(sigma * truncate));
        double[] kernel = new double[radius * 2 + 1];
        double sum = 0.0;
        for (int i = -radius; i <= radius; i++) {
            double value = Math.exp(-0.5 * (i * i) / (sigma * sigma));
            kernel[i + radius] = value;
            sum += value;
        }
        for (int i = 0; i < kernel.length; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }

    private static void convolveX(ArrayImg<DoubleType, DoubleArray> input,
                                  ArrayImg<DoubleType, DoubleArray> out, double[] kernel) {
        RandomAccess<DoubleType> src = input.randomAccess();
        RandomAccess<DoubleType> dst = out.randomAccess();
        int radius = kernel.length / 2;
        int width = width(input);
        int height = height(input);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int dx = -radius; dx <= radius; dx++) {
                    src.setPosition(reflect(x + dx, width), 0);
                    src.setPosition(y, 1);
                    sum += kernel[dx + radius] * src.get().get();
                }
                dst.setPosition(x, 0);
                dst.setPosition(y, 1);
                dst.get().set(sum);
            }
        }
    }

    private static void convolveY(ArrayImg<DoubleType, DoubleArray> input,
                                  ArrayImg<DoubleType, DoubleArray> out, double[] kernel) {
        RandomAccess<DoubleType> src = input.randomAccess();
        RandomAccess<DoubleType> dst = out.randomAccess();
        int radius = kernel.length / 2;
        int width = width(input);
        int height = height(input);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int dy = -radius; dy <= radius; dy++) {
                    src.setPosition(x, 0);
                    src.setPosition(reflect(y + dy, height), 1);
                    sum += kernel[dy + radius] * src.get().get();
                }
                dst.setPosition(x, 0);
                dst.setPosition(y, 1);
                dst.get().set(sum);
            }
        }
    }

    private static int reflect(int coordinate, int extent) {
        while (coordinate < 0 || coordinate >= extent) {
            if (coordinate < 0) {
                coordinate = -coordinate - 1;
            } else {
                coordinate = 2 * extent - coordinate - 1;
            }
        }
        return coordinate;
    }

    private static double otsuThreshold(ArrayImg<DoubleType, DoubleArray> image) {
        long[] hist = new long[256];
        long total = 0;
        for (DoubleType value : Views.iterable(image)) {
            int bin = (int) Math.max(0, Math.min(255, Math.round(value.get())));
            hist[bin]++;
            total++;
        }
        double sumTotal = 0.0;
        for (int i = 0; i < hist.length; i++) {
            sumTotal += i * hist[i];
        }
        long weightBackground = 0;
        double sumBackground = 0.0;
        int bestBin = 0;
        double bestVariance = Double.NEGATIVE_INFINITY;
        for (int bin = 0; bin < hist.length; bin++) {
            weightBackground += hist[bin];
            if (weightBackground == 0) {
                continue;
            }
            long weightForeground = total - weightBackground;
            if (weightForeground == 0) {
                break;
            }
            sumBackground += bin * hist[bin];
            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sumTotal - sumBackground) / weightForeground;
            double variance = weightBackground * (double) weightForeground
                    * Math.pow(meanBackground - meanForeground, 2.0);
            if (variance > bestVariance) {
                bestVariance = variance;
                bestBin = bin;
            }
        }
        return bestBin;
    }

    private static boolean[] thresholdMask(ArrayImg<DoubleType, DoubleArray> image, double threshold) {
        boolean[] mask = new boolean[width(image) * height(image)];
        Cursor<DoubleType> cursor = Views.iterable(image).cursor();
        int index = 0;
        while (cursor.hasNext()) {
            mask[index++] = cursor.next().get() > threshold;
        }
        return mask;
    }

    private static boolean[] removeSmallObjects(boolean[] mask, int width, int height, long minSize) {
        int[] labels = label(mask, width, height);
        long[] counts = counts(labels);
        boolean[] out = new boolean[mask.length];
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            out[i] = label > 0 && counts[label] >= minSize;
        }
        return out;
    }

    private static int[] label(boolean[] mask, int width, int height) {
        int[] labels = new int[mask.length];
        int nextLabel = 1;
        int[] offsets = {-width, -1, 1, width};
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        for (int start = 0; start < mask.length; start++) {
            if (!mask[start] || labels[start] != 0) {
                continue;
            }
            labels[start] = nextLabel;
            queue.add(start);
            while (!queue.isEmpty()) {
                int index = queue.removeFirst();
                int x = index % width;
                for (int offset : offsets) {
                    int next = index + offset;
                    if (next < 0 || next >= mask.length) {
                        continue;
                    }
                    if ((offset == -1 && x == 0) || (offset == 1 && x == width - 1)) {
                        continue;
                    }
                    if (mask[next] && labels[next] == 0) {
                        labels[next] = nextLabel;
                        queue.add(next);
                    }
                }
            }
            nextLabel++;
        }
        return labels;
    }

    private static long[] counts(int[] labels) {
        int max = 0;
        for (int label : labels) {
            max = Math.max(max, label);
        }
        long[] counts = new long[max + 1];
        for (int label : labels) {
            if (label > 0) {
                counts[label]++;
            }
        }
        return counts;
    }

    private static List<ObjectRow> measure(int[] labels, int width, int height) {
        long[] counts = counts(labels);
        long[] sumY = new long[counts.length];
        long[] sumX = new long[counts.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = labels[y * width + x];
                if (label > 0) {
                    sumY[label] += y;
                    sumX[label] += x;
                }
            }
        }
        List<ObjectRow> rows = new ArrayList<>();
        for (int label = 1; label < counts.length; label++) {
            if (counts[label] > 0) {
                rows.add(new ObjectRow(label, counts[label],
                        sumY[label] / (double) counts[label],
                        sumX[label] / (double) counts[label]));
            }
        }
        rows.sort(Comparator.comparingInt(ObjectRow::label));
        return rows;
    }

    private static void writeObjects(List<ObjectRow> rows, Path path) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(path)) {
            out.write("label,count,centroid_y,centroid_x\n");
            for (ObjectRow row : rows) {
                out.write(row.label() + "," + row.count() + "," + row.centroidY()
                        + "," + row.centroidX() + "\n");
            }
        }
    }

    private static void writeSummary(Config config, RunResult result, Path path) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(path)) {
            out.write("{\n");
            out.write("  \"input\": " + quote(config.input().toString()) + ",\n");
            out.write("  \"objects\": " + result.objects() + ",\n");
            out.write("  \"total_foreground_area\": " + result.totalArea() + ",\n");
            out.write("  \"threshold\": " + result.threshold() + ",\n");
            out.write("  \"sigma\": " + config.sigma() + ",\n");
            out.write("  \"min_size\": " + config.minSize() + ",\n");
            out.write("  \"load_seconds\": " + result.loadSeconds() + ",\n");
            out.write("  \"pipeline_seconds\": " + result.pipelineSeconds() + "\n");
            out.write("}\n");
        }
    }

    private static String quote(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static int width(ArrayImg<DoubleType, DoubleArray> image) {
        return Math.toIntExact(image.dimension(0));
    }

    private static int height(ArrayImg<DoubleType, DoubleArray> image) {
        return Math.toIntExact(image.dimension(1));
    }

    private static double secondsSince(long started) {
        return (System.nanoTime() - started) / 1_000_000_000.0;
    }

    private PipelineBench() {}
}

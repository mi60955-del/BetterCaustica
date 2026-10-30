package dev.comfyfluffy.caustica.rt;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicBoolean;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;

import dev.comfyfluffy.caustica.CausticaMod;
import net.fabricmc.loader.api.FabricLoader;
import org.lwjgl.vulkan.VK10;

/**
 * Per-pass <b>GPU</b> timestamps for the composite frame.
 *
 * <p>{@link RtFrameStats} measures how long the render thread spends <i>recording</i> a pass, which on a
 * GPU-bound renderer is a fraction of a millisecond and says nothing about where the frame time went.
 * This brackets each pass with {@code vkCmdWriteTimestamp} and reports the wall time the GPU spent in
 * it, which is the only number that can attribute a frame-time difference to a pass.
 *
 * <p>Off unless {@code -Dcaustica.rt.gpuTiming=true}. When off, every entry point is one branch on a
 * static final boolean and emits no command: the normal build is untouched.
 *
 * <p>Frames are timed into a ring of {@link #SETS} query sets. When a new frame starts, the set recorded
 * {@code SETS} frames ago is read back. It is normally finished long before then; when it is not, the
 * availability bit reports zero and the frame is skipped rather than stalling the queue on a fence.
 * Rows land in {@code <gameDir>/rt-frame-stats/gpu.csv}.
 */
public final class RtGpuTiming {
    private static final boolean ENABLED = Boolean.getBoolean("caustica.rt.gpuTiming");

    /** Ordered pass boundaries. Interval i is NAMES[i-1] -> NAMES[i]. */
    private static final String[] NAMES = {
            "setup", "skyLut", "tracePrimary", "traceIndirect", "dlss", "exposure", "bloom", "lens", "display",
            "copy"
    };
    private static final int MAX_MARKS = NAMES.length;

    /** Ring depth: how many frames of query sets are kept in flight. */
    private static final int SETS = 4;
    /** 64-bit value + 32-bit availability, padded to 16 bytes per query for alignment. */
    private static final int STRIDE = 16;

    private static final AtomicBoolean WARNED = new AtomicBoolean();

    private static VkDevice device;
    private static long pool;
    private static float periodNanos = 1.0f;

    private static long frameIndex;
    private static int markCount;
    private static boolean dropped;

    private static final long[] setFrame = new long[SETS];
    private static final int[] setMarks = new int[SETS];

    private static PrintWriter csv;
    private static boolean csvOpenAttempted;

    private RtGpuTiming() {
    }

    public static boolean enabled() {
        return ENABLED;
    }

    /**
     * Starts timing a frame. Call once per composite frame, immediately after the command buffer is
     * obtained and before anything is recorded into it.
     */
    public static void beginFrame(RtContext ctx, VkCommandBuffer cmd) {
        if (!ENABLED) {
            return;
        }
        ensurePool(ctx);
        if (pool == 0L) {
            return;
        }
        int slot = (int) (frameIndex % SETS);
        // Read the set that was filled SETS frames ago; it should be complete by now. If it is not, this
        // frame is not timed at all: resetting a pool a pending command buffer still reads is illegal.
        if (setMarks[slot] > 0 && !flushSet(slot)) {
            dropped = true;
            markCount = 0;
            return;
        }
        VK10.vkCmdResetQueryPool(cmd, pool, slot * MAX_MARKS, MAX_MARKS);
        markCount = 0;
        dropped = false;
        mark(cmd, "frame-start");
    }

    /** Records one boundary. Names are matched against {@link #NAMES}; an unknown name is ignored. */
    public static void mark(VkCommandBuffer cmd, String name) {
        if (!ENABLED || pool == 0L || dropped) {
            return;
        }
        if (markCount >= MAX_MARKS) {
            return;
        }
        int slot = (int) (frameIndex % SETS);
        VK10.vkCmdWriteTimestamp(cmd, VK10.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool,
                slot * MAX_MARKS + markCount);
        markCount++;
    }

    /** Ends the frame: stores the mark count so the matching readback knows how many queries to read. */
    public static void endFrame() {
        if (!ENABLED || pool == 0L) {
            return;
        }
        int slot = (int) (frameIndex % SETS);
        if (!dropped) {
            setFrame[slot] = frameIndex;
            setMarks[slot] = markCount;
        }
        frameIndex++;
    }

    /** Returns false when the queries are not available yet, in which case the caller must not reset them. */
    private static boolean flushSet(int slot) {
        int count = setMarks[slot];
        if (count < 2) {
            return true;
        }
        java.nio.ByteBuffer data = org.lwjgl.system.MemoryUtil.memAlloc(count * STRIDE);
        try {
            int rc = VK10.vkGetQueryPoolResults(device, pool, slot * MAX_MARKS, count,
                    data, STRIDE,
                    VK10.VK_QUERY_RESULT_64_BIT | VK10.VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (rc != VK10.VK_SUCCESS) {
                // VK_NOT_READY: the frame is still running. Skip it rather than stalling on a fence.
                return false;
            }
            long prev = -1L;
            StringBuilder row = new StringBuilder();
            row.append(setFrame[slot]);
            for (int i = 0; i < count; i++) {
                long value = data.getLong(i * STRIDE);
                int available = data.getInt(i * STRIDE + 8);
                if (available == 0) {
                    return false;
                }
                if (prev >= 0L) {
                    row.append(',').append(String.format(java.util.Locale.ROOT, "%.4f",
                            (value - prev) * (double) periodNanos / 1.0e6));
                }
                prev = value;
            }
            writeRow(row.toString());
            return true;
        } finally {
            org.lwjgl.system.MemoryUtil.memFree(data);
        }
    }

    private static void ensurePool(RtContext ctx) {
        if (pool != 0L) {
            return;
        }
        VkDevice vk = ctx.vk();
        device = vk;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
            VK10.vkGetPhysicalDeviceProperties(vk.getPhysicalDevice(), props);
            periodNanos = props.limits().timestampPeriod();
            if (periodNanos <= 0.0f) {
                if (WARNED.compareAndSet(false, true)) {
                    CausticaMod.LOGGER.warn("RtGpuTiming: device reports no timestamp support; disabled");
                }
                return;
            }
            VkQueryPoolCreateInfo ci = VkQueryPoolCreateInfo.calloc(stack).sType$Default()
                    .queryType(VK10.VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(SETS * MAX_MARKS);
            java.nio.LongBuffer out = stack.mallocLong(1);
            int rc = VK10.vkCreateQueryPool(vk, ci, null, out);
            if (rc != VK10.VK_SUCCESS) {
                if (WARNED.compareAndSet(false, true)) {
                    CausticaMod.LOGGER.warn("RtGpuTiming: vkCreateQueryPool failed ({})", rc);
                }
                return;
            }
            pool = out.get(0);
            CausticaMod.LOGGER.info(
                    "RtGpuTiming: per-pass GPU timestamps on ({} intervals, timestampPeriod {} ns)",
                    NAMES.length - 1, periodNanos);
            StringBuilder header = new StringBuilder("frame");
            for (int i = 1; i < NAMES.length; i++) {
                header.append(',').append(NAMES[i]).append("Ms");
            }
            writeRow(header.toString());
        }
    }

    private static void writeRow(String row) {
        ensureCsv();
        if (csv == null) {
            return;
        }
        csv.println(row);
        csv.flush();
    }

    private static void ensureCsv() {
        if (csvOpenAttempted) {
            return;
        }
        csvOpenAttempted = true;
        Path dir = FabricLoader.getInstance().getGameDir().resolve("rt-frame-stats");
        Path file = dir.resolve("gpu.csv");
        try {
            Files.createDirectories(dir);
            csv = new PrintWriter(new BufferedWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8)));
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("RtGpuTiming: cannot open {}: {}", file, e.toString());
            csv = null;
        }
    }
}

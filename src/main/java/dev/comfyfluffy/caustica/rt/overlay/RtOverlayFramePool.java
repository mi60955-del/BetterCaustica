package dev.comfyfluffy.caustica.rt.overlay;

import org.lwjgl.vulkan.VK10;

import java.util.ArrayList;
import java.util.List;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;

/**
 * Host-visible vertex/index scratch for overlay passes, shared by every {@link RtOverlayFeature}.
 * Acquired buffers are pooled and reused ring-style: a slot is eligible again once the frame token it
 * was last used with has completed on the RT graphics timeline, so a buffer is never rewritten (or
 * destroyed) while the GPU can still read it, and steady-state frames allocate nothing.
 */
public final class RtOverlayFramePool {
    // Vulkan requires buffer size > 0; a few zero-length overlay draws could otherwise reach acquire() with
    // bytes == 0.
    private static final long MIN_SIZE = 256;
    // Soft cap on pooled slots. Overlay draw sizes are stable frame to frame, so this is normally never
    // reached; it bounds the worst case of one-off large bursts (e.g. a momentary nametag flood) to the
    // largest handful of sizes instead of pinning them for the session.
    private static final int MAX_SLOTS = 32;

    private final List<Slot> slots = new ArrayList<>();
    /** Slots handed out this frame; excluded from reuse until {@link #endFrame} marks them. */
    private final List<Slot> busyThisFrame = new ArrayList<>();

    private static final class Slot {
        final RtBuffer buffer;
        final RtGpuExecutor.TrackedGraphicsUse lastUse = new RtGpuExecutor.TrackedGraphicsUse();

        Slot(RtBuffer buffer) {
            this.buffer = buffer;
        }
    }

    /** A host-visible vertex buffer of at least {@code bytes}, owned by the pool until teardown. */
    public RtBuffer acquireVertex(RtContext ctx, long bytes, String label) {
        return acquire(ctx, bytes, VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, label);
    }

    /** A host-visible index buffer of at least {@code bytes}, owned by the pool until teardown. */
    public RtBuffer acquireIndex(RtContext ctx, long bytes, String label) {
        return acquire(ctx, bytes, VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, label);
    }

    private RtBuffer acquire(RtContext ctx, long bytes, int usage, String label) {
        long required = Math.max(bytes, MIN_SIZE);
        long completed = ctx.gpuExecutor().completedGraphicsUseValue();
        Slot best = null;
        for (Slot slot : slots) {
            RtBuffer buffer = slot.buffer;
            if (buffer.usage == usage && buffer.size >= required && slot.lastUse.value() <= completed
                    && !busyThisFrame.contains(slot)) {
                if (best == null || buffer.size < best.buffer.size) {
                    best = slot; // best fit, so small frames keep the large slots reusable
                }
            }
        }
        if (best == null) {
            if (slots.size() >= MAX_SLOTS) {
                evictLargestIdle(usage, completed);
            }
            best = new Slot(ctx.createBuffer(required, usage, true, label));
            slots.add(best);
        }
        busyThisFrame.add(best);
        return best.buffer;
    }

    /** Destroy the largest idle slot so a new allocation cannot grow the pool past {@link #MAX_SLOTS}. */
    private void evictLargestIdle(int usage, long completed) {
        Slot largest = null;
        for (Slot slot : slots) {
            if (busyThisFrame.contains(slot) || slot.lastUse.value() > completed) {
                continue;
            }
            if (slot.buffer.usage == usage) {
                if (largest == null || slot.buffer.size > largest.buffer.size) {
                    largest = slot;
                }
            }
        }
        if (largest == null) {
            for (Slot slot : slots) {
                if (busyThisFrame.contains(slot) || slot.lastUse.value() > completed) {
                    continue;
                }
                if (largest == null || slot.buffer.size > largest.buffer.size) {
                    largest = slot;
                }
            }
        }
        if (largest != null) {
            slots.remove(largest);
            largest.buffer.destroy();
        }
    }

    /** Mark everything acquired this frame with its graphics-use token once overlay recording is done. */
    public void endFrame(RtContext ctx, RtGpuExecutor.GraphicsUse graphicsUse) {
        for (Slot slot : busyThisFrame) {
            slot.lastUse.mark(graphicsUse);
        }
        busyThisFrame.clear();
    }

    /** Immediate teardown of every pooled buffer; only valid once the device is idle. */
    public void destroy() {
        for (Slot slot : slots) {
            slot.buffer.destroy();
        }
        slots.clear();
        busyThisFrame.clear();
    }
}

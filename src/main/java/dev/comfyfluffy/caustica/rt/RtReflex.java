package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.NVLowLatency2;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkLatencySleepInfoNV;
import org.lwjgl.vulkan.VkLatencySleepModeInfoNV;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreTypeCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreWaitInfo;
import org.lwjgl.vulkan.VkSetLatencyMarkerInfoNV;

import java.nio.LongBuffer;

/**
 * NVIDIA Reflex per-frame sleep/pacing and latency markers, built on the timeline semaphore signaled
 * by {@code vkLatencySleepNV}. {@link RtDeviceBringup} enables the device support and
 * {@code VulkanGpuSurfaceMixin} configures each swapchain for latency mode.
 *
 * <p>{@code vkLatencySleepNV} does not itself block — per the extension, it schedules the driver to signal
 * a semaphore value once the paced frame-start time is reached, and the caller is expected to wait on that
 * value itself. That host-side {@code vkWaitSemaphores} in {@link #sleep} IS the actual pacing mechanism;
 * it must run once near the top of the frame, before input sampling/simulation (see
 * {@code MinecraftMixin#caustica$reflexSleep}).
 *
 * <p>Sleep mode + the timeline semaphore are scoped to a specific swapchain object ({@link #applySleepMode}
 * must be re-called whenever the swapchain is (re)created, e.g. on resize) — tracked here by comparing the
 * swapchain handle, so a stale call after a swapchain recreation is a safe no-op until re-applied.
 *
 * <p>{@code presentID} correlates markers with a specific present call. Two independent monotonic counters
 * are kept: {@link #currentSimFrameId()} (bumped once per {@link #sleep} call, i.e. once per
 * {@code runTick}) tags the SIMULATION/RENDERSUBMIT markers, while {@link #advancePresentId()} is bumped
 * once per actual {@code vkQueuePresentKHR} call and is what's chained onto the present via
 * {@code VkPresentIdKHR} plus the PRESENT_START/END markers. They must be separate: Minecraft can call
 * {@code present()} from outside the normal tick loop (e.g. {@code Minecraft.setScreenAndShow}'s synchronous
 * redraw when opening a world), which would otherwise resend a stale, already-used presentID and violate
 * {@code VUID-VkPresentIdKHR-presentIds-04999} ("each presentIds entry must be greater than all previously
 * submitted present ids").
 */
public final class RtReflex {
    public static final RtReflex INSTANCE = new RtReflex();

    /** {@code VkLatencyMarkerNV} values (VK_NV_low_latency2); mirrored here to avoid importing the enum at call sites. */
    public static final int MARKER_SIMULATION_START = NVLowLatency2.VK_LATENCY_MARKER_SIMULATION_START_NV;
    public static final int MARKER_SIMULATION_END = NVLowLatency2.VK_LATENCY_MARKER_SIMULATION_END_NV;
    public static final int MARKER_RENDERSUBMIT_START = NVLowLatency2.VK_LATENCY_MARKER_RENDERSUBMIT_START_NV;
    public static final int MARKER_RENDERSUBMIT_END = NVLowLatency2.VK_LATENCY_MARKER_RENDERSUBMIT_END_NV;
    public static final int MARKER_PRESENT_START = NVLowLatency2.VK_LATENCY_MARKER_PRESENT_START_NV;
    public static final int MARKER_PRESENT_END = NVLowLatency2.VK_LATENCY_MARKER_PRESENT_END_NV;

    private static final long SLEEP_WAIT_TIMEOUT_NS = 200_000_000L; // 200ms: generous, never expected to hit

    private long timelineSemaphore;
    private long counter;
    private long presentCounter;
    private long sleepModeSwapchain; // swapchain applySleepMode was last successfully called for; 0 = none yet
    private boolean lastBoost;
    private int lastMinIntervalUs;
    private boolean failed;

    // Diagnostics for logDiagIfDue(): how many frames were paced, how many were dropped for want of an
    // applied swapchain, and how long the driver actually held the game thread.
    private static final long DIAG_LOG_INTERVAL_NANOS = 2_000_000_000L;
    private long sleptFrames;
    private long timedOutFrames;
    private long skippedFrames;
    private long waitedNanos;
    private long lastDiagNanos;

    private RtReflex() {
    }

    public static boolean enabled() {
        return CausticaConfig.Rt.Reflex.ENABLED.value() && RtDeviceBringup.reflexEnabled();
    }

    /** Current sim frame's marker id (set by the last {@link #sleep} call) — tags SIMULATION/RENDERSUBMIT markers. */
    public long currentSimFrameId() {
        return counter;
    }

    /**
     * Advance and return a fresh, strictly-increasing presentID for the present about to happen. Call
     * exactly once per actual {@code vkQueuePresentKHR}, immediately before using the result (for the chained
     * {@code VkPresentIdKHR} and the PRESENT_START/END markers) — guarantees uniqueness even for presents
     * that don't go through {@link #sleep} first.
     */
    public long advancePresentId() {
        return ++presentCounter;
    }

    /** The swapchain {@link #applySleepMode} last successfully applied to, or 0 if none (not yet ready). */
    public long appliedSwapchain() {
        return sleepModeSwapchain;
    }

    /**
     * Apply (or refresh) the sleep-mode config on {@code swapchain}. Cheap no-op when already applied for
     * this exact swapchain handle with unchanged boost/interval config; safe to call every frame from the
     * swapchain-configure path. Must be called at least once per swapchain (creation, and again whenever the
     * swapchain is recreated, e.g. resize) before {@link #sleep}/{@link #marker} do anything.
     */
    public void applySleepMode(VkDevice device, long swapchain) {
        if (!enabled() || failed || swapchain == 0L) {
            return;
        }
        boolean boost = CausticaConfig.Rt.Reflex.LOW_LATENCY_BOOST.value();
        int minIntervalUs = CausticaConfig.Rt.Reflex.MINIMUM_INTERVAL_US.value();
        if (swapchain == sleepModeSwapchain && boost == lastBoost && minIntervalUs == lastMinIntervalUs) {
            return;
        }
        try {
            ensureSemaphore(device);
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkLatencySleepModeInfoNV info = VkLatencySleepModeInfoNV.calloc(stack).sType$Default()
                        .lowLatencyMode(true).lowLatencyBoost(boost).minimumIntervalUs(minIntervalUs);
                int r = NVLowLatency2.vkSetLatencySleepModeNV(device, swapchain, info);
                if (r != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("vkSetLatencySleepModeNV failed: " + r);
                }
            }
            sleepModeSwapchain = swapchain;
            lastBoost = boost;
            lastMinIntervalUs = minIntervalUs;
            CausticaMod.LOGGER.info("Reflex: sleep mode applied (boost={}, minIntervalUs={})", boost, minIntervalUs);
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("Reflex: applySleepMode failed; Reflex disabled for session", t);
        }
    }

    /**
     * Forget which swapchain the sleep mode was applied to, so the next
     * {@link #applySleepMode} really applies.
     *
     * <p>Call before {@code VulkanGpuSurface.configure()} builds a new swapchain. That path destroys the
     * old one first, and Vulkan is free to hand the same handle value straight back for the new object —
     * {@link #applySleepMode} keys its idempotence on the handle, so a recycled swapchain would otherwise be
     * mistaken for one that is already configured and Reflex would quietly stop pacing frames after a
     * fullscreen/windowed toggle or a resize. Latency state belongs to the swapchain object, so every new
     * object needs the call.
     */
    public void forgetSwapchain() {
        sleepModeSwapchain = 0L;
    }

    /**
     * Per-frame pacing: increment the frame counter, ask the driver to signal the timeline semaphore at the
     * paced frame-start time, then block this (the game) thread until it does. Call once, near the top of
     * the frame, before input sampling/simulation. No-op (returns immediately) unless {@link #applySleepMode}
     * has already been applied for this exact {@code swapchain}.
     */
    public void sleep(VkDevice device, long swapchain) {
        if (!enabled() || failed) {
            return;
        }
        if (swapchain == 0L || swapchain != sleepModeSwapchain || timelineSemaphore == 0L) {
            // Reflex is switched on and the device supports it, yet this frame was not paced. The usual
            // reason is that no sleep mode has been applied for the swapchain currently in use.
            skippedFrames++;
            logDiagIfDue();
            return;
        }
        counter++;
        long started = System.nanoTime();
        boolean timedOut = false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkLatencySleepInfoNV sleepInfo = VkLatencySleepInfoNV.calloc(stack).sType$Default()
                    .signalSemaphore(timelineSemaphore).value(counter);
            int r = NVLowLatency2.vkLatencySleepNV(device, swapchain, sleepInfo);
            if (r != VK10.VK_SUCCESS) {
                throw new IllegalStateException("vkLatencySleepNV failed: " + r);
            }
            VkSemaphoreWaitInfo waitInfo = VkSemaphoreWaitInfo.calloc(stack).sType$Default()
                    .semaphoreCount(1)
                    .pSemaphores(stack.longs(timelineSemaphore)).pValues(stack.longs(counter));
            r = VK12.vkWaitSemaphores(device, waitInfo, SLEEP_WAIT_TIMEOUT_NS);
            if (r != VK10.VK_SUCCESS) {
                // A single missed wait just skips pacing for this frame — not fatal, worth knowing about if
                // it happens constantly (would mean the driver never signals, which needs investigating).
                CausticaMod.LOGGER.warn("Reflex: vkWaitSemaphores({}) returned {}; pacing skipped this frame",
                        counter, r);
                timedOut = true;
            }
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("Reflex: sleep failed; Reflex disabled for session", t);
        }
        if (timedOut) {
            // Timed-out frames burn the whole timeout without telling us anything about pacing, so
            // they are counted apart; folding them into meanWaitUs would inflate exactly the frames
            // where pacing already failed.
            timedOutFrames++;
        } else if (!failed) {
            sleptFrames++;
            waitedNanos += System.nanoTime() - started;
        }
        logDiagIfDue();
    }

    /**
     * Throttled proof of what pacing is actually doing, on the same gate and cadence as the other RT
     * diagnostics. {@code meanWaitUs} is the part of the frame the driver held the game thread for: with
     * {@code minimum-interval-us = 0} it is near zero by design, because Reflex then only submits late and
     * the game's own framerate cap supplies the delay — which is why that reading, not the presence of log
     * lines, is what shows whether pacing is in play.
     */
    private void logDiagIfDue() {
        if (!CausticaConfig.Rt.FrameStats.ENABLED.value()) {
            return;
        }
        long now = System.nanoTime();
        if (now - lastDiagNanos < DIAG_LOG_INTERVAL_NANOS) {
            return;
        }
        lastDiagNanos = now;
        CausticaMod.LOGGER.info("RT reflex diag: slept={} skipped={} timedOut={} minIntervalUs={} boost={} meanWaitUs={}",
                sleptFrames, skippedFrames, timedOutFrames, CausticaConfig.Rt.Reflex.MINIMUM_INTERVAL_US.value(),
                CausticaConfig.Rt.Reflex.LOW_LATENCY_BOOST.value(),
                sleptFrames == 0 ? 0 : waitedNanos / sleptFrames / 1000);
    }

    /**
     * Set a latency marker tagged with {@code id} (pass {@link #currentSimFrameId()} for SIMULATION/
     * RENDERSUBMIT markers, or the value just returned by {@link #advancePresentId()} for PRESENT markers —
     * these are deliberately different counters, see the class doc). Markers are pure diagnostics for the
     * driver's latency analysis (incl. NVIDIA overlays) — a failure here never disables pacing.
     */
    public void marker(VkDevice device, long swapchain, int marker, long id) {
        if (!enabled() || failed || swapchain == 0L || swapchain != sleepModeSwapchain) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSetLatencyMarkerInfoNV info = VkSetLatencyMarkerInfoNV.calloc(stack).sType$Default()
                    .presentID(id).marker(marker);
            NVLowLatency2.vkSetLatencyMarkerNV(device, swapchain, info);
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("Reflex: vkSetLatencyMarkerNV({}) threw; ignoring", marker, t);
        }
    }

    private void ensureSemaphore(VkDevice device) {
        if (timelineSemaphore != 0L) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreTypeCreateInfo typeInfo = VkSemaphoreTypeCreateInfo.calloc(stack).sType$Default()
                    .semaphoreType(VK12.VK_SEMAPHORE_TYPE_TIMELINE).initialValue(0L);
            VkSemaphoreCreateInfo sci = VkSemaphoreCreateInfo.calloc(stack).sType$Default().pNext(typeInfo);
            LongBuffer p = stack.mallocLong(1);
            if (VK10.vkCreateSemaphore(device, sci, null, p) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("vkCreateSemaphore(reflex timeline) failed");
            }
            timelineSemaphore = p.get(0);
        }
    }

    /** Device teardown: destroy the timeline semaphore + reset all per-swapchain/per-session state. */
    public void destroy(VkDevice device) {
        if (timelineSemaphore != 0L) {
            VK10.vkDestroySemaphore(device, timelineSemaphore, null);
            timelineSemaphore = 0L;
        }
        sleepModeSwapchain = 0L;
        counter = 0L;
        presentCounter = 0L;
        failed = false;
        sleptFrames = 0L;
        timedOutFrames = 0L;
        skippedFrames = 0L;
        waitedNanos = 0L;
    }
}

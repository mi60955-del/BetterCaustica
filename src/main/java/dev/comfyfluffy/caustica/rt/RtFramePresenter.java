package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg;
import dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg;

import it.unimi.dsi.fastutil.longs.LongList;

import net.minecraft.client.Minecraft;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.KHRSwapchain;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDependencyInfo;
import org.lwjgl.vulkan.VkImageBlit;
import org.lwjgl.vulkan.VkImageMemoryBarrier2;
import org.lwjgl.vulkan.VkMemoryBarrier2;
import org.lwjgl.vulkan.VkPresentInfoKHR;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;

import java.nio.IntBuffer;
import java.nio.LongBuffer;

/**
 * DLSS Frame Generation present engine (slice 2). Shows more than one image per rendered frame: the
 * generated frame(s), then the real frame.
 *
 * <p>It hooks Minecraft's frame tail (Minecraft.java: {@code blitFromTexture} → {@code encoder.submit()} →
 * {@code present()}). At {@code blitFromTexture} TAIL it acquires extra swapchain image(s) and records a
 * Y-flipped blit into <em>Minecraft's own command encoder</em> (the persistent singleton), so MC's
 * once-per-frame {@code submit()} flushes our work in the same {@code vkQueueSubmit} that signals the real
 * frame — this is what makes our present semaphores actually get signaled (the deferred-submit model is why
 * a self-contained present here failed validation). Then at {@code present()} HEAD we present the extra
 * image(s) before MC presents the real one, giving display order generated-then-real.
 *
 * <p>The generated frame is {@link RtDlssFg}'s real DLSSG-interpolated output (via
 * {@link RtComposite#fgInterpolate}); when there's simply no captured RT frame this tick (menu/loading/
 * transition — routine) it falls back to duplicating the real frame for just that one frame (see
 * {@code interpFallbackDuplicate} in the present-rate log), but a genuine FG failure is fatal (see that
 * method's docs) rather than silently duplicating forever. Called from both present paths — the normal SDR
 * {@code blitFromTexture} TAIL, and (via {@code hdrBackbuffer=true}) explicitly from inside the HDR present
 * hook, since the HDR/PQ path cancels {@code blitFromTexture} at HEAD and never reaches the TAIL inject.
 * Gated by {@code caustica.rt.fg} (default off).
 */
public final class RtFramePresenter {
    public static final RtFramePresenter INSTANCE = new RtFramePresenter();

    // Non-blocking on purpose. The presents that would release an image are queued later in the same frame,
    // so a waiting acquire cannot be satisfied by anything this frame still has to do: with a long timeout an
    // over-subscribed frame stalls for the whole timeout and the frame rate collapses to one frame per
    // timeout. Returning VK_TIMEOUT immediately instead skips the extra present for this frame.
    private static final long ACQUIRE_TIMEOUT_NS = 0L;

    private static final long LOG_INTERVAL_NS = 1_000_000_000L;

    private long[] acquireSemaphores = new long[0];
    private int acquireCursor;
    private boolean failed;

    // Frames acquired + recorded this frame, awaiting present at present() HEAD (after MC's submit flush).
    private int[] pendingImageIndex = new int[0];
    private long[] pendingPresentSem = new long[0];
    private int pendingCount;

    // Present-rate diagnostics: real vs generated vkQueuePresentKHR calls per second, independent of MC's own
    // fps counter (which only counts rendered/simulated frames, not our extra presents).
    private long logWindowStartNs;
    private int realFramesInWindow;
    private int generatedFramesInWindow;
    private int interpOkInWindow;
    private int interpFallbackInWindow;
    private int acquireFailsInWindow;
    private int lastAcquireResult = VK10.VK_SUCCESS;
    private long presentFailures;
    private int lastPresentResult = VK10.VK_SUCCESS;
    private int swapchainImageCount = -1;
    // The presenter's own share of the frame. Comparing these against the flush-to-flush interval says
    // whether a long frame is spent here or inside Minecraft's frame (its own acquire or fence wait).
    private long prepareNanosSum;
    private long prepareNanosMax;
    private long flushNanosSum;
    private long flushNanosMax;
    private long acquireNanosSum;
    private long acquireNanosMax;
    private long lastFlushEntryNs;
    private long flushIntervalNanosMax;
    private int acquireNotReady;
    private int acquireTimeout;
    private int acquireOtherFail;
    private int requestedGeneratedInWindow;

    private RtFramePresenter() {
    }

    /**
     * The configured frame-generation backend, as one of the {@link CausticaConfig.Rt.Fg} names.
     * The setting sanitises anything unrecognised to {@code dlss}, so this never has to.
     */
    public static String backend() {
        return CausticaConfig.Rt.Fg.BACKEND.get();
    }

    /**
     * Whether the chosen backend wants a pre-UI copy of the frame. Both vendor backends use it to
     * keep screen-fixed content out of the motion estimate; {@code self} has no input for it and
     * {@code duplicate} generates nothing, so capturing it would be a full-resolution copy per
     * frame for nobody.
     */
    public static boolean hudlessInputWanted() {
        String backend = backend();
        return CausticaConfig.Rt.Fg.DLSS.equals(backend) || CausticaConfig.Rt.Fg.FSR.equals(backend);
    }

    /**
     * Generated frames per rendered frame. FSR 3.1.x emits one per dispatch however large the
     * request is, so it clamps itself rather than presenting the same generated frame twice.
     */
    public static int generatedFramesPerRendered() {
        if (CausticaConfig.Rt.Fg.FSR.equals(backend())) {
            return Math.min(1, CausticaConfig.Rt.Fg.MULTI_FRAME_COUNT.value());
        }
        return RtDlssFg.INSTANCE.effectiveMultiFrameCount();
    }

    /** Whether FG extra-present should run this frame (enabled, a backend that can, in a world). */
    public boolean isActive() {
        if (failed || !RtDlssFg.enabled() || Minecraft.getInstance().level == null) {
            return false;
        }
        return switch (backend()) {
            case CausticaConfig.Rt.Fg.DUPLICATE, CausticaConfig.Rt.Fg.SELF -> true;
            case CausticaConfig.Rt.Fg.FSR -> RtFsrFg.INSTANCE.isAvailable();
            default -> RtDlssFg.INSTANCE.isAvailable();
        };
    }

    /**
     * Produce generated frame {@code index} of {@code count}, or null when this index has no
     * generated content and the real frame is presented again for it. Each backend is an
     * alternative to the others, which is how the extra-present path can be exercised with no
     * interpolation at all.
     */
    private static RtImage generateFrame(VulkanCommandEncoder enc, long backbufferView, long srcImage,
            int srcW, int srcH, int swapW, int swapH, int index, int count, boolean hdrBackbuffer) {
        return switch (backend()) {
            case CausticaConfig.Rt.Fg.DUPLICATE -> null;
            case CausticaConfig.Rt.Fg.SELF -> RtComposite.INSTANCE.fgInterpolateSelf(enc, backbufferView,
                    srcImage, swapW, swapH, hdrBackbuffer);
            case CausticaConfig.Rt.Fg.FSR -> index == 1
                    ? RtComposite.INSTANCE.fgInterpolateFsr(enc, srcImage, srcW, srcH, hdrBackbuffer)
                    : null;
            default -> RtComposite.INSTANCE.fgInterpolate(enc, backbufferView, srcImage, swapW, swapH,
                    index, count, hdrBackbuffer);
        };
    }

    /**
     * Acquire {@code generatedCount} extra swapchain images and record a Y-flipped blit of {@code srcImage}
     * (the final rendered frame, GENERAL layout) into each, using Minecraft's command encoder {@code enc} so
     * the work rides MC's next {@code submit()}. The presents happen later in {@link #flushPendingPresents}.
     * Blits DLSSG's real interpolated output per generated frame, or a duplicate of the real frame when RT
     * simply isn't producing frames this tick (routine). A genuine DLSSG failure latches FG off for the
     * session — see {@link RtComposite#fgInterpolate}.
     *
     * @param hdrBackbuffer whether {@code backbufferView}/{@code srcImage} is the PQ HDR backbuffer
     *     ({@link RtComposite#hdrBackbufferView()}, already UI-composited) rather than the SDR main target —
     *     selects DLSSG's HDR backbuffer format/flag in {@link RtComposite#fgInterpolate}.
     */
    public void prepareExtraFrames(VulkanCommandEncoder enc, VulkanDevice device, long swapchain,
            LongList swapchainImages, long[] presentSemaphores, int swapW, int swapH,
            long backbufferView, long srcImage, int srcW, int srcH, int generatedCount, boolean hdrBackbuffer) {
        pendingCount = 0;
        if (failed || swapchain == 0L || srcImage == 0L || generatedCount <= 0) {
            return;
        }
        if (swapchainImages.size() != swapchainImageCount) {
            // Each frame needs one image for Minecraft's own present plus one per generated frame, all held
            // at once, so the image count bounds how many extra presents a single frame can carry.
            CausticaMod.LOGGER.info("FG present engine: swapchain image count {} -> {}",
                    swapchainImageCount, swapchainImages.size());
            swapchainImageCount = swapchainImages.size();
        }
        // Minecraft holds its own acquired image for the whole frame and every generated frame needs one
        // more, all at once, so the swapchain must be able to supply 1 + generatedCount images. Measured on
        // a 3-image swapchain: one extra present is absorbed with no failures, two exhaust it.
        requestedGeneratedInWindow = generatedCount;
        generatedCount = Math.min(generatedCount, swapchainImages.size() - 2);
        if (generatedCount <= 0) {
            return;
        }
        final long prepareStartNs = System.nanoTime();
        try {
            ensureCapacity(device, swapchainImages.size() + 1, generatedCount);
            for (int i = 0; i < generatedCount; i++) {
                // null = no generated frame for this index, so the real frame is presented again for it.
                // Routine when there is no captured RT frame to interpolate from (menu/loading/transition),
                // and also what duplicate-only asks for. A genuine failure throws instead, caught below,
                // which disables FG for the session.
                RtImage interp = generateFrame(enc, backbufferView, srcImage, srcW, srcH, swapW, swapH, i + 1,
                        generatedCount, hdrBackbuffer);
                if (interp != null) {
                    interpOkInWindow++;
                } else {
                    interpFallbackInWindow++;
                }
                long blitSrc = interp != null ? interp.image : srcImage;
                // Clip to the smaller of swapchain and source exactly as Blaze3D's own blitFromTexture does
                // (it computes min(swapchainWidth, view.getWidth(0)) and min(swapchainHeight, ...), and blits
                // with VK_FILTER_NEAREST). A different region here would put the generated frame on another
                // vertical scale than the real frame, and the two are presented on alternate vsyncs.
                int copyW = Math.min(swapW, interp != null ? interp.width : srcW);
                int copyH = Math.min(swapH, interp != null ? interp.height : srcH);

                long acquireSem = acquireSemaphores[acquireCursor];
                acquireCursor = (acquireCursor + 1) % acquireSemaphores.length;

                int imageIndex;
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    IntBuffer pIndex = stack.callocInt(1);
                    long acquireStartNs = System.nanoTime();
                    int r = KHRSwapchain.vkAcquireNextImageKHR(device.vkDevice(), swapchain, ACQUIRE_TIMEOUT_NS, acquireSem, 0L, pIndex);
                    long acquireNs = System.nanoTime() - acquireStartNs;
                    acquireNanosSum += acquireNs;
                    if (acquireNs > acquireNanosMax) {
                        acquireNanosMax = acquireNs;
                    }
                    if (r != VK10.VK_SUCCESS && r != 1000001003 /* SUBOPTIMAL */) {
                        acquireFailsInWindow++;
                        lastAcquireResult = r;
                        if (r == VK10.VK_NOT_READY) {
                            acquireNotReady++;
                        } else if (r == VK10.VK_TIMEOUT) {
                            acquireTimeout++;
                        } else {
                            acquireOtherFail++;
                        }
                        return; // no image available: present what we have, let MC recover
                    }
                    imageIndex = pIndex.get(0);
                }
                long dstImage = swapchainImages.getLong(imageIndex);
                long presentSem = presentSemaphores[imageIndex];
                recordBlit(enc, blitSrc, dstImage, copyW, copyH, acquireSem, presentSem);

                pendingImageIndex[pendingCount] = imageIndex;
                pendingPresentSem[pendingCount] = presentSem;
                pendingCount++;
            }
        } catch (Throwable t) {
            failed = true;
            pendingCount = 0;
            CausticaMod.LOGGER.error("DLSS-FG present-record failed; frame generation disabled", t);
        } finally {
            long spentNs = System.nanoTime() - prepareStartNs;
            prepareNanosSum += spentNs;
            if (spentNs > prepareNanosMax) {
                prepareNanosMax = spentNs;
            }
        }
    }

    /**
     * Present the frames acquired in {@link #prepareExtraFrames} (call at {@code present()} HEAD, after MC's
     * {@code submit()} has flushed — so the present semaphores are signaled — and before MC presents the real
     * frame, giving generated-then-real order).
     */
    public void flushPendingPresents(long swapchain, VkQueue presentQueue) {
        long flushStartNs = System.nanoTime();
        if (lastFlushEntryNs != 0L) {
            long intervalNs = flushStartNs - lastFlushEntryNs;
            if (intervalNs > flushIntervalNanosMax) {
                flushIntervalNanosMax = intervalNs;
            }
        }
        lastFlushEntryNs = flushStartNs;
        int presentedThisFrame = 0;
        if (!failed && pendingCount != 0) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                for (int i = 0; i < pendingCount; i++) {
                    VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack).sType$Default();
                    present.pWaitSemaphores(stack.longs(pendingPresentSem[i]));
                    present.swapchainCount(1);
                    present.pSwapchains(stack.longs(swapchain));
                    present.pImageIndices(stack.ints(pendingImageIndex[i]));
                    int result = KHRSwapchain.vkQueuePresentKHR(presentQueue, present);
                    if (result != VK10.VK_SUCCESS) {
                        // OUT_OF_DATE and friends mean this generated frame silently vanished; surface
                        // it in the same diagnostic line that already tracks acquire results.
                        presentFailures++;
                        lastPresentResult = result;
                        CausticaMod.LOGGER.warn("DLSS-FG generated-frame present returned {}; the frame is lost",
                                result);
                    }
                    presentedThisFrame++;
                }
            } catch (Throwable t) {
                failed = true;
                CausticaMod.LOGGER.error("DLSS-FG present failed; frame generation disabled", t);
            } finally {
                pendingCount = 0;
            }
        }
        long flushSpentNs = System.nanoTime() - flushStartNs;
        flushNanosSum += flushSpentNs;
        if (flushSpentNs > flushNanosMax) {
            flushNanosMax = flushSpentNs;
        }
        if (RtDlssFg.enabled()) {
            logPresentRate(presentedThisFrame);
        }
    }

    /**
     * Debug diagnostic (2026-07-01): tracks real vs generated {@code vkQueuePresentKHR} calls per second,
     * separate from MC's own fps counter — that counter only reflects simulated/rendered frames
     * (blitFromTexture calls), so it would NOT show an increase from FG's extra presents even if the display
     * is genuinely receiving more frames. Logged only when {@code caustica.rt.fg} is enabled.
     */
    private void logPresentRate(int generatedThisFrame) {
        realFramesInWindow++;
        generatedFramesInWindow += generatedThisFrame;
        long now = System.nanoTime();
        if (logWindowStartNs == 0L) {
            logWindowStartNs = now;
            return;
        }
        long elapsed = now - logWindowStartNs;
        if (elapsed < LOG_INTERVAL_NS) {
            return;
        }
        double seconds = elapsed / 1.0e9;
        double realFps = realFramesInWindow / seconds;
        double totalFps = (realFramesInWindow + generatedFramesInWindow) / seconds;
        CausticaMod.LOGGER.info(
                "[FG present-rate] real={} gen={} requestedGen={} realFps={} totalPresentFps={} configuredMultiFrameCount={} "
                        + "interpOk={} interpFallbackDuplicate={} backend={} acquireFails={} notReady={} "
                        + "timeout={} otherFail={} lastAcquireResult={} presentFails={} lastPresentResult={} "
                        + "swapchainImages={} windowFocused={} "
                        + "maxFlushGapMs={} prepareMs(sum/max)={}/{} flushMs(sum/max)={}/{} "
                        + "acquireMs(sum/max)={}/{}",
                realFramesInWindow, generatedFramesInWindow, requestedGeneratedInWindow,
                String.format("%.1f", realFps), String.format("%.1f", totalFps),
                generatedFramesPerRendered(), interpOkInWindow, interpFallbackInWindow,
                backend(), acquireFailsInWindow, acquireNotReady, acquireTimeout,
                acquireOtherFail, lastAcquireResult, presentFailures, lastPresentResult,
                swapchainImageCount,
                Minecraft.getInstance().getWindow().isFocused(),
                String.format("%.1f", flushIntervalNanosMax / 1e6),
                String.format("%.1f", prepareNanosSum / 1e6), String.format("%.1f", prepareNanosMax / 1e6),
                String.format("%.1f", flushNanosSum / 1e6), String.format("%.1f", flushNanosMax / 1e6),
                String.format("%.1f", acquireNanosSum / 1e6), String.format("%.1f", acquireNanosMax / 1e6));
        logWindowStartNs = now;
        realFramesInWindow = 0;
        generatedFramesInWindow = 0;
        interpOkInWindow = 0;
        interpFallbackInWindow = 0;
        acquireFailsInWindow = 0;
        lastAcquireResult = VK10.VK_SUCCESS;
        presentFailures = 0;
        lastPresentResult = VK10.VK_SUCCESS;
        acquireNotReady = 0;
        acquireTimeout = 0;
        acquireOtherFail = 0;
        prepareNanosSum = 0;
        prepareNanosMax = 0;
        flushNanosSum = 0;
        flushNanosMax = 0;
        acquireNanosSum = 0;
        acquireNanosMax = 0;
        flushIntervalNanosMax = 0;
    }

    private void recordBlit(VulkanCommandEncoder enc, long srcImage, long dstImage, int copyW, int copyH,
            long acquireSem, long presentSem) {
        VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Swapchain UNDEFINED -> TRANSFER_DST (stage/access values mirror MC's blitFromTexture).
            VkImageMemoryBarrier2.Buffer toDst = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toDst.get(0).srcStageMask(0L).srcAccessMask(0L).dstStageMask(4096L).dstAccessMask(4096L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(dstImage);
            toDst.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            // Make the source's prior writes visible to the blit read — the world render (duplicate path) or
            // the DLSSG evaluate that wrote the interp image earlier in this same submit (interp path).
            VkMemoryBarrier2.Buffer srcVis = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            srcVis.get(0).srcStageMask(65536L).srcAccessMask(65536L).dstStageMask(4096L).dstAccessMask(2048L);
            VkDependencyInfo dep1 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toDst).pMemoryBarriers(srcVis);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep1);

            // Blit final frame (GENERAL) -> swapchain (TRANSFER_DST), Y-flipped like vanilla.
            VkImageBlit.Buffer region = VkImageBlit.calloc(1, stack);
            region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).srcOffsets(1).set(copyW, copyH, 1); // srcOffsets[0] = (0,0,0) from calloc
            region.get(0).dstOffsets(0).set(0, copyH, 0);
            region.get(0).dstOffsets(1).set(copyW, 0, 1);
            VK10.vkCmdBlitImage(cmd, srcImage, VK10.VK_IMAGE_LAYOUT_GENERAL, dstImage,
                    VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region, VK10.VK_FILTER_NEAREST);

            // Swapchain TRANSFER_DST -> PRESENT_SRC_KHR (1000001002).
            VkImageMemoryBarrier2.Buffer toPresent = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toPresent.get(0).srcStageMask(4096L).srcAccessMask(4096L).dstStageMask(65536L).dstAccessMask(0L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(1000001002)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(dstImage);
            toPresent.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VkMemoryBarrier2.Buffer mem2 = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            mem2.get(0).srcStageMask(4096L).srcAccessMask(2048L).dstStageMask(65536L).dstAccessMask(98304L);
            VkDependencyInfo dep2 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toPresent).pMemoryBarriers(mem2);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep2);
        }
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(fg blit) failed");
        }
        // Register on MC's encoder (same order as MC's blitFromTexture): wait on the acquire, run the blit,
        // signal the image's present semaphore. MC's once-per-frame submit() flushes this in one
        // vkQueueSubmit, which is what actually signals presentSem (deferred-submit model).
        enc.waitSemaphore(acquireSem, 0L, 65536L);
        enc.execute(cmd);
        enc.signalSemaphore(presentSem, 0L, 4096L);
    }

    private void ensureCapacity(VulkanDevice device, int semaphoreCount, int generatedCount) {
        if (pendingImageIndex.length < generatedCount) {
            pendingImageIndex = new int[generatedCount];
            pendingPresentSem = new long[generatedCount];
        }
        if (acquireSemaphores.length >= semaphoreCount) {
            return;
        }
        destroy(device);
        acquireSemaphores = new long[semaphoreCount];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreCreateInfo sci = VkSemaphoreCreateInfo.calloc(stack).sType$Default();
            LongBuffer p = stack.mallocLong(1);
            for (int i = 0; i < semaphoreCount; i++) {
                if (VK10.vkCreateSemaphore(device.vkDevice(), sci, null, p) != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("vkCreateSemaphore(fg acquire) failed");
                }
                acquireSemaphores[i] = p.get(0);
            }
        }
        acquireCursor = 0;
    }

    /** Destroy the acquire-semaphore pool (device teardown). */
    public void destroy(VulkanDevice device) {
        for (long sem : acquireSemaphores) {
            if (sem != 0L) {
                VK10.vkDestroySemaphore(device.vkDevice(), sem, null);
            }
        }
        acquireSemaphores = new long[0];
        acquireCursor = 0;
        pendingCount = 0;
    }
}

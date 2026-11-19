package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.platform.NativeImage;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.pipeline.RtStillDenoisePipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtStillPipeline;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Util;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageBlit;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryBarrier;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The accumulated still render: one key press averages the light the player is already looking at into a
 * single PNG, while the game keeps running, DLSS-RR keeps reconstructing and frame generation keeps
 * generating.
 *
 * <p>What is summed is the scene-linear image the display chain consumes — the post-RR (or fallback
 * upscaled) trace, before exposure, bloom and the tone curve. Averaging the finished display frame instead
 * is biased by the tone curve: a curve is not linear, so a mean of tone-mapped noisy frames comes out
 * darker and flatter the more samples it holds. Summing radiance and mapping the mean once is the long
 * exposure a still render is meant to be.
 *
 * <p>Each frame of a run hands the display chain the accumulated mean instead of the live trace, so the
 * screen shows the run converging and bloom, the lens flare and the tone curve all see the mean rather
 * than being averaged themselves. The composite copies that display frame into the world target, which is
 * why a vanilla F2 capture takes the accumulated picture as well. The final frame is captured into a
 * PNG-order image and read back on a tick; the run's GPU images exist only while it is running.
 *
 * <p>A frame whose camera pose differs from the last one counted restarts the run: a mean of frames from two
 * viewpoints is neither of them. Poses are compared as quantised numbers, because the sub-pixel camera drift
 * of a held still key has to keep counting as the same viewpoint while a real turn of the head may not.
 *
 * <p>An unbounded run has no sample target and keeps summing until the key is pressed again. A run can also
 * resize the window to the configured still resolution first and put it back when it ends. Auto-exposure is
 * held for the whole run: a long exposure must not re-meter every sample, or its mean spans several
 * exposures.
 */
public final class RtStillRender {
    public static final RtStillRender INSTANCE = new RtStillRender();

    /** Progress reaches the player four times per run, not once per frame. */
    private static final int PROGRESS_STEPS = 4;
    private static final int BYTES_PER_TEXEL = 4;
    /** Ticks the window size must hold after a switch before the run may size its images to it. */
    private static final int SETTLE_TICKS = 3;
    /** Ticks a switched run waits for the resize before giving up and putting the window back. */
    private static final int SETTLE_TIMEOUT_TICKS = 100;
    /** Samples between progress overlays in an unbounded run, which has no target to divide by. */
    private static final int INFINITE_PROGRESS_STEP = 64;
    /** How long a run lets auto-exposure settle before its first sample is counted. */
    private static final long COOLDOWN_NANOS = 500_000_000L;
    /** Halton phases a still run uses when DLSS-RR is off; 32 phases repeat too early for a long exposure. */
    private static final int STILL_JITTER_PHASES = 512;
    /**
     * a-trous steps the denoise runs. Three double the support per step, so the last one reaches sixteen
     * texels without a 33x33 kernel; an odd count leaves the filtered result in the second scratch image.
     */
    private static final int DENOISE_STEPS = 3;
    /**
     * How many of the pixel's own measured noise a tap may differ by before it fades out of the kernel. The
     * spread itself is not a constant: the still pass writes the standard error of each pixel's mean into
     * the scene image's alpha, so shadow texels that are still far from converged admit the neighbours they
     * have to average with while a converged highlight keeps to itself. Three is wide enough to average a
     * flat surface down to the noise floor of a long run and still narrow enough that a shading step of a
     * few sigma, which is what a shadow edge or a texture boundary is, stays put.
     */
    private static final float DENOISE_NOISE_SCALE = 3.0f;
    /**
     * The earlier estimator's tolerance: the relative luminance a tap may differ by at one sample, divided
     * by the square root of the run's count, and bounded so it never becomes dead-strict nor a plain blur.
     * A run that accumulated with the cap records no measured spread, so the filter falls back to this.
     */
    private static final float DENOISE_SIGMA_AT_ONE_SAMPLE = 2.0f;
    private static final float DENOISE_SIGMA_MIN = 0.02f;
    private static final float DENOISE_SIGMA_MAX = 0.5f;

    private RtStillPipeline pipeline;
    private RtStillDenoisePipeline denoisePipeline;
    /** Sum of scene-linear frames, rgba32f: sixteen bytes per display texel. */
    private RtImage hdrAccumulator;
    /** The finishing frame's display image in PNG row order, rgba8, read back on a tick. */
    private RtImage resolved;
    /** Demodulated residual and the a-trous pair's intermediate, both display-res rgba16f. */
    private RtImage denoiseResidual;
    private RtImage denoiseScratch;
    private boolean accumulating;
    /** The run has every sample it wants (or was stopped) and the next frame is its last. */
    private boolean finishing;
    private boolean resolvePending;
    /** A run was asked for and the window is being resized to the configured still resolution. */
    private boolean waitingForResolution;
    private boolean resolutionSwitched;
    private int restoreScreenWidth;
    private int restoreScreenHeight;
    private int preSwitchDisplayWidth;
    private int preSwitchDisplayHeight;
    private int settleFrames;
    private int settleTicks;
    private int settledWidth;
    private int settledHeight;
    private boolean infinite;
    /** Latched at the start of a run: the earlier estimator caps each sample and records no noise. */
    private boolean legacyAccumulation;
    /** Non-zero while a started run waits for auto-exposure to settle, before its first sample. */
    private long cooldownUntilNanos;
    private int samples;
    private int target;
    private int reportedStep;
    /** Zero means "no frame counted yet", which is what makes the first frame a store instead of an add. */
    private long poseKey;

    private RtStillRender() {
    }

    /**
     * The hotkey: start a run, or finish the one in flight. Finishing resolves and saves what the run has
     * summed instead of dropping it, which is also the only way an unbounded run ends.
     */
    public void toggle() {
        if (accumulating || resolvePending || waitingForResolution || cooldownUntilNanos != 0L) {
            finish();
            return;
        }
        start();
    }

    /**
     * Phases the camera jitter should use for a still run that has no reconstructor. Zero when no run is
     * starting or running, so the caller keeps its normal (jitter-free) live path.
     */
    public int stillJitterPhases() {
        // Only while samples are actually being counted: the warm-up exists to let the exposure settle,
        // and metering a jittered frame there would latch a different exposure than the live view has.
        return accumulating ? STILL_JITTER_PHASES : 0;
    }

    /**
     * Whether a run owns the display chain's source right now — from a requested run that is still
     * settling its resolution through the last captured frame.
     *
     * <p>This is the same set of states {@link #cancel} has to drop, and it is what a feature drawn only
     * in the live view asks before it writes anything: while a run is active the image in front of the
     * display chain is a mean of sampled light, and a moving overlay has no honest place in it.
     */
    public boolean active() {
        return accumulating || resolvePending || waitingForResolution || cooldownUntilNanos != 0L;
    }

    /**
     * Sum this frame's scene-linear image into the run and write the mean into the display chain's source
     * slot. Called before exposure and bloom, so those passes run on the mean rather than on the live frame.
     * Returns false when no run is active (or it just aborted), and the caller puts the live image there.
     *
     * <p>{@code denoise} asks for the spatial filter to run over the mean before the display chain sees it,
     * with the guide views the frame that was just summed wrote. The caller passes false when the image was
     * already reconstructed — a denoised mean has no noise left for a second filter to remove, and RR's
     * output is display-res only by construction.
     */
    public boolean previewSource(RtContext ctx, VkCommandBuffer cmd, RtImage source, RtImage output,
                                 long guideNormalView, long guideAlbedoView, long guideDepthView,
                                 boolean denoise,
                                 double camX, double camY, double camZ,
                                 Matrix4fc viewRotation, Matrix4f projection) {
        if (!accumulating) {
            if (cooldownUntilNanos == 0L) {
                return false;
            }
            if (System.nanoTime() < cooldownUntilNanos) {
                return false;
            }
            // The warm-up kept the frame's normal metering alive; latch what it settled on and start
            // counting from this frame.
            cooldownUntilNanos = 0L;
            accumulating = true;
            RtComposite.INSTANCE.exposure().setHold(true);
            samples = 0;
            reportedStep = 0;
            poseKey = 0L;
        }
        if (source.width != hdrAccumulator.width || source.height != hdrAccumulator.height) {
            abort(ctx, "the scene resolution changed from " + hdrAccumulator.width + "x" + hdrAccumulator.height
                    + " to " + source.width + "x" + source.height);
            return false;
        }
        if (!finishing) {
            long pose = poseKey(camX, camY, camZ, viewRotation, projection);
            boolean first = pose != poseKey;
            poseKey = pose;
            if (first) {
                samples = 0;
                reportedStep = 0;
            }
            RtImage accumulated = source;
            // A sample orders of magnitude above its pixel's own level is what a run shows as a dot that
            // never fades: it joins the sum once and decays as 1/N from there, and no spatial filter can take
            // it back out (see {@link #denoise}). It is rejected here instead, against the frame it arrived
            // in: only a texel brighter than all eight neighbours is capped, which is what keeps a
            // one-texel-wide highlight — a chain, a fence post, a distant torch — out of the rule. Nothing
            // about the running mean is involved, so this cannot feed back and cannot flip between frames.
            // A run of the earlier estimator skips it: that path caps each sample against the running mean
            // instead, and doing both would make the old pictures unreproducible.
            if (denoise && denoisePipeline != null && !legacyAccumulation) {
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    denoisePipeline.setImages(source.view, guideNormalView, guideAlbedoView, guideDepthView,
                            denoiseScratch.view);
                    denoisePipeline.dispatch(cmd, source.width, source.height,
                            RtStillDenoisePipeline.MODE_FIREFLY_CLAMP, 1, 0.0f, 0.0f, 0);
                    memoryBarrier(cmd, stack, VK10.VK_ACCESS_SHADER_WRITE_BIT,
                            VK10.VK_ACCESS_SHADER_READ_BIT,
                            VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
                accumulated = denoiseScratch;
            }
            pipeline.setImages(hdrAccumulator.view, accumulated.view, output.view);
            // samples is still the accumulator's prior count here: zero for the first frame, which stores
            // instead of adding and so has no running mean for its second moment to be measured against.
            pipeline.dispatch(cmd, hdrAccumulator.width, hdrAccumulator.height,
                    first ? RtStillPipeline.MODE_FIRST : RtStillPipeline.MODE_ADD, 0.0f, samples,
                    legacyAccumulation ? 1 : 0);
            samples++;
            if (!infinite && samples >= target) {
                finishing = true;
            }
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still resolve mean")) {
            // The accumulate dispatch above wrote the accumulator; this read has to see it, and the display
            // chain that follows has to see the mean.
            memoryBarrier(cmd, stack, VK10.VK_ACCESS_SHADER_WRITE_BIT, VK10.VK_ACCESS_SHADER_READ_BIT,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            pipeline.dispatch(cmd, hdrAccumulator.width, hdrAccumulator.height,
                    RtStillPipeline.MODE_RESOLVE, 1.0f / samples, samples, legacyAccumulation ? 1 : 0);
            memoryBarrier(cmd, stack, VK10.VK_ACCESS_SHADER_WRITE_BIT, VK10.VK_ACCESS_SHADER_READ_BIT,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        if (denoise && CausticaConfig.Rt.Still.DENOISE.value()) {
            denoise(ctx, cmd, output, guideNormalView, guideAlbedoView, guideDepthView);
        }
        if (finishing) {
            return true;
        }
        if (infinite) {
            if (samples % INFINITE_PROGRESS_STEP == 0) {
                overlay(Component.translatable("caustica.hotkey.still.progress.infinite",
                        Component.literal(Integer.toString(samples))));
            }
            return true;
        }
        int step = samples * PROGRESS_STEPS / target;
        if (step > reportedStep) {
            reportedStep = step;
            overlay(Component.translatable("caustica.hotkey.still.progress",
                    Component.literal(Integer.toString(samples)), Component.literal(Integer.toString(target))));
        }
        return true;
    }

    /**
     * Filters the mean the resolve has just written, in place: demodulate by the surface albedo, run
     * {@link #DENOISE_STEPS} a-trous steps over the residual, and remodulate back into {@code color}.
     *
     * <p>What this pass cannot do is remove a single sample that landed orders of magnitude above its
     * pixel's own level: the kernel weights its centre at a seventh of the total, so such a spike keeps a
     * visible fraction of its excess through every step, and a median sharp enough to remove it does not
     * vary smoothly with the mean and therefore shimmers as the run grows. Spikes are rejected before they
     * enter the accumulator instead; see {@link #previewSource}.
     *
     * <p>The a-trous tolerance is the pixel's own measured noise, which the resolve wrote into the scene
     * image's alpha. Early in a run everything is noise, so the kernel is wide; a converged mean carries a
     * spread near zero, so only pixel-scale differences remain worth accepting. The pass is spatial only —
     * the mean already holds every frame's sample, so there is no history to reuse and nothing to ghost.
     */
    private void denoise(RtContext ctx, VkCommandBuffer cmd, RtImage color,
                         long guideNormalView, long guideAlbedoView, long guideDepthView) {
        if (denoisePipeline == null || denoiseResidual == null || denoiseScratch == null) {
            return;
        }
        int width = color.width;
        int height = color.height;
        float sigma = DENOISE_NOISE_SCALE;
        if (legacyAccumulation) {
            // The earlier tolerance: 2/sqrt(N) of the local level. A run that capped its samples has no
            // measured spread in the alpha, so the filter needs the width from the count itself.
            sigma = DENOISE_SIGMA_AT_ONE_SAMPLE / (float) Math.sqrt(Math.max(samples, 1));
            sigma = Math.max(DENOISE_SIGMA_MIN, Math.min(DENOISE_SIGMA_MAX, sigma));
        }
        int measured = legacyAccumulation ? 0 : 1;
        float amount = CausticaConfig.Rt.Still.DENOISE_STRENGTH.value();
        int shaderReadWrite = VK10.VK_ACCESS_SHADER_WRITE_BIT | VK10.VK_ACCESS_SHADER_READ_BIT;
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still denoise")) {
            denoisePipeline.setImages(color.view, guideNormalView, guideAlbedoView, guideDepthView,
                    denoiseResidual.view);
            denoisePipeline.dispatch(cmd, width, height, RtStillDenoisePipeline.MODE_DEMODULATE, 1, sigma, 1.0f,
                    measured);
            RtImage source = denoiseResidual;
            RtImage destination = denoiseScratch;
            for (int index = 0; index < DENOISE_STEPS; index++) {
                memoryBarrier(cmd, stack, shaderReadWrite, shaderReadWrite,
                        VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                denoisePipeline.setImages(source.view, guideNormalView, guideAlbedoView, guideDepthView,
                        destination.view);
                denoisePipeline.dispatch(cmd, width, height, RtStillDenoisePipeline.MODE_ATROUS,
                        1 << index, sigma, amount, measured);
                RtImage swap = source;
                source = destination;
                destination = swap;
            }
            // Three odd steps leave the filtered residual in `source`; the remodulate writes `color`, which is
            // what the display chain reads next.
            memoryBarrier(cmd, stack, shaderReadWrite, shaderReadWrite,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            denoisePipeline.setImages(source.view, guideNormalView, guideAlbedoView, guideDepthView, color.view);
            denoisePipeline.dispatch(cmd, width, height, RtStillDenoisePipeline.MODE_REMODULATE, 1, sigma, 1.0f,
                    measured);
            memoryBarrier(cmd, stack, shaderReadWrite, shaderReadWrite,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }

    /**
     * Capture the display frame the composite has just produced from the mean. Called at the very end of the
     * frame, after the display chain has mapped the mean; the rows are flipped into PNG order here
     * so the tick's readback stays a plain copy. Only the finishing frame captures.
     */
    public void record(RtContext ctx, VkCommandBuffer cmd, RtImage displayImage) {
        if (!finishing || !accumulating) {
            return;
        }
        if (displayImage.width != resolved.width || displayImage.height != resolved.height) {
            abort(ctx, "the display resolution changed from " + resolved.width + "x" + resolved.height
                    + " to " + displayImage.width + "x" + displayImage.height);
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still capture")) {
            memoryBarrier(cmd, stack, VK10.VK_ACCESS_SHADER_WRITE_BIT,
                    VK10.VK_ACCESS_TRANSFER_READ_BIT | VK10.VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT);
            blitFlipped(cmd, stack, displayImage, resolved);
            memoryBarrier(cmd, stack, VK10.VK_ACCESS_TRANSFER_WRITE_BIT, VK10.VK_ACCESS_TRANSFER_READ_BIT,
                    VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT);
        }
        finishing = false;
        accumulating = false;
        resolvePending = true;
        overlay(Component.translatable("caustica.hotkey.still.saving",
                Component.literal(Integer.toString(samples))));
    }

    /**
     * Resolve and save a captured run. Called from the client tick: the capture went out with its frame's
     * submission, so the readback has to wait for a point past that — and a tick still runs after the world
     * is gone, which is when a finished run must still reach the disk.
     */
    public void service() {
        if (waitingForResolution) {
            serviceResolutionWait();
            return;
        }
        if (!resolvePending) {
            return;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            abort(null, "the RT device went away before the run could be read back");
            return;
        }
        resolveAndSave(ctx);
    }

    /** Drop a run that cannot continue, e.g. a render-state invalidation that rebuilt the world. */
    public void cancel(String reason) {
        if (accumulating || resolvePending || waitingForResolution || cooldownUntilNanos != 0L
                || resolutionSwitched) {
            abort(RtContext.currentOrNull(), reason);
        }
    }

    /** Release the run's GPU objects with no player-facing message; the device is being taken down. */
    public void destroy() {
        accumulating = false;
        finishing = false;
        resolvePending = false;
        waitingForResolution = false;
        cooldownUntilNanos = 0L;
        releaseImages();
        RtComposite.INSTANCE.exposure().setHold(false);
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        if (denoisePipeline != null) {
            denoisePipeline.destroy();
            denoisePipeline = null;
        }
    }

    private void start() {
        Minecraft client = Minecraft.getInstance();
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null || client.level == null || RtComposite.INSTANCE.hasFailed()) {
            overlay(Component.translatable("caustica.hotkey.still.unavailable"));
            return;
        }
        infinite = CausticaConfig.Rt.Still.INFINITE.value();
        // Latched for the whole run: the two estimators write different things into the same accumulator and
        // the filter is told which one to expect, so a run that switched halfway would produce a mean of two
        // different quantities.
        legacyAccumulation = CausticaConfig.Rt.Still.LEGACY_ACCUMULATION.value();
        boolean switchResolution = CausticaConfig.Rt.Still.RESOLUTION_SWITCH.value();
        int width = CausticaConfig.Rt.Still.WIDTH.value();
        int height = CausticaConfig.Rt.Still.HEIGHT.value();
        if (switchResolution) {
            if (client.getWindow().isFullscreen()) {
                // Resizing a fullscreen window would leave fullscreen behind; the run still happens at the
                // display's own size rather than silently changing how the desktop is driven.
                overlay(Component.translatable("caustica.hotkey.still.fullscreen"));
            } else if (width > 0 && height > 0
                    && (client.getWindow().getScreenWidth() != width
                    || client.getWindow().getScreenHeight() != height)) {
                restoreScreenWidth = client.getWindow().getScreenWidth();
                restoreScreenHeight = client.getWindow().getScreenHeight();
                preSwitchDisplayWidth = RtComposite.INSTANCE.displayWidth();
                preSwitchDisplayHeight = RtComposite.INSTANCE.displayHeight();
                resolutionSwitched = true;
                waitingForResolution = true;
                settleFrames = 0;
                settleTicks = 0;
                settledWidth = -1;
                settledHeight = -1;
                overlay(Component.translatable("caustica.hotkey.still.resolution",
                        Component.literal(Integer.toString(width)), Component.literal(Integer.toString(height))));
                CausticaMod.LOGGER.info("Still render resizing the window from {}x{} to {}x{}",
                        restoreScreenWidth, restoreScreenHeight, width, height);
                client.getWindow().setWindowed(width, height);
                return;
            }
        }
        beginRun(ctx);
    }

    /**
     * Wait for the resize the switch asked for to reach the render target. Sizing the accumulator before
     * then would tie it to the old display and abort on the first frame of the new one.
     */
    private void serviceResolutionWait() {
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            abort(null, "the RT device went away before the resize settled");
            return;
        }
        int width = RtComposite.INSTANCE.displayWidth();
        int height = RtComposite.INSTANCE.displayHeight();
        if (width <= 0 || height <= 0) {
            settleFrames = 0;
            settledWidth = -1;
            settledHeight = -1;
        } else if (width == settledWidth && height == settledHeight) {
            settleFrames++;
        } else {
            settleFrames = 0;
            settledWidth = width;
            settledHeight = height;
        }
        // A size that never changed is still the old display, not the resize: starting there would size the
        // accumulator to it and abort on the first frame the new one arrives.
        if (settleFrames >= SETTLE_TICKS
                && (width != preSwitchDisplayWidth || height != preSwitchDisplayHeight)) {
            beginRun(ctx);
            return;
        }
        if (++settleTicks > SETTLE_TIMEOUT_TICKS) {
            abort(ctx, "the window never settled at the still resolution");
        }
    }

    private void beginRun(RtContext ctx) {
        Minecraft client = Minecraft.getInstance();
        int width = RtComposite.INSTANCE.displayWidth();
        int height = RtComposite.INSTANCE.displayHeight();
        if (client.level == null || width <= 0 || height <= 0 || RtComposite.INSTANCE.hasFailed()) {
            overlay(Component.translatable("caustica.hotkey.still.unavailable"));
            restoreResolution();
            return;
        }
        target = infinite ? Integer.MAX_VALUE : Math.max(1, CausticaConfig.Rt.Still.SAMPLES.value());
        // Drained before the new images' init barriers: a run starts between frames, and its first source
        // frame is the next frame's, so this is the only moment nothing can be in flight.
        ctx.waitIdle();
        try {
            ctx.beginImageInitBatch();
            try {
                if (pipeline == null) {
                    pipeline = RtStillPipeline.create(ctx);
                }
                if (CausticaConfig.Rt.Still.DENOISE.value()) {
                    if (denoisePipeline == null) {
                        denoisePipeline = RtStillDenoisePipeline.create(ctx);
                    }
                    denoiseResidual = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                            "still denoise residual " + width + "x" + height);
                    denoiseScratch = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                            "still denoise scratch " + width + "x" + height);
                }
                hdrAccumulator = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R32G32B32A32_SFLOAT,
                        "still HDR accumulator " + width + "x" + height);
                resolved = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R8G8B8A8_UNORM,
                        "still resolved " + width + "x" + height);
            } finally {
                ctx.endImageInitBatch();
            }
        } catch (Throwable t) {
            releaseImages();
            failed(t);
            restoreResolution();
            return;
        }
        waitingForResolution = false;
        // Let auto-exposure settle before the first sample is counted: a run that started in the middle of
        // an adaptation would hold that transient for its whole length. The live frame keeps metering
        // during this window; only when it closes does the run latch the exposure (see previewSource).
        cooldownUntilNanos = System.nanoTime() + COOLDOWN_NANOS;
        samples = 0;
        reportedStep = 0;
        poseKey = 0L;
        overlay(infinite
                ? Component.translatable("caustica.hotkey.still.started.infinite")
                : Component.translatable("caustica.hotkey.still.started",
                        Component.literal(Integer.toString(target))));
        CausticaMod.LOGGER.info("Still render started at {}x{} with {} samples{} ({})", width, height,
                infinite ? "no limit" : Integer.toString(target),
                resolutionSwitched ? " (window resized for the run)" : "",
                legacyAccumulation ? "legacy running-mean cap" : "scene-linear, measured noise");
    }

    private void finish() {
        if (waitingForResolution) {
            abort(RtContext.currentOrNull(), "stopped before the window settled");
            return;
        }
        if (cooldownUntilNanos != 0L) {
            abort(RtContext.currentOrNull(), "stopped during the warm-up");
            return;
        }
        if (resolvePending || !accumulating || finishing) {
            return;
        }
        if (samples <= 0) {
            abort(RtContext.currentOrNull(), "no frame was counted");
            return;
        }
        finishing = true;
    }

    /**
     * Copy the display image into the PNG-order image, flipped: the display chain writes rows the way a
     * texture reads them, while a PNG's first row is its top.
     */
    private static void blitFlipped(VkCommandBuffer cmd, MemoryStack stack, RtImage source, RtImage destination) {
        VkImageBlit.Buffer region = VkImageBlit.calloc(1, stack);
        region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0)
                .layerCount(1);
        region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0)
                .layerCount(1);
        region.get(0).srcOffsets(0).set(0, 0, 0);
        region.get(0).srcOffsets(1).set(destination.width, destination.height, 1);
        region.get(0).dstOffsets(0).set(0, destination.height, 0);
        region.get(0).dstOffsets(1).set(destination.width, 0, 1);
        VK10.vkCmdBlitImage(cmd, source.image, VK10.VK_IMAGE_LAYOUT_GENERAL, destination.image,
                VK10.VK_IMAGE_LAYOUT_GENERAL, region, VK10.VK_FILTER_NEAREST);
    }

    /** A full memory barrier; every image stays in GENERAL, only the access scopes change. */
    private static void memoryBarrier(VkCommandBuffer cmd, MemoryStack stack, int srcAccess, int dstAccess,
                                      int srcStage, int dstStage) {
        VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack);
        barrier.get(0).sType$Default().srcAccessMask(srcAccess).dstAccessMask(dstAccess);
        VK10.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, barrier, null, null);
    }

    private void resolveAndSave(RtContext ctx) {
        resolvePending = false;
        int width = resolved.width;
        int height = resolved.height;
        long bytes = Math.multiplyExact(Math.multiplyExact((long) width, height), (long) BYTES_PER_TEXEL);
        Path output;
        try {
            output = nextOutputPath();
        } catch (IOException e) {
            abort(ctx, "the screenshots directory could not be prepared: " + e);
            return;
        }
        // The capture blit went out with its frame's submission, so this is where it lands — and where
        // nothing else can still be reading the image.
        ctx.waitIdle();
        RtBuffer readback = ctx.createReadbackBuffer(bytes, "still render readback");
        NativeImage image;
        try {
            ctx.submitSync(cmd -> recordReadback(ctx, cmd, readback, width, height));
            readback.invalidate();
            image = new NativeImage(NativeImage.Format.RGBA, width, height, false);
            MemoryUtil.memCopy(readback.mapped, MemoryUtil.memAddress(image.getPixelBytes()), bytes);
        } catch (Throwable t) {
            readback.destroy();
            abort(ctx, "the run could not be read back: " + t);
            return;
        }
        readback.destroy();
        releaseImages();
        RtComposite.INSTANCE.exposure().setHold(false);
        restoreResolution();
        CausticaMod.LOGGER.info("Resolving accumulated still render of {} samples to {}", samples, output);
        File file = output.toFile();
        Util.ioPool().execute(() -> {
            try {
                image.writeToFile(file);
                Component link = Component.literal(file.getAbsolutePath())
                        .withStyle(ChatFormatting.UNDERLINE)
                        .withStyle(style -> style.withClickEvent(new ClickEvent.OpenFile(file)));
                Minecraft.getInstance().execute(() -> chat(Component.translatable(
                        "caustica.hotkey.still.saved", Component.literal(Integer.toString(samples)), link)));
            } catch (IOException e) {
                CausticaMod.LOGGER.warn("Still render could not be written to {}", file, e);
                Minecraft.getInstance().execute(this::failed);
            } finally {
                image.close();
            }
        });
    }

    private void recordReadback(RtContext ctx, VkCommandBuffer cmd, RtBuffer readback, int width, int height) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still render readback")) {
            // The capture blit was recorded in an earlier command buffer; the dependency still has to be
            // expressed for this submit to read that write.
            transition(ctx, cmd, stack, resolved.image, VK10.VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK10.VK_ACCESS_TRANSFER_READ_BIT, VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK10.VK_PIPELINE_STAGE_TRANSFER_BIT);

            VkBufferImageCopy.Buffer copy = VkBufferImageCopy.calloc(1, stack);
            copy.get(0).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            copy.get(0).imageExtent().set(width, height, 1);
            VK10.vkCmdCopyImageToBuffer(cmd, resolved.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    readback.handle, copy);

            VkMemoryBarrier.Buffer toHost = VkMemoryBarrier.calloc(1, stack);
            toHost.get(0).sType$Default().srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_HOST_READ_BIT);
            VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, VK10.VK_PIPELINE_STAGE_HOST_BIT,
                    0, toHost, null, null);
        }
    }

    private void transition(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack, long image,
                            int srcAccess, int dstAccess, int srcStage, int dstStage) {
        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack);
        barrier.get(0).sType$Default()
                .oldLayout(VK10.VK_IMAGE_LAYOUT_GENERAL).newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                .srcAccessMask(srcAccess).dstAccessMask(dstAccess)
                .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .image(image);
        barrier.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
        VK10.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, barrier);
    }

    private void abort(RtContext ctx, String reason) {
        // A run whose images are still up is a run the player thinks is going, even after a failed readback
        // has cleared both flags. A switched run with no images yet still has a window to put back.
        boolean active = hdrAccumulator != null || accumulating || resolvePending || waitingForResolution
                || cooldownUntilNanos != 0L;
        accumulating = false;
        finishing = false;
        resolvePending = false;
        waitingForResolution = false;
        cooldownUntilNanos = 0L;
        if (!active && !resolutionSwitched) {
            return;
        }
        if (ctx != null) {
            // The run's images are written by dispatches already submitted with their frames, so this drain
            // is what makes them safe to destroy — the same one the resize path pays.
            ctx.waitIdle();
        }
        releaseImages();
        RtComposite.INSTANCE.exposure().setHold(false);
        restoreResolution();
        CausticaMod.LOGGER.info("Still render stopped: {}", reason);
        overlay(Component.translatable("caustica.hotkey.still.cancelled"));
    }

    /**
     * Put the window back where the switch found it. Every path that ends a run calls this — a finished
     * resolve, an abort, or leaving the world — so no corner case can leave the window at the still size.
     */
    private void restoreResolution() {
        if (!resolutionSwitched) {
            return;
        }
        resolutionSwitched = false;
        Minecraft client = Minecraft.getInstance();
        if (restoreScreenWidth > 0 && restoreScreenHeight > 0) {
            client.getWindow().setWindowed(restoreScreenWidth, restoreScreenHeight);
            CausticaMod.LOGGER.info("Still render restored the window to {}x{}",
                    restoreScreenWidth, restoreScreenHeight);
        }
        restoreScreenWidth = 0;
        restoreScreenHeight = 0;
    }

    private void releaseImages() {
        if (hdrAccumulator != null) {
            hdrAccumulator.destroy();
            hdrAccumulator = null;
        }
        if (resolved != null) {
            resolved.destroy();
            resolved = null;
        }
        if (denoiseResidual != null) {
            denoiseResidual.destroy();
            denoiseResidual = null;
        }
        if (denoiseScratch != null) {
            denoiseScratch.destroy();
            denoiseScratch = null;
        }
    }

    /**
     * A pose the run can compare against another frame's. Position to a millimetre of a block, both matrices
     * element-wise to a ten-thousandth: coarse enough that the jitter and a settled view bob stay one
     * viewpoint, fine enough that any turn the average would smear is a different one.
     */
    private static long poseKey(double camX, double camY, double camZ, Matrix4fc viewRotation,
                                Matrix4f projection) {
        long hash = 1125899906842597L;
        hash = mix(hash, Math.round(camX * 1000.0));
        hash = mix(hash, Math.round(camY * 1000.0));
        hash = mix(hash, Math.round(camZ * 1000.0));
        for (int column = 0; column < 4; column++) {
            for (int row = 0; row < 4; row++) {
                hash = mix(hash, Math.round(viewRotation.get(row, column) * 10000.0));
                hash = mix(hash, Math.round(projection.get(row, column) * 10000.0));
            }
        }
        return hash == 0L ? 1L : hash;
    }

    private static long mix(long hash, long value) {
        return 1099511628211L * (hash ^ value);
    }

    /** Reserve a name now, so a run that takes minutes cannot overwrite a screenshot taken meanwhile. */
    private static Path nextOutputPath() throws IOException {
        Path directory = Minecraft.getInstance().gameDirectory.toPath().resolve("screenshots");
        Files.createDirectories(directory);
        String base = Util.getFilenameFormattedDateTime();
        for (int count = 1; ; count++) {
            String suffix = count == 1 ? "" : "_" + count;
            Path candidate = directory.resolve(base + suffix + ".png");
            if (!Files.exists(candidate)) {
                return candidate;
            }
        }
    }

    private void failed(Throwable cause) {
        CausticaMod.LOGGER.warn("Still render failed", cause);
        failed();
    }

    private void failed() {
        overlay(Component.translatable("caustica.hotkey.still.failed"));
    }

    private static void overlay(Component message) {
        Minecraft client = Minecraft.getInstance();
        if (client.player != null) {
            client.player.sendOverlayMessage(message);
        }
    }

    private static void chat(Component message) {
        Minecraft client = Minecraft.getInstance();
        if (client.player != null) {
            client.player.sendSystemMessage(message);
        }
    }
}

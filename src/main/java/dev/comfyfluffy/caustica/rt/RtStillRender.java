package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.platform.NativeImage;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
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
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryBarrier;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The accumulated still render: one key press averages the frames the player is already looking at into a
 * single PNG, while the game keeps running, DLSS-RR keeps reconstructing and frame generation keeps
 * generating.
 *
 * <p>What is summed is the finished display image — the whole frame graph's output for the screen, including
 * bloom, the tone curve and sharpening — so the run removes only what varies between frames of one
 * viewpoint: where the jitter put each sample, and whatever noise the denoiser left. Nothing waits for the
 * scene to hold still, so animated water and clouds blur the way a long exposure blurs them.
 *
 * <p>A frame whose camera pose differs from the last one counted restarts the run: a mean of frames from two
 * viewpoints is neither of them. Poses are compared as quantised numbers, because the sub-pixel camera drift
 * of a held still key has to keep counting as the same viewpoint while a real turn of the head may not.
 *
 * <p>The run's two images — an fp32 accumulator and the resolved 8-bit frame — exist only while it is running,
 * at sixteen and four bytes per display texel.
 */
public final class RtStillRender {
    public static final RtStillRender INSTANCE = new RtStillRender();

    /** Progress reaches the player four times per run, not once per frame. */
    private static final int PROGRESS_STEPS = 4;
    private static final int BYTES_PER_TEXEL = 4;

    private RtStillPipeline pipeline;
    private RtImage accumulator;
    private RtImage resolved;
    private boolean accumulating;
    private boolean resolvePending;
    private int samples;
    private int target;
    private int reportedStep;
    /** Zero means "no frame counted yet", which is what makes the first frame a store instead of an add. */
    private long poseKey;

    private RtStillRender() {
    }

    /** The hotkey: start a run, or drop the one in flight along with anything not yet saved. */
    public void toggle() {
        if (accumulating || resolvePending) {
            abort(RtContext.currentOrNull(), "dropped by a second key press");
            return;
        }
        start();
    }

    /**
     * Resolve and save a run that has all its samples. Called from the client tick: the last accumulate
     * dispatch went out with its frame's submission, so the readback has to wait for a point that is past
     * that and is not inside a frame — and a tick still runs after the world is gone, which is when a
     * finished run must still reach the disk.
     */
    public void service() {
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
        if (accumulating || resolvePending) {
            abort(RtContext.currentOrNull(), reason);
        }
    }

    /**
     * Sum this frame's finished display image into the run. Records into the frame's own command buffer, so
     * the sample is the image this frame presented rather than a copy of it.
     */
    public void record(RtContext ctx, VkCommandBuffer cmd, RtImage frame, double camX, double camY, double camZ,
                       Matrix4fc viewRotation, Matrix4f projection) {
        if (!accumulating) {
            return;
        }
        if (frame.width != accumulator.width || frame.height != accumulator.height) {
            abort(ctx, "the display resolution changed from " + accumulator.width + "x" + accumulator.height
                    + " to " + frame.width + "x" + frame.height);
            return;
        }
        long pose = poseKey(camX, camY, camZ, viewRotation, projection);
        boolean first = pose != poseKey;
        poseKey = pose;
        if (first) {
            samples = 0;
            reportedStep = 0;
        }
        pipeline.setImages(accumulator.view, frame.view, resolved.view);
        pipeline.dispatch(cmd, accumulator.width, accumulator.height,
                first ? RtStillPipeline.MODE_FIRST : RtStillPipeline.MODE_ADD, 0.0f);
        samples++;
        if (samples >= target) {
            accumulating = false;
            resolvePending = true;
            return;
        }
        int step = samples * PROGRESS_STEPS / target;
        if (step > reportedStep) {
            reportedStep = step;
            overlay(Component.translatable("caustica.hotkey.still.progress",
                    Component.literal(Integer.toString(samples)), Component.literal(Integer.toString(target))));
        }
    }

    /** Release the run's GPU objects with no player-facing message; the device is being taken down. */
    public void destroy() {
        accumulating = false;
        resolvePending = false;
        releaseImages();
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
    }

    private void start() {
        Minecraft client = Minecraft.getInstance();
        RtContext ctx = RtContext.currentOrNull();
        int width = RtComposite.INSTANCE.displayWidth();
        int height = RtComposite.INSTANCE.displayHeight();
        if (ctx == null || client.level == null || width <= 0 || height <= 0 || RtComposite.INSTANCE.hasFailed()) {
            overlay(Component.translatable("caustica.hotkey.still.unavailable"));
            return;
        }
        target = Math.max(1, CausticaConfig.Rt.Still.SAMPLES.value());
        // Drained before the new images' init barriers: a run starts between frames, and its first sample is
        // the next frame's, so this is the only moment nothing can be in flight.
        ctx.waitIdle();
        try {
            ctx.beginImageInitBatch();
            try {
                if (pipeline == null) {
                    pipeline = RtStillPipeline.create(ctx);
                }
                accumulator = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R32G32B32A32_SFLOAT,
                        "still accumulator " + width + "x" + height);
                // Rows in PNG order rather than the frame graph's: the resolve pass stores the mean flipped,
                // so the linear readback below is already the right way up.
                resolved = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R8G8B8A8_UNORM,
                        "still resolved " + width + "x" + height);
            } finally {
                ctx.endImageInitBatch();
            }
        } catch (Throwable t) {
            releaseImages();
            failed(t);
            return;
        }
        accumulating = true;
        samples = 0;
        reportedStep = 0;
        poseKey = 0L;
        overlay(Component.translatable("caustica.hotkey.still.started",
                Component.literal(Integer.toString(target))));
    }

    private void resolveAndSave(RtContext ctx) {
        resolvePending = false;
        int width = accumulator.width;
        int height = accumulator.height;
        long bytes = Math.multiplyExact(Math.multiplyExact((long) width, height), (long) BYTES_PER_TEXEL);
        Path output;
        try {
            output = nextOutputPath();
        } catch (IOException e) {
            abort(ctx, "the screenshots directory could not be prepared: " + e);
            return;
        }
        // Every accumulate dispatch of the run went out with its own frame's submission, so this is where
        // the last one of them lands — and where nothing else can still be reading the two images.
        ctx.waitIdle();
        RtBuffer readback = ctx.createReadbackBuffer(bytes, "still render readback");
        NativeImage image;
        try {
            pipeline.setImages(accumulator.view, accumulator.view, resolved.view);
            ctx.submitSync(cmd -> recordResolveAndReadback(ctx, cmd, readback, width, height));
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

    private void recordResolveAndReadback(RtContext ctx, VkCommandBuffer cmd, RtBuffer readback,
                                          int width, int height) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still render resolve")) {
            // The accumulator was last written inside earlier command buffers. The device is idle here, but
            // the dependency still has to be expressed for this submit to read those writes.
            transition(ctx, cmd, stack, accumulator.image, VK10.VK_ACCESS_SHADER_WRITE_BIT,
                    VK10.VK_ACCESS_SHADER_READ_BIT, VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            pipeline.dispatch(cmd, width, height, RtStillPipeline.MODE_RESOLVE, 1.0f / samples);
            transition(ctx, cmd, stack, resolved.image, VK10.VK_ACCESS_SHADER_WRITE_BIT,
                    VK10.VK_ACCESS_TRANSFER_READ_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
        // has cleared both flags.
        if (accumulator == null && !accumulating && !resolvePending) {
            return;
        }
        accumulating = false;
        resolvePending = false;
        if (ctx != null) {
            // The run's images are written by dispatches already submitted with their frames, so this drain
            // is what makes them safe to destroy — the same one the resize path pays.
            ctx.waitIdle();
        }
        releaseImages();
        CausticaMod.LOGGER.info("Still render stopped: {}", reason);
        overlay(Component.translatable("caustica.hotkey.still.cancelled"));
    }

    private void releaseImages() {
        if (accumulator != null) {
            accumulator.destroy();
            accumulator = null;
        }
        if (resolved != null) {
            resolved.destroy();
            resolved = null;
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

package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkImageCreateInfo;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkImageViewCreateInfo;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

/**
 * The procedural cloud stack's shape field (volume.slang's {@code cloudNoise}): one tileable
 * 512-square RGBA8 noise atlas with a full mip chain.
 *
 * <p>The atlas is generated here rather than shipped as an asset, for two reasons. It is cheap — a
 * few octaves of bilinearly upsampled value noise per channel, tens of milliseconds once per device
 * bring-up — and, more importantly, it keeps the field reproducible: the renderer owns both the
 * texture and the code that reads it, so a change to either cannot leave the two describing
 * different noise.
 *
 * <p>Each channel is an independent octave stack with its own seed, tiling at every lattice period
 * the stack is summed from (4, 8, 16, 32, 64, 128), so a sample at any tile scale wraps without a
 * seam. The channels are consumed by role: red is the fine detail the cumulus and cirrus shapes
 * threshold, green and blue are the two vertical corners of the erosion cell field that
 * volume.slang's {@code cloudCellNoise} reads through a fixed row offset, and alpha is spare. The
 * sampling itself is wrap-addressed and mip-filtered, so the shader can trade sharpness for
 * stability with distance without ever sampling outside the tile.
 */
public final class RtCloudNoise {
    public static final int RES = 512;
    /** log2(RES) + 1: every level down to the 1x1 average. */
    public static final int MIPS = 10;

    private final VkDevice vk;
    private final long vma;
    private final long image;
    private final long allocation;
    private final long view;
    private final long sampler;
    private boolean destroyed;

    private RtCloudNoise(VkDevice vk, long vma, long image, long allocation, long view, long sampler) {
        this.vk = vk;
        this.vma = vma;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.sampler = sampler;
    }

    public long view() {
        return view;
    }

    public long sampler() {
        return sampler;
    }

    public static RtCloudNoise create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        long vma = ctx.vma();
        long createdImage = 0L;
        long createdAllocation = 0L;
        long createdView = 0L;
        long createdSampler = 0L;
        RtBuffer staging = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK10.VK_IMAGE_TYPE_2D).format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .mipLevels(MIPS).arrayLayers(1).samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_SAMPLED_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            imageInfo.extent().set(RES, RES, 1);
            VmaAllocationCreateInfo allocationInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            LongBuffer imageOut = stack.mallocLong(1);
            PointerBuffer allocationOut = stack.mallocPointer(1);
            RtContext.check(Vma.vmaCreateImage(vma, imageInfo, allocationInfo, imageOut, allocationOut, null),
                    "vmaCreateImage(cloud noise)");
            createdImage = imageOut.get(0);
            createdAllocation = allocationOut.get(0);
            RtDebugLabels.nameImage(ctx, createdImage, "cloud noise atlas");

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(createdImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_2D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM);
            viewInfo.subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(MIPS).baseArrayLayer(0).layerCount(1);
            LongBuffer viewOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateImageView(vk, viewInfo, null, viewOut),
                    "vkCreateImageView(cloud noise)");
            createdView = viewOut.get(0);
            RtDebugLabels.nameImageView(ctx, createdView, "cloud noise atlas view");

            // Wrap addressing is what makes the tile repeat; the mip chain is what keeps a deck at
            // the far end of the fade band from crawling as its cells shrink below a pixel.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .anisotropyEnable(false)
                    .minLod(0f).maxLod(MIPS - 1);
            LongBuffer samplerOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSampler(vk, samplerInfo, null, samplerOut),
                    "vkCreateSampler(cloud noise)");
            createdSampler = samplerOut.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, createdSampler, "cloud noise sampler");

            long[] offsets = new long[MIPS];
            int totalBytes = 0;
            for (int mip = 0; mip < MIPS; mip++) {
                offsets[mip] = totalBytes;
                int size = RES >> mip;
                totalBytes += size * size * 4;
            }
            staging = ctx.createUploadBuffer(totalBytes, "cloud noise upload");
            ByteBuffer mapped = MemoryUtil.memByteBuffer(staging.mapped, totalBytes);
            ByteBuffer prev = generate();
            try {
                mapped.put(prev);
                for (int mip = 1; mip < MIPS; mip++) {
                    ByteBuffer down = downsample(prev, RES >> (mip - 1));
                    MemoryUtil.memFree(prev);
                    prev = down;
                    mapped.put(prev);
                }
            } finally {
                MemoryUtil.memFree(prev);
            }
            mapped.position(0).limit(totalBytes);
            staging.flush();

            long uploadImage = createdImage;
            long uploadBuffer = staging.handle;
            ctx.submitSync(cmd -> {
                try (MemoryStack uploadStack = MemoryStack.stackPush()) {
                    VkImageMemoryBarrier.Buffer toTransfer = VkImageMemoryBarrier.calloc(1, uploadStack);
                    toTransfer.get(0).sType$Default()
                            .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
                            .newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                            .srcAccessMask(0).dstAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                            .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED).image(uploadImage);
                    toTransfer.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(MIPS).baseArrayLayer(0).layerCount(1);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, toTransfer);

                    VkBufferImageCopy.Buffer copies = VkBufferImageCopy.calloc(MIPS, uploadStack);
                    for (int mip = 0; mip < MIPS; mip++) {
                        int size = RES >> mip;
                        copies.get(mip).bufferOffset(offsets[mip])
                                .bufferRowLength(0).bufferImageHeight(0);
                        copies.get(mip).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                                .mipLevel(mip).baseArrayLayer(0).layerCount(1);
                        copies.get(mip).imageOffset().set(0, 0, 0);
                        copies.get(mip).imageExtent().set(size, size, 1);
                    }
                    VK10.vkCmdCopyBufferToImage(cmd, uploadBuffer, uploadImage,
                            VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copies);

                    // GENERAL, not SHADER_READ_ONLY_OPTIMAL, to match every other sampled image in
                    // this codebase (see RtToneLut) — the descriptor write must use the same layout
                    // or validation flags a mismatch.
                    VkImageMemoryBarrier.Buffer toRead = VkImageMemoryBarrier.calloc(1, uploadStack);
                    toRead.get(0).sType$Default()
                            .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                            .newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                            .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT)
                            .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                            .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED).image(uploadImage);
                    toRead.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(MIPS).baseArrayLayer(0).layerCount(1);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, null, null, toRead);
                }
            });
        } catch (Throwable t) {
            if (createdSampler != 0L) VK10.vkDestroySampler(vk, createdSampler, null);
            if (createdView != 0L) VK10.vkDestroyImageView(vk, createdView, null);
            if (createdImage != 0L) Vma.vmaDestroyImage(vma, createdImage, createdAllocation);
            throw t;
        } finally {
            if (staging != null) staging.destroy();
        }
        return new RtCloudNoise(vk, vma, createdImage, createdAllocation, createdView, createdSampler);
    }

    /** Octave periods, in texels of the 512 level: every one divides {@link #RES}, so all tile. */
    private static final int[] PERIODS = {4, 8, 16, 32, 64, 128};
    /** Per-channel seeds: independent fields, which is what lets the shader use each by role. */
    private static final int[] SEEDS = {101, 211, 307, 401};
    /** Around-0.5 gain, so a channel spans the range the coverage thresholds assume it does. */
    private static final float GAIN = 1.45f;

    /**
     * Builds mip 0: four independent octave stacks, one per channel. Each octave is a small random
     * lattice upsampled with a smooth bilinear weight, which is both cheaper and smoother than
     * evaluating a hash per texel — the lattice is generated once per octave and read sixteen texels
     * at a time.
     */
    private static ByteBuffer generate() {
        ByteBuffer out = MemoryUtil.memAlloc(RES * RES * 4);
        float[] lattice = new float[PERIODS[PERIODS.length - 1] * PERIODS[PERIODS.length - 1]];
        float[] sum = new float[RES * RES * 4];
        float norm = 0f;
        float amp = 1f;
        for (int octave = 0; octave < PERIODS.length; octave++) {
            int period = PERIODS[octave];
            float step = (float) period / RES;
            for (int channel = 0; channel < 4; channel++) {
                fillLattice(lattice, period, SEEDS[channel] + octave * 977);
                int seedOffset = channel;
                for (int y = 0; y < RES; y++) {
                    float fy = (y + 0.5f) * step;
                    int y0 = (int) fy;
                    float ty = fy - y0;
                    ty = ty * ty * (3f - 2f * ty);
                    int y1 = (y0 + 1) % period;
                    y0 %= period;
                    int row0 = y0 * period;
                    int row1 = y1 * period;
                    int dst = (y * RES) * 4 + seedOffset;
                    for (int x = 0; x < RES; x++) {
                        float fx = (x + 0.5f) * step;
                        int x0 = (int) fx;
                        float tx = fx - x0;
                        tx = tx * tx * (3f - 2f * tx);
                        int x1 = (x0 + 1) % period;
                        x0 %= period;
                        float v0 = lattice[row0 + x0] + (lattice[row0 + x1] - lattice[row0 + x0]) * tx;
                        float v1 = lattice[row1 + x0] + (lattice[row1 + x1] - lattice[row1 + x0]) * tx;
                        sum[dst] += amp * (v0 + (v1 - v0) * ty);
                        dst += 4;
                    }
                }
            }
            norm += amp;
            amp *= 0.5f;
        }
        for (int i = 0; i < RES * RES; i++) {
            for (int channel = 0; channel < 4; channel++) {
                float v = sum[i * 4 + channel] / norm;
                v = 0.5f + (v - 0.5f) * GAIN;
                int q = Math.round(Math.min(1f, Math.max(0f, v)) * 255f);
                out.put(i * 4 + channel, (byte) q);
            }
        }
        out.position(0).limit(RES * RES * 4);
        return out;
    }

    private static void fillLattice(float[] target, int period, int seed) {
        int cells = period * period;
        for (int i = 0; i < cells; i++) {
            target[i] = hash01(i % period, i / period, seed);
        }
    }

    /** Deterministic 0..1 hash: an integer avalanche, so the field is identical on every machine. */
    private static float hash01(int x, int y, int seed) {
        int h = x * 374761393 + y * 668265263 + seed * 2147483;
        h = (h ^ (h >>> 13)) * 1274126177;
        h ^= h >>> 16;
        return (h >>> 8) * (1.0f / 16777216.0f);
    }

    /** Box-filtered half-resolution copy, wrapping at the edges so the tile stays seamless. */
    private static ByteBuffer downsample(ByteBuffer src, int size) {
        int half = size / 2;
        ByteBuffer out = MemoryUtil.memAlloc(half * half * 4);
        for (int y = 0; y < half; y++) {
            int y0 = (y * 2) % size;
            int y1 = (y * 2 + 1) % size;
            for (int x = 0; x < half; x++) {
                int x0 = (x * 2) % size;
                int x1 = (x * 2 + 1) % size;
                int t00 = (y0 * size + x0) * 4;
                int t10 = (y0 * size + x1) * 4;
                int t01 = (y1 * size + x0) * 4;
                int t11 = (y1 * size + x1) * 4;
                for (int c = 0; c < 4; c++) {
                    int sum = (src.get(t00 + c) & 0xFF) + (src.get(t10 + c) & 0xFF)
                            + (src.get(t01 + c) & 0xFF) + (src.get(t11 + c) & 0xFF);
                    out.put((y * half + x) * 4 + c, (byte) ((sum + 2) >> 2));
                }
            }
        }
        out.position(0).limit(half * half * 4);
        return out;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyImageView(vk, view, null);
        Vma.vmaDestroyImage(vma, image, allocation);
        destroyed = true;
    }
}

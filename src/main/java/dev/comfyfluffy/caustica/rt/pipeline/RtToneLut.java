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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;

/**
 * A baked ACES color-pipeline 3D LUT (scene-referred look or display transform; see
 * {@code tools/bake_display_lut.py}). RGBA16F, one mip, loaded whole from a classpath
 * resource and uploaded once via a staging buffer — same shape as {@code RtMaterialPageTexture}
 * but 3D and self-describing (the resource carries its own size + shaper range in a small header,
 * see {@link #load}).
 */
public final class RtToneLut {
    private static final int MAGIC = 0x54554C43; // "CLUT" little-endian
    private static final int HEADER_BYTES = 4 + 4 + 4 + 4 + 4; // magic, version, size, loStops, hiStops
    // Display shader contract. Reject incompatible resources at load time instead of silently sampling
    // them with shaders/pipelines/display/main.comp.slang's fixed shaper.
    private static final float SHADER_SHAPER_LO_STOPS = -12.0f;
    private static final float SHADER_SHAPER_HI_STOPS = 12.0f;

    private final VkDevice vk;
    private final long vma;
    private final long image;
    private final long allocation;
    private final long view;
    private final long sampler;
    public final int size;
    private boolean destroyed;

    private RtToneLut(VkDevice vk, long vma, long image, long allocation, long view, long sampler,
                       int size) {
        this.vk = vk;
        this.vma = vma;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.sampler = sampler;
        this.size = size;
    }

    public long view() {
        return view;
    }

    public long sampler() {
        return sampler;
    }

    /** Loads a display-transform resource from {@code /bettercaustica/color/luts/}. */
    public static RtToneLut load(RtContext ctx, String resourceName) {
        return loadResource(ctx, "/bettercaustica/color/luts/" + resourceName);
    }

    /** Loads an absolute classpath LUT resource, including an LMT owned by a look package. */
    public static RtToneLut loadResource(RtContext ctx, String path) {
        if (path == null || !path.startsWith("/")) {
            throw new IllegalArgumentException("LUT resource path must be absolute: " + path);
        }
        ByteBuffer data = readResource(path);
        try {
            data.order(ByteOrder.LITTLE_ENDIAN);
            int magic = data.getInt(0);
            if (magic != MAGIC) {
                throw new IllegalStateException(path + ": bad magic 0x" + Integer.toHexString(magic));
            }
            int version = data.getInt(4);
            if (version != 1) {
                throw new IllegalStateException(path + ": unsupported version " + version);
            }
            int size = data.getInt(8);
            float loStops = data.getFloat(12);
            float hiStops = data.getFloat(16);
            if (size < 2) {
                throw new IllegalStateException(path + ": invalid LUT size " + size);
            }
            if (loStops != SHADER_SHAPER_LO_STOPS || hiStops != SHADER_SHAPER_HI_STOPS) {
                throw new IllegalStateException(path + ": LUT shaper " + loStops + ".." + hiStops
                        + " does not match display shader " + SHADER_SHAPER_LO_STOPS + ".."
                        + SHADER_SHAPER_HI_STOPS);
            }
            long texelCount = (long) size * size * size;
            long expectedBytes = HEADER_BYTES + texelCount * 4L * 2L; // RGBA16F
            if (data.remaining() != expectedBytes) {
                throw new IllegalStateException(path + ": expected " + expectedBytes + " bytes, got "
                        + data.remaining() + " (size=" + size + ")");
            }
            ByteBuffer texels = data.slice(HEADER_BYTES, (int) (expectedBytes - HEADER_BYTES));
            return upload(ctx, size, texels, path);
        } finally {
            MemoryUtil.memFree(data);
        }
    }

    private static RtToneLut upload(RtContext ctx, int size, ByteBuffer texels, String label) {
        VkDevice vk = ctx.vk();
        long vma = ctx.vma();
        long createdImage = 0L;
        long createdAllocation = 0L;
        long createdView = 0L;
        long createdSampler = 0L;
        RtBuffer staging = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK10.VK_IMAGE_TYPE_3D).format(VK10.VK_FORMAT_R16G16B16A16_SFLOAT)
                    .mipLevels(1).arrayLayers(1).samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_SAMPLED_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            imageInfo.extent().set(size, size, size);
            VmaAllocationCreateInfo allocationInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            LongBuffer imageOut = stack.mallocLong(1);
            PointerBuffer allocationOut = stack.mallocPointer(1);
            RtContext.check(Vma.vmaCreateImage(vma, imageInfo, allocationInfo, imageOut, allocationOut, null),
                    "vmaCreateImage(tone lut " + label + ")");
            createdImage = imageOut.get(0);
            createdAllocation = allocationOut.get(0);
            RtDebugLabels.nameImage(ctx, createdImage, "tone LUT " + label);

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(createdImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_3D)
                    .format(VK10.VK_FORMAT_R16G16B16A16_SFLOAT);
            viewInfo.subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            LongBuffer viewOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateImageView(vk, viewInfo, null, viewOut),
                    "vkCreateImageView(tone lut " + label + ")");
            createdView = viewOut.get(0);
            RtDebugLabels.nameImageView(ctx, createdView, "tone LUT " + label + " view");

            // Edge-aligned LUT: the shaper's [0,1] domain maps texel 0's centre to input 0 and texel
            // (size-1)'s centre to input 1 (see tools/bake_display_lut.py). CLAMP_TO_EDGE holds the
            // boundary texel for any exposed value outside the shaper's ±stops range instead of
            // wrapping or reading black.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0f).maxLod(0f);
            LongBuffer samplerOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSampler(vk, samplerInfo, null, samplerOut),
                    "vkCreateSampler(tone lut " + label + ")");
            createdSampler = samplerOut.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, createdSampler, "tone LUT " + label + " sampler");

            int totalBytes = texels.remaining();
            staging = ctx.createUploadBuffer(totalBytes, "tone lut " + label + " upload");
            ByteBuffer mapped = MemoryUtil.memByteBuffer(staging.mapped, totalBytes);
            mapped.put(texels.duplicate());
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
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, toTransfer);

                    VkBufferImageCopy.Buffer copy = VkBufferImageCopy.calloc(1, uploadStack);
                    copy.get(0).bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
                    copy.get(0).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .mipLevel(0).baseArrayLayer(0).layerCount(1);
                    copy.get(0).imageOffset().set(0, 0, 0);
                    copy.get(0).imageExtent().set(size, size, size);
                    VK10.vkCmdCopyBufferToImage(cmd, uploadBuffer, uploadImage,
                            VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);

                    // GENERAL, not SHADER_READ_ONLY_OPTIMAL, to match every other sampled/storage
                    // image in this codebase (see RtMaterialPageTexture, RtContext.createStorageImage)
                    // — the descriptor write below must use the same layout or validation flags a
                    // mismatch.
                    VkImageMemoryBarrier.Buffer toRead = VkImageMemoryBarrier.calloc(1, uploadStack);
                    toRead.get(0).sType$Default()
                            .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                            .newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                            .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT)
                            .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                            .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED).image(uploadImage);
                    toRead.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
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
        return new RtToneLut(vk, vma, createdImage, createdAllocation, createdView, createdSampler,
                size);
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

    private static ByteBuffer readResource(String path) {
        try (InputStream in = RtToneLut.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("missing LUT resource: " + path);
            }
            byte[] bytes = in.readAllBytes();
            ByteBuffer buf = MemoryUtil.memAlloc(bytes.length);
            buf.put(bytes);
            buf.flip();
            return buf;
        } catch (IOException e) {
            throw new IllegalStateException("failed to read LUT resource: " + path, e);
        }
    }
}

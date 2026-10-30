package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaMod;
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
 * The baked cloud-density volumes the volumetric cloud layer raymarches (the clouds_unreal
 * scheme: Houdini-authored VDB clouds converted to per-cloud 3D density grids). All shapes
 * share one R8 image shaped as a 2D array — {@link #RES} layers per shape, one layer per
 * horizontal slice — so the world pipeline needs a single combined-image-sampler descriptor
 * for all of them, and volume.slang fetches a slice as layer {@code shape * RES + z}. The
 * shader interpolates between adjacent layers itself, because 2D-array filtering is nearest
 * along the layer axis.
 *
 * <p>Each {@code .vol} resource is self-describing: 4-byte magic "BVOL", u8 version, u8
 * reserved, u16 resolution, then {@code res^3} R8 voxels ordered x fastest, then y, then z —
 * z is the volume's VERTICAL axis: it becomes the array layer here, and volume.slang maps the
 * layer to height with layer 0 at the cloud's bottom, while the in-layer x/y are the instance
 * box's horizontal footprint. tools/bake_cloud_vols.py writes this format and validates the
 * layout against {@link #RESOURCES}. Resources are loaded from
 * {@code /assets/bettercaustica/textures/cloud/}; a missing or malformed one zero-fills its shape
 * rather than failing the device bring-up — an empty shape renders no cloud, which is the
 * correct degradation, and the descriptor stays valid either way.
 */
public final class RtCloudVolumes {
    public static final int SHAPE_COUNT = 9;
    public static final int RES = 128;
    private static final int HEADER_BYTES = 8;
    private static final int LAYER_BYTES = RES * RES;
    private static final int SHAPE_BYTES = LAYER_BYTES * RES;

    /**
     * Resource order IS the {@code caustica.rt.volume.cloudType} slider order: the slider's
     * 0..1 maps to {@code round(type * (SHAPE_COUNT - 1))}. The hero cumulus sits at the
     * default's index (0.6 → 5) and the storm deck last.
     */
    private static final String[] RESOURCES = {
            "cumulus_small.vol", "cumulus_towering.vol", "cumulus_heavy.vol",
            "cumulus_rising.vol", "cumulus_wispy.vol", "cumulus_mediocris.vol",
            "cauliflower.vol", "cauliflower_multi.vol", "storm.vol"};

    private final VkDevice vk;
    private final long vma;
    private final long image;
    private final long allocation;
    private final long view;
    private final long sampler;
    /**
     * Shapes this atlas actually holds: {@link #SHAPE_COUNT} clamped down to what the device's
     * {@code maxImageArrayLayers} can hold (the Vulkan spec minimum is 256 layers — two shapes). The
     * pushed shape index rides this, so a clamped device selects among fewer shapes instead of
     * failing bring-up.
     */
    private final int shapeCount;
    private boolean destroyed;

    private RtCloudVolumes(VkDevice vk, long vma, long image, long allocation, long view, long sampler,
                           int shapeCount) {
        this.vk = vk;
        this.vma = vma;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.sampler = sampler;
        this.shapeCount = shapeCount;
    }

    public long view() {
        return view;
    }

    public long sampler() {
        return sampler;
    }

    /** Shapes the atlas holds; may be below {@link #SHAPE_COUNT} on layer-limited devices. */
    public int shapeCount() {
        return shapeCount;
    }

    public static RtCloudVolumes create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        long vma = ctx.vma();
        // The spec only guarantees 256 array layers, so the full nine-shape atlas (1152 layers) is a
        // cap, not an assumption: the atlas shrinks to what the device can hold and the pushed shape
        // index clamps with it.
        int shapes = Math.max(1, Math.min(SHAPE_COUNT, ctx.maxImageArrayLayers() / RES));
        long createdImage = 0L;
        long createdAllocation = 0L;
        long createdView = 0L;
        long createdSampler = 0L;
        RtBuffer staging = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int layers = shapes * RES;
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK10.VK_IMAGE_TYPE_2D).format(VK10.VK_FORMAT_R8_UNORM)
                    .mipLevels(1).arrayLayers(layers).samples(VK10.VK_SAMPLE_COUNT_1_BIT)
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
                    "vmaCreateImage(cloud volumes)");
            createdImage = imageOut.get(0);
            createdAllocation = allocationOut.get(0);
            RtDebugLabels.nameImage(ctx, createdImage, "cloud volume atlas");

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(createdImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_2D_ARRAY)
                    .format(VK10.VK_FORMAT_R8_UNORM);
            viewInfo.subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(layers);
            LongBuffer viewOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateImageView(vk, viewInfo, null, viewOut),
                    "vkCreateImageView(cloud volumes)");
            createdView = viewOut.get(0);
            RtDebugLabels.nameImageView(ctx, createdView, "cloud volume atlas view");

            // Trilinear in-slice filtering; the shader lerps between adjacent layers by hand. Clamp
            // addressing: instance UVs are normalized box coordinates, never a repeat domain.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0f).maxLod(0f);
            LongBuffer samplerOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSampler(vk, samplerInfo, null, samplerOut),
                    "vkCreateSampler(cloud volumes)");
            createdSampler = samplerOut.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, createdSampler, "cloud volume atlas sampler");

            int totalBytes = shapes * SHAPE_BYTES;
            staging = ctx.createUploadBuffer(totalBytes, "cloud volume atlas upload");
            ByteBuffer mapped = MemoryUtil.memByteBuffer(staging.mapped, totalBytes);
            for (int shape = 0; shape < shapes; shape++) {
                ByteBuffer voxels = readShape(RESOURCES[shape]);
                try {
                    mapped.position(shape * SHAPE_BYTES);
                    mapped.put(voxels);
                } finally {
                    MemoryUtil.memFree(voxels);
                }
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
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(layers);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, toTransfer);

                    VkBufferImageCopy.Buffer copies = VkBufferImageCopy.calloc(shapes, uploadStack);
                    for (int shape = 0; shape < SHAPE_COUNT; shape++) {
                        copies.get(shape).bufferOffset(shape * SHAPE_BYTES)
                                .bufferRowLength(0).bufferImageHeight(0);
                        copies.get(shape).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                                .mipLevel(0).baseArrayLayer(shape * RES).layerCount(RES);
                        copies.get(shape).imageOffset().set(0, 0, 0);
                        copies.get(shape).imageExtent().set(RES, RES, 1);
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
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(layers);
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
        return new RtCloudVolumes(vk, vma, createdImage, createdAllocation, createdView, createdSampler,
                shapes);
    }

    /** Reads one shape into an exact {@link #SHAPE_BYTES} buffer, zero-filling on any problem. */
    private static ByteBuffer readShape(String resource) {
        ByteBuffer out = MemoryUtil.memAlloc(SHAPE_BYTES);
        try (InputStream in = RtCloudVolumes.class
                .getResourceAsStream("/assets/bettercaustica/textures/cloud/" + resource)) {
            if (in == null) {
                // Zero-filled: the shape renders no cloud, the descriptor stays valid. Loud, because
                // a packaging mistake would otherwise look exactly like an empty sky.
                CausticaMod.LOGGER.warn("missing cloud volume resource {} - that shape renders nothing",
                        resource);
                return out;
            }
            byte[] bytes = in.readAllBytes();
            if (bytes.length < HEADER_BYTES) {
                return out;
            }
            ByteBuffer header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            if (header.getInt(0) != 0x4C4F5642 /* "BVOL" */ || header.get(4) != 1
                    || Short.toUnsignedInt(header.getShort(6)) != RES
                    || bytes.length != HEADER_BYTES + SHAPE_BYTES) {
                CausticaMod.LOGGER.warn("malformed cloud volume resource {} - that shape renders nothing",
                        resource);
                return out;
            }
            out.put(bytes, HEADER_BYTES, SHAPE_BYTES);
            out.position(0).limit(SHAPE_BYTES);
            return out;
        } catch (IOException e) {
            return out;
        }
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

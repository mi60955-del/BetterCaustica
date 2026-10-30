package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.LensFlarePushData;
import dev.comfyfluffy.caustica.rt.gen.LensSourcePushData;
import dev.comfyfluffy.caustica.rt.pipeline.RtComputePipelineSupport.SingleSet;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The lens flare chain: the starburst, the aperture halos and the lens dirt, all rendered into one
 * scene-referred image the display mapper adds before the tone curve.
 *
 * <p>Two dispatches, both at the display size divided by a resolution divisor (an eighth by default, so 1/64
 * of the pixels), which is what makes a gather of a few hundred taps per texel affordable. {@code source.comp}
 * reduces the exposed image
 * to the field of light that overflows the lens; {@code main.comp} gathers its three artifacts out of that
 * field. The gather has to read the field's neighbourhood, so the field cannot be averaged inside it: that
 * would multiply the box average by the number of taps the gather spends.
 *
 * <p>The source texture and the flare texture are separate images because the gather reads the field and
 * writes the flare, and because the display mapper only needs the latter. Both are sampled with a
 * black-border sampler: a tap that leaves the frame reads no light, not the frame's edge, which would drag a
 * source at the border into a streak along it.
 */
public final class RtLensPipeline {
    private static final String SOURCE_SHADER = "/bettercaustica/shaders/pipelines/lens/source.comp.spv";
    private static final String FLARE_SHADER = "/bettercaustica/shaders/pipelines/lens/main.comp.spv";

    private final RtContext ctx;
    private final SingleSet sourcePipeline;
    private final SingleSet flarePipeline;
    private final long edgeSampler;
    private final long borderSampler;
    private long boundRtView;
    private long boundExposureView;
    private long boundHotView;
    private long boundFlareView;
    private long boundGlowView;
    private boolean destroyed;

    private RtLensPipeline(RtContext ctx, SingleSet sourcePipeline, SingleSet flarePipeline,
                           long edgeSampler, long borderSampler) {
        this.ctx = ctx;
        this.sourcePipeline = sourcePipeline;
        this.flarePipeline = flarePipeline;
        this.edgeSampler = edgeSampler;
        this.borderSampler = borderSampler;
    }

    public static RtLensPipeline create(RtContext ctx) {
        SingleSet sourcePipeline = createSourcePipeline(ctx);
        SingleSet flarePipeline = createFlarePipeline(ctx);
        long edgeSampler = createSampler(ctx, VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK10.VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, "lens edge sampler");
        long borderSampler = createSampler(ctx, VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                VK10.VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, "lens border sampler");
        return new RtLensPipeline(ctx, sourcePipeline, flarePipeline, edgeSampler, borderSampler);
    }

    private static SingleSet createSourcePipeline(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(LENS_SOURCE_BINDING_COUNT, stack);
            bindings.get(LENS_SOURCE_OUTPUT).binding(LENS_SOURCE_OUTPUT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(LENS_SOURCE_SOURCE).binding(LENS_SOURCE_SOURCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(LENS_SOURCE_EXPOSURE).binding(LENS_SOURCE_EXPOSURE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            return RtComputePipelineSupport.createSingleSetPipeline(ctx, stack, "lens source",
                    bindings, poolSizes, LensSourcePushData.BYTE_SIZE, SOURCE_SHADER);
        }
    }

    private static SingleSet createFlarePipeline(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(LENS_FLARE_BINDING_COUNT, stack);
            bindings.get(LENS_FLARE_OUTPUT).binding(LENS_FLARE_OUTPUT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(LENS_FLARE_HOT).binding(LENS_FLARE_HOT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(LENS_FLARE_GLOW).binding(LENS_FLARE_GLOW)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);
            return RtComputePipelineSupport.createSingleSetPipeline(ctx, stack, "lens flare",
                    bindings, poolSizes, LensFlarePushData.BYTE_SIZE, FLARE_SHADER);
        }
    }

    private static long createSampler(RtContext ctx, int addressMode, int borderColor, String label) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handle = stack.mallocLong(1);
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(addressMode)
                    .addressModeV(addressMode)
                    .addressModeW(addressMode)
                    .borderColor(borderColor)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle), "vkCreateSampler(" + label + ")");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, label);
            return sampler;
        }
    }

    /**
     * The sampler the display mapper uses for the flare image. Black border: the display pass samples
     * well inside the image, and this keeps the two passes on one sampler object.
     */
    public long sampler() {
        return borderSampler;
    }

    /**
     * Point both dispatches at this frame's images. {@code hot} is the field the source pass writes and the
     * flare pass reads; {@code glow} is the bloom pyramid's level 0, which is what lights the dust.
     */
    public void setImages(long rtView, long exposureView, RtImage hot, RtImage flare, long glowView) {
        if (boundRtView == rtView && boundExposureView == exposureView && boundHotView == hot.view
                && boundFlareView == flare.view && boundGlowView == glowView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(6, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(6, stack);
            int index = 0;

            images.get(index).imageView(hot.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(sourcePipeline.descriptorSet())
                    .dstBinding(LENS_SOURCE_OUTPUT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;
            images.get(index).imageView(rtView).sampler(edgeSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(sourcePipeline.descriptorSet())
                    .dstBinding(LENS_SOURCE_SOURCE).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;
            images.get(index).imageView(exposureView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(sourcePipeline.descriptorSet())
                    .dstBinding(LENS_SOURCE_EXPOSURE).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;

            images.get(index).imageView(flare.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(flarePipeline.descriptorSet())
                    .dstBinding(LENS_FLARE_OUTPUT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;
            images.get(index).imageView(hot.view).sampler(borderSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(flarePipeline.descriptorSet())
                    .dstBinding(LENS_FLARE_HOT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;
            images.get(index).imageView(glowView).sampler(borderSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(flarePipeline.descriptorSet())
                    .dstBinding(LENS_FLARE_GLOW).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
            index++;

            VkWriteDescriptorSet.Buffer used = VkWriteDescriptorSet.create(writes.address(), index);
            VK10.vkUpdateDescriptorSets(ctx.vk(), used, null);
        }
        boundRtView = rtView;
        boundExposureView = exposureView;
        boundHotView = hot.view;
        boundFlareView = flare.view;
        boundGlowView = glowView;
    }

    /** Reduce the exposed frame to the field of light that overflows the lens. */
    public void dispatchSource(VkCommandBuffer cmd, RtImage hot, LensSourcePushData pushData) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "lens source")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, sourcePipeline.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    sourcePipeline.pipelineLayout(), 0, stack.longs(sourcePipeline.descriptorSet()), null);
            ByteBuffer push = stack.malloc(LensSourcePushData.BYTE_SIZE);
            pushData.write(push);
            VK10.vkCmdPushConstants(cmd, sourcePipeline.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (hot.width + 7) / 8, (hot.height + 7) / 8, 1);
        }
    }

    /** Gather the three artifacts out of that field into the image the display mapper reads. */
    public void dispatchFlare(VkCommandBuffer cmd, RtImage flare, LensFlarePushData pushData) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "lens flare")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, flarePipeline.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    flarePipeline.pipelineLayout(), 0, stack.longs(flarePipeline.descriptorSet()), null);
            ByteBuffer push = stack.malloc(LensFlarePushData.BYTE_SIZE);
            pushData.write(push);
            VK10.vkCmdPushConstants(cmd, flarePipeline.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (flare.width + 7) / 8, (flare.height + 7) / 8, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        sourcePipeline.destroy(vk);
        flarePipeline.destroy(vk);
        VK10.vkDestroySampler(vk, edgeSampler, null);
        VK10.vkDestroySampler(vk, borderSampler, null);
        destroyed = true;
    }
}

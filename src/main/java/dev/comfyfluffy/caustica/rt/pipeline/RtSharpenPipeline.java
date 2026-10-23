package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.SharpenPushData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Adds back the detail DLSS-RR's upscale averages away. Runnable only after the display mapper: the pass
 * reads the unsharpened SDR and PQ HDR images that mapper wrote and writes the finished pair the present
 * paths consume, so sharpening happens in display code values, where the softness it undoes exists.
 *
 * <p>Both channels are bound and written by one dispatch. That keeps the pass a single step in the frame
 * graph and lets the HDR channel be skipped from the push constant while the descriptor stays legal, which
 * is what the 1x1 placeholder view is for.
 */
public final class RtSharpenPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/sharpen/";
    /** Push constants: the detail gain and whether the PQ HDR image is live this frame. */
    private static final int PUSH_BYTES = SharpenPushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    /**
     * Loads are unfiltered, so this sampler only has to satisfy the combined-image descriptor. CLAMP_TO_EDGE
     * matches how the pass clamps its own border taps, keeping a future filtered fetch consistent with the
     * unfiltered path.
     */
    private final long sampler;
    private long boundOutputView;
    private long boundHdrOutputView;
    private long boundSourceView;
    private long boundHdrSourceView;
    private boolean destroyed;

    private RtSharpenPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu, long sampler) {
        this.ctx = ctx;
        this.gpu = gpu;
        this.sampler = sampler;
    }

    public static RtSharpenPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(SHARPEN_BINDING_COUNT, stack);
            binds.get(SHARPEN_OUTPUT).binding(SHARPEN_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(SHARPEN_HDR_OUTPUT).binding(SHARPEN_HDR_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(SHARPEN_SOURCE).binding(SHARPEN_SOURCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(SHARPEN_HDR_SOURCE).binding(SHARPEN_HDR_SOURCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);

            RtComputePipelineSupport.SingleSet gpu = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "display sharpen", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv");

            LongBuffer handle = stack.mallocLong(1);
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_NEAREST)
                    .minFilter(VK10.VK_FILTER_NEAREST)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle),
                    "vkCreateSampler(display sharpen)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "display sharpen sampler");

            return new RtSharpenPipeline(ctx, gpu, sampler);
        }
    }

    /**
     * Binds the display mapper's unsharpened pair ({@code source*}, GENERAL layout, sampled) and the finished
     * pair ({@code output*}, storage). Callers pass the 1x1 placeholder in the HDR slots while HDR output is
     * off; the dispatch's flag then keeps that channel unwritten.
     */
    public void setImages(long outputImageView, long hdrOutputImageView, long sourceImageView, long hdrSourceImageView) {
        if (boundOutputView == outputImageView && boundHdrOutputView == hdrOutputImageView
                && boundSourceView == sourceImageView && boundHdrSourceView == hdrSourceImageView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer hdrOutputInfo = VkDescriptorImageInfo.calloc(1, stack);
            hdrOutputInfo.get(0).imageView(hdrOutputImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer sourceInfo = VkDescriptorImageInfo.calloc(1, stack);
            sourceInfo.get(0).imageView(sourceImageView).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer hdrSourceInfo = VkDescriptorImageInfo.calloc(1, stack);
            hdrSourceInfo.get(0).imageView(hdrSourceImageView).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(SHARPEN_BINDING_COUNT, stack);
            writes.get(SHARPEN_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(SHARPEN_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);
            writes.get(SHARPEN_HDR_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(SHARPEN_HDR_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(hdrOutputInfo);
            writes.get(SHARPEN_SOURCE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(SHARPEN_SOURCE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(sourceInfo);
            writes.get(SHARPEN_HDR_SOURCE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(SHARPEN_HDR_SOURCE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(hdrSourceInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutputView = outputImageView;
        boundHdrOutputView = hdrOutputImageView;
        boundSourceView = sourceImageView;
        boundHdrSourceView = hdrSourceImageView;
    }

    /**
     * Runs the 3x3 high pass over both channels. The dispatch is unconditional: {@code enabled} folds into
     * the push-constant gain, so a disabled pass still rewrites its outputs with an exact copy of its inputs
     * instead of leaving the finished images one frame stale. {@code hdrEnabled} must match which images are
     * bound in the HDR slots.
     */
    public void dispatch(VkCommandBuffer cmd, int width, int height, boolean enabled, float amount, boolean hdrEnabled) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "display sharpen")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0, stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            new SharpenPushData(hdrEnabled ? 1 : 0, enabled ? amount : 0.0f).write(push);
            VK10.vkCmdPushConstants(cmd, gpu.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        gpu.destroy(ctx.vk());
        VK10.vkDestroySampler(ctx.vk(), sampler, null);
        destroyed = true;
    }
}

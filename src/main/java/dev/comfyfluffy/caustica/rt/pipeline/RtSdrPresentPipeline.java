package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;

import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Compute pass that converts Minecraft's SDR main target (rgba8, sRGB-encoded, sampled) to a PQ-encoded
 * rgba16f image at paper white. Used by the HDR present path for non-RT frames (menus, loading screen, title
 * panorama) so they present correctly to the PQ swapchain instead of being raw-copied (misdisplayed). Same
 * descriptor shape as {@link RtHdrCompositePipeline}: binding 0 = storage out, binding 1 = sampled SDR in.
 */
public final class RtSdrPresentPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/sdr_present/";
    private static final int PUSH_BYTES = Float.BYTES; // float uiNits

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private long boundOutView;
    private long boundSdrView;
    private long boundSampler;
    private boolean destroyed;

    private RtSdrPresentPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu) {
        this.ctx = ctx;
        this.gpu = gpu;
    }

    public static RtSdrPresentPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(PRESENT_BINDING_COUNT, stack);
            binds.get(PRESENT_OUTPUT).binding(PRESENT_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(PRESENT_SOURCE).binding(PRESENT_SOURCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);

            return new RtSdrPresentPipeline(ctx, RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "sdr present", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv"));
        }
    }

    /** Bind the destination PQ image (storage) and the SDR source (combined image sampler, GENERAL layout). */
    public void setImages(long outImageView, long sdrImageView, long sampler) {
        if (boundOutView == outImageView && boundSdrView == sdrImageView && boundSampler == sampler) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outInfo = VkDescriptorImageInfo.calloc(1, stack);
            outInfo.get(0).imageView(outImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer sdrInfo = VkDescriptorImageInfo.calloc(1, stack);
            sdrInfo.get(0).imageView(sdrImageView).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(PRESENT_BINDING_COUNT, stack);
            writes.get(PRESENT_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(PRESENT_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outInfo);
            writes.get(PRESENT_SOURCE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(PRESENT_SOURCE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(sdrInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutView = outImageView;
        boundSdrView = sdrImageView;
        boundSampler = sampler;
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height, float uiNits) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "sdr present")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0, stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            push.putFloat(0, uiNits);
            VK10.vkCmdPushConstants(cmd, gpu.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        gpu.destroy(ctx.vk());
        destroyed = true;
    }
}

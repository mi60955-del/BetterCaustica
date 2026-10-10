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

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * Midpoint frame interpolation: reads the current and previous displayed frames plus the ray-traced motion
 * guide and writes the half-way exposure into an output image sized like the backbuffer, which the
 * multi-present engine then presents as the generated frame.
 *
 * <p>Unlike the DLSS and FSR paths this keeps no history of its own: the caller supplies the previous
 * frame, so the image pair and the guide must all describe the same frame boundary.
 */
public final class RtFgInterpPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/fg_interp/";
    private static final int PUSH_BYTES = Integer.BYTES; // int debugView

    private static final int BINDING_OUTPUT = 0;
    private static final int BINDING_CURRENT = 1;
    private static final int BINDING_PREVIOUS = 2;
    private static final int BINDING_MOTION = 3;
    private static final int BINDING_COUNT = 4;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private final long linearSampler;
    private long boundOutView;
    private long boundCurrentView;
    private long boundPreviousView;
    private long boundMotionView;
    private boolean destroyed;

    private RtFgInterpPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu, long linearSampler) {
        this.ctx = ctx;
        this.gpu = gpu;
        this.linearSampler = linearSampler;
    }

    public static RtFgInterpPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(BINDING_COUNT, stack);
            bind(binds.get(BINDING_OUTPUT), BINDING_OUTPUT, VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            bind(binds.get(BINDING_CURRENT), BINDING_CURRENT, VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
            bind(binds.get(BINDING_PREVIOUS), BINDING_PREVIOUS, VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
            bind(binds.get(BINDING_MOTION), BINDING_MOTION, VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);

            RtComputePipelineSupport.SingleSet gpu = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "fg interp", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv");

            // The previous frame is sampled at fractional uv, so it needs filtering; the shared UI sampler
            // is nearest and would turn every sub-pixel shift into a snap to the nearest texel.
            VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            java.nio.LongBuffer out = stack.mallocLong(1);
            check(VK10.vkCreateSampler(ctx.vk(), sci, null, out), "vkCreateSampler(fg interp)");
            long sampler = out.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "fg interp linear sampler");

            return new RtFgInterpPipeline(ctx, gpu, sampler);
        }
    }

    private static void bind(VkDescriptorSetLayoutBinding b, int index, int type) {
        b.binding(index).descriptorType(type).descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
    }

    /**
     * Bind the interpolated output, both displayed frames, and the motion guide. All are GENERAL layout.
     *
     * <p>The single descriptor set is referenced by submits still in flight ({@code execute()} does not
     * fence), so a real rewrite while a prior frame's dispatch may still be running is the classic
     * update-while-in-use hazard. View changes are resize-like rare events, so idling the device before
     * the rewrite is the cheap, obviously-correct guard; the steady state (same views every frame) never
     * reaches it.
     */
    public void setImages(long outImageView, long currentImageView, long previousImageView, long motionImageView) {
        if (boundOutView == outImageView && boundCurrentView == currentImageView
                && boundPreviousView == previousImageView && boundMotionView == motionImageView) {
            return;
        }
        ctx.waitIdle();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer currentInfo = VkDescriptorImageInfo.calloc(1, stack);
            currentInfo.get(0).imageView(currentImageView).sampler(linearSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer previousInfo = VkDescriptorImageInfo.calloc(1, stack);
            previousInfo.get(0).imageView(previousImageView).sampler(linearSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkDescriptorImageInfo.Buffer outInfo = VkDescriptorImageInfo.calloc(1, stack);
            outInfo.get(0).imageView(outImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer motionInfo = VkDescriptorImageInfo.calloc(1, stack);
            motionInfo.get(0).imageView(motionImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(BINDING_COUNT, stack);
            write(writes.get(BINDING_OUTPUT), gpu.descriptorSet(), BINDING_OUTPUT, VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, outInfo);
            write(writes.get(BINDING_CURRENT), gpu.descriptorSet(), BINDING_CURRENT, VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, currentInfo);
            write(writes.get(BINDING_PREVIOUS), gpu.descriptorSet(), BINDING_PREVIOUS, VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, previousInfo);
            write(writes.get(BINDING_MOTION), gpu.descriptorSet(), BINDING_MOTION, VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, motionInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutView = outImageView;
        boundCurrentView = currentImageView;
        boundPreviousView = previousImageView;
        boundMotionView = motionImageView;
    }

    private static void write(VkWriteDescriptorSet w, long set, int binding, int type,
            VkDescriptorImageInfo.Buffer info) {
        w.sType$Default().dstSet(set).dstBinding(binding).descriptorCount(1).descriptorType(type)
                .pImageInfo(info);
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height, int debugView) {
        try (MemoryStack stack = MemoryStack.stackPush();
                RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fg interp")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0,
                    stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            push.putInt(0, debugView);
            VK10.vkCmdPushConstants(cmd, gpu.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VK10.vkDestroySampler(ctx.vk(), linearSampler, null);
        gpu.destroy(ctx.vk());
        destroyed = true;
    }
}

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
import dev.comfyfluffy.caustica.rt.gen.StillPushData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The accumulated still render's one pass: it sums a scene-linear frame into an fp32 accumulator, and
 * divides that accumulator out into the display-res half-float mean the display chain renders.
 *
 * <p>Adding and resolving share a pipeline because the accumulator is the pass's only state, and two
 * pipelines would each have to describe it. In the resolve the source slot is bound to whatever the run
 * last accumulated even though the resolve does not read it: the descriptor layout needs a live view in
 * every slot, and that source is the one image guaranteed still there when the resolve is recorded.
 */
public final class RtStillPipeline {
    /** Add this frame to the accumulator. */
    public static final int MODE_ADD = 0;
    /** Store this frame as the run's first sample, which is what keeps the accumulator never-cleared. */
    public static final int MODE_FIRST = 1;
    /** Divide the accumulator by the sample count into the display-res half-float mean. */
    public static final int MODE_RESOLVE = 2;

    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/still/";
    private static final int PUSH_BYTES = StillPushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    /** Loads are unfiltered, so this only has to satisfy the combined-image descriptor. */
    private final long sampler;
    private long boundAccumulatorView;
    private long boundSourceView;
    private long boundOutputView;
    private boolean destroyed;

    private RtStillPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu, long sampler) {
        this.ctx = ctx;
        this.gpu = gpu;
        this.sampler = sampler;
    }

    public static RtStillPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(STILL_BINDING_COUNT, stack);
            binds.get(STILL_ACCUMULATOR).binding(STILL_ACCUMULATOR).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(STILL_SOURCE).binding(STILL_SOURCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(STILL_OUTPUT).binding(STILL_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);

            RtComputePipelineSupport.SingleSet gpu = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "still accumulate", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv");

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
                    "vkCreateSampler(still accumulate)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "still accumulate sampler");

            return new RtStillPipeline(ctx, gpu, sampler);
        }
    }

    public void setImages(long accumulatorImageView, long sourceImageView, long outputImageView) {
        if (boundAccumulatorView == accumulatorImageView && boundSourceView == sourceImageView
                && boundOutputView == outputImageView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer accumulatorInfo = VkDescriptorImageInfo.calloc(1, stack);
            accumulatorInfo.get(0).imageView(accumulatorImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer sourceInfo = VkDescriptorImageInfo.calloc(1, stack);
            sourceInfo.get(0).imageView(sourceImageView).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(STILL_BINDING_COUNT, stack);
            writes.get(STILL_ACCUMULATOR).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(STILL_ACCUMULATOR)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(accumulatorInfo);
            writes.get(STILL_SOURCE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(STILL_SOURCE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(sourceInfo);
            writes.get(STILL_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(STILL_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(outputInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundAccumulatorView = accumulatorImageView;
        boundSourceView = sourceImageView;
        boundOutputView = outputImageView;
    }

    /**
     * Runs one step over the whole accumulator. {@code weight} is {@code 1 / samples} and only the resolve
     * reads it, so the adding dispatches pass zero. {@code count} is how many samples the accumulator
     * already holds: the add dispatch divides its running mean by it, the resolve derives the standard error
     * of the mean from it, and the first sample of a run passes zero because it stores instead of adding.
     * {@code legacy} selects the earlier estimator — a cap against the running mean, and no second moment in
     * the alpha — for the whole run; it is latched when the run starts, since the two write different
     * things into the same image.
     */
    public void dispatch(VkCommandBuffer cmd, int width, int height, int mode, float weight, int count,
                         int legacy) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "still accumulate")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0,
                    stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            new StillPushData(mode, weight, count, legacy).write(push);
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

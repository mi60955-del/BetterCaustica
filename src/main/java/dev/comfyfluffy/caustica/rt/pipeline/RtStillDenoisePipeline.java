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
import dev.comfyfluffy.caustica.rt.gen.StillDenoisePushData;

import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Spatial filtering over scene-linear radiance, shared by the still render and the DLSS-RR input. Modes 0-2
 * demodulate the still run's accumulated mean by the surface albedo, run a-trous steps over the residual
 * with the world pass's normal/albedo/depth guides as edge stops, and remodulate. Mode 3 clamps isolated single-texel outliers in the one-sample trace before
 * RR reconstructs it.
 *
 * <p>One pipeline and one descriptor set serve every mode, so the run is a fixed sequence of the same pass
 * rather than several passes with several layouts. Every slot is a storage image loaded unfiltered, which is
 * what lets the kernel place its own taps instead of leaning on a sampler's borders. A consumer that runs
 * mode 3 on every frame owns its own pipeline instance: the descriptor set is not ringed, so sharing one
 * with a still run would let the two sequences overwrite each other's bindings.
 */
public final class RtStillDenoisePipeline {
    /** Divide the mean by the surface albedo, leaving the irradiance the surface receives. */
    public static final int MODE_DEMODULATE = 0;
    /** One a-trous step over the demodulated residual. */
    public static final int MODE_ATROUS = 1;
    /** Multiply the filtered residual by the albedo again. */
    public static final int MODE_REMODULATE = 2;
    /** Clamp isolated single-texel outliers in the one-sample frame handed to DLSS-RR. */
    public static final int MODE_FIREFLY_CLAMP = 3;

    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/still_denoise/";
    private static final int PUSH_BYTES = StillDenoisePushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private long boundSourceView;
    private long boundNormalView;
    private long boundAlbedoView;
    private long boundDepthView;
    private long boundOutputView;
    private boolean destroyed;

    private RtStillDenoisePipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu) {
        this.ctx = ctx;
        this.gpu = gpu;
    }

    public static RtStillDenoisePipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds =
                    VkDescriptorSetLayoutBinding.calloc(STILL_DENOISE_BINDING_COUNT, stack);
            int[] storageBindings = {
                    STILL_DENOISE_SOURCE, STILL_DENOISE_GUIDE_NORMAL, STILL_DENOISE_GUIDE_ALBEDO,
                    STILL_DENOISE_GUIDE_DEPTH, STILL_DENOISE_OUTPUT
            };
            for (int binding : storageBindings) {
                binds.get(binding).binding(binding).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(storageBindings.length);

            RtComputePipelineSupport.SingleSet gpu = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "still denoise", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv");
            return new RtStillDenoisePipeline(ctx, gpu);
        }
    }

    /**
     * Binds the image this step reads ({@code source}) and the one it writes ({@code output}), plus the
     * guides the run's frame wrote. The two image slots can be the mean and a scratch image in either
     * order; every image must already be in GENERAL layout, which is what the still render keeps its own.
     */
    public void setImages(long sourceImageView, long normalImageView, long albedoImageView, long depthImageView,
                          long outputImageView) {
        if (boundSourceView == sourceImageView && boundNormalView == normalImageView
                && boundAlbedoView == albedoImageView && boundDepthView == depthImageView
                && boundOutputView == outputImageView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int[] bindings = {
                    STILL_DENOISE_SOURCE, STILL_DENOISE_GUIDE_NORMAL, STILL_DENOISE_GUIDE_ALBEDO,
                    STILL_DENOISE_GUIDE_DEPTH, STILL_DENOISE_OUTPUT
            };
            long[] views = {sourceImageView, normalImageView, albedoImageView, depthImageView, outputImageView};
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(bindings.length, stack);
            for (int i = 0; i < bindings.length; i++) {
                VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
                info.get(0).imageView(views[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(i).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(bindings[i])
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(info);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundSourceView = sourceImageView;
        boundNormalView = normalImageView;
        boundAlbedoView = albedoImageView;
        boundDepthView = depthImageView;
        boundOutputView = outputImageView;
    }

    /**
     * Runs one step over the whole image. {@code step} is the a-trous spacing in texels and is read only by
     * the a-trous mode; {@code sigma} is the luminance spread a tap may differ by before it fades out, and
     * {@code amount} blends the filtered result over the input. {@code measured} says whether that spread
     * comes from the scene image's alpha (a still run of the current estimator: the tolerance is
     * {@code sigma} times that per-pixel noise) or is {@code sigma} times the local level on its own (a run
     * that accumulated with the earlier one, which records no noise). The firefly-clamp mode reads none of
     * them.
     */
    public void dispatch(VkCommandBuffer cmd, int width, int height, int mode, int step, float sigma,
                         float amount, int measured) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd,
                     mode == MODE_FIREFLY_CLAMP ? "rr input firefly clamp" : "still denoise")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0,
                    stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            new StillDenoisePushData(mode, step, sigma, amount, measured).write(push);
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

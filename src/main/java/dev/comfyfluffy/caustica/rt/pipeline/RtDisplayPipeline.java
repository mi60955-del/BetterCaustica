package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.DisplayPushData;

import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/** Maps the display-res scene-linear ACEScg RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR. */
public final class RtDisplayPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/display/";
    /** Push constants: output/look LUT state plus gamma and HDR peak nits. */
    private static final int PUSH_BYTES = DisplayPushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private long boundOutputView;
    private long boundRtView;
    private long boundExposureView;
    private long boundHdrView;
    private long boundLutView;
    private long boundLutSampler;
    private long boundHdrLutView;
    private long boundHdrLutSampler;
    private long boundLookLutView;
    private long boundLookLutSampler;
    private long boundBloomView;
    private long boundBloomSampler;
    private long boundExposureStateBuffer;
    private boolean destroyed;

    private RtDisplayPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu) {
        this.ctx = ctx;
        this.gpu = gpu;
    }

    public static RtDisplayPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(DISPLAY_BINDING_COUNT, stack);
            binds.get(DISPLAY_OUTPUT).binding(DISPLAY_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_RT_IMAGE).binding(DISPLAY_RT_IMAGE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_EXPOSURE).binding(DISPLAY_EXPOSURE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_HDR_OUTPUT).binding(DISPLAY_HDR_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            // Baked ACES 2.0 display-transform LUTs; see RtToneLut.
            binds.get(DISPLAY_SDR_TONE_LUT).binding(DISPLAY_SDR_TONE_LUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_HDR_TONE_LUT).binding(DISPLAY_HDR_TONE_LUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_LOOK_LUT).binding(DISPLAY_LOOK_LUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DISPLAY_BLOOM).binding(DISPLAY_BLOOM).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            // Auto-exposure state: the Purkinje shift keys off the metered absolute scene luminance,
            // which the applied exposure scalar has already normalised out of the image itself.
            binds.get(DISPLAY_EXPOSURE_STATE).binding(DISPLAY_EXPOSURE_STATE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(3, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(4);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(4);
            poolSizes.get(2).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);

            return new RtDisplayPipeline(ctx, RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "rt display", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv"));
        }
    }

    public void setImages(long outputImageView, long rtImageView, long exposureImageView, long hdrImageView,
                           long lutView, long lutSampler, long hdrLutView, long hdrLutSampler,
                           long lookLutView, long lookLutSampler, long bloomView, long bloomSampler,
                           RtBuffer exposureState) {
        if (boundOutputView == outputImageView && boundRtView == rtImageView
                && boundExposureView == exposureImageView && boundHdrView == hdrImageView
                && boundLutView == lutView && boundLutSampler == lutSampler
                && boundHdrLutView == hdrLutView && boundHdrLutSampler == hdrLutSampler
                && boundLookLutView == lookLutView && boundLookLutSampler == lookLutSampler
                && boundBloomView == bloomView && boundBloomSampler == bloomSampler
                && boundExposureStateBuffer == exposureState.handle) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer rtInfo = VkDescriptorImageInfo.calloc(1, stack);
            rtInfo.get(0).imageView(rtImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer exposureInfo = VkDescriptorImageInfo.calloc(1, stack);
            exposureInfo.get(0).imageView(exposureImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer hdrInfo = VkDescriptorImageInfo.calloc(1, stack);
            hdrInfo.get(0).imageView(hdrImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer lutInfo = VkDescriptorImageInfo.calloc(1, stack);
            lutInfo.get(0).imageView(lutView).sampler(lutSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer hdrLutInfo = VkDescriptorImageInfo.calloc(1, stack);
            hdrLutInfo.get(0).imageView(hdrLutView).sampler(hdrLutSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer lookLutInfo = VkDescriptorImageInfo.calloc(1, stack);
            lookLutInfo.get(0).imageView(lookLutView).sampler(lookLutSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer bloomInfo = VkDescriptorImageInfo.calloc(1, stack);
            bloomInfo.get(0).imageView(bloomView).sampler(bloomSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorBufferInfo.Buffer stateInfo = VkDescriptorBufferInfo.calloc(1, stack);
            stateInfo.get(0).buffer(exposureState.handle).offset(0).range(exposureState.size);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(DISPLAY_BINDING_COUNT, stack);
            writes.get(DISPLAY_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);
            writes.get(DISPLAY_RT_IMAGE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_RT_IMAGE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(rtInfo);
            writes.get(DISPLAY_EXPOSURE).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_EXPOSURE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(exposureInfo);
            writes.get(DISPLAY_HDR_OUTPUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_HDR_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(hdrInfo);
            writes.get(DISPLAY_SDR_TONE_LUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_SDR_TONE_LUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(lutInfo);
            writes.get(DISPLAY_HDR_TONE_LUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_HDR_TONE_LUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(hdrLutInfo);
            writes.get(DISPLAY_LOOK_LUT).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_LOOK_LUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(lookLutInfo);
            writes.get(DISPLAY_BLOOM).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DISPLAY_BLOOM)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(bloomInfo);
            writes.get(DISPLAY_EXPOSURE_STATE).sType$Default().dstSet(gpu.descriptorSet())
                    .dstBinding(DISPLAY_EXPOSURE_STATE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(stateInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutputView = outputImageView;
        boundRtView = rtImageView;
        boundExposureView = exposureImageView;
        boundHdrView = hdrImageView;
        boundLutView = lutView;
        boundLutSampler = lutSampler;
        boundHdrLutView = hdrLutView;
        boundHdrLutSampler = hdrLutSampler;
        boundLookLutView = lookLutView;
        boundLookLutSampler = lookLutSampler;
        boundBloomView = bloomView;
        boundBloomSampler = bloomSampler;
        boundExposureStateBuffer = exposureState.handle;
    }

    /**
     * Run the display mapping. {@code advancedGrading} selects the SDR chain: false applies the look LMT
     * ({@code lookEnabled}) and samples the baked ACES 2.0 LUT, true runs AgX with {@code agxGain} and
     * {@code agxSaturation}. The PQ HDR image always comes from the HDR LUT, which is baked for a fixed
     * mastering-nits peak (see {@code CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS}) and selected host-side by
     * which LUT resource is bound. SDR (binding 0) always writes; the HDR image (binding 3) also writes
     * when {@code hdrEnabled}.
     *
     * <p>{@code purkinjeEnabled} re-weights the exposed scene-linear image by the CIE mesopic response
     * before either display path, and {@code purkinjeStrength} scales that deviation (1 is the CIE
     * response itself, 0 leaves the image untouched). It reads the metered absolute scene luminance
     * from the bound exposure state, so it keys on adaptation rather than on how bright the frame was
     * exposed to look.
     */
    public void dispatch(VkCommandBuffer cmd, int width, int height, boolean hdrEnabled, int lutSize,
                         float gamma, float hdrPeakNits, boolean lookEnabled, int lookLutSize,
                         float bloomStrength, boolean advancedGrading, float agxGain, float agxSaturation,
                         boolean purkinjeEnabled, float purkinjeStrength) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "display compute")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0, stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(DisplayPushData.BYTE_SIZE);
            new DisplayPushData(hdrEnabled ? 1 : 0, (float) lutSize, gamma, hdrPeakNits,
                    lookEnabled ? 1 : 0, (float) lookLutSize, bloomStrength,
                    advancedGrading ? 1 : 0, agxGain, agxSaturation,
                    purkinjeEnabled ? 1 : 0, purkinjeStrength).write(push);
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

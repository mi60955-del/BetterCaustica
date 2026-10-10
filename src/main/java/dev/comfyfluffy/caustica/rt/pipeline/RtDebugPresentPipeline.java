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
import dev.comfyfluffy.caustica.rt.gen.DebugPresentPushData;

import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Computes and presents {@code debugView} content as a downstream inspection pass after
 * {@link RtDisplayPipeline}, separate from the primary raygen. It nearest-samples the real
 * render-resolution guide buffers while RR remains enabled, keeping debug-only register pressure out
 * of the primary raygen. Diagnostic output is written after exposure and ACES, so literal colors remain literal
 * without perturbing the exposure controller's history.
 */
public final class RtDebugPresentPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/debug_present/";
    private static final int PUSH_BYTES = DebugPresentPushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private long boundOutputView;
    private long boundNormalView;
    private long boundAlbedoView;
    private long boundDepthView;
    private long boundMotionView;
    private long boundSpecAlbedoView;
    private long boundSpecMotionView;
    private long boundSceneView;
    private long boundExposureView;
    private long boundExposureStateBuffer;
    private long boundVolumeDebugView;
    private boolean destroyed;

    private RtDebugPresentPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gpu) {
        this.ctx = ctx;
        this.gpu = gpu;
    }

    public static RtDebugPresentPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // 0: output (SDR display target). 1..6: guide buffers. 7: post-RR scene image.
            // 8: same-frame display exposure. 9: exposure state, including the sky metering scale.
            // 10: volumetric diagnostics written by the primary raygen.
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(DEBUG_PRESENT_BINDING_COUNT, stack);
            for (int i = DEBUG_PRESENT_OUTPUT; i < DEBUG_PRESENT_EXPOSURE_STATE; i++) {
                binds.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            binds.get(DEBUG_PRESENT_EXPOSURE_STATE).binding(DEBUG_PRESENT_EXPOSURE_STATE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DEBUG_PRESENT_VOLUME_DEBUG).binding(DEBUG_PRESENT_VOLUME_DEBUG)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(10);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);

            return new RtDebugPresentPipeline(ctx, RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "rt debug present", binds, poolSizes, PUSH_BYTES, SHADER_DIR + "main.comp.spv"));
        }
    }

    public void setImages(long outputImageView, long normalView, long albedoView, long depthView,
                           long motionView, long specAlbedoView, long specMotionView,
                           long sceneView, long exposureView, long volumeDebugView,
                           RtBuffer exposureState) {
        if (boundOutputView == outputImageView && boundNormalView == normalView && boundAlbedoView == albedoView
                && boundDepthView == depthView && boundMotionView == motionView
                && boundSpecAlbedoView == specAlbedoView && boundSpecMotionView == specMotionView
                && boundSceneView == sceneView && boundExposureView == exposureView
                && boundExposureStateBuffer == exposureState.handle
                && boundVolumeDebugView == volumeDebugView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Image bindings 0..8 run linearly; the exposure state sits at binding 9 and the volumetric
            // image at binding 10, so those two are written by their generated binding numbers.
            long[] views = {outputImageView, normalView, albedoView, depthView, motionView,
                    specAlbedoView, specMotionView, sceneView, exposureView};
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(DEBUG_PRESENT_BINDING_COUNT, stack);
            for (int i = 0; i < views.length; i++) {
                VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
                info.get(0).imageView(views[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(i).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(i)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(info);
            }
            VkDescriptorImageInfo.Buffer volumeInfo = VkDescriptorImageInfo.calloc(1, stack);
            volumeInfo.get(0).imageView(volumeDebugView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(DEBUG_PRESENT_VOLUME_DEBUG).sType$Default().dstSet(gpu.descriptorSet())
                    .dstBinding(DEBUG_PRESENT_VOLUME_DEBUG)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(volumeInfo);
            VkDescriptorBufferInfo.Buffer stateInfo = VkDescriptorBufferInfo.calloc(1, stack);
            stateInfo.get(0).buffer(exposureState.handle).offset(0).range(exposureState.size);
            writes.get(DEBUG_PRESENT_EXPOSURE_STATE).sType$Default().dstSet(gpu.descriptorSet())
                    .dstBinding(DEBUG_PRESENT_EXPOSURE_STATE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(stateInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutputView = outputImageView;
        boundNormalView = normalView;
        boundAlbedoView = albedoView;
        boundDepthView = depthView;
        boundMotionView = motionView;
        boundSpecAlbedoView = specAlbedoView;
        boundSpecMotionView = specMotionView;
        boundSceneView = sceneView;
        boundExposureView = exposureView;
        boundExposureStateBuffer = exposureState.handle;
        boundVolumeDebugView = volumeDebugView;
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height, int debugView,
                         float centerWeightSigma, float centerWeightFloor) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "debug present compute")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(), 0, stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(DebugPresentPushData.BYTE_SIZE);
            new DebugPresentPushData(debugView, centerWeightSigma, centerWeightFloor).write(push);
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

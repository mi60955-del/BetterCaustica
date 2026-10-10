package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.ExposureHistPushData;
import dev.comfyfluffy.caustica.rt.gen.ExposureResolvePushData;

import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/** Compute pipelines for histogram auto-exposure over the RT HDR trace output. */
final class RtExposurePipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/";

    private final RtContext ctx;
    private final long histDescriptorSetLayout;
    private final long histDescriptorPool;
    private final long histDescriptorSet;
    private final long histPipelineLayout;
    private final long histPipeline;
    private final long resolveDescriptorSetLayout;
    private final long resolveDescriptorPool;
    private final long resolveDescriptorSet;
    private final long resolvePipelineLayout;
    private final long resolvePipeline;

    private long boundColorView;
    private long boundDepthView;
    private long boundAlbedoView;
    private long boundHistogramBufferForHist;
    private long boundHistogramBufferForResolve;
    private long boundExposureView;
    private long boundStateBuffer;
    private boolean destroyed;

    private RtExposurePipeline(RtContext ctx,
                               long histDescriptorSetLayout, long histDescriptorPool, long histDescriptorSet,
                               long histPipelineLayout, long histPipeline,
                               long resolveDescriptorSetLayout, long resolveDescriptorPool, long resolveDescriptorSet,
                               long resolvePipelineLayout, long resolvePipeline) {
        this.ctx = ctx;
        this.histDescriptorSetLayout = histDescriptorSetLayout;
        this.histDescriptorPool = histDescriptorPool;
        this.histDescriptorSet = histDescriptorSet;
        this.histPipelineLayout = histPipelineLayout;
        this.histPipeline = histPipeline;
        this.resolveDescriptorSetLayout = resolveDescriptorSetLayout;
        this.resolveDescriptorPool = resolveDescriptorPool;
        this.resolveDescriptorSet = resolveDescriptorSet;
        this.resolvePipelineLayout = resolvePipelineLayout;
        this.resolvePipeline = resolvePipeline;
    }

    static RtExposurePipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer histBinds = VkDescriptorSetLayoutBinding.calloc(EXPOSURE_HIST_BINDING_COUNT, stack);
            histBinds.get(EXPOSURE_HIST_COLOR).binding(EXPOSURE_HIST_COLOR).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            histBinds.get(EXPOSURE_HIST_BINS).binding(EXPOSURE_HIST_BINS).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            histBinds.get(EXPOSURE_HIST_DEPTH).binding(EXPOSURE_HIST_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            histBinds.get(EXPOSURE_HIST_ALBEDO).binding(EXPOSURE_HIST_ALBEDO).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long histDsl = RtComputePipelineSupport.createDescriptorSetLayout(ctx, stack, histBinds, "rt exposure hist");
            VkDescriptorPoolSize.Buffer histSizes = VkDescriptorPoolSize.calloc(2, stack);
            histSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(3);
            histSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);
            long histPool = RtComputePipelineSupport.createDescriptorPool(ctx, stack, histSizes, 1, "rt exposure hist");
            long histSet = RtComputePipelineSupport.allocateDescriptorSets(ctx, stack, histPool, histDsl, 1, "rt exposure hist")[0];
            long histLayout = RtComputePipelineSupport.createPipelineLayout(ctx, stack, histDsl,
                    ExposureHistPushData.BYTE_SIZE, "rt exposure hist");
            long histPipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, histLayout,
                    SHADER_DIR + "exposure_hist/main.comp.spv", "rt exposure hist pipeline");

            VkDescriptorSetLayoutBinding.Buffer resolveBinds = VkDescriptorSetLayoutBinding.calloc(EXPOSURE_RESOLVE_BINDING_COUNT, stack);
            resolveBinds.get(EXPOSURE_RESOLVE_HIST_BINS).binding(EXPOSURE_RESOLVE_HIST_BINS).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            resolveBinds.get(EXPOSURE_RESOLVE_IMAGE).binding(EXPOSURE_RESOLVE_IMAGE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            resolveBinds.get(EXPOSURE_RESOLVE_STATE).binding(EXPOSURE_RESOLVE_STATE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long resolveDsl = RtComputePipelineSupport.createDescriptorSetLayout(ctx, stack, resolveBinds, "rt exposure resolve");
            VkDescriptorPoolSize.Buffer resolveSizes = VkDescriptorPoolSize.calloc(2, stack);
            resolveSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            resolveSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(2);
            long resolvePool = RtComputePipelineSupport.createDescriptorPool(ctx, stack, resolveSizes, 1, "rt exposure resolve");
            long resolveSet = RtComputePipelineSupport.allocateDescriptorSets(ctx, stack, resolvePool, resolveDsl, 1, "rt exposure resolve")[0];
            long resolveLayout = RtComputePipelineSupport.createPipelineLayout(ctx, stack, resolveDsl,
                    ExposureResolvePushData.BYTE_SIZE, "rt exposure resolve");
            long resolvePipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, resolveLayout,
                    SHADER_DIR + "exposure_resolve/main.comp.spv", "rt exposure resolve pipeline");

            return new RtExposurePipeline(ctx, histDsl, histPool, histSet, histLayout, histPipeline,
                    resolveDsl, resolvePool, resolveSet, resolveLayout, resolvePipeline);
        }
    }

    void setResources(long colorView, long depthView, long albedoView,
                      RtBuffer histogram, long exposureView, RtBuffer state) {
        if (boundColorView != colorView || boundDepthView != depthView || boundAlbedoView != albedoView
                || boundHistogramBufferForHist != histogram.handle) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkDescriptorImageInfo.Buffer colorInfo = VkDescriptorImageInfo.calloc(1, stack);
                colorInfo.get(0).imageView(colorView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorBufferInfo.Buffer histInfo = VkDescriptorBufferInfo.calloc(1, stack);
                histInfo.get(0).buffer(histogram.handle).offset(0).range(histogram.size);
                VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
                depthInfo.get(0).imageView(depthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorImageInfo.Buffer albedoInfo = VkDescriptorImageInfo.calloc(1, stack);
                albedoInfo.get(0).imageView(albedoView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(EXPOSURE_HIST_BINDING_COUNT, stack);
                writes.get(EXPOSURE_HIST_COLOR).sType$Default().dstSet(histDescriptorSet).dstBinding(EXPOSURE_HIST_COLOR)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(colorInfo);
                writes.get(EXPOSURE_HIST_BINS).sType$Default().dstSet(histDescriptorSet).dstBinding(EXPOSURE_HIST_BINS)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(histInfo);
                writes.get(EXPOSURE_HIST_DEPTH).sType$Default().dstSet(histDescriptorSet).dstBinding(EXPOSURE_HIST_DEPTH)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(depthInfo);
                writes.get(EXPOSURE_HIST_ALBEDO).sType$Default().dstSet(histDescriptorSet).dstBinding(EXPOSURE_HIST_ALBEDO)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(albedoInfo);
                VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
            }
            boundColorView = colorView;
            boundDepthView = depthView;
            boundAlbedoView = albedoView;
            boundHistogramBufferForHist = histogram.handle;
        }
        if (boundHistogramBufferForResolve != histogram.handle || boundExposureView != exposureView
                || boundStateBuffer != state.handle) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkDescriptorBufferInfo.Buffer histInfo = VkDescriptorBufferInfo.calloc(1, stack);
                histInfo.get(0).buffer(histogram.handle).offset(0).range(histogram.size);
                VkDescriptorImageInfo.Buffer exposureInfo = VkDescriptorImageInfo.calloc(1, stack);
                exposureInfo.get(0).imageView(exposureView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorBufferInfo.Buffer stateInfo = VkDescriptorBufferInfo.calloc(1, stack);
                stateInfo.get(0).buffer(state.handle).offset(0).range(state.size);
                VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(EXPOSURE_RESOLVE_BINDING_COUNT, stack);
                writes.get(EXPOSURE_RESOLVE_HIST_BINS).sType$Default().dstSet(resolveDescriptorSet).dstBinding(EXPOSURE_RESOLVE_HIST_BINS)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(histInfo);
                writes.get(EXPOSURE_RESOLVE_IMAGE).sType$Default().dstSet(resolveDescriptorSet).dstBinding(EXPOSURE_RESOLVE_IMAGE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(exposureInfo);
                writes.get(EXPOSURE_RESOLVE_STATE).sType$Default().dstSet(resolveDescriptorSet).dstBinding(EXPOSURE_RESOLVE_STATE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(stateInfo);
                VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
            }
            boundHistogramBufferForResolve = histogram.handle;
            boundExposureView = exposureView;
            boundStateBuffer = state.handle;
        }
    }

    void dispatchHistogram(org.lwjgl.vulkan.VkCommandBuffer cmd, int width, int height,
                           RtExposure.AutoConfig config) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure histogram")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, histPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, histPipelineLayout, 0,
                    stack.longs(histDescriptorSet), null);
            ByteBuffer push = stack.malloc(ExposureHistPushData.BYTE_SIZE);
            new ExposureHistPushData(config.stride(), config.centerWeightSigma(), config.centerWeightFloor())
                    .write(push);
            VK10.vkCmdPushConstants(cmd, histPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            int stride = config.stride();
            int sampleWidth = (width + stride - 1) / stride;
            int sampleHeight = (height + stride - 1) / stride;
            VK10.vkCmdDispatch(cmd, (sampleWidth + 15) / 16, (sampleHeight + 15) / 16, 1);
        }
    }

    void dispatchResolve(org.lwjgl.vulkan.VkCommandBuffer cmd, RtExposure.AutoConfig config, float frameTimeSeconds) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure resolve")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipelineLayout, 0,
                    stack.longs(resolveDescriptorSet), null);
            ByteBuffer push = stack.malloc(ExposureResolvePushData.BYTE_SIZE);
            RtExposure.ExposureCurve curve = config.curve();
            new ExposureResolvePushData(
                    config.key(), config.minEv(), config.maxEv(), config.adaptDarken(), config.adaptBrighten(),
                    frameTimeSeconds, config.evBias(), config.lowPercentile(), config.highPercentile(),
                    config.skyWeightCap(), curve.scene0(), curve.compensation0(), curve.scene1(),
                    curve.compensation1(), curve.scene2(), curve.compensation2(), curve.scene3(),
                    curve.compensation3(), config.emissiveWeightCap(), config.evOffset(), config.preExposure(),
                    config.resetSequence()
            ).write(push);
            VK10.vkCmdPushConstants(cmd, resolvePipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, 1, 1, 1);
        }
    }

    void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, resolvePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, resolvePipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, resolveDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, resolveDescriptorSetLayout, null);
        VK10.vkDestroyPipeline(vk, histPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, histPipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, histDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, histDescriptorSetLayout, null);
        destroyed = true;
    }
}

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
import dev.comfyfluffy.caustica.rt.gen.DofPostGatherPushData;
import dev.comfyfluffy.caustica.rt.gen.DofPostRecombinePushData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The post-process depth of field's two dispatches: the half-res gather that accumulates the near and
 * far defocus fields for both display channels, and the full-res recombine that layers them over the
 * display mapper's compose pair in place. Mode 2 of the depth of field setting runs them; modes 0 and
 * 1 leave the compose pair exactly as the mapper wrote it, which is why the recombine targets the
 * compose pair rather than a copy — the sharpen pass reads the same images either way.
 *
 * <p>The compose images are bound on both sides of the recombine (storage output and nothing else —
 * the gather reads them through its own sampled bindings before the recombine ever runs). Each
 * recombine thread loads and stores exactly its own texel and no other thread touches it, which is
 * what makes the in-place pair a filter and not a race; the barrier between the dispatches is what
 * orders it against the gather.
 */
public final class RtDofPostPipeline {
    private static final String GATHER_SHADER = "/bettercaustica/shaders/pipelines/dof_post_gather/gather.comp.spv";
    private static final String RECOMBINE_SHADER = "/bettercaustica/shaders/pipelines/dof_post_recombine/recombine.comp.spv";
    private static final int GATHER_PUSH_BYTES = DofPostGatherPushData.BYTE_SIZE;
    private static final int RECOMBINE_PUSH_BYTES = DofPostRecombinePushData.BYTE_SIZE;

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gather;
    private final RtComputePipelineSupport.SingleSet recombine;
    /**
     * Linear and clamp-to-edge, shared by both dispatches: the gather samples the compose pair inside
     * its kernels, the recombine upsamples the half-res fields with it. The depth image is read
     * through storage loads instead, because r32f has no linear filter guarantee.
     */
    private final long sampler;
    private long boundGatherFar;
    private long boundGatherNear;
    private long boundGatherHdrFar;
    private long boundGatherHdrNear;
    private long boundGatherSource;
    private long boundGatherHdrSource;
    private long boundGatherDepth;
    private long boundOutput;
    private long boundHdrOutput;
    private long boundRecombineFar;
    private long boundRecombineNear;
    private long boundRecombineHdrFar;
    private long boundRecombineHdrNear;
    private boolean destroyed;

    private RtDofPostPipeline(RtContext ctx, RtComputePipelineSupport.SingleSet gather,
                              RtComputePipelineSupport.SingleSet recombine, long sampler) {
        this.ctx = ctx;
        this.gather = gather;
        this.recombine = recombine;
        this.sampler = sampler;
    }

    public static RtDofPostPipeline create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer gatherBinds =
                    VkDescriptorSetLayoutBinding.calloc(DOF_POST_GATHER_BINDING_COUNT, stack);
            int[] gatherStorage = {DOF_POST_GATHER_FAR, DOF_POST_GATHER_NEAR,
                    DOF_POST_GATHER_HDR_FAR, DOF_POST_GATHER_HDR_NEAR, DOF_POST_GATHER_FOCUS_DEPTH};
            for (int binding : gatherStorage) {
                gatherBinds.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            int[] gatherSampled = {DOF_POST_GATHER_SOURCE, DOF_POST_GATHER_HDR_SOURCE};
            for (int binding : gatherSampled) {
                gatherBinds.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorPoolSize.Buffer gatherPool = VkDescriptorPoolSize.calloc(2, stack);
            gatherPool.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(5);
            gatherPool.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);
            RtComputePipelineSupport.SingleSet gather = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "dof post gather", gatherBinds, gatherPool,
                    GATHER_PUSH_BYTES, GATHER_SHADER);

            VkDescriptorSetLayoutBinding.Buffer recombineBinds =
                    VkDescriptorSetLayoutBinding.calloc(DOF_POST_RECOMBINE_BINDING_COUNT, stack);
            int[] recombineStorage = {DOF_POST_RECOMBINE_OUTPUT, DOF_POST_RECOMBINE_HDR_OUTPUT};
            for (int binding : recombineStorage) {
                recombineBinds.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            int[] recombineSampled = {DOF_POST_RECOMBINE_FAR, DOF_POST_RECOMBINE_NEAR,
                    DOF_POST_RECOMBINE_HDR_FAR, DOF_POST_RECOMBINE_HDR_NEAR};
            for (int binding : recombineSampled) {
                recombineBinds.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorPoolSize.Buffer recombinePool = VkDescriptorPoolSize.calloc(2, stack);
            recombinePool.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            recombinePool.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(4);
            RtComputePipelineSupport.SingleSet recombine = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "dof post recombine", recombineBinds, recombinePool,
                    RECOMBINE_PUSH_BYTES, RECOMBINE_SHADER);

            LongBuffer handle = stack.mallocLong(1);
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle),
                    "vkCreateSampler(dof post)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "dof post sampler");

            return new RtDofPostPipeline(ctx, gather, recombine, sampler);
        }
    }

    /**
     * Binds everything both dispatches touch. The compose pair arrives twice by design: sampled for
     * the gather's kernels and stored as the recombine's in-place target, and the half-res fields
     * arrive twice as well, storage for the gather and sampled for the recombine. The HDR slots take
     * the 1x1 placeholder while HDR output is off; the dispatch's flag then keeps them unwritten.
     * {@code focusDepthView} is the cur side of the lens depth pair — the one this frame's primary
     * pass published.
     */
    public void setImages(long farView, long nearView, long hdrFarView, long hdrNearView,
                          long sourceView, long hdrSourceView, long focusDepthView,
                          long outputView, long hdrOutputView) {
        if (boundGatherFar == farView && boundGatherNear == nearView
                && boundGatherHdrFar == hdrFarView && boundGatherHdrNear == hdrNearView
                && boundGatherSource == sourceView && boundGatherHdrSource == hdrSourceView
                && boundGatherDepth == focusDepthView
                && boundOutput == outputView && boundHdrOutput == hdrOutputView
                && boundRecombineFar == farView && boundRecombineNear == nearView
                && boundRecombineHdrFar == hdrFarView && boundRecombineHdrNear == hdrNearView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer farStorage = storageView(stack, farView);
            VkDescriptorImageInfo.Buffer nearStorage = storageView(stack, nearView);
            VkDescriptorImageInfo.Buffer hdrFarStorage = storageView(stack, hdrFarView);
            VkDescriptorImageInfo.Buffer hdrNearStorage = storageView(stack, hdrNearView);
            VkDescriptorImageInfo.Buffer depthStorage = storageView(stack, focusDepthView);
            VkDescriptorImageInfo.Buffer outputStorage = storageView(stack, outputView);
            VkDescriptorImageInfo.Buffer hdrOutputStorage = storageView(stack, hdrOutputView);
            VkDescriptorImageInfo.Buffer sourceSampled = sampledView(stack, sourceView);
            VkDescriptorImageInfo.Buffer hdrSourceSampled = sampledView(stack, hdrSourceView);
            VkDescriptorImageInfo.Buffer farSampled = sampledView(stack, farView);
            VkDescriptorImageInfo.Buffer nearSampled = sampledView(stack, nearView);
            VkDescriptorImageInfo.Buffer hdrFarSampled = sampledView(stack, hdrFarView);
            VkDescriptorImageInfo.Buffer hdrNearSampled = sampledView(stack, hdrNearView);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(
                    DOF_POST_GATHER_BINDING_COUNT + DOF_POST_RECOMBINE_BINDING_COUNT, stack);
            int w = 0;
            long gatherSet = gather.descriptorSet();
            w = write(writes, w, gatherSet, DOF_POST_GATHER_FAR,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, farStorage);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_NEAR,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, nearStorage);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_HDR_FAR,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, hdrFarStorage);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_HDR_NEAR,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, hdrNearStorage);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_FOCUS_DEPTH,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, depthStorage);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_SOURCE,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sourceSampled);
            w = write(writes, w, gatherSet, DOF_POST_GATHER_HDR_SOURCE,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hdrSourceSampled);
            long recombineSet = recombine.descriptorSet();
            w = write(writes, w, recombineSet, DOF_POST_RECOMBINE_OUTPUT,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, outputStorage);
            w = write(writes, w, recombineSet, DOF_POST_RECOMBINE_HDR_OUTPUT,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, hdrOutputStorage);
            w = write(writes, w, recombineSet, DOF_POST_RECOMBINE_FAR,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, farSampled);
            w = write(writes, w, recombineSet, DOF_POST_RECOMBINE_NEAR,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nearSampled);
            w = write(writes, w, recombineSet, DOF_POST_RECOMBINE_HDR_FAR,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hdrFarSampled);
            write(writes, w, recombineSet, DOF_POST_RECOMBINE_HDR_NEAR,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hdrNearSampled);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundGatherFar = farView;
        boundGatherNear = nearView;
        boundGatherHdrFar = hdrFarView;
        boundGatherHdrNear = hdrNearView;
        boundGatherSource = sourceView;
        boundGatherHdrSource = hdrSourceView;
        boundGatherDepth = focusDepthView;
        boundOutput = outputView;
        boundHdrOutput = hdrOutputView;
        boundRecombineFar = farView;
        boundRecombineNear = nearView;
        boundRecombineHdrFar = hdrFarView;
        boundRecombineHdrNear = hdrNearView;
    }

    /** One storage-image descriptor: the view in GENERAL layout, as every image here lives. */
    private static VkDescriptorImageInfo.Buffer storageView(MemoryStack stack, long view) {
        return VkDescriptorImageInfo.calloc(1, stack)
                .imageView(view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
    }

    /** One combined-image-sampler descriptor over the shared linear sampler. */
    private VkDescriptorImageInfo.Buffer sampledView(MemoryStack stack, long view) {
        return VkDescriptorImageInfo.calloc(1, stack)
                .imageView(view).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
    }

    private static int write(VkWriteDescriptorSet.Buffer writes, int index, long set, int binding,
                             int type, VkDescriptorImageInfo.Buffer info) {
        writes.get(index).sType$Default().dstSet(set).dstBinding(binding)
                .descriptorCount(1).descriptorType(type).pImageInfo(info);
        return index + 1;
    }

    /**
     * Runs the gather at the half-res dispatch size over both display channels; {@code hdrEnabled}
     * must match which images are bound in the HDR slots.
     */
    public void dispatchGather(VkCommandBuffer cmd, int halfWidth, int halfHeight,
                               DofPostGatherPushData push) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "dof post gather")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gather.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    gather.pipelineLayout(), 0, stack.longs(gather.descriptorSet()), null);
            ByteBuffer pushBytes = stack.malloc(GATHER_PUSH_BYTES);
            push.write(pushBytes);
            VK10.vkCmdPushConstants(cmd, gather.pipelineLayout(),
                    VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes);
            VK10.vkCmdDispatch(cmd, (halfWidth + 15) / 16, (halfHeight + 15) / 16, 1);
        }
    }

    /** Runs the recombine at full display resolution, layering the fields over the compose pair. */
    public void dispatchRecombine(VkCommandBuffer cmd, int width, int height,
                                  DofPostRecombinePushData push) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "dof post recombine")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, recombine.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    recombine.pipelineLayout(), 0, stack.longs(recombine.descriptorSet()), null);
            ByteBuffer pushBytes = stack.malloc(RECOMBINE_PUSH_BYTES);
            push.write(pushBytes);
            VK10.vkCmdPushConstants(cmd, recombine.pipelineLayout(),
                    VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        gather.destroy(ctx.vk());
        recombine.destroy(ctx.vk());
        VK10.vkDestroySampler(ctx.vk(), sampler, null);
        destroyed = true;
    }
}

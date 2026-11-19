package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.RainPushData;
import dev.comfyfluffy.caustica.rt.overlay.RtOverlayPipelines;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.KHRDynamicRendering;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkOffset2D;
import org.lwjgl.vulkan.VkRect2D;
import org.lwjgl.vulkan.VkRenderingAttachmentInfo;
import org.lwjgl.vulkan.VkRenderingInfo;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkViewport;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.RAIN_DEPTH;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.RAIN_SET;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.RAIN_SKY_VIEW;

/**
 * The rain-streak overlay: one graphics pipeline that composites procedural streaks onto the
 * reconstructed scene image.
 *
 * <p>The pass is drawn after DLSS-RR (or the fallback upscale) and before the display chain reads that
 * image, so the streaks are metered by auto-exposure, smeared by bloom and rolled off by the tone curve
 * exactly like the light they are mixed with. It owns its pipeline, layout, descriptor set and depth
 * sampler; the two images it reads — the trace's depth guide and the sky-view LUT — belong to {@code
 * RtComposite} and {@code RtSkyLut} and are only ever written into the descriptor set here.
 *
 * <p>Depth is read from that guide rather than attached: the scene is already resolved, and a depth
 * attachment would force the streak draws to be ordered against a depth buffer that only describes the
 * traced surfaces (rain has no depth of its own to write). The fragment shader performs the same
 * comparison a rasterizer would.
 */
public final class RtRainPipeline {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/rain/";

    private final RtContext ctx;
    private final RtOverlayPipelines.Pipeline pipeline;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    /**
     * Nearest/clamp sampler for the depth guide: the test wants the exact texel under the pixel, and a
     * filtered depth would invent surfaces between two real ones.
     */
    private final long depthSampler;
    /** The sky-view LUT's own linear/clamp sampler, owned by {@code RtSkyLut}. */
    private final long skySampler;

    private RtRainPipeline(RtContext ctx, RtOverlayPipelines.Pipeline pipeline, long descriptorSetLayout,
                           long descriptorPool, long descriptorSet, long depthSampler, long skySampler) {
        this.ctx = ctx;
        this.pipeline = pipeline;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.depthSampler = depthSampler;
        this.skySampler = skySampler;
    }

    public static RtRainPipeline create(RtContext ctx, long skySampler) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int stages = VK10.VK_SHADER_STAGE_VERTEX_BIT | VK10.VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(2, stack);
            bindings.get(0).binding(RAIN_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(stages);
            bindings.get(1).binding(RAIN_SKY_VIEW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(stages);
            long descriptorSetLayout = RtComputePipelineSupport.createDescriptorSetLayout(ctx, stack, bindings,
                    "rain streaks");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);
            long descriptorPool = RtComputePipelineSupport.createDescriptorPool(ctx, stack, poolSizes, 1,
                    "rain streaks");
            long descriptorSet = RtComputePipelineSupport.allocateDescriptorSets(ctx, stack, descriptorPool,
                    descriptorSetLayout, 1, "rain streaks")[0];

            long depthSampler = createNearestClampSampler(ctx, stack);

            // The attachment is the reconstructed scene image itself (RGBA16F, GENERAL layout), loaded
            // rather than cleared: the streaks are composited over it by the premultiplied "over" op.
            RtOverlayPipelines.Pipeline pipeline = new RtOverlayPipelines.Spec(
                    "rain/rain.vert.spv", "rain/rain.frag.spv")
                    .vertex(RtOverlayPipelines.VertexFormat.NONE)
                    .blend(RtOverlayPipelines.Blend.PREMULTIPLIED_ALPHA)
                    .attachment(VK10.VK_FORMAT_R16G16B16A16_SFLOAT)
                    .push(RainPushData.BYTE_SIZE, stages)
                    .descriptorSetLayout(descriptorSetLayout)
                    .build(ctx, "rain streaks");
            return new RtRainPipeline(ctx, pipeline, descriptorSetLayout, descriptorPool, descriptorSet,
                    depthSampler, skySampler);
        }
    }

    /**
     * Point the two sampler bindings at this frame's images.
     *
     * <p>Written every frame rather than only when the handles change: the depth guide is a mod-owned
     * render target that is destroyed and recreated on a resize, and a destroyed {@code VkImageView} handle
     * can be handed straight back for a different image. A handle comparison would then consider the
     * descriptor current while it named the old, unrelated object. Two {@code vkUpdateDescriptorSets}
     * calls are cheaper than the class of bug the check would be guarding against.
     */
    public void bindViews(long depthView, long skyView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(2, stack);
            info.get(0).sampler(depthSampler).imageView(depthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            info.get(1).sampler(skySampler).imageView(skyView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            int[] targets = {RAIN_DEPTH, RAIN_SKY_VIEW};
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(2, stack);
            for (int i = 0; i < targets.length; i++) {
                writes.get(i).sType$Default().dstSet(descriptorSet).dstBinding(targets[i])
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.create(info.address(i), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    /**
     * Composite {@code dropCount} streaks into {@code target} and leave it in GENERAL layout with the
     * attachment written. The caller is responsible for the barrier that makes those writes visible to
     * whatever reads the image next.
     */
    public void draw(VkCommandBuffer cmd, MemoryStack stack, RtImage target, RainPushData push) {
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "rain streaks")) {
            VkRenderingAttachmentInfo.Buffer colorAttach = VkRenderingAttachmentInfo.calloc(1, stack).sType$Default()
                    .imageView(target.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                    .loadOp(VK10.VK_ATTACHMENT_LOAD_OP_LOAD)
                    .storeOp(VK10.VK_ATTACHMENT_STORE_OP_STORE);
            VkRect2D renderArea = VkRect2D.calloc(stack);
            renderArea.offset(VkOffset2D.calloc(stack).set(0, 0));
            renderArea.extent().set(target.width, target.height);
            VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack).sType$Default()
                    .renderArea(renderArea).layerCount(1).pColorAttachments(colorAttach);
            KHRDynamicRendering.vkCmdBeginRenderingKHR(cmd, renderingInfo);

            VkViewport.Buffer viewport = VkViewport.calloc(1, stack);
            viewport.get(0).x(0).y(0).width(target.width).height(target.height).minDepth(0f).maxDepth(1f);
            VK10.vkCmdSetViewport(cmd, 0, viewport);
            VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack);
            scissor.get(0).offset(VkOffset2D.calloc(stack).set(0, 0));
            scissor.get(0).extent().set(target.width, target.height);
            VK10.vkCmdSetScissor(cmd, 0, scissor);

            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout,
                    RAIN_SET, stack.longs(descriptorSet), null);
            ByteBuffer constants = stack.malloc(RainPushData.BYTE_SIZE);
            push.write(constants);
            VK10.vkCmdPushConstants(cmd, pipeline.layout,
                    VK10.VK_SHADER_STAGE_VERTEX_BIT | VK10.VK_SHADER_STAGE_FRAGMENT_BIT, 0, constants);
            // Six vertices per drop, generated from SV_VertexID: the pass has no vertex buffer to bind.
            VK10.vkCmdDraw(cmd, push.dropCount() * 6, 1, 0, 0);
            KHRDynamicRendering.vkCmdEndRenderingKHR(cmd);
        }
    }

    private static long createNearestClampSampler(RtContext ctx, MemoryStack stack) {
        VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle), "vkCreateSampler(rain depth)");
        long sampler = handle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "rain depth sampler");
        return sampler;
    }

    public void destroy() {
        VkDevice vk = ctx.vk();
        VK10.vkDestroySampler(vk, depthSampler, null);
        pipeline.destroy(vk);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
    }
}

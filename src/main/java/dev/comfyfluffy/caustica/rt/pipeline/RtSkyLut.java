package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.PushAddrData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The sky's LUTs (Hillaire 2020) and the compute passes that bake them. See
 * {@code shaders/pipelines/world/sky.slang} for the physics.
 *
 * <ul>
 *   <li><b>Transmittance</b> 256x64 — point-to-space extinction by (altitude, cos zenith). Static.</li>
 *   <li><b>Multiple scattering</b> 32x32 — the static closed series of second and higher scattering
 *       orders that lights twilight and the night sky.</li>
 *   <li><b>Sky view</b> 192x216 — the dome itself, one 192x108 slice per celestial body, rebuilt every
 *       frame from the sun/moon angles in {@code WorldPush}.</li>
 *   <li><b>Sky irradiance</b> 9x1 — the dome convolved into order-2 spherical harmonics with the Lambert
 *       transfer applied, so a diffuse shading point reads its whole unoccluded sky irradiance from one
 *       small table instead of one sampled continuation ray.</li>
 * </ul>
 *
 * <p>The two static LUTs are baked on the first frame rather than at construction: they read the ground
 * albedo from the look package through the frame's {@code WorldPush} slot, so they need a frame's push
 * buffer to exist. All four passes share one descriptor set and one pipeline layout — every pass declares
 * all seven bindings, and each uses the subset it needs.
 *
 * <p>The images are single-buffered. The bake and the trace that reads it run in the same submission with
 * a barrier between, and the composite's end-of-frame barrier orders the next frame's overwrite against
 * this frame's trace — the same argument that lets every other GPU-written per-frame image here be
 * single-buffered.
 */
public final class RtSkyLut {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/sky_lut/";
    // Keep in lock-step with the same-named constants in shaders/pipelines/world/sky.slang.
    public static final int TRANSMITTANCE_WIDTH = 256;
    public static final int TRANSMITTANCE_HEIGHT = 64;
    public static final int MULTISCATTER_WIDTH = 32;
    public static final int MULTISCATTER_HEIGHT = 32;
    public static final int SKY_VIEW_WIDTH = 192;
    public static final int SKY_VIEW_SLICE_HEIGHT = 108;
    public static final int SKY_VIEW_BODY_COUNT = 2;
    public static final int SKY_VIEW_HEIGHT = SKY_VIEW_SLICE_HEIGHT * SKY_VIEW_BODY_COUNT;
    public static final int SKY_IRRADIANCE_COEFFS = 9;
    private static final int GROUP_SIZE = 8;

    private final RtContext ctx;
    private final RtImage transmittance;
    private final RtImage multiScatter;
    private final RtImage skyView;
    private final RtImage skyIrradiance;
    private final long sampler;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long transmittancePipeline;
    private final long multiScatterPipeline;
    private final long skyViewPipeline;
    private final long irradiancePipeline;
    private boolean staticLutsBaked;
    private boolean destroyed;

    private RtSkyLut(RtContext ctx, RtImage transmittance, RtImage multiScatter, RtImage skyView,
                     RtImage skyIrradiance, long sampler, long descriptorSetLayout, long descriptorPool,
                     long descriptorSet, long pipelineLayout, long transmittancePipeline,
                     long multiScatterPipeline, long skyViewPipeline, long irradiancePipeline) {
        this.ctx = ctx;
        this.transmittance = transmittance;
        this.multiScatter = multiScatter;
        this.skyView = skyView;
        this.skyIrradiance = skyIrradiance;
        this.sampler = sampler;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.pipelineLayout = pipelineLayout;
        this.transmittancePipeline = transmittancePipeline;
        this.multiScatterPipeline = multiScatterPipeline;
        this.skyViewPipeline = skyViewPipeline;
        this.irradiancePipeline = irradiancePipeline;
    }

    public static RtSkyLut create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        // The four LUT images are created before the stack try and have no owner until the constructor
        // runs, so a failure anywhere below must destroy them along with whatever the try created; their
        // init barriers batch into one submit instead of one per image.
        ctx.beginImageInitBatch();
        RtImage transmittance;
        RtImage multiScatter;
        RtImage skyView;
        RtImage skyIrradiance;
        try {
            transmittance = ctx.createStorageImage(TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sky transmittance LUT");
            multiScatter = ctx.createStorageImage(MULTISCATTER_WIDTH, MULTISCATTER_HEIGHT,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sky multiple-scattering LUT");
            skyView = ctx.createStorageImage(SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sky view LUT");
            skyIrradiance = ctx.createStorageImage(SKY_IRRADIANCE_COEFFS, 1,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sky irradiance SH");
        } finally {
            ctx.endImageInitBatch();
        }
        java.util.ArrayDeque<Runnable> undo = new java.util.ArrayDeque<>();
        try {
            try (MemoryStack stack = MemoryStack.stackPush()) {
            // CLAMP on both axes. The sky-view LUT's U axis is angle-from-the-light, which is folded on
            // itself rather than wrapped, and its V axis stacks the two body slices — a REPEAT here would
            // let the sun's rows bleed into the moon's at the seam.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(sky LUT)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "sky LUT sampler");
            undo.push(() -> VK10.vkDestroySampler(vk, sampler, null));

            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(SKY_LUT_BINDING_COUNT, stack);
            for (int i = SKY_LUT_TRANSMITTANCE_IMAGE; i <= SKY_LUT_SKY_IRRADIANCE_IMAGE; i++) {
                bindings.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            for (int i = SKY_LUT_TRANSMITTANCE_SAMPLER; i < SKY_LUT_BINDING_COUNT; i++) {
                bindings.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            long descriptorSetLayout = RtComputePipelineSupport.createDescriptorSetLayout(ctx, stack, bindings, "sky LUT");
            undo.push(() -> VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null));

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(4);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(3);
            long descriptorPool = RtComputePipelineSupport.createDescriptorPool(ctx, stack, poolSizes, 1, "sky LUT");
            undo.push(() -> VK10.vkDestroyDescriptorPool(vk, descriptorPool, null));

            long descriptorSet = RtComputePipelineSupport.allocateDescriptorSets(
                    ctx, stack, descriptorPool, descriptorSetLayout, 1, "sky LUT")[0];

            // Same inline address block the trace uses: each pass dereferences the frame's WorldPush
            // through pcAddr.worldPushAddr exactly like the RT stages do.
            long pipelineLayout = RtComputePipelineSupport.createPipelineLayout(ctx, stack, descriptorSetLayout,
                    PushAddrData.BYTE_SIZE, "sky LUT");
            undo.push(() -> VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null));

            long transmittancePipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, pipelineLayout,
                    SHADER_DIR + "transmittance.comp.spv", "sky transmittance pipeline");
            undo.push(() -> VK10.vkDestroyPipeline(vk, transmittancePipeline, null));
            long multiScatterPipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, pipelineLayout,
                    SHADER_DIR + "multiscatter.comp.spv", "sky multiple-scattering pipeline");
            undo.push(() -> VK10.vkDestroyPipeline(vk, multiScatterPipeline, null));
            long skyViewPipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, pipelineLayout,
                    SHADER_DIR + "view.comp.spv", "sky view pipeline");
            undo.push(() -> VK10.vkDestroyPipeline(vk, skyViewPipeline, null));
            long irradiancePipeline = RtComputePipelineSupport.createComputePipeline(ctx, stack, pipelineLayout,
                    SHADER_DIR + "irradiance.comp.spv", "sky irradiance pipeline");
            undo.push(() -> VK10.vkDestroyPipeline(vk, irradiancePipeline, null));

            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(SKY_LUT_BINDING_COUNT, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(SKY_LUT_BINDING_COUNT, stack);
            long[] storageViews = {transmittance.view, multiScatter.view, skyView.view, skyIrradiance.view};
            int[] storageBindings = {SKY_LUT_TRANSMITTANCE_IMAGE, SKY_LUT_MULTISCATTER_IMAGE,
                    SKY_LUT_SKY_VIEW_IMAGE, SKY_LUT_SKY_IRRADIANCE_IMAGE};
            for (int i = 0; i < storageViews.length; i++) {
                int binding = storageBindings[i];
                images.get(binding).imageView(storageViews[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(binding).sType$Default().dstSet(descriptorSet).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(binding), 1));
            }
            // The sky-view LUT is written as a storage image by its own bake and read back as a texture by
            // the irradiance bake — different dispatches, with a barrier between, so the two bindings of the
            // same view never race.
            long[] sampledViews = {transmittance.view, multiScatter.view, skyView.view};
            int[] sampledBindings = {SKY_LUT_TRANSMITTANCE_SAMPLER, SKY_LUT_MULTISCATTER_SAMPLER,
                    SKY_LUT_SKY_VIEW_SAMPLER};
            for (int i = 0; i < sampledViews.length; i++) {
                int binding = sampledBindings[i];
                images.get(binding).imageView(sampledViews[i]).sampler(sampler)
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(binding).sType$Default().dstSet(descriptorSet).dstBinding(binding)
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(binding), 1));
            }
            VK10.vkUpdateDescriptorSets(vk, writes, null);

            return new RtSkyLut(ctx, transmittance, multiScatter, skyView, skyIrradiance, sampler,
                    descriptorSetLayout, descriptorPool, descriptorSet, pipelineLayout,
                    transmittancePipeline, multiScatterPipeline, skyViewPipeline, irradiancePipeline);
            }
        } catch (Throwable t) {
            for (Runnable r : undo) {
                try {
                    r.run();
                } catch (Throwable ignored) {
                }
            }
            transmittance.destroy();
            multiScatter.destroy();
            skyView.destroy();
            skyIrradiance.destroy();
            throw t;
        }
    }

    public long sampler() {
        return sampler;
    }

    public long transmittanceView() {
        return transmittance.view;
    }

    public long skyViewView() {
        return skyView.view;
    }

    public long skyIrradianceView() {
        return skyIrradiance.view;
    }

    /**
     * Record this frame's sky LUT work: the two static LUTs on the first frame, then the sky-view LUT and
     * the irradiance basis convolved from it. Must be recorded before the trace, with a barrier after (the
     * caller's) — the miss and raygen stages sample all of these.
     */
    public void record(VkCommandBuffer cmd, long worldPushAddress) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "sky LUTs")) {
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(descriptorSet), null);
            ByteBuffer push = stack.malloc(PushAddrData.BYTE_SIZE);
            new PushAddrData(worldPushAddress).write(push);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);

            if (!staticLutsBaked) {
                dispatch(cmd, stack, transmittancePipeline, TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT);
                // The multiple-scattering bake samples the transmittance LUT the previous dispatch just
                // wrote, and the sky-view bake samples both.
                dispatch(cmd, stack, multiScatterPipeline, MULTISCATTER_WIDTH, MULTISCATTER_HEIGHT);
                staticLutsBaked = true;
            }
            dispatch(cmd, stack, skyViewPipeline, SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);
            // One workgroup of nine, one thread per SH coefficient; the dispatch helper rounds up to whole
            // 8-wide groups and the kernel's own bound discards the overhang.
            dispatch(cmd, stack, irradiancePipeline, SKY_IRRADIANCE_COEFFS, 1);
        }
    }

    private void dispatch(VkCommandBuffer cmd, MemoryStack stack, long pipeline, int width, int height) {
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdDispatch(cmd, (width + GROUP_SIZE - 1) / GROUP_SIZE,
                (height + GROUP_SIZE - 1) / GROUP_SIZE, 1);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, irradiancePipeline, null);
        VK10.vkDestroyPipeline(vk, skyViewPipeline, null);
        VK10.vkDestroyPipeline(vk, multiScatterPipeline, null);
        VK10.vkDestroyPipeline(vk, transmittancePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, sampler, null);
        skyIrradiance.destroy();
        skyView.destroy();
        multiScatter.destroy();
        transmittance.destroy();
        destroyed = true;
    }
}

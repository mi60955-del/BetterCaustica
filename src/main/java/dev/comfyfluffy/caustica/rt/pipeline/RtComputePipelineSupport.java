package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * Shared Vulkan plumbing for the RT compute pipelines: SPIR-V module loading, descriptor-set
 * layout/pool/set creation, pipeline layouts carrying one push-constant range, and single-shader
 * compute pipeline creation. Every helper labels its objects via {@link RtDebugLabels} and checks
 * return codes; teardown mirrors creation through {@link SingleSet#destroy} or the owning class's
 * own destroy path for the primitives.
 */
final class RtComputePipelineSupport {
    private RtComputePipelineSupport() {
    }

    /** The Vulkan objects owned by one single-set compute pipeline. */
    record SingleSet(long descriptorSetLayout, long descriptorPool, long descriptorSet,
                     long pipelineLayout, long pipeline) {
        void destroy(VkDevice vk) {
            VK10.vkDestroyPipeline(vk, pipeline, null);
            VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
            VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
            VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        }
    }

    /**
     * Create a one-set, one-shader compute pipeline: descriptor set layout from {@code bindings}, a
     * pool sized by {@code poolSizes} holding exactly one set, a pipeline layout with a single
     * compute-stage push-constant range of {@code pushConstantBytes} (0 omits the range), and the
     * pipeline itself from the SPIR-V resource at {@code shaderResource}.
     */
    static SingleSet createSingleSetPipeline(RtContext ctx, MemoryStack stack, String label,
                                             VkDescriptorSetLayoutBinding.Buffer bindings,
                                             VkDescriptorPoolSize.Buffer poolSizes,
                                             int pushConstantBytes, String shaderResource) {
        long setLayout = createDescriptorSetLayout(ctx, stack, bindings, label);
        long pool = createDescriptorPool(ctx, stack, poolSizes, 1, label);
        long[] sets = allocateDescriptorSets(ctx, stack, pool, setLayout, 1, label);
        long layout = createPipelineLayout(ctx, stack, setLayout, pushConstantBytes, label);
        long pipeline = createComputePipeline(ctx, stack, layout, shaderResource, label);
        return new SingleSet(setLayout, pool, sets[0], layout, pipeline);
    }

    static long createDescriptorSetLayout(RtContext ctx, MemoryStack stack,
                                          VkDescriptorSetLayoutBinding.Buffer bindings, String label) {
        VkDescriptorSetLayoutCreateInfo ci = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType$Default().pBindings(bindings);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorSetLayout(ctx.vk(), ci, null, handle),
                "vkCreateDescriptorSetLayout(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, handle.get(0),
                label + " descriptor set layout");
        return handle.get(0);
    }

    static long createDescriptorPool(RtContext ctx, MemoryStack stack,
                                     VkDescriptorPoolSize.Buffer poolSizes, int maxSets, String label) {
        VkDescriptorPoolCreateInfo ci = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType$Default().maxSets(maxSets).pPoolSizes(poolSizes);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorPool(ctx.vk(), ci, null, handle),
                "vkCreateDescriptorPool(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, handle.get(0),
                label + " descriptor pool");
        return handle.get(0);
    }

    static long[] allocateDescriptorSets(RtContext ctx, MemoryStack stack, long pool,
                                         long setLayout, int count, String label) {
        LongBuffer layouts = stack.mallocLong(count);
        for (int i = 0; i < count; i++) {
            layouts.put(i, setLayout);
        }
        VkDescriptorSetAllocateInfo ai = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default().descriptorPool(pool).pSetLayouts(layouts);
        LongBuffer handles = stack.mallocLong(count);
        check(VK10.vkAllocateDescriptorSets(ctx.vk(), ai, handles),
                "vkAllocateDescriptorSets(" + label + ")");
        long[] sets = new long[count];
        for (int i = 0; i < count; i++) {
            sets[i] = handles.get(i);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, sets[i],
                    label + " descriptor set " + i);
        }
        return sets;
    }

    /**
     * One descriptor set plus a single compute-stage push-constant range of {@code pushConstantBytes};
     * {@code pushConstantBytes == 0} omits the range.
     */
    static long createPipelineLayout(RtContext ctx, MemoryStack stack, long setLayout,
                                     int pushConstantBytes, String label) {
        VkPipelineLayoutCreateInfo ci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                .pSetLayouts(stack.longs(setLayout));
        if (pushConstantBytes > 0) {
            VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushConstantBytes);
            ci.pPushConstantRanges(range);
        }
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreatePipelineLayout(ctx.vk(), ci, null, handle),
                "vkCreatePipelineLayout(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, handle.get(0),
                label + " pipeline layout");
        return handle.get(0);
    }

    /**
     * Create the compute pipeline for the {@code main} entry point of the SPIR-V resource at
     * {@code shaderResource}; the shader module is destroyed once the pipeline exists. The pipeline is
     * labelled {@code label} and its module {@code label + " module"}.
     */
    static long createComputePipeline(RtContext ctx, MemoryStack stack, long layout,
                                      String shaderResource, String label) {
        long module = loadModule(ctx.vk(), stack, shaderResource);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module)
                .pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer ci = VkComputePipelineCreateInfo.calloc(1, stack);
        ci.get(0).sType$Default().stage(stage).layout(layout);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(ctx.vk(), VK10.VK_NULL_HANDLE, ci, null, handle),
                "vkCreateComputePipelines(" + label + ")");
        VK10.vkDestroyShaderModule(ctx.vk(), module, null);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, handle.get(0), label);
        return handle.get(0);
    }

    /** Read a SPIR-V classpath resource and turn it into a shader module the caller must destroy. */
    static long loadModule(VkDevice vk, MemoryStack stack, String resource) {
        byte[] bytes;
        try (InputStream in = RtComputePipelineSupport.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + resource);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + resource, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo ci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, ci, null, module),
                    "vkCreateShaderModule(" + resource + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}

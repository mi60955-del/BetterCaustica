package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferMemoryBarrier;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.DofFocusPushData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.DOF_FOCUS_BINDING_COUNT;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.DOF_FOCUS_DEPTH;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.DOF_FOCUS_METER;

/**
 * The lens's autofocus meter: one workgroup that reads the centre of the depth the primary pass just
 * published and answers the harmonic mean of the distances in it, which is the focal plane a lens would
 * pick for that window.
 *
 * <p>The answer is a GPU value the host cannot use this frame — reading it now would stall on the trace
 * that produced it. So each frame copies it into its own slot of a ring and the host consumes a slot only
 * once that frame's graphics timeline value has completed, which is the same shape
 * {@link RtExposure} uses for its controller state. The measurement is therefore a handful of frames old,
 * and that is invisible: the focal plane glides to its target over
 * {@code dof.rack-time}, which is longer than the ring.
 *
 * <p>Metering the traced depth rather than raycasting the block state is what makes the focus match the
 * picture. A block-shape probe can only ever answer one block's voxel face, so it walks through entities,
 * particles, painting, water and the half of a block a fence or a stair actually occupies, and reports
 * whatever is behind them as what the frame shows.
 *
 * <p>The depth image also answers for a pixel whose picture is a reflection: the mirror's own distance is
 * what it holds by default, and with {@code dof.reflection-focus} on, the pixels whose mirror lobe carries
 * them hold the distance to the reflected image instead. This class neither knows nor cares which — one
 * distance per pixel is one answer, and the lens is placed on the pinhole ray either way.
 */
public final class RtFocusProbe {
    private static final String SHADER_DIR = "/bettercaustica/shaders/pipelines/dof_focus/";
    /** The meter's one sample: the harmonic-mean distance and the number of taps behind it. */
    private static final long METER_BYTES = 2L * Float.BYTES;
    /**
     * The closest range a tap may answer. Below it a distance is a numeric artifact rather than a surface:
     * nothing closer than the projection's near plane can be drawn, and one such texel would otherwise
     * dominate a mean taken in reciprocal metres.
     */
    private static final float NEAR_BLOCK = 0.05f;
    /** The centre window is an eighth of the shorter frame edge, so it covers the crosshair's target and its neighbours. */
    private static final int WINDOW_DIVISOR = 8;
    private static final int RING = 6;

    private static final class ReadbackSlot {
        final RtBuffer buffer;
        final RtGpuExecutor.TrackedGraphicsUse graphicsUse = new RtGpuExecutor.TrackedGraphicsUse();
        boolean armed;

        ReadbackSlot(RtBuffer buffer) {
            this.buffer = buffer;
        }
    }

    private final RtContext ctx;
    private final RtComputePipelineSupport.SingleSet gpu;
    private final RtBuffer meter;
    private final ReadbackSlot[] readbacks;
    private int readbackIndex;
    private ReadbackSlot pending;
    /** Whether this frame's copy was recorded, so a frame that skipped the meter never arms a slot. */
    private boolean copied;
    /** This frame's latched measurement in blocks, or 0 while no completed copy exists yet. */
    private float distanceBlocks;
    private long boundDepthView;
    private boolean destroyed;

    private RtFocusProbe(RtContext ctx, RtComputePipelineSupport.SingleSet gpu, RtBuffer meter,
                         ReadbackSlot[] readbacks) {
        this.ctx = ctx;
        this.gpu = gpu;
        this.meter = meter;
        this.readbacks = readbacks;
    }

    public static RtFocusProbe create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer binds =
                    VkDescriptorSetLayoutBinding.calloc(DOF_FOCUS_BINDING_COUNT, stack);
            binds.get(DOF_FOCUS_DEPTH).binding(DOF_FOCUS_DEPTH)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(DOF_FOCUS_METER).binding(DOF_FOCUS_METER)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);
            RtComputePipelineSupport.SingleSet gpu = RtComputePipelineSupport.createSingleSetPipeline(
                    ctx, stack, "dof focus meter", binds, poolSizes, DofFocusPushData.BYTE_SIZE,
                    SHADER_DIR + "main.comp.spv");

            RtBuffer meterBuffer = ctx.createBuffer(METER_BYTES, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, "dof focus meter");
            bindMeter(gpu.descriptorSet(), meterBuffer, ctx, stack);
            ReadbackSlot[] readbacks = new ReadbackSlot[RING];
            for (int i = 0; i < RING; i++) {
                readbacks[i] = new ReadbackSlot(ctx.createReadbackBuffer(METER_BYTES,
                        "dof focus readback " + i));
            }
            return new RtFocusProbe(ctx, gpu, meterBuffer, readbacks);
        }
    }

    private static void bindMeter(long descriptorSet, RtBuffer meter, RtContext ctx, MemoryStack stack) {
        VkDescriptorBufferInfo.Buffer info = VkDescriptorBufferInfo.calloc(1, stack);
        info.get(0).buffer(meter.handle).offset(0L).range(meter.size);
        VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
        write.get(0).sType$Default().dstSet(descriptorSet).dstBinding(DOF_FOCUS_METER)
                .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .pBufferInfo(info);
        VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
    }

    /**
     * Meters this frame's depth and copies the answer into this frame's readback slot. {@code depthView} is
     * the image the primary pass has just written, {@code farBlock} the distance the lens treats as
     * infinity. Call only while the lens is on: with it off nothing writes that image, and the meter would
     * average whatever the allocator left in it.
     */
    public void record(VkCommandBuffer cmd, MemoryStack stack, long depthView, int renderW, int renderH,
                       float farBlock) {
        if (depthView != boundDepthView) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).imageView(depthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().dstSet(gpu.descriptorSet()).dstBinding(DOF_FOCUS_DEPTH)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(info);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
            boundDepthView = depthView;
        }

        int window = Math.max(16, Math.min(renderW, renderH) / WINDOW_DIVISOR);
        int originX = Math.max(0, (renderW - window) / 2);
        int originY = Math.max(0, (renderH - window) / 2);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "dof focus meter")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline());
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipelineLayout(),
                    0, stack.longs(gpu.descriptorSet()), null);
            ByteBuffer push = stack.malloc(DofFocusPushData.BYTE_SIZE);
            new DofFocusPushData(originX, originY, window, window, NEAR_BLOCK, farBlock).write(push);
            VK10.vkCmdPushConstants(cmd, gpu.pipelineLayout(), VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, 1, 1, 1);
        }

        if (pending == null) {
            return;
        }
        // The meter's write has to reach the transfer before the copy reads it, and no later pass may see
        // the copy as a hazard, so this is the same explicit buffer barrier the exposure state uses.
        VkBufferMemoryBarrier.Buffer toTransfer = VkBufferMemoryBarrier.calloc(1, stack);
        toTransfer.get(0).sType$Default()
                .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK10.VK_ACCESS_TRANSFER_READ_BIT)
                .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .buffer(meter.handle).offset(0L).size(METER_BYTES);
        VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, toTransfer, null);
        VkBufferCopy.Buffer copy = VkBufferCopy.calloc(1, stack)
                .srcOffset(0L).dstOffset(0L).size(METER_BYTES);
        VK10.vkCmdCopyBuffer(cmd, meter.handle, pending.buffer.handle, copy);
        copied = true;
    }

    /** Arm the slot this frame copied into, once the command buffer has been accepted for submission. */
    public void markReadbackUse(RtGpuExecutor.GraphicsUse graphicsUse) {
        if (pending == null || !copied) {
            return;
        }
        pending.armed = true;
        pending.graphicsUse.mark(graphicsUse);
        pending = null;
        copied = false;
    }

    /**
     * Advance the ring and latch whichever slot has already completed, before this frame's push constants
     * are written. A slot nobody armed means the meter was switched off for a lap, so the last real
     * measurement is kept rather than dropped to zero.
     */
    public void beginFrame(RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter) {
        pending = null;
        copied = false;
        readbackIndex = (readbackIndex + 1) % readbacks.length;
        ReadbackSlot slot = readbacks[readbackIndex];
        graphicsUseWaiter.await(slot.graphicsUse);
        if (slot.armed) {
            slot.buffer.invalidate();
            ByteBuffer read = MemoryUtil.memByteBuffer(slot.buffer.mapped, (int) METER_BYTES)
                    .order(ByteOrder.nativeOrder());
            float distance = read.getFloat(0);
            if (Float.isFinite(distance) && distance > 0.0f) {
                distanceBlocks = distance;
            }
        }
        slot.armed = false;
        pending = slot;
    }

    /** The focal distance the last completed measurement answered, in blocks; 0 until there is one. */
    public float distanceBlocks() {
        return distanceBlocks;
    }

    /** Forget the history: used when the depth pair the meter reads is rebuilt. */
    public void reset() {
        for (ReadbackSlot slot : readbacks) {
            slot.armed = false;
        }
        distanceBlocks = 0.0f;
        pending = null;
        copied = false;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        gpu.destroy(ctx.vk());
        meter.destroy();
        for (ReadbackSlot slot : readbacks) {
            slot.buffer.destroy();
        }
        destroyed = true;
    }
}

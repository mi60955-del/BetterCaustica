package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.KHRPipelineExecutableProperties;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineExecutableInfoKHR;
import org.lwjgl.vulkan.VkPipelineExecutableInternalRepresentationKHR;
import org.lwjgl.vulkan.VkPipelineExecutablePropertiesKHR;
import org.lwjgl.vulkan.VkPipelineExecutableStatisticKHR;
import org.lwjgl.vulkan.VkPipelineInfoKHR;

import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.vkGetPipelineExecutableInternalRepresentationsKHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.vkGetPipelineExecutablePropertiesKHR;
import static org.lwjgl.vulkan.KHRPipelineExecutableProperties.vkGetPipelineExecutableStatisticsKHR;
import static org.lwjgl.vulkan.VK10.VK_INCOMPLETE;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;

/**
 * Diagnostic dump of what the driver compiled for a pipeline: one block per executable (stage), with
 * the shader name, the driver's statistics (register count, spills, and whatever else it reports) and
 * its internal representation (assembly), written to {@code rt-frame-stats/pipeline-stats.txt}.
 *
 * <p>Only reachable while {@code -Dcaustica.rt.pipelineStats=true}: the feature is enabled at device
 * creation and the pipelines ask for the capture bits at creation, both of which this class's caller
 * gates on {@link RtDeviceBringup#pipelineExecutableInfoEnabled()}. A driver that exposes the
 * extension but reports no statistics produces a block with an empty list, which is itself the answer.
 */
final class RtPipelineStats {

    private static boolean loggedFailure;

    private RtPipelineStats() {
    }

    /** Dumps every executable of {@code pipeline}; never throws. */
    static void dump(VkDevice vk, long pipeline, String label) {
        if (!RtDeviceBringup.pipelineExecutableInfoEnabled()) {
            return;
        }
        List<String> out = new ArrayList<>();
        out.add("=== " + label + " ===");
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineInfoKHR info = VkPipelineInfoKHR.calloc(stack).sType$Default().pipeline(pipeline);
            IntBuffer count = stack.mallocInt(1);
            int rc = vkGetPipelineExecutablePropertiesKHR(vk, info, count, null);
            if (rc != VK_SUCCESS || count.get(0) <= 0) {
                out.add("executables: none (rc=" + rc + ")");
                write(out);
                return;
            }
            int executables = count.get(0);
            VkPipelineExecutablePropertiesKHR.Buffer props = VkPipelineExecutablePropertiesKHR.calloc(executables, stack);
            for (int i = 0; i < executables; i++) {
                props.get(i).sType$Default();
            }
            rc = vkGetPipelineExecutablePropertiesKHR(vk, info, count, props);
            if (rc != VK_SUCCESS) {
                out.add("properties failed (rc=" + rc + ")");
                write(out);
                return;
            }
            for (int i = 0; i < Math.min(count.get(0), executables); i++) {
                VkPipelineExecutablePropertiesKHR prop = props.get(i);
                out.add("");
                out.add("-- [" + i + "] " + prop.nameString() + " (stages=0x"
                        + Integer.toHexString(prop.stages()) + ", subgroup=" + prop.subgroupSize() + ") ");
                out.add("   " + prop.descriptionString());
                VkPipelineExecutableInfoKHR exInfo = VkPipelineExecutableInfoKHR.calloc(stack).sType$Default()
                        .pipeline(pipeline).executableIndex(i);
                appendStatistics(vk, exInfo, stack, out);
                appendRepresentations(vk, exInfo, stack, out);
            }
        } catch (Throwable t) {
            if (!loggedFailure) {
                loggedFailure = true;
                CausticaMod.LOGGER.warn("RT pipeline statistics dump failed", t);
            }
            out.add("dump failed: " + t);
        }
        write(out);
    }

    private static void appendStatistics(VkDevice vk, VkPipelineExecutableInfoKHR exInfo,
                                         MemoryStack stack, List<String> out) {
        IntBuffer count = stack.mallocInt(1);
        int rc = vkGetPipelineExecutableStatisticsKHR(vk, exInfo, count, null);
        if (rc != VK_SUCCESS || count.get(0) <= 0) {
            out.add("   statistics: none (rc=" + rc + ")");
            return;
        }
        int n = count.get(0);
        VkPipelineExecutableStatisticKHR.Buffer stats = VkPipelineExecutableStatisticKHR.calloc(n, stack);
        for (int i = 0; i < n; i++) {
            stats.get(i).sType$Default();
        }
        rc = vkGetPipelineExecutableStatisticsKHR(vk, exInfo, count, stats);
        if (rc != VK_SUCCESS) {
            out.add("   statistics: query failed (rc=" + rc + ")");
            return;
        }
        for (int i = 0; i < Math.min(count.get(0), n); i++) {
            VkPipelineExecutableStatisticKHR stat = stats.get(i);
            String value;
            switch (stat.format()) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR -> value = Boolean.toString(stat.value().b32());
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR -> value = Long.toString(stat.value().i64());
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR -> value = Long.toUnsignedString(stat.value().u64());
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR -> value = Double.toString(stat.value().f64());
                default -> value = "?";
            }
            out.add("   " + stat.nameString() + " = " + value);
        }
    }

    private static void appendRepresentations(VkDevice vk, VkPipelineExecutableInfoKHR exInfo,
                                              MemoryStack stack, List<String> out) {
        IntBuffer count = stack.mallocInt(1);
        int rc = vkGetPipelineExecutableInternalRepresentationsKHR(vk, exInfo, count, null);
        if (rc != VK_SUCCESS || count.get(0) <= 0) {
            out.add("   representations: none (rc=" + rc + ")");
            return;
        }
        int n = count.get(0);
        VkPipelineExecutableInternalRepresentationKHR.Buffer reps =
                VkPipelineExecutableInternalRepresentationKHR.calloc(n, stack);
        // The driver fills pData in place, so every entry needs a buffer before the call. 1 MiB covers
        // every assembly listing this renderer produces; a short read comes back with a smaller dataSize.
        ByteBuffer data = ByteBuffer.allocateDirect(1 << 20);
        for (int i = 0; i < n; i++) {
            VkPipelineExecutableInternalRepresentationKHR rep = reps.get(i);
            rep.sType$Default();
            MemoryUtil.memPutLong(rep.address() + VkPipelineExecutableInternalRepresentationKHR.DATASIZE, data.capacity());
            MemoryUtil.memPutAddress(rep.address() + VkPipelineExecutableInternalRepresentationKHR.PDATA,
                    MemoryUtil.memAddress0(data));
        }
        rc = vkGetPipelineExecutableInternalRepresentationsKHR(vk, exInfo, count, reps);
        if (rc != VK_SUCCESS) {
            out.add("   representations: query failed (rc=" + rc + (rc == VK_INCOMPLETE ? ", incomplete" : "") + ")");
            return;
        }
        for (int i = 0; i < Math.min(count.get(0), n); i++) {
            VkPipelineExecutableInternalRepresentationKHR rep = reps.get(i);
            long size = rep.dataSize();
            out.add("   representation: " + rep.nameString() + " (text=" + rep.isText()
                    + ", bytes=" + size + ")");
            if (size > 0 && size <= data.capacity()) {
                byte[] bytes = new byte[(int) size];
                ByteBuffer view = data.duplicate();
                view.position(0);
                view.get(bytes);
                String text = new String(bytes, StandardCharsets.UTF_8);
                out.add(text);
            }
        }
    }

    private static void write(List<String> lines) {
        Path dir = Path.of("rt-frame-stats");
        try {
            Files.createDirectories(dir);
            Files.write(dir.resolve("pipeline-stats.txt"), lines, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not write rt-frame-stats/pipeline-stats.txt", e);
        }
    }
}

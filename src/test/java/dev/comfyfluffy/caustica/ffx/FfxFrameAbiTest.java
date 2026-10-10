package dev.comfyfluffy.caustica.ffx;

import org.junit.jupiter.api.Test;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the three copies of the {@code ffxshim_frame} ABI against each other: the
 * {@link FfxLibrary.FrameRequest} record, the Java downcall
 * {@linkplain FfxLibrary#frameDescriptor() descriptor}, and the C signature in
 * {@code native/ffx_shim/ffx_shim.cpp}. Any of the three drifting (reordered field, changed type,
 * added/removed parameter) fails here instead of surfacing as a runtime
 * {@code WrongMethodTypeException} or silently misread arguments on the native side.
 */
final class FfxFrameAbiTest {
    /** Index of the first float parameter in the C signature ({@code jitterX}). */
    private static final int JITTER_ARG_INDEX = 7;
    private static final int JITTER_ARG_COUNT = 2;

    @Test
    void frameDescriptorMatchesFrameRequestRecord() {
        List<MemoryLayout> expected = recordLayouts();
        FunctionDescriptor descriptor = FfxLibrary.frameDescriptor();
        List<MemoryLayout> actual = descriptor.argumentLayouts();
        assertEquals(expected.size(), actual.size(),
                "ffxshim_frame argument count drifted from FrameRequest (record components + "
                        + JITTER_ARG_COUNT + " jitter floats)");
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(((ValueLayout) expected.get(i)).carrier(),
                    ((ValueLayout) actual.get(i)).carrier(),
                    "ffxshim_frame argument " + i + " drifted from FrameRequest");
        }
    }

    @Test
    void frameDescriptorMatchesShimSignature() {
        List<ShimParam> params = parseShimFrameParams();
        FunctionDescriptor descriptor = FfxLibrary.frameDescriptor();
        List<MemoryLayout> actual = descriptor.argumentLayouts();
        assertEquals(params.size(), actual.size(), "shim parameter count drifted from the Java descriptor");
        for (int i = 0; i < params.size(); i++) {
            assertEquals(params.get(i).carrier(), ((ValueLayout) actual.get(i)).carrier(),
                    "shim parameter " + i + " (" + params.get(i).name() + ") drifted from the Java descriptor");
        }
    }

    @Test
    void shimSignatureMatchesFrameRequestRecord() {
        List<ShimParam> params = parseShimFrameParams();
        RecordComponent[] components = FfxLibrary.FrameRequest.class.getRecordComponents();
        assertEquals(components.length + JITTER_ARG_COUNT, params.size(),
                "shim parameter count drifted from FrameRequest");
        for (int i = 0; i < components.length; i++) {
            int paramIndex = i < JITTER_ARG_INDEX ? i : i + JITTER_ARG_COUNT;
            ShimParam param = params.get(paramIndex);
            assertEquals(recordCarrier(components[i]), param.carrier(),
                    "FrameRequest component " + components[i].getName() + " disagrees with shim parameter "
                            + param.name());
            assertEquals(components[i].getName(), param.javaName(),
                    "shim parameter name drifted from FrameRequest (position " + paramIndex + ")");
        }
    }

    /**
     * The descriptor layout sequence the record implies: one native argument per component in
     * declaration order, with the two C-side jitter floats inserted after {@code displayHeight}.
     */
    private static List<MemoryLayout> recordLayouts() {
        RecordComponent[] components = FfxLibrary.FrameRequest.class.getRecordComponents();
        List<MemoryLayout> layouts = new ArrayList<>(components.length + JITTER_ARG_COUNT);
        for (int i = 0; i < components.length; i++) {
            if (i == JITTER_ARG_INDEX) {
                layouts.add(ValueLayout.JAVA_FLOAT);
                layouts.add(ValueLayout.JAVA_FLOAT);
            }
            layouts.add(layoutFor(components[i]));
        }
        return layouts;
    }

    /**
     * The native layout a record component binds to. {@code commandBuffer} is the one component that is
     * pointer-wrapped at the call ({@code MemorySegment.ofAddress}) because the shim takes it as
     * {@code void*}; every other {@code long} (the VkImage handles) is a {@code uint64_t} on the C side
     * and passes as a raw integer.
     */
    private static MemoryLayout layoutFor(RecordComponent component) {
        Class<?> type = component.getType();
        if (type == MemorySegment.class) return ValueLayout.ADDRESS;
        if (type == long.class) {
            return component.getName().equals("commandBuffer") ? ValueLayout.ADDRESS : ValueLayout.JAVA_LONG;
        }
        if (type == int.class || type == boolean.class) return ValueLayout.JAVA_INT;
        if (type == float.class) return ValueLayout.JAVA_FLOAT;
        throw new AssertionError("FrameRequest component type has no native carrier: " + type);
    }

    private static Class<?> recordCarrier(RecordComponent component) {
        Class<?> type = component.getType();
        if (type == MemorySegment.class) return MemorySegment.class;
        if (type == long.class) {
            return component.getName().equals("commandBuffer") ? MemorySegment.class : long.class;
        }
        // Booleans are passed as C ints (0/1): reset() ? 1 : 0 at the downcall.
        if (type == boolean.class || type == int.class) return int.class;
        if (type == float.class) return float.class;
        throw new AssertionError("FrameRequest component type has no native carrier: " + type);
    }

    /** Every {@code ffxshim_frame} parameter parsed from the shim source, in declaration order. */
    private static List<ShimParam> parseShimFrameParams() {
        Path shim = locateShimSource();
        String source;
        try {
            source = Files.readString(shim);
        } catch (java.io.IOException e) {
            throw new AssertionError("Cannot read " + shim, e);
        }
        Matcher start = Pattern.compile("\\bffxshim_frame\\s*\\(").matcher(source);
        assertTrue(start.find(), "ffxshim_frame definition not found in " + shim);
        int open = source.indexOf('(', start.start());
        int close = source.indexOf(')', open);
        assertTrue(close > open, "Unbalanced ffxshim_frame parameter list in " + shim);
        List<ShimParam> params = new ArrayList<>();
        for (String part : source.substring(open + 1, close).split(",")) {
            String[] tokens = part.trim().split("\\s+");
            assertTrue(tokens.length >= 2, "Cannot parse ffxshim_frame parameter '" + part + "'");
            params.add(new ShimParam(String.join(" ", java.util.Arrays.copyOfRange(tokens, 0, tokens.length - 1)),
                    tokens[tokens.length - 1]));
        }
        return params;
    }

    private static Path locateShimSource() {
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            Path candidate = dir.resolve("native").resolve("ffx_shim").resolve("ffx_shim.cpp");
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        throw new AssertionError("native/ffx_shim/ffx_shim.cpp not found above the working directory");
    }

    /** One parsed C parameter: the shim-side type string plus the Java-side carrier it must bind to. */
    private record ShimParam(String type, String name) {
        Class<?> carrier() {
            return switch (type) {
                case "void*", "const float*" -> MemorySegment.class;
                case "uint64_t" -> long.class;
                case "int32_t", "uint32_t" -> int.class;
                case "float" -> float.class;
                default -> throw new AssertionError("Unknown ffxshim_frame parameter type: " + type);
            };
        }

        /** The record-side name: the shim qualifies format parameters with the vendor prefix. */
        String javaName() {
            return name.replace("VkFormat", "Format");
        }
    }
}

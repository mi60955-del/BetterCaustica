package dev.comfyfluffy.caustica.ffx;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * FFM bindings for the FidelityFX shim ({@code ffxshim.dll}).
 *
 * <p>The shim wraps AMD's unified C API behind a flat one (see {@code native/ffx_shim/ffx_shim.cpp}).
 * All six FFX descriptor structs, including the nested {@code FfxApiResource} values, live in the
 * shim, and Java passes only primitives and raw Vulkan handles as {@code long}. Going through a
 * shim rather than binding {@code amd_fidelityfx_vk.dll} directly is deliberate: a Panama layout
 * that disagrees with a C struct by four bytes returns OK and renders garbage, whereas the shim's
 * layout is checked against AMD's own headers by the C++ compiler.
 *
 * <p>Every {@code ffxshim_*} status is an {@code FfxApiReturnCodes} value where {@link #OK} is
 * success.
 */
public final class FfxLibrary {
    /** FFX_API_RETURN_OK. */
    public static final int OK = 0;

    /** The shim's {@code g_lastMessage} buffer size; see {@code native/ffx_shim/ffx_shim.cpp}. */
    private static final int MESSAGE_CAPACITY = 512;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final ValueLayout.OfInt C_INT = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong C_LONG = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfFloat C_FLOAT = ValueLayout.JAVA_FLOAT;
    private static final AddressLayout C_PTR = ValueLayout.ADDRESS;
    // One native argument per FrameRequest component, plus the two C-side jitter floats.
    private static final int FRAME_ARG_COUNT = FrameRequest.class.getRecordComponents().length + 2;

    private final MethodHandle available;
    private final MethodHandle load;
    private final MethodHandle create;
    private final MethodHandle frameSpread;
    private final MethodHandle destroy;
    private final MethodHandle unload;
    private final MethodHandle lastMessage;
    private final MethodHandle lastResult;
    private final MethodHandle version;

    /** Version name of the provider that was selected, filled in by {@link #create}. */
    private volatile String providerVersion = "";

    private FfxLibrary(SymbolLookup lookup) {
        available = bind(lookup, "ffxshim_available", FunctionDescriptor.of(C_INT));
        load = bind(lookup, "ffxshim_load", FunctionDescriptor.of(C_INT, C_PTR));
        // int ffxshim_create(device, physicalDevice, getDeviceProcAddr,
        //   displayW, displayH, maxRenderW, maxRenderH, backbufferFmt, hudlessFmt,
        //   depthInverted, depthInfinite, hdr, displayResMotionVectors, jitteredMotionVectors,
        //   debugChecking, uint64* outVramBytes, char* versionOut, int versionOutLength)
        // Eighteen parameters: three handles, twelve scalars, then the two out-parameters and the
        // capacity that goes with the second one.
        create = bind(lookup, "ffxshim_create", FunctionDescriptor.of(C_INT,
                C_PTR, C_PTR, C_PTR,
                C_INT, C_INT, C_INT, C_INT, C_INT, C_INT,
                C_INT, C_INT, C_INT, C_INT, C_INT, C_INT,
                C_PTR, C_PTR, C_INT));
        // int ffxshim_frame(cmd, frameId, reset, renderW, renderH, displayW, displayH,
        //   jitterX, jitterY, mvScaleX, mvScaleY, frameTimeDeltaMs,
        //   cameraNear, cameraFar, fovY, viewSpaceToMeters,
        //   cameraPosition, cameraUp, cameraRight, cameraForward,
        //   depthImage, depthFormat, motionImage, motionFormat, presentColorImage,
        //   presentColorFormat, hudlessImage, hudlessFormat, hudlessW, hudlessH,
        //   outputImage, outputFormat, debugView)
        MethodHandle frame = bind(lookup, "ffxshim_frame", frameDescriptor());
        frameSpread = frame.asSpreader(Object[].class, FRAME_ARG_COUNT);
        destroy = bind(lookup, "ffxshim_destroy", FunctionDescriptor.ofVoid());
        unload = bind(lookup, "ffxshim_unload", FunctionDescriptor.ofVoid());
        lastMessage = bind(lookup, "ffxshim_last_message", FunctionDescriptor.of(C_PTR));
        lastResult = bind(lookup, "ffxshim_last_result", FunctionDescriptor.of(C_INT));
        version = bind(lookup, "ffxshim_version", FunctionDescriptor.of(C_PTR));
    }

    public static FfxLibrary load(Path dll) {
        return new FfxLibrary(SymbolLookup.libraryLookup(dll, Arena.global()));
    }

    private static MethodHandle bind(SymbolLookup lookup, String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(
                lookup.find(name).orElseThrow(() -> new IllegalStateException("ffxshim missing export " + name)),
                descriptor);
    }

    /**
     * One generated frame. The field names, and so the units, are AMD's.
     *
     * @param frameId            must rise by exactly one per rendered frame; any other step makes
     *                           FSR reset its history, which costs quality silently
     * @param motionVectorScaleX motionVectorScaleY: 1.0 when the vectors are in render pixels,
     *                           which is what {@code gMotion} holds. The runtime divides the scale
     *                           by the motion vector target size and adds the result to a UV, so
     *                           pixels with scale 1 is what yields a UV displacement
     * @param hudlessImage       the frame before the UI was composited, or 0 together with format 0
     *                           to let FSR separate UI by itself
     */
    public record FrameRequest(
            long commandBuffer,
            long frameId,
            boolean reset,
            int renderWidth,
            int renderHeight,
            int displayWidth,
            int displayHeight,
            float motionVectorScaleX,
            float motionVectorScaleY,
            float frameTimeDeltaMs,
            float cameraNear,
            float cameraFar,
            float cameraFovAngleVertical,
            float viewSpaceToMetersFactor,
            MemorySegment cameraPosition,
            MemorySegment cameraUp,
            MemorySegment cameraRight,
            MemorySegment cameraForward,
            long depthImage,
            int depthFormat,
            long motionImage,
            int motionFormat,
            long presentColorImage,
            int presentColorFormat,
            long hudlessImage,
            int hudlessFormat,
            int hudlessWidth,
            int hudlessHeight,
            long outputImage,
            int outputFormat,
            boolean debugView) {
    }

    public boolean available() {
        try {
            return ((int) available.invokeExact()) != 0;
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_available failed", t);
        }
    }

    /**
     * Load {@code amd_fidelityfx_vk.dll}. {@code dllPath} may be a full path, a directory to look
     * in, or null for "next to ffxshim.dll".
     */
    public int load(String dllPath) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment path = dllPath == null || dllPath.isBlank()
                    ? MemorySegment.NULL : arena.allocateFrom(dllPath);
            return (int) load.invokeExact(path);
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_load failed", t);
        }
    }

    /**
     * Create the frame-generation context, which is a process-wide singleton in the shim. Pass 0
     * for {@code hudlessFormat} when there is no HUD-less input.
     *
     * @param outVramBytes receives the SDK's own device-memory estimate, or pass null
     */
    public int create(long vkDevice, long vkPhysicalDevice, long vkGetDeviceProcAddr,
                      int displayWidth, int displayHeight, int maxRenderWidth, int maxRenderHeight,
                      int backbufferFormat, int hudlessFormat,
                      boolean depthInverted, boolean depthInfinite, boolean hdrColor,
                      boolean displayResolutionMotionVectors, boolean jitteredMotionVectors,
                      boolean debugChecking, MemorySegment outVramBytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment versionOut = arena.allocate(64);
            // Both of these have to be locals rather than inline conditional expressions: a
            // reference conditional passed straight to invokeExact is typed by its target type,
            // which is Object, and invokeExact matches on static type, so the call fails with
            // WrongMethodTypeException before the native function is reached.
            MemorySegment device = MemorySegment.ofAddress(vkDevice);
            MemorySegment physicalDevice = MemorySegment.ofAddress(vkPhysicalDevice);
            MemorySegment procAddr = MemorySegment.ofAddress(vkGetDeviceProcAddr);
            MemorySegment vram = outVramBytes == null ? MemorySegment.NULL : outVramBytes;
            int rc = (int) create.invokeExact(
                    device, physicalDevice, procAddr,
                    displayWidth, displayHeight, maxRenderWidth, maxRenderHeight,
                    backbufferFormat, hudlessFormat,
                    depthInverted ? 1 : 0, depthInfinite ? 1 : 0, hdrColor ? 1 : 0,
                    displayResolutionMotionVectors ? 1 : 0, jitteredMotionVectors ? 1 : 0,
                    debugChecking ? 1 : 0,
                    vram, versionOut, 64);
            // Read the version defensively: the context now exists whether or not this succeeds,
            // and letting a bookkeeping read throw would report a successful create as a failure
            // while leaving the shim holding a context it will not hand out twice.
            try {
                providerVersion = readFixed(versionOut, 64);
            } catch (RuntimeException ignored) {
                providerVersion = "";
            }
            return rc;
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_create failed", t);
        }
    }

    /**
     * The downcall descriptor of {@code ffxshim_frame}. Package-visible so the ABI consistency test can
     * pin it against both the {@link FrameRequest} record and the shim's C signature.
     */
    static FunctionDescriptor frameDescriptor() {
        return FunctionDescriptor.of(C_INT,
                C_PTR, C_LONG, C_INT, C_INT, C_INT, C_INT, C_INT,
                C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT,
                C_PTR, C_PTR, C_PTR, C_PTR,
                C_LONG, C_INT, C_LONG, C_INT, C_LONG, C_INT, C_LONG, C_INT, C_INT, C_INT,
                C_LONG, C_INT, C_INT);
    }

    public int frame(FrameRequest request) {
        try {
            return (int) frameSpread.invokeExact(frameArgs(request));
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_frame failed", t);
        }
    }

    /**
     * The native argument list of {@code ffxshim_frame}, derived from the record so the downcall cannot
     * drift from it: the components in declaration order, with the two C-side jitter floats inserted
     * after {@code displayHeight} (always zero — Caustica jitters the primary ray itself and reports
     * jitter-free motion vectors, so the runtime must not cancel a jitter that was never applied).
     * Booleans are passed as C ints, and buffer addresses are wrapped without copying.
     */
    private static Object[] frameArgs(FrameRequest r) {
        Object[] args = new Object[FRAME_ARG_COUNT];
        int i = 0;
        args[i++] = MemorySegment.ofAddress(r.commandBuffer());
        args[i++] = r.frameId();
        args[i++] = r.reset() ? 1 : 0;
        args[i++] = r.renderWidth();
        args[i++] = r.renderHeight();
        args[i++] = r.displayWidth();
        args[i++] = r.displayHeight();
        args[i++] = 0.0f; // jitterX
        args[i++] = 0.0f; // jitterY
        args[i++] = r.motionVectorScaleX();
        args[i++] = r.motionVectorScaleY();
        args[i++] = r.frameTimeDeltaMs();
        args[i++] = r.cameraNear();
        args[i++] = r.cameraFar();
        args[i++] = r.cameraFovAngleVertical();
        args[i++] = r.viewSpaceToMetersFactor();
        args[i++] = r.cameraPosition();
        args[i++] = r.cameraUp();
        args[i++] = r.cameraRight();
        args[i++] = r.cameraForward();
        args[i++] = r.depthImage();
        args[i++] = r.depthFormat();
        args[i++] = r.motionImage();
        args[i++] = r.motionFormat();
        args[i++] = r.presentColorImage();
        args[i++] = r.presentColorFormat();
        args[i++] = r.hudlessImage();
        args[i++] = r.hudlessFormat();
        args[i++] = r.hudlessWidth();
        args[i++] = r.hudlessHeight();
        args[i++] = r.outputImage();
        args[i++] = r.outputFormat();
        args[i++] = r.debugView() ? 1 : 0;
        return args;
    }

    public void destroy() {
        try {
            destroy.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_destroy failed", t);
        }
    }

    public void unload() {
        try {
            unload.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("ffxshim_unload failed", t);
        }
    }

    /** The SDK's own last diagnostic line, or an empty string when it has not spoken. */
    public String lastMessage() {
        try {
            return readString((MemorySegment) lastMessage.invokeExact());
        } catch (Throwable t) {
            return "(ffxshim_last_message failed: " + t + ")";
        }
    }

    /**
     * The most recent non-OK return code the shim recorded, for failure logs: some SDK failure
     * paths only reach the message callback, and this names a return code even when the dispatch
     * entry itself already returned one.
     */
    public int lastResult() {
        try {
            return (int) lastResult.invokeExact();
        } catch (Throwable t) {
            return Integer.MIN_VALUE;
        }
    }

    public String version() {
        if (!providerVersion.isEmpty()) {
            return providerVersion;
        }
        try {
            return readString((MemorySegment) version.invokeExact());
        } catch (Throwable t) {
            return "?";
        }
    }

    /**
     * A NUL-terminated string the SDK owns: the downcall hands back a zero-length segment, so
     * widening it is both legal and the only way to find the terminator.
     */
    private static String readString(MemorySegment segment) {
        if (segment == null || segment.equals(MemorySegment.NULL)) {
            return "";
        }
        return segment.reinterpret(MESSAGE_CAPACITY).getString(0);
    }

    /**
     * A NUL-terminated string inside a buffer this class allocated and therefore already knows the
     * size of. {@link MemorySegment#reinterpret(long)} cannot grow a non-empty segment, so reading
     * a caller-owned buffer through {@link #readString} throws instead of returning the text.
     */
    private static String readFixed(MemorySegment segment, int capacity) {
        int length = 0;
        while (length < capacity && segment.get(ValueLayout.JAVA_BYTE, length) != 0) {
            length++;
        }
        return new String(segment.asSlice(0, length).toArray(ValueLayout.JAVA_BYTE),
                java.nio.charset.StandardCharsets.UTF_8);
    }
}

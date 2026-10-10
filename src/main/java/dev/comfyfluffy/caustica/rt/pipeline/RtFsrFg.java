package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.systems.GpuDeviceBackend;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.ffx.FfxLibrary;
import dev.comfyfluffy.caustica.ffx.FfxRuntime;
import dev.comfyfluffy.caustica.mixin.GpuDeviceAccessor;
import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.RtContext;

import org.joml.Matrix4fc;
import org.lwjgl.vulkan.VK10;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * AMD FidelityFX Frame Generation backend.
 *
 * <p>Unlike DLSS-FG this needs no per-vendor feature handle negotiated through a runtime: one
 * process-wide FFX context covers it, so this class owns the context lifecycle and the frame
 * counter and leaves the swapchain to {@code RtFramePresenter}. The context is created with
 * {@code FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY}, which is what lets Minecraft keep
 * its own {@code VkSwapchainKHR}.
 *
 * <p>Two things here are load-bearing and easy to get wrong silently, so they are stated in terms
 * of what the SDK does rather than what its documentation says:
 * <ul>
 *   <li>the motion vector scale is 1.0, because {@code gMotion} is already in render pixels and
 *       the runtime divides the scale by the vector target size before adding it to a UV;</li>
 *   <li>{@code frameId} must rise by exactly one per generated frame. A gap is not an error - FSR
 *       quietly drops its history and the frame loses temporal stability - so the counter is owned
 *       here and only advances when a frame was actually dispatched.</li>
 * </ul>
 */
public final class RtFsrFg {
    public static final RtFsrFg INSTANCE = new RtFsrFg();

    /** One block holding the four world-space camera triples FSR asks for, in its order. */
    private static final int CAMERA_FLOATS = 12;

    private FfxLibrary lib;
    private boolean contextUp;
    private boolean failed;
    private boolean probed;
    private boolean available;

    private int displayWidth = -1;
    private int displayHeight = -1;
    private int renderWidth = -1;
    private int renderHeight = -1;
    private int backbufferFormat = Integer.MIN_VALUE;

    /**
     * A window drag changes the context shape every frame, and rebuilding the FFX context at that
     * cadence would build pipelines and allocate device memory per frame. A changed shape is only
     * rebuilt once it has held steady for {@link #RESIZE_SETTLE_FRAMES} consecutive calls; until
     * then {@link #ensureContext} returns false and the caller presents the real frame again.
     */
    private static final int RESIZE_SETTLE_FRAMES = 8;
    private int pendingDisplayWidth = -1;
    private int pendingDisplayHeight = -1;
    private int pendingRenderWidth = -1;
    private int pendingRenderHeight = -1;
    private int pendingFormat = Integer.MIN_VALUE;
    private int settleFramesRemaining;

    /**
     * One failed dispatch is usually transient, while a context that can never dispatch fails
     * identically every frame; only a streak is treated as fatal so a single bad frame does not
     * disable frame generation for the rest of the session.
     */
    private static final int FRAME_FAILURE_LATCH_THRESHOLD = 3;
    private int consecutiveFrameFailures;

    private boolean loggedProjectionFallback;

    private long frameId;
    private long framesDispatched;
    private long lastFrameNanos;
    private long vramBytes;

    private RtFsrFg() {
    }

    /** Whether this backend was chosen in settings. */
    public static boolean selected() {
        return CausticaConfig.Rt.Fg.FSR.equals(CausticaConfig.Rt.Fg.BACKEND.get());
    }

    /** Whether the probe succeeded and no failure has latched; drives the extra-present path. */
    public boolean isAvailable() {
        return available && !failed;
    }

    public boolean isReady() {
        return contextUp && !failed;
    }

    /** FSR 3.1.x generates one frame per dispatch, so extra generated frames are not on offer. */
    public static int maxGeneratedFramesPerRender() {
        return 1;
    }

    /**
     * Load the shim and ask it whether the SDK resolved, once per session. Safe to call every
     * tick while FSR is selected; the answer is cached because a missing DLL does not become
     * present, and the log line is written once.
     */
    public void probeAvailabilityOnce() {
        if (probed || failed) {
            return;
        }
        if (!(device() instanceof VulkanDevice vk)) {
            return;
        }
        FfxLibrary l = FfxRuntime.INSTANCE.acquire(vk);
        if (l == null) {
            return; // not up yet; try again next tick
        }
        probed = true;
        lib = l;
        available = l.available();
        CausticaMod.LOGGER.info("FSR Frame Generation available: {}", available);
    }

    /**
     * Bring the FFX context up for this display/render size and backbuffer format, recreating it
     * when any of them changes. Runs outside the recorded frame (its own submit and wait), because
     * context creation allocates device memory and builds pipelines.
     */
    public boolean ensureContext(int displayW, int displayH, int renderW, int renderH, int vkFormat) {
        if (failed) {
            return false;
        }
        if (!(device() instanceof VulkanDevice device)) {
            return false;
        }
        try {
            if (lib == null) {
                probeAvailabilityOnce();
            }
            if (lib == null || !available) {
                throw new IllegalStateException("FidelityFX is not available on this system");
            }
            if (contextUp && displayWidth == displayW && displayHeight == displayH
                    && renderWidth == renderW && renderHeight == renderH && backbufferFormat == vkFormat) {
                return true;
            }
            // An existing context is only rebuilt once the requested shape has stopped changing, so
            // dragging the window costs duplication instead of a rebuild per frame. A first creation
            // is not debounced: there is nothing to churn against.
            if (contextUp && !resizeSettled(displayW, displayH, renderW, renderH, vkFormat)) {
                return false;
            }
            RtContext ctx = RtContext.currentOrNull();
            if (ctx != null) {
                ctx.waitIdle();
            } else {
                VK10.vkDeviceWaitIdle(device.vkDevice());
            }
            if (contextUp) {
                lib.destroy();
                contextUp = false;
            }
            boolean hdr = vkFormat == VK10.VK_FORMAT_R16G16B16A16_SFLOAT;
            int rc;
            long vram;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment vramOut = arena.allocate(8);
                rc = lib.create(device.vkDevice().address(),
                        device.vkDevice().getPhysicalDevice().address(),
                        FfxRuntime.deviceProcAddr(device),
                        displayW, displayH, renderW, renderH,
                        vkFormat, hdr ? vkFormat : VK10.VK_FORMAT_R8G8B8A8_UNORM,
                        true /* reversed-Z */, false /* finite far plane */, hdr,
                        false /* vectors are at render resolution */, false /* vectors carry no jitter */,
                        true /* ask the SDK to validate its inputs */, vramOut);
                vram = vramOut.get(ValueLayout.JAVA_LONG, 0);
            }
            if (rc != FfxLibrary.OK) {
                throw new IllegalStateException("ffxshim_create returned " + rc + " " + lib.lastMessage());
            }
            contextUp = true;
            vramBytes = vram;
            displayWidth = displayW;
            displayHeight = displayH;
            renderWidth = renderW;
            renderHeight = renderH;
            backbufferFormat = vkFormat;
            // A fresh context has no history, and the frame sequence restarts with it.
            frameId = 0;
            lastFrameNanos = 0;
            consecutiveFrameFailures = 0;
            CausticaMod.LOGGER.info("FSR-FG context created: display {}x{}, render {}x{}, format {}, "
                            + "provider {}, {} bytes of device memory",
                    displayW, displayH, renderW, renderH, vkFormat, lib.version(), vramBytes);
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("FSR-FG setup failed; frame generation disabled", t);
            return false;
        }
    }

    /**
     * Tracks whether a changed context shape has held steady for {@link #RESIZE_SETTLE_FRAMES}
     * consecutive calls. Only called while a context is already up and its shape no longer matches.
     */
    private boolean resizeSettled(int displayW, int displayH, int renderW, int renderH, int vkFormat) {
        if (pendingDisplayWidth != displayW || pendingDisplayHeight != displayH
                || pendingRenderWidth != renderW || pendingRenderHeight != renderH
                || pendingFormat != vkFormat) {
            pendingDisplayWidth = displayW;
            pendingDisplayHeight = displayH;
            pendingRenderWidth = renderW;
            pendingRenderHeight = renderH;
            pendingFormat = vkFormat;
            settleFramesRemaining = RESIZE_SETTLE_FRAMES;
            return false;
        }
        if (settleFramesRemaining > 0) {
            settleFramesRemaining--;
            return false;
        }
        return true;
    }

    /**
     * Record one generated frame into {@code commandBuffer}: FSR's prepare and generate dispatches
     * for the pair ({@code presentColor}, {@code hudless}) -&gt; {@code output}.
     *
     * <p>{@code hudless} is the frame before the user interface was composited and may be 0, which
     * leaves FSR to separate screen-fixed content by itself. Passing it is what keeps the HUD from
     * being smeared into the world motion.
     */
    public boolean frame(long commandBuffer,
                         long presentColorImage, int presentColorFormat,
                         long hudlessImage, int hudlessFormat, int hudlessWidth, int hudlessHeight,
                         long outputImage, int outputFormat,
                         long depthImage, int depthFormat,
                         long motionImage, int motionFormat,
                         boolean reset) {
        if (!isReady() || lib == null) {
            return false;
        }
        try (Arena arena = Arena.ofConfined()) {
            CameraFrame camera = readCamera(arena);
            long now = System.nanoTime();
            float frameTimeDeltaMs = lastFrameNanos == 0 ? 16.7f : (now - lastFrameNanos) / 1_000_000.0f;
            lastFrameNanos = now;

            int rc = lib.frame(new FfxLibrary.FrameRequest(
                    commandBuffer, frameId + 1, reset || frameId == 0,
                    renderWidth, renderHeight, displayWidth, displayHeight,
                    1.0f, 1.0f,
                    Math.clamp(frameTimeDeltaMs, 0.5f, 500.0f),
                    camera.near, camera.far, camera.fovYRadians, 1.0f,
                    camera.position, camera.up, camera.right, camera.forward,
                    depthImage, depthFormat,
                    motionImage, motionFormat,
                    presentColorImage, presentColorFormat,
                    hudlessImage, hudlessFormat, hudlessWidth, hudlessHeight,
                    outputImage, outputFormat,
                    CausticaConfig.Rt.Fg.DEBUG_VIEW.value() != 0));
            if (rc != FfxLibrary.OK) {
                throw new IllegalStateException("ffxshim_frame returned " + rc + " " + lib.lastMessage());
            }
            // Only a dispatch that happened counts towards the sequence FSR expects.
            frameId++;
            framesDispatched++;
            consecutiveFrameFailures = 0;
            // What FSR is actually being told. Two of the reported symptoms turn on inputs that are
            // derived rather than configured - whether a HUD-less source reached the dispatch at all
            // (without it FSR interpolates the presented image, HUD included), and the frustum scalars
            // reverse-engineered from the projection matrix, which drive every depth-based decision
            // the algorithm makes.
            if (framesDispatched == 1 || framesDispatched % 300 == 0) {
                CausticaMod.LOGGER.info("FSR-FG frame {}: display {}x{} render {}x{} hudlessAttached={} "
                                + "near={} far={} fovYdeg={} frameTimeMs={} basis r=({},{},{}) u=({},{},{}) f=({},{},{})",
                        frameId, displayWidth, displayHeight, renderWidth, renderHeight,
                        hudlessImage != 0L,
                        fmt(camera.near), fmt(camera.far), fmt((float) Math.toDegrees(camera.fovYRadians)),
                        fmt(Math.clamp(frameTimeDeltaMs, 0.5f, 500.0f)),
                        fmt(camera.right.get(ValueLayout.JAVA_FLOAT, 0)),
                        fmt(camera.right.get(ValueLayout.JAVA_FLOAT, 4)),
                        fmt(camera.right.get(ValueLayout.JAVA_FLOAT, 8)),
                        fmt(camera.up.get(ValueLayout.JAVA_FLOAT, 0)),
                        fmt(camera.up.get(ValueLayout.JAVA_FLOAT, 4)),
                        fmt(camera.up.get(ValueLayout.JAVA_FLOAT, 8)),
                        fmt(camera.forward.get(ValueLayout.JAVA_FLOAT, 0)),
                        fmt(camera.forward.get(ValueLayout.JAVA_FLOAT, 4)),
                        fmt(camera.forward.get(ValueLayout.JAVA_FLOAT, 8)));
            }
            return true;
        } catch (Throwable t) {
            consecutiveFrameFailures++;
            if (consecutiveFrameFailures >= FRAME_FAILURE_LATCH_THRESHOLD) {
                failed = true;
                CausticaMod.LOGGER.error("FSR-FG frame failed {} times in a row; frame generation disabled",
                        consecutiveFrameFailures, t);
            } else {
                CausticaMod.LOGGER.warn("FSR-FG frame failed ({}/{} consecutive); retrying",
                        consecutiveFrameFailures, FRAME_FAILURE_LATCH_THRESHOLD, t);
            }
            return false;
        }
    }

    /**
     * Release the context and reset the per-context counters. Failure latches are deliberately not
     * touched here: {@link #resetFailure()} owns them, and full teardown calls the two together.
     * The SDK itself is unloaded by {@link FfxRuntime} at device teardown.
     */
    public void destroy() {
        if (lib != null && contextUp) {
            try {
                lib.destroy();
            } catch (Throwable t) {
                CausticaMod.LOGGER.warn("FSR-FG context release failed", t);
            }
        }
        contextUp = false;
        frameId = 0;
        framesDispatched = 0;
        lastFrameNanos = 0;
        consecutiveFrameFailures = 0;
        settleFramesRemaining = 0;
        displayWidth = -1;
        displayHeight = -1;
        renderWidth = -1;
        renderHeight = -1;
        backbufferFormat = Integer.MIN_VALUE;
    }

    /**
     * Drop a latched failure so the next render-state invalidation (F3+A, dimension change) gets
     * another attempt. No-op while healthy, so an invalidation never rebuilds a working context.
     * On a real reset the context is released too: the failure that latched may have left it in a
     * state the SDK rejects, and {@link #ensureContext} recreates it from scratch.
     */
    public void resetFailure() {
        if (!failed) {
            return;
        }
        destroy();
        failed = false;
        probed = false;
        available = false;
        lib = null;
    }

    /**
     * Release the context when the frame-generation backend setting no longer names FSR, so
     * switching to DLSS does not leave this context holding device memory. No-op otherwise.
     */
    public void destroyUnlessSelected() {
        if (contextUp && !selected()) {
            destroy();
        }
    }

    // --- camera ------------------------------------------------------------------------

    /** The four triples FSR wants, already in the off-heap layout the shim reads. */
    private static final class CameraFrame {
        MemorySegment position;
        MemorySegment up;
        MemorySegment right;
        MemorySegment forward;
        float fovYRadians = 1.0f;
        float near = 0.05f;
        float far = 1000.0f;
    }

    /**
     * Derive the frame's camera from the matrices the composite already captured.
     *
     * <p>{@code frameViewRotation} maps world to view, so its rows are the camera axes in world
     * space, and the view axis is negated because the camera looks down -z. The frustum scalars
     * come out of the projection rather than being re-read from settings: for a reversed-Z,
     * zero-to-one perspective matrix the two depth entries satisfy {@code far = m32 / m22} and
     * {@code near = m32 / (1 + m22)}, which is what the traced depth in {@code gDepth} is actually
     * consistent with.
     */
    private CameraFrame readCamera(Arena arena) {
        CameraFrame out = new CameraFrame();
        RtComposite composite = RtComposite.INSTANCE;
        Matrix4fc view = composite.frameViewRotation();
        if (view != null) {
            out.right = vector(arena, view.m00(), view.m01(), view.m02());
            out.up = vector(arena, view.m10(), view.m11(), view.m12());
            out.forward = vector(arena, -view.m20(), -view.m21(), -view.m22());
        } else {
            out.right = vector(arena, 1, 0, 0);
            out.up = vector(arena, 0, 1, 0);
            out.forward = vector(arena, 0, 0, -1);
        }
        out.position = vector(arena, (float) composite.cameraX(), (float) composite.cameraY(),
                (float) composite.cameraZ());

        Matrix4fc projection = composite.frameProjection();
        if (projection != null) {
            float m11 = projection.m11();
            if (Float.isFinite(m11) && Math.abs(m11) > 1e-6f) {
                out.fovYRadians = 2.0f * (float) Math.atan(1.0f / Math.abs(m11));
            }
            float m22 = projection.m22();
            float m32 = projection.m32();
            float far = m22 == 0 ? Float.NaN : m32 / m22;
            float near = (1.0f + m22) == 0 ? Float.NaN : m32 / (1.0f + m22);
            if (Float.isFinite(near) && Float.isFinite(far) && near > 0.0f && far > near) {
                out.near = near;
                out.far = far;
            } else if (!loggedProjectionFallback) {
                loggedProjectionFallback = true;
                // A reversed-Z matrix with an infinite far plane decodes to NaN, and the defaults
                // then drive every depth-based FSR decision; say so once instead of silently.
                CausticaMod.LOGGER.warn("FSR-FG: projection does not decode to a finite reversed-Z frustum "
                                + "(m22={}, m32={}); using default near={}, far={}",
                        m22, m32, out.near, out.far);
            }
        }
        return out;
    }

    private static String fmt(float value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
    }

    private static MemorySegment vector(Arena arena, float x, float y, float z) {
        MemorySegment segment = arena.allocate(CAMERA_FLOATS / 4L * Float.BYTES);
        segment.set(ValueLayout.JAVA_FLOAT, 0, x);
        segment.set(ValueLayout.JAVA_FLOAT, 4, y);
        segment.set(ValueLayout.JAVA_FLOAT, 8, z);
        return segment;
    }

    private static GpuDeviceBackend device() {
        return ((GpuDeviceAccessor) RenderSystem.getDevice()).caustica$getBackend();
    }
}

package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import com.mojang.blaze3d.vulkan.VulkanGpuTexture;
import com.mojang.blaze3d.vulkan.VulkanGpuTextureView;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.client.CausticaJitter;
import dev.comfyfluffy.caustica.mixin.CommandEncoderAccessor;
import dev.comfyfluffy.caustica.rt.gen.WorldPushConstantsData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.BreakEntry;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float2;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float3;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float4;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Int4;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.BiomeColors;
import net.minecraft.client.renderer.EndFlashState;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.resources.model.ModelBakery;
import net.minecraft.core.BlockPos;
import net.minecraft.data.AtlasIds;
import net.minecraft.resources.Identifier;
import net.minecraft.tags.FluidTags;
import net.minecraft.util.Mth;
import net.minecraft.world.attribute.EnvironmentAttributes;
import net.minecraft.world.level.MoonPhase;
import net.minecraft.world.level.material.FluidState;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkClearColorValue;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDependencyInfo;
import org.lwjgl.vulkan.VkImageBlit;
import org.lwjgl.vulkan.VkImageCopy;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkImageMemoryBarrier2;
import org.lwjgl.vulkan.VkImageSubresourceRange;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryBarrier2;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

import dev.comfyfluffy.caustica.rt.accel.RtAccel;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.entity.RtEntities;
import dev.comfyfluffy.caustica.rt.entity.RtEntityTextures;
import dev.comfyfluffy.caustica.rt.material.RtBlockMaterials;
import dev.comfyfluffy.caustica.rt.material.RtEmissionSemantics;
import dev.comfyfluffy.caustica.rt.material.RtMaterialOverrides;
import dev.comfyfluffy.caustica.rt.material.RtMaterialRegistry;
import dev.comfyfluffy.caustica.rt.pipeline.RtDebugPresentPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtBloomPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtSkyLut;
import dev.comfyfluffy.caustica.rt.pipeline.RtDisplayPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg;
import dev.comfyfluffy.caustica.rt.pipeline.RtDlssRr;
import dev.comfyfluffy.caustica.rt.pipeline.RtFgInterpPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg;
import dev.comfyfluffy.caustica.rt.overlay.RtWorldOverlay;
import dev.comfyfluffy.caustica.rt.pipeline.RtHdrCompositePipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtSdrPresentPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtExposure;
import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtToneLut;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Objects;

/**
 * On-screen composite. Each frame, ray-trace into a render-res storage image (+ guide buffers), use
 * DLSS Ray Reconstruction to denoise and upscale it to display res, write that into a storage-capable
 * copy of the world color, and copy the result back to the world target at the
 * end-of-world seam. Gated by {@code -Dcaustica.rt=true}.
 *
 * <p>The path tracer and its guide buffers run at the configured render scale of display res with a per-frame
 * sub-pixel camera jitter; DLSS-RR ({@link RtDlssRr}) reconstructs the display-res image. With RR
 * disabled the trace runs at 1:1 and a linear blit stands in for the upscale (a raw, noisy reference).
 *
 * <p>Traces the extracted {@link RtTerrain} with perspective camera rays (camera matrices captured
 * each frame via {@link #captureFrame}); writes nothing until terrain is available.
 * Pipelines/SBT/descriptors are built once; sized images rebuilt on resize.
 */
public final class RtComposite {
    public static final RtComposite INSTANCE = new RtComposite();

    public static boolean enabled() {
        return CausticaConfig.Rt.ENABLED.value();
    }

    // WorldPushData and its serializer are generated from Slang's reflected Std430DataLayout. Java never
    // owns or calculates a shader byte offset, struct size, array stride, or fixed-array capacity.
    private static final int WORLD_PUSH_SIZE = WorldPushData.BYTE_SIZE;
    // Real inline push constants (fast constant-bank reads), separate from the WorldPush BDA ring above.
    // Hot addresses/frameIndex avoid unnecessary global-memory dereferences; WorldPushConstantsData is
    // generated from the same Slang module and owns this second ABI as well. debugView is no longer
    // part of it -- no world shader reads it anymore; debug views are a downstream compute pass.
    private static final long PATH_RECORD_BYTES = 48L;
    private static int debugView() {
        return CausticaConfig.Rt.Composite.DEBUG_VIEW.value();
    }

    private static int spp() {
        return CausticaConfig.Rt.Composite.SPP.value();
    }

    private static int maxBounces() {
        return CausticaConfig.Rt.Composite.MAX_BOUNCES.value();
    }

    private static boolean waterWaves() {
        return CausticaConfig.Rt.Composite.WATER_WAVES.value();
    }

    private static final int WATER_ANCHOR_MASK = 4095;
    // worldPush.flags volumetric bits, mirroring WORLD_FLAG_FOG / WORLD_FLAG_VOLUME_SHADOW /
    // WORLD_FLAG_VOLUME_DEBUG in world_common.slang (VolumeFogTest checks the two sides agree). The
    // cloud, spray and march-tier bits have no Java setter until their features exist.
    private static final int WORLD_FLAG_FOG = 1 << 7;
    private static final int WORLD_FLAG_CLOUD = 1 << 8;
    private static final int WORLD_FLAG_SPRAY = 1 << 9;
    private static final int WORLD_FLAG_VOLUME_SHADOW = 1 << 10;
    private static final int WORLD_FLAG_VOLUME_MARCH = 1 << 11;
    private static final int WORLD_FLAG_VOLUME_DEBUG = 1 << 12;
    /** The volumetric cloud slab (Phase 5) walks the layer instead of crossing it once. */
    private static final int WORLD_FLAG_CLOUD_VOLUMETRIC = 1 << 13;
    // Debug views the primary raygen fills the volume debug image for; one channel per view.
    private static final int DEBUG_VIEW_VOLUME_FIRST = 10;
    private static final int DEBUG_VIEW_VOLUME_LAST = 13;
    // The versioned look package owns every photometric anchor and the sky geometry. Its sun illuminance is the
    // photometric solar constant at the top of the atmosphere; the shader's transmittance LUT brings that
    // to ~117,000 lux under a zenith sun and reddens/dims it through sunset, and because world.rmiss tints
    // the visible disc from the same LUT, the light on terrain and the sky's sunset are one number.
    //
    // world.rgen consumes it as ILLUMINANCE at normal incidence (lux) — the NEE term is brdf·E·ndl with no
    // solid-angle factor, and the diffuse BRDF's 1/π turns 100,000 lux into
    // 31,800 cd/m² white / 5,730 cd/m² 18%-grey noon surface. It is therefore independent of the sky
    // package's angular radii, which only jitter the shadow ray and so only set penumbra softness.
    private static final RtLookPackage LOOK = RtLookPackage.current();
    /**
     * The night ramp for {@link #nightFactor()}: a quarter of a day separates midnight from either horizon
     * crossing, and the lift reaches full once {@code NIGHT_RAMP} of that has been crossed. 0.08 puts full
     * night from about 1900 ticks after sunset to the same before sunrise, and leaves both ramps roughly
     * 96 seconds long, so a sunrise still looks like one instead of a light switch.
     */
    private static final float NIGHT_RAMP = 0.08f;
    private static final long DAY_TICKS = 24000L;
    private static final Identifier SUN_ID = Identifier.withDefaultNamespace("sun");
    private static final Identifier[] MOON_IDS = createMoonIds();
    private static final Identifier END_FLASH_ID = Identifier.withDefaultNamespace("end_flash");
    // Vanilla's standalone end-sky texture (SkyRenderer.END_SKY_LOCATION), loaded by the TextureManager
    // when the client's SkyRenderer is constructed, i.e. always before the first traced frame.
    private static final Identifier END_SKY_LOCATION =
            Identifier.withDefaultNamespace("textures/environment/end_sky.png");
    // Sign of the sub-pixel jitter as reported to DLSS-RR + applied to the primary ray, mirroring the
    // validated DLSS-SR convention (Vulkan flipped clip space wants Y negated).
    private static float jitterSignX() {
        return CausticaConfig.Rt.Composite.JITTER_SIGN_X.value();
    }

    private static float jitterSignY() {
        return CausticaConfig.Rt.Composite.JITTER_SIGN_Y.value();
    }

    // Monotonic per-composite frame counter used for cache eviction, shader sampling, and diagnostics.
    private static volatile long frameCounter;

    public static long frameCounter() {
        return frameCounter;
    }

    private RtPipeline worldPipeline;
    // Set at the HEAD of Minecraft.reloadResourcePacks() (mixin): a resource reload recreates the block
    // atlas + entity textures. We tear down the world pipeline there (drops all descriptor references) and
    // rebuild it once the NEW atlas is in place — detected by the atlas view handle changing away from
    // boundBlockAlbedoAtlasHandle to a fresh non-zero value (MC's deferred free keeps the old handle live for a few
    // frames, so "handle != 0" alone isn't enough to tell old from new).
    private volatile boolean reloadRebindRequested;
    // The block-atlas view handle currently bound into the world pipeline (set by bindWorldTextures).
    private long boundBlockAlbedoAtlasHandle;
    private int bindlessTextureCapacity;
    // True after the LabPBR atlases have been resolved/bound for the currently alive world pipeline.
    private boolean materialBindingsReady;
    // Set when a new material epoch is published. The first composite returns to vanilla so the next
    // client tick can apply RtTerrain's full-clear before any old-epoch primitive IDs are traced.
    private boolean materialEpochTraceGate;
    // World push data lives in a host-visible BDA ring; only the slot address and a small hot subset are
    // pushed inline (the full generated structure exceeds NVIDIA's 256-byte push-constant ceiling).
    // Exact graphics completion guards host writes; ring depth only avoids routine waits.
    private static final int PUSH_RING = 6;
    private PushSlot[] pushRing;
    private int pushSlot;
    private RtDisplayPipeline displayPipeline;
    private RtBloomPipeline bloomPipeline;
    // Atmosphere LUTs (transmittance + multiple scattering + this frame's sky view). Device-lifetime; the
    // two static tables are baked on the first frame that records the pass.
    private RtSkyLut skyLut;
    private RtDebugPresentPipeline debugPresentPipeline;
    private RtToneLut sdrToneLut;
    private RtToneLut hdrToneLut;
    private RtToneLut lookLut;
    // Absolute classpath path of the LMT currently uploaded into lookLut (null until first load), so
    // the swap check above is a string compare on the steady-state path and a matching name is never
    // re-read from the Resource.
    private String loadedLookLutName;
    private int loadedHdrLutNits = -1;
    private RtImage output;
    // Packed primary -> indirect continuations. Pass A is fixed at one sample and owns two records per
    // render pixel (base + optional transmission); Pass B resamples them at the configured SPP.
    private RtBuffer continuationQueue;
    private RtImage displayImage;
    // Bloom pyramid, finest first: level 0 is half display resolution and each level halves again. The
    // display mapper reads level 0, which the upsample sweep leaves holding the sum of every band.
    private RtImage[] bloomLevels = new RtImage[0];
    // Parallel PQ-encoded ([0,1], ST.2084) HDR display image. Written alongside displayImage only while
    // HDR output is enabled (allocated on demand for exactly that state — see ensureOutput). When the PQ
    // swapchain is active, the combined UI overlay is composited over this image, then this image is
    // blitted straight to the swapchain.
    private RtImage hdrDisplayImage;
    /**
     * 1x1 stand-in bound into the display pipeline's HDR slot while {@link #hdrDisplayImage} does not
     * exist: the shader binding is format-qualified rgba16f, so the descriptor always needs a legal view
     * even though the dispatch's hdrEnabled flag keeps the write gated off.
     */
    private RtImage hdrBindingPlaceholder;
    // Set true after this frame's display dispatch wrote hdrDisplayImage (HDR enabled + RT ran); gates the
    // HDR present blit so a frame where RT did not run falls back to the vanilla SDR present.
    private boolean hdrWrittenThisFrame;
    // DLSS-FG "hudless" resource: a copy of the main render target before the combined UI overlay
    // composites back on top. Lazily allocated (only meaningful once FG + the UI overlay redirect are both
    // active), resized on demand.
    private RtImage fgHudlessImage;
    // Same idea as fgHudlessImage but for the HDR present path: a copy of hdrDisplayImage taken in
    // presentHdr right before its own combined-UI composite dispatch overwrites it in place (see
    // captureFgHdrHudless). Already PQ-encoded (same as hdrDisplayImage), so this is a plain image copy, not
    // a format conversion — DLSS-FG requires a display-ready EOTF-encoded [0,1] signal (its programming
    // guide explicitly disallows scRGB), and PQ is exactly that.
    private RtImage fgHdrHudlessImage;
    // Composites the combined UI overlay over hdrDisplayImage at paper white, just before present.
    private RtHdrCompositePipeline hdrCompositePipeline;
    private long hdrUiSampler;

    private static final class PushSlot {
        final RtBuffer buffer;
        final RtGpuExecutor.TrackedGraphicsUse graphicsUse = new RtGpuExecutor.TrackedGraphicsUse();

        PushSlot(RtBuffer buffer) {
            this.buffer = buffer;
        }
    }
    // Menu/non-RT present: converts the SDR main target (sRGB) to PQ-encoded at paper white so menus,
    // the title panorama and the loading screen present correctly to the PQ swapchain instead of being
    // raw-copied (misdisplayed). Lazily created; the image is sized to the swapchain.
    private RtSdrPresentPipeline sdrPresentPipeline;
    private RtImage sdrPresentImage;
    // DLSS Frame Generation: per-generated-frame interpolated output images (backbuffer size/format), and
    // the jitter-free reprojection matrices derived from the MV view-projections each frame. In HDR mode
    // these hold DLSSG's raw PQ-encoded output, which is blitted straight to the (PQ) swapchain — no decode
    // needed since the swapchain itself is PQ-native.
    private RtImage[] fgInterp = new RtImage[0];
    private int fgInterpW = -1;
    private int fgInterpH = -1;
    private int fgInterpFormat = Integer.MIN_VALUE;
    private boolean fgReset = true;
    private final Matrix4f fgClipToPrev = new Matrix4f();
    private final Matrix4f fgPrevToClip = new Matrix4f();
    private final Matrix4f fgMatTmp = new Matrix4f();
    // Self-made interpolation instead of a vendor backend: the generated frame plus the previous displayed
    // frame it blends against. The history is written only after the interpolation has read it.
    private RtFgInterpPipeline fgInterpPipeline;
    /** Packs the swapchain size of the last logged missing-HUD-less warning, so it warns once per size. */
    private int fgHudlessMismatch = -1;
    private RtImage fgSelfInterp;
    private RtImage fgSelfHistory;
    private boolean fgSelfHistoryWritten;
    // Guide buffers (first-hit attributes for DLSS-RR): normal+roughness, albedo, depth, motion,
    // specular albedo, and reflection motion.
    private RtImage gNormal;
    private RtImage gAlbedo;
    private RtImage gDepth;
    private RtImage gMotion;
    /** The two bounce-0 reservoir images, and which index this frame writes. */
    private RtImage[] reservoirImages = new RtImage[0];
    /** The companion (s,t) rectangle coordinates, one image per side, advanced on the same index. */
    private RtImage[] reservoirSampleImages = new RtImage[0];
    private int reservoirWrite;
    private RtImage gSpecAlbedo;
    private RtImage gSpecMotion;
    // Per-pixel volumetric diagnostics (debug views 10..13), render-res like the guides. Written by the
    // primary raygen only on frames whose WorldPush carries WORLD_FLAG_VOLUME_DEBUG; the debug present
    // pass shows one channel per view.
    private RtImage volumeDebug;
    // Display-res RT image the display mapper reads: DLSS-RR writes it (render -> display denoise+upscale), or a
    // linear blit of `output` fills it when RR is off/unavailable (the no-RR reference / fallback).
    private RtImage rrOutput;
    private final RtExposure exposure = new RtExposure();

    // Trace + guide buffers run at render res; composite (display-mapping) runs at display res.
    private int displayW = -1;
    private int displayH = -1;
    private int renderW = -1;
    private int renderH = -1;
    // What ensureOutput last sized the render/guide images for, so a quality change (or RR being
    // toggled) at a fixed window size is noticed even though displayW/displayH didn't change.
    private boolean renderSizeRrEnabled;
    private int renderSizeRrQuality = Integer.MIN_VALUE;
    // Same idea for the HDR output toggle: it decides whether hdrDisplayImage exists at all, so a runtime
    // switch must trigger the ensureOutput rebuild that allocates or frees it.
    private boolean renderSizeHdrEnabled;

    // Motion-vector reprojection state: the previous frame's camera-relative view-projection and
    // camera position, read into the push constant each frame then advanced at frame end.
    private final Matrix4f mvPrevProjView = new Matrix4f();
    private final Matrix4f mvCurProjView = new Matrix4f();
    private final Matrix4f mvPushMatrix = new Matrix4f();
    private final Matrix4f frameInvViewProj = new Matrix4f();
    private final BlockPos.MutableBlockPos cameraBlockPos = new BlockPos.MutableBlockPos();
    private double mvPrevCamX;
    private double mvPrevCamY;
    private double mvPrevCamZ;
    private float mvCamDeltaX;
    private float mvCamDeltaY;
    private float mvCamDeltaZ;
    private boolean mvHasPrev;
    private float previousWaterWaveTime;
    private boolean waterWaveTimeValid;
    // The rain level pushed last frame, so WorldPush.weather can carry both ends of the transition
    // instead of forcing the shaders to diff against a history they do not have.
    private float previousRainLevel;
    // Accumulated wave clock (s) and the timestamp it was last advanced from. Accumulating is what lets
    // the speed control change the rate without jumping the phase to a different wall-clock multiple.
    private float waterWaveClock;
    private long waveClockNanos;
    // Per-axis wind-drift accumulators (blocks, double precision). The fog pair wraps each axis in
    // whole 4096-block periods — the fog noise lattice's exact period — so a wrap is invisible by
    // construction (the old shader-side speed x clock product snapped the whole field at the wave
    // clock's hourly wrap). The cloud pair never wraps: the streak lattice has no period to wrap
    // into, and the doubles only shed sub-block precision past millions of blocks of drift.
    private double fogDriftX;
    private double fogDriftZ;
    private double cloudDriftX;
    private double cloudDriftZ;
    /**
     * The cloud field's evolution phase, in noise-lattice units. Advanced by the same clamped step as
     * the drift accumulators and never wrapped: the lattices it offsets are not periodic, so a wrap
     * would jump every cloud in the sky, and at this rate fp32 keeps sub-cell precision for days of
     * continuous play - long past any session that would notice the crawl freezing.
     */
    private double cloudEvolve;
    /**
     * Evolution rate in lattice units per second; see {@link #cloudEvolve}. One unit is one warp-lattice
     * cell, i.e. a full reorganisation of the streak shapes, so 0.03 puts one of those every ~33 s -
     * clouds that visibly live rather than a fixed decal sliding along.
     */
    private static final double CLOUD_EVOLVE_SPEED = 0.030d;
    private long atlasSampler;
    private long endSkySampler;
    private boolean failed;
    private boolean loggedActive;

    // Camera captured each frame from GameRenderer (unjittered level projection + camera rotation + pos).
    private final Matrix4f frameProjection = new Matrix4f();
    private final Matrix4f frameViewRotation = new Matrix4f();
    private double camX;
    private double camY;
    private double camZ;
    private boolean frameCaptured;
    private long celestialUvAtlasHandle;
    private int celestialUvMoonPhase = -1;
    private float sunU0;
    private float sunV0;
    private float sunU1 = 1f;
    private float sunV1 = 1f;
    private float moonU0;
    private float moonV0;
    private float moonU1 = 1f;
    private float moonV1 = 1f;
    private float endFlashU0;
    private float endFlashV0;
    private float endFlashU1 = 1f;
    private float endFlashV1 = 1f;

    // Per-frame TLAS resources, rebuilt in place from a small ring of persistent slots (see
    // RtAccel.TlasRing — replaces the old create-and-defer-destroy-per-frame churn whose VMA slow path
    // showed up as rare multi-ms prepareTlas spikes).
    private final RtAccel.TlasRing tlasRing = new RtAccel.TlasRing();

    // This frame's TLAS handle, published after prepareTlas so the world-overlay pass (block outline's
    // rayQueryEXT occlusion test) can bind the exact same acceleration structure the primary trace used —
    // same-queue submission order (RtWorldOverlay's transient buffer runs later, same graphics queue)
    // makes the TLAS build's writes visible without an extra semaphore, matching every other overlay
    // feature's reliance on in-order queue execution for this frame's world content.
    private volatile long currentTlasHandle;
    private RtGpuExecutor.GraphicsUse pendingGraphicsUse;

    private RtComposite() {
    }

    /** This frame's TLAS handle (0 if none built yet), for {@code dev.comfyfluffy.caustica.rt.overlay} occlusion queries. */
    public long currentTlasHandle() {
        return currentTlasHandle;
    }

    private static Identifier[] createMoonIds() {
        MoonPhase[] phases = MoonPhase.values();
        Identifier[] ids = new Identifier[phases.length];
        for (int i = 0; i < phases.length; i++) {
            ids[i] = Identifier.withDefaultNamespace("moon/" + phases[i].getSerializedName());
        }
        return ids;
    }

    public boolean hasFailed() {
        return this.failed;
    }

    /** Read-only access to the auto-exposure controller, for diagnostics (F3 entry, frame stats log). */
    public RtExposure exposure() {
        return exposure;
    }

    /**
     * Export the latest RT scene image at the exact input seam of the Look/LMT stage.
     *
     * <p>The GPU image stores {@code sceneLinear * preExposure} in fp16. This readback multiplies RGB by
     * the display shader's current 1x1 {@code residualExposure}, in float32, then quantizes the resulting
     * exposure-adjusted scene-linear image to fp16 EXR. Metadata keeps both factors so the original scene-linear
     * values can be reconstructed with {@code RGB / (preExposure * residualExposure)}.
     *
     * @return {@code true} when a current RT frame was available and written
     */
    public boolean exportLatestResidualExposureExr(Path outputPath) throws java.io.IOException {
        RenderSystem.assertOnRenderThread();
        RtContext ctx = RtContext.currentOrNull();
        if (!enabled() || failed || ctx == null || rrOutput == null || exposure.image() == null
                || displayW <= 0 || displayH <= 0 || pendingGraphicsUse != null) {
            return false;
        }

        long pixelCount = Math.multiplyExact((long) displayW, (long) displayH);
        long rgbaBytes = Math.multiplyExact(pixelCount, 4L * Short.BYTES);
        long totalBytes = Math.addExact(rgbaBytes, Float.BYTES);
        if (pixelCount > Integer.MAX_VALUE / 4L) {
            throw new IllegalArgumentException("EXR capture is too large for a Java array: "
                    + displayW + "x" + displayH);
        }

        // All ordinary frame commands have been submitted before the F2 key is handled. Drain them before
        // a private one-shot copy so rrOutput and the exposure image describe the same completed frame.
        ctx.waitIdle();
        RtBuffer readback = ctx.createReadbackBuffer(totalBytes, "residual-exposure EXR readback");
        try {
            ctx.submitSync(cmd -> recordExrReadback(ctx, cmd, readback, rgbaBytes));
            readback.invalidate();

            float residualExposure = MemoryUtil.memGetFloat(readback.mapped + rgbaBytes);
            RtExposure.CaptureMetadata exposureMetadata = exposure.captureMetadata(residualExposure);
            short[] exposedRgba = new short[Math.toIntExact(pixelCount * 4L)];
            for (int sample = 0; sample < exposedRgba.length; sample++) {
                short storedHalf = MemoryUtil.memGetShort(readback.mapped + (long) sample * Short.BYTES);
                float value = Float.float16ToFloat(storedHalf);
                if ((sample & 3) != 3) {
                    value *= residualExposure;
                }
                // Residual exposure is expected to keep this seam comfortably centred in fp16. Clamp only
                // true outliers/infinities so a pathological light cannot poison a grading application.
                value = Math.clamp(value, -65504.0f, 65504.0f);
                exposedRgba[sample] = Float.floatToFloat16(value);
            }

            RtOpenExrWriter.write(outputPath, displayW, displayH, exposedRgba,
                    new RtOpenExrWriter.Metadata(
                            exposureMetadata.preExposure(),
                            exposureMetadata.residualExposure(),
                            exposureMetadata.absoluteExposure(),
                            exposureMetadata.mode(),
                            exposureMetadata.evScene(),
                            exposureMetadata.evTarget(),
                            exposureMetadata.evApplied(),
                            LOOK.id() + "@" + LOOK.packageVersion(),
                            frameCounter));
            return true;
        } finally {
            readback.destroy();
        }
    }

    private void recordExrReadback(RtContext ctx, VkCommandBuffer cmd, RtBuffer readback, long exposureOffset) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd,
                     "residual-exposure EXR readback")) {
            VkImageMemoryBarrier.Buffer imageBarriers = VkImageMemoryBarrier.calloc(2, stack);
            imageBarriers.get(0).sType$Default()
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_GENERAL).newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                    .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT | VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_TRANSFER_READ_BIT)
                    .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .image(rrOutput.image);
            imageBarriers.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .levelCount(1).layerCount(1);
            imageBarriers.get(1).sType$Default()
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_GENERAL).newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                    .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT | VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_TRANSFER_READ_BIT)
                    .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .image(exposure.image().image);
            imageBarriers.get(1).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .levelCount(1).layerCount(1);
            VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, imageBarriers);

            VkBufferImageCopy.Buffer sceneCopy = VkBufferImageCopy.calloc(1, stack);
            sceneCopy.get(0).bufferOffset(0L);
            sceneCopy.get(0).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            sceneCopy.get(0).imageExtent().set(displayW, displayH, 1);
            VK10.vkCmdCopyImageToBuffer(cmd, rrOutput.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    readback.handle, sceneCopy);

            VkBufferImageCopy.Buffer exposureCopy = VkBufferImageCopy.calloc(1, stack);
            exposureCopy.get(0).bufferOffset(exposureOffset);
            exposureCopy.get(0).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            exposureCopy.get(0).imageExtent().set(1, 1, 1);
            VK10.vkCmdCopyImageToBuffer(cmd, exposure.image().image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    readback.handle, exposureCopy);

            VkMemoryBarrier.Buffer hostBarrier = VkMemoryBarrier.calloc(1, stack);
            hostBarrier.get(0).sType$Default().srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_HOST_READ_BIT);
            VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK10.VK_PIPELINE_STAGE_HOST_BIT, 0, hostBarrier, null, null);
        }
    }

    /**
     * Whether the current frame must retain vanilla world rendering while RT resource state converges.
     *
     * <p>The composite still runs at the normal seam so it can consume the one-frame epoch gate or observe
     * the newly uploaded atlas. This method only prevents {@code LevelRenderer} from being cancelled before
     * a deliberately transient {@link #composite} return. Such a return is not a renderer failure and must
     * not trip {@code VanillaRenderController}'s permanent safety latch.</p>
     */
    public boolean requiresVanillaWorldFallback() {
        // Pipeline creation publishes a new material epoch and deliberately makes composite() return
        // false once so RtTerrain can apply the matching full clear. Keep vanilla alive for that bring-up
        // frame; otherwise LevelRenderer is cancelled before composite() discovers it must fall back and
        // VanillaRenderController permanently latches the resulting missing replacement frame.
        if (worldPipeline == null || !materialBindingsReady) {
            return true;
        }
        if (materialEpochTraceGate) {
            return true;
        }
        if (RtEntityTextures.maxTextures() > bindlessTextureCapacity) {
            return true;
        }
        if (reloadRebindRequested) {
            long atlas = blockAlbedoAtlasView();
            return atlas == 0L || atlas == boundBlockAlbedoAtlasHandle;
        }
        return false;
    }

    /**
     * Clear the failure latch on an explicit render-state invalidation (F3+A, dimension change) so RT
     * re-arms after a transient error instead of staying on vanilla until restart. A deterministic
     * failure just latches again on the next frame (bounded log spam: one error line per invalidation).
     */
    public void resetFailureLatch() {
        if (failed) {
            failed = false;
            CausticaMod.LOGGER.info("RT failure latch cleared by render-state invalidation; retrying RT");
        }
        // RR and the frame-generation backends latch independently of the composite; hand them the
        // same recovery path. Each reset is a no-op while its backend is healthy.
        RtFsrFg.INSTANCE.resetFailure();
        RtDlssFg.INSTANCE.resetFailure();
        RtDlssRr.INSTANCE.resetFailure();
        RtWorldOverlay.INSTANCE.resetFailureLatch();
        RtUiOverlay.resetFailureLatch();
    }

    /** Capture the frame's camera for the next composite. Called from GameRendererMixin. */
    public void captureFrame(Matrix4f projection, Matrix4fc viewRotation, double cameraX, double cameraY, double cameraZ) {
        frameProjection.set(projection);
        frameViewRotation.set(viewRotation);
        camX = cameraX;
        camY = cameraY;
        camZ = cameraZ;
        frameCaptured = true;
    }

    /** Reset exposure filtering after an explicit render-state invalidation such as F3+A. */
    public void resetExposureHistory() {
        exposure.requestReset();
    }

    /**
     * The frame's forward camera-relative view-projection (jitter-free), exactly what {@code world.rgen}
     * traced with — overlay raster passes ({@code dev.comfyfluffy.caustica.rt.overlay}) reuse it so their content lands
     * pixel-exact on the RT image. Valid after {@code updateMotion} ran this frame; do not mutate.
     */
    public Matrix4fc currentViewProjection() {
        return mvCurProjView;
    }

    /**
     * This frame's world-to-view rotation and projection, and the camera position they were built
     * for. {@link dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg} needs all three because FidelityFX
     * asks for a camera basis and frustum scalars rather than matrices, and re-deriving them from
     * Minecraft's settings would risk disagreeing with what the tracer actually used.
     *
     * <p>Do not mutate what these return; they are the live fields.
     */
    public Matrix4fc frameViewRotation() {
        return frameCaptured ? frameViewRotation : null;
    }

    public Matrix4fc frameProjection() {
        return frameCaptured ? frameProjection : null;
    }

    public double cameraX() {
        return camX;
    }

    public double cameraY() {
        return camY;
    }

    public double cameraZ() {
        return camZ;
    }

    /**
     * Reset per-frame present state at the very start of {@link net.minecraft.client.renderer.GameRenderer}
     * render (before any RT work). Critical for menu/no-world frames: {@link #composite()} is only called
     * while a level is rendering ({@code WorldRenderScaler} opens its window in {@code renderLevel}), so on
     * menu frames {@code composite} never runs and {@code hdrWrittenThisFrame} would otherwise keep its stale
     * {@code true} from the last world frame — presenting a black/stale HDR image behind the menu. Clearing it
     * here every frame makes {@link #isHdrPresentActive()} false on menu frames so the SDR convert-present path
     * runs instead.
     */
    public void beginFrame() {
        if (pendingGraphicsUse != null) {
            throw new IllegalStateException("Previous RT graphics use was never completed");
        }
        RtFrameStats.FRAME.beginIfInactive();
        hdrWrittenThisFrame = false;
    }

    /** This frame's completion token, valid until {@link #finishGraphicsUse()} signals it. */
    public RtGpuExecutor.GraphicsUse currentGraphicsUse() {
        RenderSystem.assertOnRenderThread();
        return pendingGraphicsUse;
    }

    /** Signal this RT frame's shared completion token after its final TLAS consumer (world overlay). */
    public void finishGraphicsUse() {
        RtGpuExecutor.GraphicsUse graphicsUse = pendingGraphicsUse;
        if (graphicsUse == null) {
            return;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            throw new IllegalStateException("RT context disappeared before graphics use completed");
        }
        var encoder = (VulkanCommandEncoder) ((CommandEncoderAccessor) RenderSystem.getDevice()
                .createCommandEncoder()).caustica$getBackend();
        ctx.gpuExecutor().endGraphicsUse(encoder, graphicsUse);
        pendingGraphicsUse = null;
    }

    public void endFrame() {
        RtFrameStats.FRAME.end();
    }

    public boolean composite(GpuTexture nativeColor, int width, int height) {
        frameCounter++; // global frame serial used by remaining per-frame/entity rings and diagnostics
        VulkanDiagnostics.setInFlight("graphics-latest", "frame=" + frameCounter + " size=" + width + "x" + height);
        hdrWrittenThisFrame = false; // set true again below once this frame's HDR display image is written
        if (failed) {
            return false;
        }
        RtContext ctx = RtContext.get();
        if (ctx == null) {
            return false;
        }
        ctx.gpuExecutor().throwIfFailed();
        // Count-bounded terrain streaming (dispatch/drain/build kick) runs here once per render frame — before
        // the ready gate below, because it is what MAKES terrain ready during the initial fill.
        try {
            RtTerrain.frame(ctx);
        } catch (Throwable t) {
            ctx.gpuExecutor().throwIfFailed();
            failed = true;
            CausticaMod.LOGGER.error("RT terrain streaming failed; reverting to vanilla path", t);
            return false;
        }
        if (RtTerrain.currentOrNull() == null || !frameCaptured || Minecraft.getInstance().level == null) {
            // No world this frame (incl. after quitting to the title — terrain residency + frameCaptured can
            // linger until an explicit invalidate, which would otherwise present a stale/empty HDR image as a
            // black menu background). Skip RT so the present path falls back to vanilla SDR / the PQ SDR
            // convert path, which shows the menu + panorama correctly.
            return false;
        }
        try {
            if (displayPipeline == null) {
                displayPipeline = RtDisplayPipeline.create(ctx);
            }
            if (bloomPipeline == null) {
                bloomPipeline = RtBloomPipeline.create(ctx);
            }
            if (skyLut == null) {
                // Normally already created by ensureWorld before the pipeline exists at all; this only
                // fires if render() somehow runs before the tick-driven ensureResourcesReady has, which
                // ensureWorld's own binding order otherwise guarantees never happens.
                skyLut = RtSkyLut.create(ctx);
            }
            if (debugPresentPipeline == null) {
                debugPresentPipeline = RtDebugPresentPipeline.create(ctx);
            }
            if (sdrToneLut == null) {
                sdrToneLut = RtToneLut.load(ctx, "sdr_aces2_rec709.bin");
            }
            // The mastering target is live, so track it each frame.
            int wantedHdrNits = CausticaConfig.Rt.Hdr.PEAK_NITS.value();
            if (hdrToneLut == null || loadedHdrLutNits != wantedHdrNits) {
                RtToneLut newHdrLut = RtToneLut.load(ctx, "hdr_aces2_rec2020_" + wantedHdrNits + "nit.bin");
                if (newHdrLut.size != sdrToneLut.size) {
                    // display.comp's lutSize push constant is shared by both LUT samples (see
                    // lutTexCoord()); bake_display_lut.py currently always sizes both the same, but
                    // this would silently misalign one LUT's edge texels if that ever changed.
                    newHdrLut.destroy();
                    throw new IllegalStateException("SDR/HDR tone LUT size mismatch: "
                            + sdrToneLut.size + " vs " + newHdrLut.size);
                }
                if (hdrToneLut != null) {
                    ctx.waitIdle(); // nits-step change is rare; no in-flight frame may sample the old LUT
                    hdrToneLut.destroy();
                }
                hdrToneLut = newHdrLut;
                loadedHdrLutNits = wantedHdrNits;
            }
            // The scene-referred LMT is part of the immutable versioned look package and shared by
            // both SDR and HDR output transforms. Unlike the HDR LUT above it CAN be swapped
            // independently at runtime — tonemap.look-override names another .bin in the same
            // directory — because an LMT is a scene-referred curve: it never changes how much of the
            // scene the display transform sees, only its shape, so the exposure history stays valid.
            // Re-checked every frame rather than only when null, so iterating on a look does not need
            // a rebuild. Reading the Resource is skipped when the name already matches.
            String wantedLmtName = lookQualifiedLmtName();
            if (!wantedLmtName.equals(loadedLookLutName)) {
                RtToneLut newLookLut = tryLoadLookLut(ctx, wantedLmtName);
                if (newLookLut != null) {
                    if (lookLut != null) {
                        ctx.waitIdle(); // no in-flight frame may sample the old LUT
                        lookLut.destroy();
                    }
                    lookLut = newLookLut;
                    loadedLookLutName = wantedLmtName;
                }
            }
            if (lookLut == null) {
                // First load failed and there was nothing to fall back to (should not happen: the
                // packaged LMT is validated by RtLookPackage's static initializer).
                lookLut = RtToneLut.loadResource(ctx, LOOK.lmtResource());
                loadedLookLutName = LOOK.lmtResource();
            }
            // A resource reload re-stitches the block atlas. We've already torn down the world pipeline
            // (onResourceReloadStart) so nothing references the old atlas, but MC's deferred free keeps the
            // old view handle live for a few frames, then swaps in the new atlas (whose GPU upload may lag,
            // leaving the handle 0 transiently). Skip RT — vanilla renders — until the handle becomes a
            // fresh, non-zero value different from what we last bound; only then rebuild against it.
            if (reloadRebindRequested) {
                long atlas = blockAlbedoAtlasView();
                if (atlas == 0L || atlas == boundBlockAlbedoAtlasHandle) {
                    return false;
                }
            }
            // RR resolved its true optimal render size after the first feature creation (the DLSSD
            // library registers the optimal-settings callback only once loaded) — invalidate the
            // cached render size so ensureOutput re-queries and rebuilds at it.
            if (RtDlssRr.INSTANCE.takeRefreshRequest()) {
                renderSizeRrQuality = Integer.MIN_VALUE;
            }
            ensureOutput(ctx, width, height);
            // ensureOutput's rebuild path (only taken on resize/RR-setting change) already rebinds
            // displayPipeline's descriptor set; this covers the case ensureOutput early-returned but
            // hdrToneLut/lookLut may have been hot-swapped just above; setImages is a no-op if the bound
            // views already match, so this is cheap on every other frame.
            RtToneLut boundLookLut = lookLut;
            displayPipeline.setImages(displayImage.view, rrOutput.view, exposure.image().view, hdrDisplayBindingView(ctx),
                    sdrToneLut.view(), sdrToneLut.sampler(), hdrToneLut.view(), hdrToneLut.sampler(),
                    boundLookLut.view(), boundLookLut.sampler(), bloomLevels[0].view, bloomPipeline.sampler(),
                    exposure.stateBuffer());
            bloomPipeline.setImages(rrOutput.view, exposure.image().view, bloomLevels);
            debugPresentPipeline.setImages(displayImage.view, gNormal.view, gAlbedo.view, gDepth.view,
                    gMotion.view, gSpecAlbedo.view, gSpecMotion.view, rrOutput.view, exposure.image().view,
                    volumeDebug.view, exposure.stateBuffer());
            // Cheap idempotent check every frame (not just on resize): if the exposure mode is switched
            // manual -> auto at runtime (video settings), the auto-mode histogram/state/pipeline must be
            // allocated before recordFrame's exposure.record() below needs them, or it throws.
            exposure.ensureResources(ctx);
            refreshPipelineShapeIfNeeded(ctx);
            RtPipeline active = ensureWorld(ctx);
            if (materialEpochTraceGate) {
                materialEpochTraceGate = false;
                return false;
            }
            refreshMaterialBindingsIfNeeded(ctx);
            updateMotion();
            recordFrame(ctx, active, nativeColor);
            if (!loggedActive) {
                loggedActive = true;
                CausticaMod.LOGGER.info("RT composite active (terrain): {}x{}, RT output replaces the world target", width, height);
            }
            return true;
        } catch (Throwable t) {
            // Close a graphics-use claimed by a half-run recordFrame before anything else: an open
            // token would make the next beginFrame() throw, turning one bad frame into a crash loop.
            if (pendingGraphicsUse != null) {
                try {
                    finishGraphicsUse();
                } catch (Throwable cleanup) {
                    pendingGraphicsUse = null;
                    CausticaMod.LOGGER.warn("RT graphics use could not be closed after a composite failure",
                            cleanup);
                }
            }
            ctx.gpuExecutor().throwIfFailed();
            failed = true;
            CausticaMod.LOGGER.error("RT composite failed; reverting to vanilla path", t);
            return false;
        }
    }

    /**
     * Bring the world pipeline + LabPBR atlases up as soon as we're in a world and the block atlas is
     * loaded — <em>before</em> terrain tessellates — so the immutable material snapshot is available to
     * the first worker section. Driven from the client tick ahead of {@link RtTerrain#update}. No-op once
     * the pipeline exists, while a reload rebuild is pending (the reload path rebuilds against the new
     * atlas), or until we're in a world with the atlas ready. The heavy {@code _s}/{@code _n} atlases are
     * deliberately not built at the menu — only once a world is entered.
     */
    public void ensureResourcesReady(RtContext ctx) {
        if (failed || worldPipeline != null || reloadRebindRequested) {
            return;
        }
        if (Minecraft.getInstance().level == null || blockAlbedoAtlasView() == 0L) {
            return;
        }
        try {
            ensureWorld(ctx);
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("RT resource bring-up failed; reverting to vanilla path", t);
        }
    }

    private RtPipeline ensureWorld(RtContext ctx) {
        if (worldPipeline == null) {
            // Must exist before bindWorldTextures below writes the sky-LUT descriptors. This is the
            // earliest possible bind: ensureResourcesReady drives this from the client tick, ahead of the
            // render()/composite path. bindWorldTextures only ever runs again on a
            // resource reload, so a skyLut that is still null on this first call stays permanently unbound
            // and every miss/raygen sky sample reads the pre-vkUpdateDescriptorSets undefined descriptor
            // (VUID-vkCmdTraceRaysKHR-None-08114).
            if (skyLut == null) {
                skyLut = RtSkyLut.create(ctx);
            }
            bindlessTextureCapacity = RtEntityTextures.maxTextures();
            worldPipeline = RtPipeline.create(ctx, new String[]{
                            RtDeviceBringup.worldPrimaryRaygenShader(),
                            RtDeviceBringup.worldRaygenShader()},
                    new String[]{"sky.rmiss.spv", "guide.rmiss.spv"},
                    "closest_hit.rchit.spv", "any_hit.rahit.spv",
                    WorldPushConstantsData.BYTE_SIZE, bindlessTextureCapacity);
            // Per-frame world data lives in this BDA ring; the pipeline pushes its address and hot fields.
            if (pushRing == null) {
                pushRing = new PushSlot[PUSH_RING];
                for (int i = 0; i < PUSH_RING; i++) {
                    pushRing[i] = new PushSlot(ctx.createBuffer(WORLD_PUSH_SIZE,
                            VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "rt world push " + i));
                }
            }
            if (output != null) {
                worldPipeline.setStorageImage(output.view);
                bindGuideImages();
            }
            bindWorldTextures(ctx);
            reloadRebindRequested = false;
        }
        // The TLAS is rebuilt and bound per frame in recordFrame since dynamic entity content animates
        // the instance set every frame.
        return worldPipeline;
    }

    private void refreshPipelineShapeIfNeeded(RtContext ctx) {
        if (worldPipeline == null || reloadRebindRequested) {
            return;
        }
        int desiredBindlessCapacity = RtEntityTextures.maxTextures();
        if (desiredBindlessCapacity <= bindlessTextureCapacity) {
            return;
        }
        ctx.waitIdle();
        worldPipeline.destroy();
        worldPipeline = null;
        bindlessTextureCapacity = 0;
        materialBindingsReady = false;
    }

    /**
     * Resolve + bind every world-pipeline texture: the block atlas (binding 2 + bindless fallback slot 0)
     * and the canonical material page bundles in reserved bindless slots. Shared by first creation and
     * the post-reload rebind. Resets the entity bindless registry, recreates material pages, builds
     * the shared material registry, and invalidates old-epoch geometry before tracing resumes.
     */
    private void bindWorldTextures(RtContext ctx) {
        long sampler = atlasSampler(ctx);
        long atlasView = blockAlbedoAtlasView();
        boundBlockAlbedoAtlasHandle = atlasView; // remember what we bound so a reload can detect the new atlas
        worldPipeline.setBlockAlbedoAtlas(atlasView, sampler);
        // Bindless slot 0 = fallback texture (the block atlas) so an entity whose texture can't be
        // resolved samples something defined rather than an unbound (partially-bound) descriptor.
        RtBlockMaterials.INSTANCE.reset();
        RtMaterialOverrides materialOverrides = RtMaterialOverrides.load();
        RtEmissionSemantics emissionSemantics = RtEmissionSemantics.analyze();
        RtBlockMaterials.INSTANCE.prepareAll(ctx, bindlessTextureCapacity, emissionSemantics, materialOverrides);
        RtEntityTextures.INSTANCE.reset(bindlessTextureCapacity);
        worldPipeline.setEntityAlbedoTexture(0, atlasView, sampler);
        RtBlockMaterials.INSTANCE.bindPages(worldPipeline, sampler);
        RtMaterialRegistry.INSTANCE.rebuild(ctx, RtBlockMaterials.INSTANCE, emissionSemantics, materialOverrides);
        materialBindingsReady = true;
        // Sky rewrite: bind the vanilla celestials atlas (sun + moon phases) for world.rmiss. The view
        // handle is stable across frames; the shader only samples it inside the sun/moon discs (sky
        // directions), so the block-atlas fallback is never read if the celestials atlas isn't ready.
        long celView = celestialsAtlasView();
        worldPipeline.setSkyAtlas(celView != 0L ? celView : atlasView, sampler);
        // The end sky's own texture: linear + wrapping (the cube repeats it 16x16 per face), so it cannot
        // share the nearest-filtered atlas sampler. Falls back to the block atlas the same way the
        // celestials binding does — the descriptor must always be written — and is only ever sampled by
        // the miss shader's end-sky branch, which requires the End's own skybox.
        long endSkyView = endSkyAtlasView();
        worldPipeline.setEndSkyTexture(endSkyView != 0L ? endSkyView : atlasView, endSkySampler(ctx));
        // Atmosphere LUTs live for the device's lifetime, but the world pipeline's descriptor sets do
        // not (a resource reload rebuilds it), so rebind them alongside the atlas.
        if (skyLut != null) {
            worldPipeline.setSkyLuts(skyLut.skyViewView(), skyLut.transmittanceView(),
                    skyLut.skyIrradianceView(), skyLut.sampler());
        }
        setCelestialUvAtlas(celView);
        // Atlas UVs and material IDs are one resource epoch. Drop old terrain as a unit rather than
        // incrementally displaying old UVs/IDs against the new atlas/table.
        RtTerrain.requestFullClear();
        materialEpochTraceGate = true;
    }

    private void refreshMaterialBindingsIfNeeded(RtContext ctx) {
        if (worldPipeline == null || reloadRebindRequested) {
            return;
        }
        if (!materialBindingsReady) {
            bindWorldTextures(ctx);
        }
    }

    /** Vulkan image-view of the vanilla celestials atlas (sun + moon-phase sprites), or 0 if unavailable. */
    private static long celestialsAtlasView() {
        try {
            GpuTextureView view = Minecraft.getInstance().getAtlasManager()
                    .getAtlasOrThrow(AtlasIds.CELESTIALS).getTextureView();
            return vkImageView(view);
        } catch (Exception e) {
            return 0L;
        }
    }

    /** Vulkan image-view of vanilla's end_sky.png, or 0 if it is not registered yet. */
    private static long endSkyAtlasView() {
        try {
            GpuTextureView view = Minecraft.getInstance().getTextureManager()
                    .getTexture(END_SKY_LOCATION).getTextureView();
            return vkImageView(view);
        } catch (Exception e) {
            return 0L;
        }
    }

    /**
     * Hooked at the HEAD of {@link net.minecraft.client.Minecraft#reloadResourcePacks()} (mixin). A
     * resource reload re-stitches the block atlas (and reloads entity textures): MC frees the old GPU
     * images via its deferred destruction queue, which refuses while any descriptor set still references
     * them ("in use by VkDescriptorSet" → device lost). So we drain in-flight frames and then <b>destroy
     * the world pipeline outright</b> — dropping every descriptor reference (block atlas binding 2 +
     * bindless set) — so MC can free its textures cleanly. The pipeline is cheap to rebuild (no terrain
     * re-upload); {@code ensureWorld} recreates it on the first world frame after the reload, once the new
     * atlas is ready (gated in {@link #composite}). The new material epoch clears terrain before trace.
     */
    public void onResourceReloadStart() {
        reloadRebindRequested = true;
        materialBindingsReady = false;
        setCelestialUvAtlas(0L);
        RtEntities.INSTANCE.onResourceReload();
        RtContext ctx = RtContext.currentOrNull();
        if (ctx != null) {
            ctx.waitIdle();
            RtWorldOverlay.INSTANCE.onResourceReload();
            if (worldPipeline != null) {
                worldPipeline.destroy();
                worldPipeline = null;
                bindlessTextureCapacity = 0;
            }
            RtMaterialRegistry.INSTANCE.destroy();
        }
    }

    /** Bind the guide buffers into the world pipeline's extra storage-image slots. */
    private void bindGuideImages() {
        if (worldPipeline == null || gNormal == null) {
            return;
        }
        worldPipeline.setExtraStorageImage(0, gNormal.view);
        worldPipeline.setExtraStorageImage(1, gAlbedo.view);
        worldPipeline.setExtraStorageImage(2, gDepth.view);
        worldPipeline.setExtraStorageImage(3, gMotion.view);
        worldPipeline.setExtraStorageImage(4, gSpecAlbedo.view);
        worldPipeline.setExtraStorageImage(5, gSpecMotion.view);
        bindReservoirImages();
        if (volumeDebug != null) {
            worldPipeline.setVolumeDebugImage(volumeDebug.view);
        }
    }

    /** Rebind the reservoir pair so the trace reads the image last frame wrote and writes the other. */
    private void bindReservoirImages() {
        if (worldPipeline == null || reservoirImages.length != 2 || reservoirSampleImages.length != 2) {
            return;
        }
        worldPipeline.setReservoirImages(reservoirImages[reservoirWrite ^ 1].view,
                reservoirImages[reservoirWrite].view,
                reservoirSampleImages[reservoirWrite ^ 1].view,
                reservoirSampleImages[reservoirWrite].view);
    }

    /** Zero the images this frame's trace is about to write, so uncovered texels read as empty history. */
    private void clearReservoir(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack) {
        if (reservoirImages.length != 2 || reservoirSampleImages.length != 2) {
            return;
        }
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "reservoir clear")) {
            VkClearColorValue color = VkClearColorValue.calloc(stack);
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack);
            range.get(0).aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VK10.vkCmdClearColorImage(cmd, reservoirImages[reservoirWrite].image,
                    VK10.VK_IMAGE_LAYOUT_GENERAL, color, range);
            VK10.vkCmdClearColorImage(cmd, reservoirSampleImages[reservoirWrite].image,
                    VK10.VK_IMAGE_LAYOUT_GENERAL, color, range);
            // The indirect trace writes these same images from the raygen. Submission order alone does not
            // order a transfer write against a shader write, so without this barrier the clear can land
            // after the trace and erase (or tear) the reservoir this frame just built.
            VkMemoryBarrier.Buffer cleared = VkMemoryBarrier.calloc(1, stack);
            cleared.get(0).sType$Default()
                    .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT);
            VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, cleared, null, null);
        }
    }

    private void destroyGuideImages() {
        for (RtImage reservoir : reservoirImages) {
            reservoir.destroy();
        }
        reservoirImages = new RtImage[0];
        for (RtImage reservoir : reservoirSampleImages) {
            reservoir.destroy();
        }
        reservoirSampleImages = new RtImage[0];
        reservoirWrite = 0;
        if (gNormal != null) {
            gNormal.destroy();
            gNormal = null;
        }
        if (gAlbedo != null) {
            gAlbedo.destroy();
            gAlbedo = null;
        }
        if (gDepth != null) {
            gDepth.destroy();
            gDepth = null;
        }
        if (gMotion != null) {
            gMotion.destroy();
            gMotion = null;
        }
        if (gSpecAlbedo != null) {
            gSpecAlbedo.destroy();
            gSpecAlbedo = null;
        }
        if (gSpecMotion != null) {
            gSpecMotion.destroy();
            gSpecMotion = null;
        }
        if (volumeDebug != null) {
            volumeDebug.destroy();
            volumeDebug = null;
        }
        if (rrOutput != null) {
            rrOutput.destroy();
            rrOutput = null;
        }
    }

    private void ensureOutput(RtContext ctx, int width, int height) {
        // Debug presentation is downstream of the ordinary frame graph and must not change the image
        // being inspected. In particular, toggling it must not rebuild at native resolution or disable
        // the RR path whose render-resolution guide inputs the debug pass visualizes.
        boolean rrEnabled = RtDlssRr.enabled();
        int rrQuality = rrEnabled ? RtDlssRr.quality() : Integer.MIN_VALUE;
        boolean hdrEnabled = CausticaConfig.Rt.Hdr.enabled();
        if (output != null && continuationQueue != null
                && displayImage != null && rrOutput != null
                && bloomLevels.length > 0 && exposure.ready()
                && displayW == width && displayH == height
                && renderSizeRrEnabled == rrEnabled && renderSizeRrQuality == rrQuality
                && renderSizeHdrEnabled == hdrEnabled && hdrEnabled == (hdrDisplayImage != null)) {
            return;
        }
        ctx.waitIdle(); // resize is rare; no in-flight frame may use the old image/descriptor
        // Every image created below defers its UNDEFINED->GENERAL init barrier into one synchronous
        // submit, instead of one submit+wait per image across a 15+-image rebuild.
        ctx.beginImageInitBatch();
        try {
            if (displayImage != null) {
                displayImage.destroy();
            }
            if (hdrDisplayImage != null) {
                hdrDisplayImage.destroy();
                hdrDisplayImage = null;
            }
            destroyBloomLevels();
            if (output != null) {
                output.destroy();
            }
            if (continuationQueue != null) {
                continuationQueue.destroy();
                continuationQueue = null;
            }
            destroyGuideImages();

            displayW = width;
            displayH = height;
            // The path tracer + its guide buffers run at render res; DLSS-RR (or a fallback blit) upscales
            // to display res. With RR off there is no reconstruction pass, so trace at 1:1 for a faithful reference.
            // With RR on, ask NGX what render resolution its chosen quality mode actually expects rather
            // than assuming a fixed ratio: different quality modes (and driver versions) use different
            // ratios, and DLSSD's own optimal-settings query is the source of truth for what it will accept.
            int[] optimal = rrEnabled ? RtDlssRr.INSTANCE.queryOptimalRenderSize(width, height) : null;
            renderW = optimal != null ? optimal[0] : width;
            renderH = optimal != null ? optimal[1] : height;
            renderSizeRrEnabled = rrEnabled;
            renderSizeRrQuality = rrQuality;
            renderSizeHdrEnabled = hdrEnabled;

            // RT traces and DLSS-RR reconstruct scene-linear ACEScg in an HDR R16G16B16A16_SFLOAT target,
            // so radiance > 1 and wide-gamut colour survive to the display seam. displayImage stays
            // R8G8B8A8 to match the main target it is copied into
            // (vkCmdCopyImage requires texel-size-compatible formats).
            output = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "trace color " + renderW + "x" + renderH);
            long pixelRecords = Math.multiplyExact((long) renderW, (long) renderH);
            long continuationBytes = Math.multiplyExact(
                    Math.multiplyExact(pixelRecords, 2L), PATH_RECORD_BYTES);
            continuationQueue = ctx.createBuffer(continuationBytes,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                    "path continuation queue " + renderW + "x" + renderH + "x2");
            displayImage = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R8G8B8A8_UNORM, "RT display image " + width + "x" + height);
            // PQ-encoded ([0,1], ST.2084) HDR display image, written in parallel by display.comp when HDR
            // mode is active. Allocated only while HDR output is enabled: at display resolution it costs
            // ~29 MB at 1440p / ~66 MB at 4K and is never read when the flag is off. While it is absent
            // the display pipeline's HDR slot binds the 1x1 placeholder (see hdrDisplayBindingView).
            if (hdrEnabled) {
                hdrDisplayImage = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "RT HDR display image " + width + "x" + height);
            }
            // Bloom pyramid. Level 0 is half display resolution (the prefilter's 13-tap already covers a 5x5
            // display-pixel footprint, so nothing is lost by starting there); each further level halves again
            // until the look package's level count or the smallest useful size is reached.
            int bloomWidth = Math.max(1, (width + 1) / 2);
            int bloomHeight = Math.max(1, (height + 1) / 2);
            int bloomLevelCount = RtBloomPipeline.levelsFor(bloomWidth, bloomHeight, LOOK.bloom().levels());
            bloomLevels = new RtImage[bloomLevelCount];
            for (int level = 0; level < bloomLevelCount; level++) {
                bloomLevels[level] = ctx.createStorageImage(bloomWidth, bloomHeight,
                        VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "RT bloom level " + level + " " + bloomWidth + "x" + bloomHeight);
                bloomWidth = Math.max(1, bloomWidth / 2);
                bloomHeight = Math.max(1, bloomHeight / 2);
            }
            // Guide buffers match the trace (render) resolution; DLSS-RR consumes them at render res.
            gNormal = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "guide normal roughness " + renderW + "x" + renderH);
            gAlbedo = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "guide diffuse albedo " + renderW + "x" + renderH);
            gDepth = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R32_SFLOAT, "guide linear depth " + renderW + "x" + renderH);
            gMotion = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16_SFLOAT, "guide motion " + renderW + "x" + renderH);
            gSpecAlbedo = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "guide specular albedo " + renderW + "x" + renderH);
            gSpecMotion = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16_SFLOAT, "guide specular motion " + renderW + "x" + renderH);
            // Bounce-0 RIS reservoir ping-pong. Both sides are render-res RGBA32F: the trace writes one and
            // reads the other, and they alternate per traced frame. They are not DLSS-RR guides, so they get
            // their own bindings rather than a guide slot. The format must match the rgba32f the shaders
            // declare — a narrower texel would make every reservoir access alias its neighbour.
            reservoirImages = new RtImage[]{
                    ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R32G32B32A32_SFLOAT,
                            "reservoir A " + renderW + "x" + renderH),
                    ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R32G32B32A32_SFLOAT,
                            "reservoir B " + renderW + "x" + renderH)};
            // The survivor's rectangle coordinates. Half precision is deliberate: a quadrature audit of the
            // merge showed quantising (s,t) to fp16 changes the estimator's bias by less than its own noise.
            reservoirSampleImages = new RtImage[]{
                    ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16_SFLOAT,
                            "reservoir sample A " + renderW + "x" + renderH),
                    ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16_SFLOAT,
                            "reservoir sample B " + renderW + "x" + renderH)};
            reservoirWrite = 0;
            // Per-pixel volumetric diagnostics, same render resolution as the guides the primary pass
            // writes them from.
            volumeDebug = ctx.createStorageImage(renderW, renderH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "volume debug " + renderW + "x" + renderH);
            // Display-res RT image the display mapper reads. Always present (DLSS-RR target, or blit-upscale fallback).
            rrOutput = ctx.createStorageImage(width, height, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "DLSS-RR output " + width + "x" + height);
            // Also creates the 1x1 exposure image, so it stays inside the init-batch window.
            exposure.ensureResources(ctx);
        } finally {
            ctx.endImageInitBatch();
        }

        mvHasPrev = false; // recreated images -> first MV frame is zero
        waterWaveTimeValid = false;
        if (worldPipeline != null) {
            worldPipeline.setStorageImage(output.view);
            bindGuideImages();
        }
        RtToneLut boundLookLut = lookLut;
        displayPipeline.setImages(displayImage.view, rrOutput.view, exposure.image().view, hdrDisplayBindingView(ctx),
                sdrToneLut.view(), sdrToneLut.sampler(), hdrToneLut.view(), hdrToneLut.sampler(),
                boundLookLut.view(), boundLookLut.sampler(), bloomLevels[0].view, bloomPipeline.sampler(),
                exposure.stateBuffer());
        bloomPipeline.setImages(rrOutput.view, exposure.image().view, bloomLevels);
        debugPresentPipeline.setImages(displayImage.view, gNormal.view, gAlbedo.view, gDepth.view,
                gMotion.view, gSpecAlbedo.view, gSpecMotion.view, rrOutput.view, exposure.image().view,
                volumeDebug.view, exposure.stateBuffer());
    }

    /**
     * The view bound into the display pipeline's HDR output slot: the real HDR display image while HDR
     * output is enabled, otherwise the persistent 1x1 placeholder that keeps the format-qualified
     * descriptor legally bound without pinning a display-resolution image.
     */
    private long hdrDisplayBindingView(RtContext ctx) {
        if (hdrDisplayImage != null) {
            return hdrDisplayImage.view;
        }
        if (hdrBindingPlaceholder == null) {
            hdrBindingPlaceholder = ctx.createStorageImage(1, 1, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "RT HDR display placeholder");
        }
        return hdrBindingPlaceholder.view;
    }

    private void destroyBloomLevels() {
        for (RtImage level : bloomLevels) {
            if (level != null) {
                level.destroy();
            }
        }
        bloomLevels = new RtImage[0];
    }

    /**
     * Compute this frame's motion-vector push data: the matrix that projects a current world point
     * into the previous frame's clip space, plus the per-frame camera translation. On the first frame
     * (or after a reset) push the current view-projection with zero delta so MVs come out zero.
     */
    private void updateMotion() {
        // The reservoir pair advances on the same cadence as the motion-vector history: exactly one step
        // per frame that runs the trace, so "the image the previous traced frame wrote" is always the one
        // bound as prev.
        reservoirWrite ^= 1;
        bindReservoirImages();
        mvCurProjView.set(frameProjection).mul(frameViewRotation);
        if (mvHasPrev) {
            mvPushMatrix.set(mvPrevProjView);
            mvCamDeltaX = (float) (camX - mvPrevCamX);
            mvCamDeltaY = (float) (camY - mvPrevCamY);
            mvCamDeltaZ = (float) (camZ - mvPrevCamZ);
        } else {
            mvPushMatrix.set(mvCurProjView);
            mvCamDeltaX = 0f;
            mvCamDeltaY = 0f;
            mvCamDeltaZ = 0f;
        }
        mvPrevProjView.set(mvCurProjView);
        mvPrevCamX = camX;
        mvPrevCamY = camY;
        mvPrevCamZ = camZ;
        mvHasPrev = true;
    }

    private void recordFrame(RtContext ctx, RtPipeline active, GpuTexture nativeColor) {
        long dstImage = vkImage(nativeColor);
        var encoder = (VulkanCommandEncoder) ((CommandEncoderAccessor) RenderSystem.getDevice().createCommandEncoder()).caustica$getBackend();
        RtGpuExecutor gpuExecutor = ctx.gpuExecutor();
        // Reserve the graphics-use value that guards this frame's reusable TLAS and entity resources.
        RtGpuExecutor.GraphicsUse graphicsUse = gpuExecutor.beginGraphicsUse(encoder);
        RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter = gpuExecutor.graphicsUseWaiter();
        // Reuse a completed readback slot, then latch one pre-exposure value for both raygen and resolve.
        // This belongs after the timeline snapshot and before any world push data is written.
        exposure.beginFrame(graphicsUseWaiter, nightFactor());
        pendingGraphicsUse = graphicsUse;
        RtEntities.FrameEntities frameEntities = null;
        VkCommandBuffer cmd = encoder.allocateAndBeginTransientCommandBuffer();
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_COMMAND_BUFFER, cmd.address(), "composite command buffer");
        int debugView = debugView();
        RtTerrain terrain = RtTerrain.currentOrNull();
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope frameLabel = RtDebugLabels.scope(ctx, cmd, "composite frame")) {
            // RR drives the upscale: trace + jitter at render res, DLSS-RR denoises+upscales to display.
            // A debug view observes this ordinary path; it never changes jitter or disables RR.
            boolean rrPath = RtDlssRr.enabled();
            float jitterX = 0f;
            float jitterY = 0f;
            if (rrPath) {
                CausticaJitter.INSTANCE.prepare(renderW, renderH, displayW);
                jitterX = CausticaJitter.INSTANCE.jitterPixelsX() * jitterSignX();
                jitterY = CausticaJitter.INSTANCE.jitterPixelsY() * jitterSignY();
            }

            boolean rrDone = false;
            // Select the next BDA ring slot; the generated WorldPushData serializer fills it once all
            // frame-derived values (including entity addresses and block-breaking entries) are known.
            pushSlot = (pushSlot + 1) % PUSH_RING;
            PushSlot selectedPushSlot = pushRing[pushSlot];
            graphicsUseWaiter.await(selectedPushSlot.graphicsUse);
            selectedPushSlot.graphicsUse.mark(graphicsUse);
            RtBuffer pushBuf = selectedPushSlot.buffer;
            ByteBuffer push = MemoryUtil.memByteBuffer(pushBuf.mapped, WORLD_PUSH_SIZE);
            frameInvViewProj.set(frameProjection).mul(frameViewRotation).invert();
            // flags: camera-in-water (so the path tracer starts in the water medium when the eye is
            // submerged, fixing the air→water first-segment orientation) and animated water normals.
            // Bit 1 remains unused to avoid conflicting with stale external readers.
            int flags = 0;
            var level = Minecraft.getInstance().level;
            if (level != null) {
                cameraBlockPos.set(Mth.floor(camX), Mth.floor(camY), Mth.floor(camZ));
                // Height-aware, mirroring vanilla's own Camera.getFluidInCamera(): a plain block-granular
                // test wrongly flags the eye submerged anywhere in a water column's top block, even well
                // above its actual surface (shallow/flowing water, or standing with your head just over a
                // source block).
                FluidState fs = level.getFluidState(cameraBlockPos);
                if (fs.is(FluidTags.WATER) && camY < cameraBlockPos.getY() + fs.getHeight(level, cameraBlockPos)) {
                    flags |= 0b01;
                }
            }
            if (waterWaves()) {
                flags |= 0b10000; // animated water wave normals
            }
            if (CausticaConfig.Rt.Sky.VANILLA_SUN.value()) {
                flags |= 0b100000; // draw the sun from the celestials sprite rather than as a disc
            }
            if (CausticaConfig.Rt.Sky.VANILLA_MOON.value()) {
                flags |= 0b1000000; // likewise for the moon
            }

            // Water parameters: camera-biome tint plus wrapped animation time. Per-water-body tint
            // comes from the primitive; this is the fallback for a camera already inside the medium.
            float wtr = 0.25f, wtg = 0.46f, wtb = 0.9f; // neutral ocean-ish default if no level/biome
            if (level != null) {
                int wc = BiomeColors.getAverageWaterColor(level, cameraBlockPos);
                wtr = ((wc >> 16) & 0xFF) / 255f;
                wtg = ((wc >> 8) & 0xFF) / 255f;
                wtb = (wc & 0xFF) / 255f;
            }
            // Accumulate the wave clock from a clamped per-frame step rather than reading it off the wall
            // clock, so changing the speed control alters the rate without teleporting the phase. The step
            // clamp keeps a stall or a pause from fast-forwarding the sea.
            long waveClockNow = System.nanoTime();
            float waveClockStep = waveClockNanos == 0L ? 0f
                    : Math.min(Math.max((waveClockNow - waveClockNanos) / 1.0e9f, 0f), 0.25f);
            waveClockNanos = waveClockNow;
            // Wrapped hourly to keep the phase argument well inside float32's precise range.
            waterWaveClock = (waterWaveClock + waveClockStep) % 3600.0f;
            // Wind drift for the fog billows and the cloud streaks: integrate speed x direction
            // per frame (the same clamped step the wave clock uses, so a pause cannot teleport the
            // fields), then wrap the fog pair per axis. Unconditional — the drift stays coherent
            // across tier toggles, and the accumulators are two multiplies a frame.
            if (waveClockStep > 0f) {
                double fogSpeed = CausticaConfig.Rt.Volume.FOG_WIND_SPEED.value();
                double fogAzim = Math.toRadians(CausticaConfig.Rt.Volume.FOG_WIND_ANGLE.value());
                fogDriftX += fogSpeed * Math.cos(fogAzim) * waveClockStep;
                fogDriftZ += fogSpeed * Math.sin(fogAzim) * waveClockStep;
                fogDriftX -= 4096.0 * Math.floor(fogDriftX / 4096.0);
                fogDriftZ -= 4096.0 * Math.floor(fogDriftZ / 4096.0);
                double cloudSpeed = CausticaConfig.Rt.Volume.CLOUD_WIND_SPEED.value();
                double cloudAzim = Math.toRadians(CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE.value());
                cloudDriftX += cloudSpeed * Math.cos(cloudAzim) * waveClockStep;
                cloudDriftZ += cloudSpeed * Math.sin(cloudAzim) * waveClockStep;
                // The rate is a warp-lattice cell every ~80 s at 60 fps: fast enough that a player who
                // looks away and back sees different shapes, slow enough that nothing boils and RR reads
                // the field as the same slowly-moving signal it reads the drift as.
                cloudEvolve += CLOUD_EVOLVE_SPEED * waveClockStep;
            }
            float waterWaveTime = waterWaveClock;
            float waterWaveDelta = waterWaveTime - previousWaterWaveTime;
            // A first frame, long pause, or one-hour phase wrap has no adjacent wave frame to reproject.
            // Use the current phase so the reflection MV is neutral instead of manufacturing a huge jump.
            float priorWaterWaveTime = waterWaveTimeValid
                    && waterWaveDelta >= 0f && waterWaveDelta <= 0.25f
                    ? previousWaterWaveTime : waterWaveTime;
            previousWaterWaveTime = waterWaveTime;
            waterWaveTimeValid = true;
            Float4 waterParams = linearAcesCgFromSrgb(wtr, wtg, wtb, waterWaveTime);
            // Wave-shape controls, read per frame: height and wavelength scale the spectrum in the shader,
            // clock rate scales the accumulated time above. All 1 is the authored look. The fourth lane is
            // not a wave control but rides along in the same push: how much of its Fresnel reflection water
            // keeps.
            Float4 waterWave = new Float4(CausticaConfig.Rt.Composite.WATER_WAVE_HEIGHT.value(),
                    CausticaConfig.Rt.Composite.WATER_WAVE_SCALE.value(),
                    CausticaConfig.Rt.Composite.WATER_WAVE_SPEED.value(),
                    CausticaConfig.Rt.Composite.WATER_REFLECTION.value());
            // Wave-domain anchor: the terrain rebase origin reduced mod 4096 (kept small for shader
            // float precision). hitPos.xz (rebased) + anchor reconstructs a world-pinned coordinate, so the
            // ripple pattern stays fixed in the world as the player moves and the rebase origin shifts.
            Float4 waterAnchor = new Float4(terrain.blockX & WATER_ANCHOR_MASK,
                    terrain.blockZ & WATER_ANCHOR_MASK, priorWaterWaveTime, 0f);

            // Rebuild the TLAS this frame from static section instances merged with dynamic entity
            // instances, bind it into the pipeline's descriptor ring, record the build, then barrier so
            // the trace sees the finished TLAS. Section BLASes are already built (async, by RtTerrain);
            // only the cheap instance-level TLAS is rebuilt per frame. Retired terrain geometry/table
            // generations are reclaimed by graphics-timeline completion.
            // Entity BLASes are built inline below and merged into the per-frame TLAS. geomTableAddr
            // feeds the hit shader entity path (per-prim normal/tint) and motion vectors.
            RtEntities.FrameEntities fe = RtEntities.INSTANCE.beginFrame(ctx, terrain.staticInstances(),
                    terrain.blockX, terrain.blockY, terrain.blockZ, camX, camY, camZ, frameProjection, frameViewRotation);
            frameEntities = fe;
            // Block-breaking overlay: resolves each destroy-stage RenderType's texture into the
            // SAME bindless entity-texture array (destroy_stage_N.png is a standalone Sampler0 texture,
            // not a block-atlas sprite — see ModelBakery.BREAKING_LOCATIONS/DESTROY_TYPES), so any newly
            // resolved slot rides along with the uploadPending() call right below.
            BreakEntry[] breaking = breakingEntries(terrain);
            SkyPush sky = skyPush();
            // Volumetric flags. Fog follows the sky LUT's own sky-light gate: its ambient term reads the
            // sky irradiance basis, which is only baked where the sky lights the world, so a dimension
            // without sky light must not read it. The volume debug views ask the primary raygen to fill
            // the volume debug image this frame.
            boolean skyLit = sky.skyEnvironment().x() > 0f;
            if (CausticaConfig.Rt.Volume.FOG_ENABLED.value() && skyLit) {
                flags |= WORLD_FLAG_FOG;
            }
            if (CausticaConfig.Rt.Volume.FOG_SHADOW.value()) {
                flags |= WORLD_FLAG_VOLUME_SHADOW;
            }
            boolean fogMarchOn = CausticaConfig.Rt.Volume.FOG_ENABLED.value() && skyLit
                    && CausticaConfig.Rt.Volume.FOG_MARCH.value();
            if (fogMarchOn) {
                flags |= WORLD_FLAG_VOLUME_MARCH;
            }
            // Spray rides the marched fog integrator (a wavefield density has no closed form), so the
            // flag only goes up when that tier is actually running and the strength is non-zero.
            boolean sprayOn = fogMarchOn && CausticaConfig.Rt.Composite.WATER_SPRAY.value() > 0f;
            if (sprayOn) {
                flags |= WORLD_FLAG_SPRAY;
            }
            // The planar cloud sheet and its shadow field. Same sky-light gate as the fog (the
            // sheet's ambient reads the sky dome), independent of every fog switch — cloud shadows
            // work with the fog off. The coverage lane doubles as the shader gate, so a coverage
            // of zero leaves every consumer inert even with the flag up.
            boolean cloudOn = CausticaConfig.Rt.Volume.CLOUD_ENABLED.value() && skyLit;
            if (cloudOn) {
                flags |= WORLD_FLAG_CLOUD;
            }
            // The volumetric tier is a second bit, not a replacement of the first: every consumer that
            // only wants the layer's mean opacity (the NEE probes, the fog sun legs, a continuation ray)
            // keeps reading WORLD_FLAG_CLOUD, and only sky.rmiss branches on this one.
            boolean volumetricCloudOn = cloudOn
                    && CausticaConfig.Rt.Volume.CLOUD_VOLUMETRIC.value();
            if (volumetricCloudOn) {
                flags |= WORLD_FLAG_CLOUD_VOLUMETRIC;
            }
            int debugViewValue = debugView();
            if (skyLit && debugViewValue >= DEBUG_VIEW_VOLUME_FIRST && debugViewValue <= DEBUG_VIEW_VOLUME_LAST) {
                flags |= WORLD_FLAG_VOLUME_DEBUG;
            }
            // The volume debug image reads the push block rather than the flags, so the fog density is
            // pushed as zero while the fog is off — otherwise the debug views would keep showing a fog
            // the flags no longer render.
            boolean fogOn = CausticaConfig.Rt.Volume.FOG_ENABLED.value() && skyLit;
            // Weather drives the volume block: the levels are pushed whether or not a feature currently
            // consumes them, so every volume shader reads one consistent weather snapshot. The previous
            // frame's rain level rides along so temporal-stability work can interpolate.
            float partialTick = Minecraft.getInstance().getDeltaTracker().getGameTimeDeltaPartialTick(false);
            float rainLevel = level.getRainLevel(partialTick);
            float thunderLevel = level.getThunderLevel(partialTick);
            float storm = CausticaConfig.Rt.Volume.stormLevel(rainLevel, thunderLevel);
            float cloudRainBoost = CausticaConfig.Rt.Volume.CLOUD_RAIN_BOOST.value();
            // Fog density over the day: dawn and dusk are the peak (low-sun ground mist is the look
            // people expect from fog), while noon and night keep a fraction of it. Sun elevation is
            // sin(2π·dayFraction) against the same fraction the GameTime lane pushes: +1 at noon,
            // 0 at 06:00/18:00, -1 at midnight. A gaussian in elevation (σ ≈ 14°) is the twilight
            // term; smoothstep plateaus hold noon at 20% and night at 25% of the peak so the fog
            // never fully leaves. Both plateaus ramp in past ±0.10 elevation so the twilight peak
            // owns the horizon stretch unopposed.
            float timeOfDayFrac = (level.getGameTime() % 24000L + partialTick) / 24000.0f;
            float sunElevation = (float) Math.sin(timeOfDayFrac * 2.0 * Math.PI);
            float twilight = (float) Math.exp(-(sunElevation * sunElevation) / (0.25f * 0.25f));
            float dayPlateau = Math.clamp((sunElevation - 0.10f) / 0.30f, 0.0f, 1.0f);
            dayPlateau = dayPlateau * dayPlateau * (3.0f - 2.0f * dayPlateau);
            float nightPlateau = Math.clamp((-sunElevation - 0.10f) / 0.30f, 0.0f, 1.0f);
            nightPlateau = nightPlateau * nightPlateau * (3.0f - 2.0f * nightPlateau);
            float fogDayFactor = Math.max(twilight,
                    Math.max(dayPlateau * 0.20f, nightPlateau * 0.25f));
            new WorldPushData(
                    frameInvViewProj,
                    new Float3((float) (camX - terrain.blockX), (float) (camY - terrain.blockY),
                            (float) (camZ - terrain.blockZ)),
                    (int) frameCounter,
                    mvPushMatrix,
                    new Float3(mvCamDeltaX, mvCamDeltaY, mvCamDeltaZ),
                    spp(),
                    new Float2(jitterX, jitterY),
                    flags,
                    maxBounces(),
                    sky.celestial(),
                    sky.look0(),
                    sky.look1(),
                    sky.look2(),
                    sky.look3(),
                    sky.skyEnvironment(),
                    sky.skyboxState(),
                    sky.dimensionLight(),
                    sky.dimensionAmbient(),
                    sky.dimensionFog(),
                    sky.look4(),
                    sky.sunUv(),
                    sky.moonUv(),
                    sky.endFlashUv(),
                    waterParams,
                    waterAnchor,
                    waterWave,
                    mvCurProjView,
                    breaking.length,
                    breaking,
                    // RIS emitter NEE: candidate count (0 = emitter NEE off; the shader also requires
                    // lightCount > 0, so an empty buffer leaves only direct-hit emission). The light buffer
                    // device addresses themselves are pc.light*Addr — every 64-bit address lives in the
                    // push-constant block now, not here.
                    new Float4(terrain.lightRebaseOffsetX(), terrain.lightRebaseOffsetY(),
                            terrain.lightRebaseOffsetZ(), terrain.lightInvGlobalPowerSum()),
                    new Float4(terrain.lightGridOriginX(), terrain.lightGridOriginY(), terrain.lightGridOriginZ(), 16f),
                    new Int4(terrain.lightGridDimX(), terrain.lightGridDimY(), terrain.lightGridDimZ(), 0),
                    terrain.lightCount(),
                    terrain.lightTranslationCount(),
                    terrain.lightTranslationApplies() ? 1 : 0,
                    CausticaConfig.Rt.Lights.RIS_CANDIDATES.value(),
                    CausticaConfig.Rt.Lights.TEMPORAL_REUSE.value() ? 1 : 0,
                    CausticaConfig.Rt.Lights.TEMPORAL_HISTORY.value(),
                    // Must be the SAME value the exposure resolve divides out this frame (it reads it
                    // from the same RtExposure accessor), or the two stop cancelling.
                    exposure.preExposure(),
                    CausticaConfig.Rt.Composite.WATER_OPACITY.value(),
                    // Vanilla's own GameTime. The end portal's layer scroll is authored against a value
                    // that completes one cycle per in-game day, so a monotonic seconds clock churns it
                    // more than a thousand times too fast. Same fraction the fog day factor consumed.
                    timeOfDayFrac,
                    // ---- Volumetric block (volume.slang). Fog zero means no fog: the shaders gate on
                    // the flags bits, but a zero density keeps the WorldPush honest on its own. The fog
                    // base is pushed already rebased so the height profile is world-pinned. Cloud and
                    // spray lanes ride along at zero until their phases arrive. The density rides the
                    // fog day factor computed above (dawn/dusk peak, noon 20%, night 25% of it).
                    new Float4(fogOn
                            ? CausticaConfig.Rt.Volume.FOG_DENSITY.value() * fogDayFactor
                                    * (1f + CausticaConfig.Rt.Volume.FOG_RAIN_BOOST.value() * rainLevel)
                            : 0f,
                            CausticaConfig.Rt.Volume.FOG_HEIGHT.value(),
                            CausticaConfig.Rt.Volume.FOG_BASE_Y.value() - terrain.blockY,
                            CausticaConfig.Rt.Volume.FOG_ANISOTROPY.value()),
                    // x/z the Java-accumulated wind drift (per-axis, pre-wrapped in whole
                    // 4096-block periods — the fog lattice's exact period), z FBM noise amplitude
                    // 0..1, w the shadow-cull floor (both fog tiers read it, so unlike the noise
                    // lanes it rides ungated). The noise amplitude is zero unless the marched tier
                    // runs, so the analytic frame samples no noise.
                    new Float4((float) fogDriftX,
                            (float) fogDriftZ,
                            fogMarchOn ? CausticaConfig.Rt.Volume.FOG_NOISE.value() : 0f,
                            CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR.value()),
                    linearAcesCgFromSrgb(CausticaConfig.Rt.Volume.FOG_ALBEDO.value(),
                            CausticaConfig.Rt.Volume.FOG_ALBEDO.value(),
                            CausticaConfig.Rt.Volume.FOG_ALBEDO.value(),
                            CausticaConfig.Rt.Volume.FOG_AMBIENT.value()),
                    // ---- Cloud block. x coverage (0 = inert: every consumer early-outs), y the
                    // cloud base already rebased like fog0.z, z the slab thickness the sheet's and
                    // the shadow's optical depths scale with, w the density multiplier (the shadow
                    // strength).
                    // Coverage and density are the WEATHER-LIFTED values, not the sliders': rain decks
                    // the sky over and shadows harder, and every consumer of these two lanes - sheet,
                    // walk, shadow field, NEE, fog legs - reads the same lifted numbers because they were
                    // lifted before the push rather than re-derived per consumer.
                    new Float4(cloudOn ? CausticaConfig.Rt.Volume.rainCoverage(
                            CausticaConfig.Rt.Volume.CLOUD_COVERAGE.value(), storm, cloudRainBoost) : 0f,
                            CausticaConfig.Rt.Volume.CLOUD_ALTITUDE.value() - terrain.blockY,
                            CausticaConfig.Rt.Volume.CLOUD_THICKNESS.value(),
                            CausticaConfig.Rt.Volume.rainDensity(
                                    CausticaConfig.Rt.Volume.CLOUD_DENSITY.value(), storm, cloudRainBoost)),
                    // x the volumetric tier's vertical profile mix (0 while that tier is off, so the
                    // planar frame pushes exactly the block it always did), y the wind azimuth (the
                    // streak frame's direction; the drift itself accumulates into cloud2.zw Java-side),
                    // z the localized-coverage strength, w the fade distance (blocks). Gated lanes ride
                    // zero while off.
                    new Float4(volumetricCloudOn
                                    ? CausticaConfig.Rt.Volume.CLOUD_TYPE.value() : 0f,
                            cloudOn ? (float) Math.toRadians(
                                    CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE.value()) : 0f,
                            cloudOn ? CausticaConfig.Rt.Volume.CLOUD_PATCHINESS.value() : 0f,
                            cloudOn ? CausticaConfig.Rt.Volume.CLOUD_FADE_DISTANCE.value() : 0f),
                    // ---- Cloud anchoring block. xy the EXACT terrain origin (no mod-4096: the
                    // streak lattice has no period, so the anchor must not wrap either), zw the
                    // Java-accumulated cloud wind drift per axis (unwrapped doubles). MUST stay
                    // before cloudColor: the generated record orders this block cloud0, cloud1,
                    // cloud2, cloudColor (see world_common.slang), and the arguments here are
                    // positional — swapping these two pushed the drift into the ambient-weight
                    // lane and the ambient weight into the anchor (clouds froze, then burned).
                    new Float4((float) terrain.blockX, (float) terrain.blockZ,
                            (float) cloudDriftX, (float) cloudDriftZ),
                    // x the field's evolution phase (unconditional, like the drift: a tier toggle must
                    // not rewind the sky), yz unused, w the sheet's ambient-term weight.
                    new Float4((float) cloudEvolve, 0f, 0f, 1f),
                    // z spray strength: the setting's base lifted by rain (rain is what spray is for),
                    // and zero unless the spray flag went up with the marched tier.
                    new Float4(rainLevel, thunderLevel,
                            sprayOn ? CausticaConfig.Rt.Composite.WATER_SPRAY.value()
                                    * (0.5f + 0.5f * rainLevel) : 0f,
                            previousRainLevel),
                    // x march steps per segment, y jitter strength 0..1, z reserved (max integration
                    // distance — left at 0: truncating the integral would desynchronise the out fogT
                    // from the throughput attenuation riding it), w the volumetric cloud slab's step
                    // count (zero unless that tier is on, which is how the shader's own early-out and
                    // the flag agree). Zero on the analytic fog tier, whose integrator takes no march
                    // parameters.
                    new Float4(fogMarchOn ? (float) CausticaConfig.Rt.Volume.FOG_MARCH_STEPS.value() : 0f,
                            fogMarchOn ? CausticaConfig.Rt.Volume.FOG_JITTER.value() : 0f,
                            0f,
                            volumetricCloudOn
                                    ? (float) CausticaConfig.Rt.Volume.CLOUD_STEPS.value() : 0f),
                    // ---- Water-volume block. x turbidity (the water's sigmaS/sigmaT fraction), y the
                    // water phase's Henyey–Greenstein g (forward-biased per the plan's 0.5..0.9 range),
                    // z spray band height in blocks, w the band's base: the sea level, already rebased
                    // like fog0.z so the band stays pinned as the terrain rebase shifts.
                    new Float4(CausticaConfig.Rt.Composite.WATER_TURBIDITY.value(), 0.6f, 1.5f,
                            (float) (level.getSeaLevel() - terrain.blockY))
            ).write(push);
            previousRainLevel = rainLevel;
            pushBuf.flush(0L, WORLD_PUSH_SIZE);
            // Upload any entity textures registered this frame into the bindless set before the trace.
            RtEntityTextures.INSTANCE.uploadPending(active, atlasSampler(ctx));
            // Build the entity BLAS, the TLAS that references it and the terrain BLAS, then the trace.
            // Barriers separate each stage; the graphics-use timeline guards resource reuse.
            if (!fe.blas().isEmpty()) {
                try (RtFrameStats.Scope ignored = RtFrameStats.FRAME.stage("entity.blasRecord")) {
                    RtAccel.recordBlasBuilds(ctx, cmd, fe.blas());
                }
                VulkanCommandEncoder.memoryBarrier(cmd, stack); // entity BLAS writes visible to the TLAS build
            }
            RtAccel.PreparedTlas frameTlas;
            try (RtFrameStats.Scope ignored = RtFrameStats.FRAME.stage("frame.prepareTlas")) {
                frameTlas = RtAccel.prepareTlas(ctx, fe.baseInstances(), fe.dynamicInstances(), tlasRing,
                        graphicsUse);
            }
            active.setTlas(frameTlas.accel.handle, graphicsUse, graphicsUseWaiter);
            currentTlasHandle = frameTlas.accel.handle;
            try (RtFrameStats.Scope ignored = RtFrameStats.FRAME.stage("frame.recordTlas")) {
                RtAccel.recordTlasBuild(ctx, cmd, frameTlas);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // TLAS build visible to the trace

            // Push the BDA ring slot's address plus the small hot subset used directly by the shaders.
            // Every 64-bit device address the trace needs lives here, not behind worldPushAddr: the
            // section/entity/material tables are read from world.rahit/world.rchit, which never load
            // WorldPush at all, and the RIS light buffers are read from world.rgen's hot inner loop, so
            // none of them should cost an extra BDA dereference to find.
            ByteBuffer pushConstants = stack.malloc(WorldPushConstantsData.BYTE_SIZE);
            // The Materials page's switches, packed into one word for the same reason wetness and the
            // parallax depth ride in this block: world.rchit evaluates materials and never dereferences
            // WorldPush, so a per-hit world load would be paid on the hottest path in the frame.
            int materialToggles =
                    (CausticaConfig.Rt.Materials.TEXTURE_EMISSION.value()
                            ? CausticaConfig.Rt.Materials.TOGGLE_TEXTURE_EMISSION : 0)
                    | (CausticaConfig.Rt.Materials.BLOCK_EMISSION.value()
                            ? CausticaConfig.Rt.Materials.TOGGLE_BLOCK_EMISSION : 0)
                    | (CausticaConfig.Rt.Materials.SUBSURFACE.value()
                            ? CausticaConfig.Rt.Materials.TOGGLE_SUBSURFACE : 0)
                    | (CausticaConfig.Rt.Materials.POROSITY.value()
                            ? CausticaConfig.Rt.Materials.TOGGLE_POROSITY : 0)
                    | (CausticaConfig.Rt.Materials.ENTITY_PARALLAX.value()
                            ? CausticaConfig.Rt.Materials.TOGGLE_ENTITY_PARALLAX : 0);
            new WorldPushConstantsData(pushBuf.deviceAddress, terrain.tableAddress(), fe.geomTableAddr(),
                    RtMaterialRegistry.INSTANCE.tableAddress(),
                    terrain.lightBufferAddress(), terrain.lightAliasBufferAddress(),
                    terrain.lightLocalAliasBufferAddress(), terrain.lightGridCellBufferAddress(),
                    terrain.lightGridSpanBufferAddress(), terrain.lightTranslationAddress(),
                    continuationQueue.deviceAddress,
                    (int) frameCounter,
                    CausticaConfig.Rt.Composite.WETNESS.value(),
                    CausticaConfig.Rt.Composite.PARALLAX_DEPTH.value(),
                    CausticaConfig.Rt.Materials.PARALLAX_LAYERS.value(),
                    materialToggles).write(pushConstants);
            // Sky LUTs, from the same WorldPush slot the trace is about to read: the sky the LUT holds and
            // the sky the frame shades are built from one set of angles, not two. Recorded here (after the
            // push flush, before the trace) so the miss shader's very first fetch sees this frame's dome.
            // A dimension with no sky light blends the whole atmosphere out, so nothing reads these and the
            // bake is skipped; the first frame back in a lit dimension re-bakes before its own trace.
            if (sky.skyEnvironment().x() > 0f) {
                try (RtFrameStats.Scope ignored = RtFrameStats.FRAME.stage("frame.skyLut")) {
                    skyLut.record(cmd, pushBuf.deviceAddress);
                }
                VulkanCommandEncoder.memoryBarrier(cmd, stack); // sky LUT writes visible to raygen/miss
            }

            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "world primary trace");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.tracePrimary")) {
                active.trace(cmd, renderW, renderH, pushConstants, 0);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // continuation/guide writes visible to pass B
            // The reservoir write only reaches pixels that get a bounce-0 diffuse vertex, so clear this
            // frame's image first: an unwritten texel has to read as "no history" next frame, not as
            // whatever vkAllocateMemory returned, which interpreted as RGBA32F can be NaN or Inf.
            clearReservoir(ctx, cmd, stack);
            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "world indirect trace");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.traceIndirect")) {
                active.trace(cmd, renderW, renderH, pushConstants, 1);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // RT writes visible to DLSS reads
            // DLSS-RR denoise + upscale. The RT pass wrote noisy color (render res) + guides;
            // RR reads them and writes the display-res denoised result straight into rrOutput.
            if (rrPath && RtDlssRr.INSTANCE.ensureFeature(cmd.address(), renderW, renderH, displayW, displayH)) {
                try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "DLSS-RR evaluate");
                     RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.dlssRr")) {
                    rrDone = RtDlssRr.INSTANCE.evaluate(cmd.address(), output, gDepth, gMotion, gAlbedo,
                            gSpecAlbedo, gNormal, gSpecMotion, rrOutput, renderW, renderH, displayW, displayH,
                            -jitterX, -jitterY, frameViewRotation, frameProjection);
                }
            }

            // When DLSS-RR did not produce the display-res image (disabled or a runtime failure), bring
            // the render-res trace up to display res with a linear blit so the display mapper and
            // downstream debug pass always have a valid display-res scene image. With RR off
            // render == display, so this is a 1:1 copy.
            if (!rrDone) {
                VulkanCommandEncoder.memoryBarrier(cmd, stack);
                try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fallback upscale");
                     RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.upscale")) {
                    blitUpscale(cmd, stack, output, rrOutput);
                }
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // rrOutput visible to exposure histogram

            // Auto-exposure meters rrOutput (the post-RR, denoised/converged image), not the raw
            // pre-RR trace: RR has no notion of exposure (DLSS-RR Integration Guide §3.7 — ignore
            // exposure/auto-exposure/sharpness entirely for RR), so this is purely our own metering
            // choice, independent of RR's pipeline placement. Metering the noisy pre-RR buffer made
            // the histogram's log-luminance average biased by Monte-Carlo noise (Jensen's inequality
            // on the concave log()), so the computed exposure drifted with SPP; rrOutput is stable
            // regardless of SPP, keeping exposure consistent.
            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.exposure")) {
                exposure.record(ctx, cmd, stack, rrOutput, gDepth, gAlbedo);
                exposure.recordStateReadback(cmd, stack);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // exposure image visible to the display mapper

            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "bloom");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.bloom")) {
                RtLookPackage.Bloom bloom = LOOK.bloom();
                // The tent radius is in source texels, so it needs no resolution scaling: the pyramid's
                // reach is set by its level count, and each level's texel already scales with the frame.
                bloomPipeline.dispatch(cmd, bloomLevels,
                        bloom.thresholdSceneLinear(), bloom.softKneeFraction(), bloom.radius());
            }

            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "map RT to display");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.displayMap")) {
                displayPipeline.dispatch(cmd, displayW, displayH, CausticaConfig.Rt.Hdr.enabled(),
                        sdrToneLut.size, CausticaConfig.Rt.Tonemap.GAMMA.value(), loadedHdrLutNits,
                        CausticaConfig.Rt.Tonemap.LOOK_ENABLED.value(), lookLut.size,
                        LOOK.bloom().strength() / bloomLevels.length,
                        CausticaConfig.Rt.Tonemap.ADVANCED_GRADING.value(),
                        CausticaConfig.Rt.Tonemap.AGX_GAIN.value(),
                        CausticaConfig.Rt.Tonemap.AGX_SATURATION.value(),
                        // Needs the meter, not just the setting: in manual exposure mode evScene is
                        // never refreshed, so there is no absolute luminance to adapt to.
                        CausticaConfig.Rt.Tonemap.PURKINJE.value() && exposure.isAuto(),
                        CausticaConfig.Rt.Tonemap.PURKINJE_STRENGTH.value());
            }
            hdrWrittenThisFrame = CausticaConfig.Rt.Hdr.enabled();
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // display output visible to debug composite

            if (debugView != 0) {
                // Debug content is composited only after the real scene has completed trace, RR/fallback,
                // exposure, and display mapping. It therefore observes the renderer without perturbing
                // exposure history or feeding literal diagnostic colors through ACES. Debug presentation
                // remains SDR for now; a PQ swapchain uses the existing SDR->PQ conversion path.
                try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "debug present");
                     RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.debugPresent")) {
                    debugPresentPipeline.dispatch(cmd, displayW, displayH, debugView,
                            CausticaConfig.Rt.Exposure.CENTER_WEIGHT_SIGMA.value(),
                            CausticaConfig.Rt.Exposure.CENTER_WEIGHT_FLOOR.value());
                }
                hdrWrittenThisFrame = false;
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "copy composite to main target");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.copyOutput")) {
                VK10.vkCmdCopyImage(cmd, displayImage.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                        dstImage, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack, displayW, displayH));
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(rt composite) failed");
        }
        encoder.execute(cmd); // deferred into the frame's submission — correct for per-frame work
        // Do not attach a merely reserved token: failed recording may never signal it. Once execute succeeds,
        // every owner in this frame's manifest is protected through the final overlay consumer.
        RtEntities.INSTANCE.markGraphicsUse(frameEntities, graphicsUse);
        exposure.markStateReadbackUse(graphicsUse);
    }

    /**
     * Block-breaking overlay: mirrors vanilla's {@code ClientLevel.destructionProgress()} (populated
     * by network packets, independent of the cancelled {@code LevelRenderer.render()}) into the push's
     * {@code breaking[]} list, so {@code world.rchit} can blend
     * the matching destroy-stage crack texture into a hit terrain block's albedo. Each block's own
     * destroy-stage texture ({@code minecraft:textures/block/destroy_stage_N.png}, resolved via
     * {@link ModelBakery#DESTROY_TYPES}) is a standalone {@code Sampler0} texture, not a block-atlas sprite,
     * so it rides the same bindless entity-texture array as entity textures ({@link RtEntityTextures}).
     */
    private BreakEntry[] breakingEntries(RtTerrain terrain) {
        BreakEntry[] result = new BreakEntry[WorldPushData.BREAKING_CAPACITY];
        int count = 0;
        var level = Minecraft.getInstance().level;
        if (level != null) {
            for (var entry : level.destructionProgress().long2ObjectEntrySet()) {
                if (count >= result.length) {
                    break;
                }
                var progresses = entry.getValue();
                if (progresses == null || progresses.isEmpty()) {
                    continue;
                }
                int stage = Mth.clamp(progresses.last().getProgress(), 0, 9);
                BlockPos pos = BlockPos.of(entry.getLongKey());
                int slot = RtEntityTextures.INSTANCE.slotFor(ModelBakery.DESTROY_TYPES.get(stage));
                result[count++] = new BreakEntry(new Int4(
                        pos.getX() - terrain.blockX,
                        pos.getY() - terrain.blockY,
                        pos.getZ() - terrain.blockZ,
                        slot));
            }
        }
        return count == result.length ? result : java.util.Arrays.copyOf(result, count);
    }

    private record SkyPush(Float4 celestial, Float4 look0, Float4 look1, Float4 look2, Float4 look3,
                           Float4 skyEnvironment, Float4 skyboxState, Float4 dimensionLight,
                           Float4 dimensionAmbient, Float4 dimensionFog, Float4 look4,
                           Float4 sunUv, Float4 moonUv, Float4 endFlashUv) {}

    private record CelestialUv(Float4 sun, Float4 moon) {}

    /**
     * This frame's sky state: Minecraft's four eased celestial angles, its star brightness, the moon
     * phase, and the look package's sky constants. Nothing else.
     *
     * <p>Every direction, colour, level and atmospheric transmittance is derived in {@code sky.slang}
     * from these values, keeping atmospheric evaluation in one implementation.
     *
     * <p>The angles come from the camera's {@link EnvironmentAttributeProbe} rather than from the tick:
     * in 26.2 they are timeline tracks driven through a cubic-bezier ease, and a datapack can replace the
     * track outright, so the probe is the only source that stays correct for a custom dimension.
     *
     * <p>The sky-view LUT's viewer altitude tracks the camera's real world height above sea level, not
     * the look package's fixed reference altitude: a build-limit mod or a rocket/space mod climbing
     * toward the 100 km shell should see the atmosphere actually thin out. The block-to-km scale is
     * exaggerated 10x (100 blocks = 1 km, not the literal 1000) — vanilla's build range is under half a
     * real km, which would put the whole playable height range within a rounding error of one LUT texel
     * row; at 100:1 the same climb is a few km, enough to see the horizon and zenith actually shift.
     * Clamped to [0, 99] km so an absurd Y (or one beyond the modelled 100 km shell) degrades to the
     * shell edge instead of an LUT sample outside its baked domain. The shader applies its own lower
     * floor — see {@code sky.MIN_VIEWER_ALTITUDE_KM}, which is set by what fp32 can resolve at planet
     * radius, not by anything visual — so zero here is safe and means "at or below sea level".
     */
    /**
     * How dark the world's sky is: 0 from sunrise to sunset, 1 through the middle of the night, ramping
     * across {@link #NIGHT_RAMP} of a day at either end.
     *
     * <p>Derived from the dimension's own clock rather than from an attribute, because the attributes that
     * look like they carry it do not: {@code STAR_BRIGHTNESS} measured 0.24 at midnight instead of 1, and
     * {@code SKY_LIGHT_LEVEL} moves with where the camera stands. A control the player dials in EV has to
     * reach the EV it says, so the ramp is computed here.
     *
     * <p>{@code hasSkyLight} is the dimension gate. The Nether fixes its clock, so without it a Nether
     * session would sit at whatever the frozen time happens to read — permanently lifted, or never.
     */
    private float nightFactor() {
        Minecraft mc = Minecraft.getInstance();
        if (mc.level == null || !mc.level.dimensionType().hasSkyLight()) {
            return 0.0f;
        }
        // Minecraft's clock: 0 sunrise, 6000 noon, 12000 sunset, 18000 midnight.
        float timeOfDay = (float) (Math.floorMod(mc.level.getDefaultClockTime(), DAY_TICKS) / (double) DAY_TICKS);
        // Distance from midnight, so the two horizon crossings are the same distance and noon is furthest.
        float fromMidnight = Math.abs(timeOfDay - 0.75f);
        float distance = Math.min(fromMidnight, 1.0f - fromMidnight);
        return Math.clamp((0.25f - distance) / NIGHT_RAMP, 0.0f, 1.0f);
    }

    private SkyPush skyPush() {
        Minecraft mc = Minecraft.getInstance();
        float partial = mc.getDeltaTracker().getGameTimeDeltaPartialTick(false);
        var probe = mc.gameRenderer.mainCamera().attributeProbe();
        int seaLevel = mc.level != null ? mc.level.getSeaLevel() : 0;
        float viewerAltitudeKm = Math.clamp((float) ((camY - seaLevel) / 100.0), 0.0f, 99.0f);
        float toRadians = (float) (Math.PI / 180.0);
        float sunAngle = probe.getValue(EnvironmentAttributes.SUN_ANGLE, partial) * toRadians;
        float moonAngle = probe.getValue(EnvironmentAttributes.MOON_ANGLE, partial) * toRadians;
        // Stars use Minecraft's own celestial rotation and brightness (the values vanilla's SkyRenderer
        // uses), so the field wheels about the celestial pole tied to world time and fades in and out at
        // dusk/dawn exactly like vanilla's.
        float starAngle = probe.getValue(EnvironmentAttributes.STAR_ANGLE, partial) * toRadians;
        float starBrightness = probe.getValue(EnvironmentAttributes.STAR_BRIGHTNESS, partial);
        float moonPhase = probe.getValue(EnvironmentAttributes.MOON_PHASE, partial).index(); // 0 full .. 4 new
        // Whether this dimension is lit by its sky at all, and what colour that sky carries. The Nether and
        // the End report a zero factor: no sun, no moon, no stars, no atmosphere. Reading it from the probe
        // rather than from a dimension id keeps a datapack-authored dimension correct for free, and a
        // partial value is honoured because the shader blends on it.
        float skyLightFactor = probe.getValue(EnvironmentAttributes.SKY_LIGHT_FACTOR, partial);
        int skyColor = probe.getValue(EnvironmentAttributes.SKY_COLOR, partial);
        Float4 skyEnvironment = new Float4(skyLightFactor,
                (float) srgbToLinear(((skyColor >> 16) & 0xFF) / 255.0),
                (float) srgbToLinear(((skyColor >> 8) & 0xFF) / 255.0),
                (float) srgbToLinear((skyColor & 0xFF) / 255.0));

        // Which sky a sky-light-free dimension renders, and what the End layers on top of it. The Skybox
        // ordinal is the only thing that can tell the Nether from the End — skyLightFactor is zero in
        // both, and they need opposite skies. The flash state is exactly what vanilla's SkyRenderer
        // consumes (eased intensity plus the two raw angles), gated on the dimension's own
        // hasEndFlashes and the player's lightning-flash option. The three attribute colours are the
        // same ones vanilla's lightmap tints the end flash and its ambient floor with; a datapack
        // dimension's custom values arrive here untouched.
        var dimensionType = mc.level.dimensionType();
        boolean flashEnabled = dimensionType.hasEndFlashes() && !mc.options.hideLightningFlash().get();
        // Explicit mapping rather than ordinal(): the shader's SKYBOX_* constants are the contract, not
        // Mojang's declaration order.
        int skyboxOrdinal = switch (dimensionType.skybox()) {
            case NONE -> 0;
            case OVERWORLD -> 1;
            case END -> 2;
        };
        // Vanilla only constructs and consumes the flash state for an END skybox — ClientLevel
        // .endFlashState() is null in every other dimension — so the reads are gated the same way.
        float flashIntensity = 0.0f;
        float flashXAngle = 0.0f;
        float flashYAngle = 0.0f;
        if (skyboxOrdinal == 2 && flashEnabled) {
            EndFlashState endFlash = mc.level.endFlashState();
            if (endFlash != null) {
                flashIntensity = Math.clamp(endFlash.getIntensity(partial), 0.0f, 1.0f);
                flashXAngle = endFlash.getXAngle() * toRadians;
                flashYAngle = endFlash.getYAngle() * toRadians;
            }
        }
        Float4 skyboxState = new Float4(skyboxOrdinal, flashIntensity, flashXAngle, flashYAngle);
        int skyLightColor = probe.getValue(EnvironmentAttributes.SKY_LIGHT_COLOR, partial);
        int ambientLightColor = probe.getValue(EnvironmentAttributes.AMBIENT_LIGHT_COLOR, partial);
        int fogColor = probe.getValue(EnvironmentAttributes.FOG_COLOR, partial);
        Float4 dimensionLight = new Float4(
                (float) srgbToLinear(((skyLightColor >> 16) & 0xFF) / 255.0),
                (float) srgbToLinear(((skyLightColor >> 8) & 0xFF) / 255.0),
                (float) srgbToLinear((skyLightColor & 0xFF) / 255.0),
                dimensionType.ambientLight());
        Float4 dimensionAmbient = new Float4(
                (float) srgbToLinear(((ambientLightColor >> 16) & 0xFF) / 255.0),
                (float) srgbToLinear(((ambientLightColor >> 8) & 0xFF) / 255.0),
                (float) srgbToLinear((ambientLightColor & 0xFF) / 255.0),
                0f);
        Float4 dimensionFog = new Float4(
                (float) srgbToLinear(((fogColor >> 16) & 0xFF) / 255.0),
                (float) srgbToLinear(((fogColor >> 8) & 0xFF) / 255.0),
                (float) srgbToLinear((fogColor & 0xFF) / 255.0),
                0f);

        RtLookPackage.Sky sky = LOOK.sky();
        RtLookPackage.Lighting lighting = LOOK.lighting();
        RtLookPackage.End end = LOOK.end();
        CelestialUv uv = celestialUv(moonPhase);
        return new SkyPush(
                new Float4(sunAngle, moonAngle, starAngle, starBrightness),
                new Float4(lighting.sunIlluminanceLux(), lighting.moonIlluminanceLux(),
                        lighting.nightAirglowLuminanceCdM2(), lighting.starLuminanceCdM2()),
                new Float4(sky.sunNoonSouthTiltDegrees() * toRadians,
                        sky.sunAngularRadiusDegrees() * toRadians,
                        sky.moonAngularRadiusDegrees() * toRadians,
                        lighting.moonPhaseFixedFraction()),
                new Float4(sky.sunDiscHalfAngleDegrees() * toRadians,
                        sky.moonDiscHalfAngleDegrees() * toRadians,
                        viewerAltitudeKm, moonPhase),
                new Float4(sky.groundAlbedo(), sky.horizonSoftenDegrees() * toRadians,
                        end.flashAngularRadiusDegrees() * toRadians, end.skyGradientGain()),
                skyEnvironment,
                skyboxState,
                dimensionLight,
                dimensionAmbient,
                dimensionFog,
                new Float4(end.flashIlluminanceLux(), end.flashQuadLuminanceCdM2(),
                        end.ambientIlluminanceLux(), end.skyBoxGain()),
                uv.sun(),
                uv.moon(),
                new Float4(endFlashU0, endFlashV0, endFlashU1, endFlashV1));
    }

    /**
     * Push the celestials-atlas UV rects (u0,v0,u1,v1) for the sun sprite and the current moon-phase
     * sprite, so world.rmiss can sample the real vanilla textures on the discs. Atlas-not-ready (early
     * boot / no resources) leaves full-range UVs and the shader's block-atlas fallback covers it.
     */
    private CelestialUv celestialUv(float moonPhaseIndex) {
        if (celestialUvAtlasHandle == 0L) {
            setCelestialUvAtlas(celestialsAtlasView());
        }
        int phase = Math.clamp((int) moonPhaseIndex, 0, MOON_IDS.length - 1);
        if (phase != celestialUvMoonPhase) {
            refreshCelestialUvCache(phase);
        }
        return new CelestialUv(
                new Float4(sunU0, sunV0, sunU1, sunV1),
                new Float4(moonU0, moonV0, moonU1, moonV1));
    }

    private void setCelestialUvAtlas(long atlasHandle) {
        if (celestialUvAtlasHandle == atlasHandle) {
            return;
        }
        celestialUvAtlasHandle = atlasHandle;
        celestialUvMoonPhase = -1;
        sunU0 = 0f; sunV0 = 0f; sunU1 = 1f; sunV1 = 1f;
        moonU0 = 0f; moonV0 = 0f; moonU1 = 1f; moonV1 = 1f;
        endFlashU0 = 0f; endFlashV0 = 0f; endFlashU1 = 1f; endFlashV1 = 1f;
    }

    private void refreshCelestialUvCache(int moonPhase) {
        sunU0 = 0f; sunV0 = 0f; sunU1 = 1f; sunV1 = 1f;
        moonU0 = 0f; moonV0 = 0f; moonU1 = 1f; moonV1 = 1f;
        endFlashU0 = 0f; endFlashV0 = 0f; endFlashU1 = 1f; endFlashV1 = 1f;
        try {
            if (celestialUvAtlasHandle != 0L) {
                TextureAtlas atlas = Minecraft.getInstance().getAtlasManager().getAtlasOrThrow(AtlasIds.CELESTIALS);
                TextureAtlasSprite sun = atlas.getSprite(SUN_ID);
                sunU0 = sun.getU0(); sunV0 = sun.getV0(); sunU1 = sun.getU1(); sunV1 = sun.getV1();
                TextureAtlasSprite moon = atlas.getSprite(MOON_IDS[moonPhase]);
                moonU0 = moon.getU0(); moonV0 = moon.getV0(); moonU1 = moon.getU1(); moonV1 = moon.getV1();
                TextureAtlasSprite endFlash = atlas.getSprite(END_FLASH_ID);
                endFlashU0 = endFlash.getU0(); endFlashV0 = endFlash.getV0();
                endFlashU1 = endFlash.getU1(); endFlashV1 = endFlash.getV1();
            }
        } catch (Exception ignored) {
            // celestials atlas not yet loaded — keep full-range UVs (fallback texture is the block atlas)
        }
        celestialUvMoonPhase = moonPhase;
    }

    private static Float4 linearAcesCgFromSrgb(double r, double g, double b, float w) {
        return linearAcesCgFromBt709(
                srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), w);
    }

    /** OCIO cg-config-v4.0.0 ACES 2.0: Linear Rec.709 (sRGB)/D65 to ACEScg/AP1/D60. */
    private static Float4 linearAcesCgFromBt709(double r, double g, double b, float w) {
        return new Float4(
                (float) (0.61309743 * r + 0.33952314 * g + 0.04737945 * b),
                (float) (0.07019372 * r + 0.91635388 * g + 0.01345240 * b),
                (float) (0.02061559 * r + 0.10956977 * g + 0.86981463 * b),
                w);
    }

    private static double srgbToLinear(double value) {
        return value <= 0.04045 ? value / 12.92
                : Math.pow((value + 0.055) / 1.055, 2.4);
    }

    /**
     * The look-package-relative LMT selected by config, or the packaged name when no override is set.
     * The sanitizer in {@code CausticaConfig.Rt.Tonemap} has already rejected anything with a path
     * separator, so this concatenation cannot leave the package directory.
     */
    private static String lookQualifiedLmtName() {
        String override = CausticaConfig.Rt.Tonemap.LOOK_OVERRIDE.get();
        if (override == null || override.isEmpty()) {
            return LOOK.lmtResource();
        }
        String packaged = LOOK.lmtResource();
        return packaged.substring(0, packaged.lastIndexOf('/') + 1) + override;
    }

    /**
     * Loads an LMT, returning null when an override names something unreadable or contract-violating so
     * the caller keeps the LUT it already has. A mistyped file name should cost a warning and one
     * wrong-looking frame, not the whole renderer — {@code RtToneLut.loadResource} throws on bad magic,
     * version, size or shaper. The packaged LMT is not caught: that failing is a build problem.
     */
    private static RtToneLut tryLoadLookLut(RtContext ctx, String resource) {
        try {
            return RtToneLut.loadResource(ctx, resource);
        } catch (RuntimeException e) {
            CausticaMod.LOGGER.warn("Cannot use tonemap.look-override '{}': {}; keeping the current LMT",
                    resource, e.getMessage());
            return null;
        }
    }

    public void destroy() {
        // Teardown runs after the device is idle (CLIENT_STOPPING waits), so the TLAS ring's slots are no
        // longer in flight and can be freed immediately.
        tlasRing.destroy();
        if (RtDlssRr.enabled()) {
            RtDlssRr.INSTANCE.destroy();
        }
        if (displayImage != null) {
            displayImage.destroy();
            displayImage = null;
        }
        if (hdrDisplayImage != null) {
            hdrDisplayImage.destroy();
            hdrDisplayImage = null;
        }
        if (hdrBindingPlaceholder != null) {
            hdrBindingPlaceholder.destroy();
            hdrBindingPlaceholder = null;
        }
        renderSizeHdrEnabled = false;
        destroyBloomLevels();
        if (fgHudlessImage != null) {
            fgHudlessImage.destroy();
            fgHudlessImage = null;
        }
        if (fgHdrHudlessImage != null) {
            fgHdrHudlessImage.destroy();
            fgHdrHudlessImage = null;
        }
        RtWorldOverlay.INSTANCE.destroy(); // overlay features/pipelines/scratch live on the same device lifetime
        if (output != null) {
            output.destroy();
            output = null;
        }
        if (continuationQueue != null) {
            continuationQueue.destroy();
            continuationQueue = null;
        }
        destroyGuideImages();
        exposure.destroy();
        if (displayPipeline != null) {
            displayPipeline.destroy();
            displayPipeline = null;
        }
        if (bloomPipeline != null) {
            bloomPipeline.destroy();
            bloomPipeline = null;
        }
        if (skyLut != null) {
            skyLut.destroy();
            skyLut = null;
        }
        if (debugPresentPipeline != null) {
            debugPresentPipeline.destroy();
            debugPresentPipeline = null;
        }
        if (sdrToneLut != null) {
            sdrToneLut.destroy();
            sdrToneLut = null;
        }
        if (hdrToneLut != null) {
            hdrToneLut.destroy();
            hdrToneLut = null;
        }
        if (lookLut != null) {
            lookLut.destroy();
            lookLut = null;
        }
        loadedLookLutName = null;
        loadedHdrLutNits = -1;
        if (hdrCompositePipeline != null) {
            hdrCompositePipeline.destroy();
            hdrCompositePipeline = null;
        }
        if (hdrUiSampler != 0L) {
            RtContext hdrCtx = RtContext.currentOrNull();
            if (hdrCtx != null) {
                VK10.vkDestroySampler(hdrCtx.vk(), hdrUiSampler, null);
            }
            hdrUiSampler = 0L;
        }
        if (sdrPresentPipeline != null) {
            sdrPresentPipeline.destroy();
            sdrPresentPipeline = null;
        }
        if (sdrPresentImage != null) {
            sdrPresentImage.destroy();
            sdrPresentImage = null;
        }
        for (RtImage img : fgInterp) {
            if (img != null) {
                img.destroy();
            }
        }
        fgInterp = new RtImage[0];
        fgInterpW = -1;
        fgInterpH = -1;
        fgInterpFormat = Integer.MIN_VALUE;
        if (fgInterpPipeline != null) {
            fgInterpPipeline.destroy();
            fgInterpPipeline = null;
        }
        if (fgSelfInterp != null) {
            fgSelfInterp.destroy();
            fgSelfInterp = null;
        }
        if (fgSelfHistory != null) {
            fgSelfHistory.destroy();
            fgSelfHistory = null;
        }
        fgSelfHistoryWritten = false;
        if (worldPipeline != null) {
            worldPipeline.destroy();
            worldPipeline = null;
        }
        bindlessTextureCapacity = 0;
        materialBindingsReady = false;
        materialEpochTraceGate = false;
        RtMaterialRegistry.INSTANCE.destroy();
        if (pushRing != null) {
            for (PushSlot slot : pushRing) {
                if (slot != null) {
                    slot.buffer.destroy();
                }
            }
            pushRing = null;
        }
        if (atlasSampler != 0L) {
            RtContext ctx = RtContext.currentOrNull();
            if (ctx != null) {
                VK10.vkDestroySampler(ctx.vk(), atlasSampler, null);
            }
            atlasSampler = 0L;
        }
        if (endSkySampler != 0L) {
            RtContext ctx = RtContext.currentOrNull();
            if (ctx != null) {
                VK10.vkDestroySampler(ctx.vk(), endSkySampler, null);
            }
            endSkySampler = 0L;
        }
    }

    private long atlasSampler(RtContext ctx) {
        if (atlasSampler == 0L) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                        .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                        .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_LINEAR)
                        .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .minLod(0f).maxLod(16f);
                LongBuffer p = stack.mallocLong(1);
                if (VK10.vkCreateSampler(ctx.vk(), sci, null, p) != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("vkCreateSampler(block atlas) failed");
                }
                atlasSampler = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, atlasSampler, "block atlas sampler");
            }
        }
        return atlasSampler;
    }

    /**
     * Linear + wrapping sampler for the end-sky cube: vanilla's buildEndSky() tiles end_sky.png 16x16
     * per face and magnifies it heavily, so REPEAT addressing is load-bearing and NEAREST (the atlas
     * sampler) would render it as hard pixels rather than vanilla's soft cloudscape.
     */
    private long endSkySampler(RtContext ctx) {
        if (endSkySampler == 0L) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                        .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                        .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                        .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                        .minLod(0f).maxLod(0.25f);
                LongBuffer p = stack.mallocLong(1);
                if (VK10.vkCreateSampler(ctx.vk(), sci, null, p) != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("vkCreateSampler(end sky) failed");
                }
                endSkySampler = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, endSkySampler, "end sky sampler");
            }
        }
        return endSkySampler;
    }

    private static long blockAlbedoAtlasView() {
        GpuTextureView view = Minecraft.getInstance().getTextureManager()
                .getTexture(TextureAtlas.LOCATION_BLOCKS).getTextureView();
        return vkImageView(view);
    }

    private static long vkImageView(GpuTextureView view) {
        if (view instanceof VulkanGpuTextureView vulkanView) {
            return vulkanView.vkImageView();
        }
        throw new IllegalStateException("cannot resolve VkImageView for " + view);
    }

    private static long vkImage(GpuTexture texture) {
        if (texture instanceof VulkanGpuTexture vulkanTexture) {
            return vulkanTexture.vkImage();
        }
        throw new IllegalStateException("cannot resolve VkImage for " + texture);
    }

    private static VkImageCopy.Buffer copyRegion(MemoryStack stack, int width, int height) {
        VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
        region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).extent().set(width, height, 1);
        return region;
    }

    /** Whether the HDR present path (HDR image + combined UI -> PQ swapchain) should replace the vanilla SDR blit. */
    public boolean isHdrPresentActive() {
        return CausticaConfig.Rt.Hdr.enabled()
                && hdrWrittenThisFrame
                && hdrDisplayImage != null;
    }

    /**
     * DLSS-FG: the PQ-encoded HDR backbuffer (view/image), valid only right after {@link #presentHdr} has run
     * this frame (it's the same image {@code presentHdr} just composited UI into and blitted to the
     * swapchain) — used as the interpolation source for HDR frame generation instead of the SDR main target.
     * Already display-ready PQ, so it's fed to DLSSG directly with no extra encode step. 0 if HDR isn't
     * active this frame.
     */
    public long hdrBackbufferView() {
        return hdrDisplayImage != null ? hdrDisplayImage.view : 0L;
    }

    public long hdrBackbufferImage() {
        return hdrDisplayImage != null ? hdrDisplayImage.image : 0L;
    }

    /**
     * Blit this frame's PQ-encoded HDR image straight into the swapchain image, replacing Minecraft's SDR
     * blit. Replicates {@code VulkanGpuSurface.blitFromTexture}'s barrier + acquire-wait/present-signal
     * sequence with the HDR {@link RtImage} as the (GENERAL-layout) source; an added memory barrier makes the
     * display-compute writes visible to the blit read. The SDR main target is bypassed; the combined UI image
     * is blended over the HDR image here at paper white before the swapchain blit. The magic stage/access
     * values mirror vanilla {@code blitFromTexture} exactly. Y is flipped to match the vanilla swapchain blit.
     */
    public void presentHdr(VulkanCommandEncoder enc, long swapchainImage, int swapW, int swapH, long acquireSem, long presentSem) {
        RtImage src = hdrDisplayImage;
        int copyW = Math.min(swapW, src.width);
        int copyH = Math.min(swapH, src.height);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();

            // DLSS-FG "hudless" capture: hdrDisplayImage right now holds the RT world before the combined
            // UI overlay is blended in. Snapshot it before that composite overwrites it in place, mirroring
            // captureFgHudless's SDR pattern (pre-UI copy) but reusing this frame's already-open command
            // buffer.
            if (RtDlssFg.enabled()) {
                captureFgHdrHudless(cmd, stack, src);
            }

            // Step C.2: composite the combined UI overlay over the HDR world image (in place) at paper white,
            // before the swapchain blit. The overlay is an MC render target kept in GENERAL layout, sampled by
            // the compute pass. A memory barrier first makes the overlay writes + the world HDR writes visible
            // to the compute; the dep1 barrier below (ALL writes -> transfer read) then covers the compute's
            // HDR write for the blit.
            long overlayView = RtUiOverlay.populatedThisFrame() ? RtUiOverlay.overlayColorView() : 0L;
            if (overlayView != 0L) {
                ensureHdrUiResources();
                if (hdrCompositePipeline != null) {
                    VkMemoryBarrier2.Buffer pre = VkMemoryBarrier2.calloc(1, stack).sType$Default();
                    pre.get(0).srcStageMask(65536L).srcAccessMask(65536L).dstStageMask(2048L).dstAccessMask(98304L);
                    VkDependencyInfo preDep = VkDependencyInfo.calloc(stack).sType$Default().pMemoryBarriers(pre);
                    KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, preDep);
                    hdrCompositePipeline.setImages(hdrDisplayImage.view, overlayView, hdrUiSampler);
                    hdrCompositePipeline.dispatch(cmd, src.width, src.height, CausticaConfig.Rt.Hdr.uiNits());
                }
                RtUiOverlay.markConsumed();
            }
            // Swapchain UNDEFINED -> TRANSFER_DST, plus make the HDR compute writes visible to the blit read.
            VkImageMemoryBarrier2.Buffer toDst = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toDst.get(0).srcStageMask(0L).srcAccessMask(0L).dstStageMask(4096L).dstAccessMask(4096L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(swapchainImage);
            toDst.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VkMemoryBarrier2.Buffer srcVis = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            srcVis.get(0).srcStageMask(65536L).srcAccessMask(65536L).dstStageMask(4096L).dstAccessMask(2048L);
            VkDependencyInfo dep1 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toDst).pMemoryBarriers(srcVis);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep1);

            // Blit HDR (GENERAL) -> swapchain (TRANSFER_DST), Y-flipped like vanilla.
            VkImageBlit.Buffer region = VkImageBlit.calloc(1, stack);
            region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).srcOffsets(1).set(copyW, copyH, 1); // srcOffsets[0] = (0,0,0) from calloc
            region.get(0).dstOffsets(0).set(0, copyH, 0);
            region.get(0).dstOffsets(1).set(copyW, 0, 1);
            VK10.vkCmdBlitImage(cmd, src.image, VK10.VK_IMAGE_LAYOUT_GENERAL, swapchainImage,
                    VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region, VK10.VK_FILTER_NEAREST);

            // Swapchain TRANSFER_DST -> PRESENT_SRC_KHR (1000001002).
            VkImageMemoryBarrier2.Buffer toPresent = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toPresent.get(0).srcStageMask(4096L).srcAccessMask(4096L).dstStageMask(65536L).dstAccessMask(0L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(1000001002)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(swapchainImage);
            toPresent.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VkMemoryBarrier2.Buffer mem2 = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            mem2.get(0).srcStageMask(4096L).srcAccessMask(2048L).dstStageMask(65536L).dstAccessMask(98304L);
            VkDependencyInfo dep2 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toPresent).pMemoryBarriers(mem2);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep2);

            if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("vkEndCommandBuffer(hdr present) failed");
            }
            enc.waitSemaphore(acquireSem, 0L, 65536L);
            enc.execute(cmd);
            enc.signalSemaphore(presentSem, 0L, 4096L);
        }
    }

    /** Lazily create the HDR UI-composite compute pipeline + its nearest/clamp sampler (first HDR present). */
    private void ensureHdrUiResources() {
        if (hdrCompositePipeline != null) {
            return;
        }
        RtContext ctx = RtContext.get();
        if (ctx == null || !ensureUiSampler(ctx)) {
            return;
        }
        hdrCompositePipeline = RtHdrCompositePipeline.create(ctx);
    }

    /** Ensure the shared nearest/clamp sampler used to sample SDR/overlay targets in the present compute. */
    private boolean ensureUiSampler(RtContext ctx) {
        if (hdrUiSampler != 0L) {
            return true;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            var p = stack.mallocLong(1);
            if (VK10.vkCreateSampler(ctx.vk(), sci, null, p) != VK10.VK_SUCCESS) {
                return false;
            }
            hdrUiSampler = p.get(0);
        }
        return true;
    }

    /**
     * Whether a non-RT frame (menu, title panorama, loading screen) should be SDR-&gt;PQ converted for
     * present instead of vanilla's raw SDR blit. True when the PQ swapchain is active but this frame did
     * not produce an HDR image ({@link #isHdrPresentActive()} false).
     */
    public boolean isPqSdrPresentActive() {
        // The conversion is needed only while the CURRENT swapchain is PQ and this frame has no HDR
        // image (menus/loading, or the short interval after the toggle changed but before configure()).
        // Once configure recreates a native-SDR swapchain, vanilla's ordinary blit is correct.
        return CausticaConfig.Rt.Hdr.swapchainPqActive()
                && !isHdrPresentActive();
    }

    /**
     * Present a non-RT (menu/loading) frame to the PQ swapchain: convert the SDR main target (sRGB-encoded
     * rgba8, GENERAL layout, already holding the composited panorama + UI) to PQ-encoded at paper white via
     * a compute pass into {@link #sdrPresentImage}, then blit that into the swapchain. Mirrors
     * {@link #presentHdr} barrier-for-barrier; returns false (keep vanilla SDR blit) if resources are
     * unavailable.
     */
    public boolean presentSdrToPq(VulkanCommandEncoder enc, long swapchainImage, int swapW, int swapH,
            long sdrMainView, long acquireSem, long presentSem) {
        if (sdrMainView == 0L || failed) {
            return false;
        }
        RtContext ctx = RtContext.get();
        if (ctx == null || !ensureUiSampler(ctx)) {
            return false;
        }
        if (sdrPresentPipeline == null) {
            sdrPresentPipeline = RtSdrPresentPipeline.create(ctx);
        }
        if (sdrPresentImage == null || sdrPresentImage.width != swapW || sdrPresentImage.height != swapH) {
            if (sdrPresentImage != null) {
                sdrPresentImage.destroy();
            }
            sdrPresentImage = ctx.createStorageImage(swapW, swapH, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "RT SDR->PQ present image " + swapW + "x" + swapH);
        }
        RtImage dst = sdrPresentImage;
        int copyW = Math.min(swapW, dst.width);
        int copyH = Math.min(swapH, dst.height);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();

            // Make the prior GUI/overlay writes to the SDR main target visible to the compute sample.
            VkMemoryBarrier2.Buffer pre = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            pre.get(0).srcStageMask(65536L).srcAccessMask(65536L).dstStageMask(2048L).dstAccessMask(98304L);
            VkDependencyInfo preDep = VkDependencyInfo.calloc(stack).sType$Default().pMemoryBarriers(pre);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, preDep);

            sdrPresentPipeline.setImages(dst.view, sdrMainView, hdrUiSampler);
            sdrPresentPipeline.dispatch(cmd, dst.width, dst.height, CausticaConfig.Rt.Hdr.uiNits());

            // Swapchain UNDEFINED -> TRANSFER_DST, plus make the compute write visible to the blit read.
            VkImageMemoryBarrier2.Buffer toDst = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toDst.get(0).srcStageMask(0L).srcAccessMask(0L).dstStageMask(4096L).dstAccessMask(4096L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(swapchainImage);
            toDst.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VkMemoryBarrier2.Buffer srcVis = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            srcVis.get(0).srcStageMask(65536L).srcAccessMask(65536L).dstStageMask(4096L).dstAccessMask(2048L);
            VkDependencyInfo dep1 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toDst).pMemoryBarriers(srcVis);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep1);

            // Blit converted PQ image (GENERAL) -> swapchain (TRANSFER_DST), Y-flipped like vanilla.
            VkImageBlit.Buffer region = VkImageBlit.calloc(1, stack);
            region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).srcOffsets(1).set(copyW, copyH, 1); // srcOffsets[0] = (0,0,0) from calloc
            region.get(0).dstOffsets(0).set(0, copyH, 0);
            region.get(0).dstOffsets(1).set(copyW, 0, 1);
            VK10.vkCmdBlitImage(cmd, dst.image, VK10.VK_IMAGE_LAYOUT_GENERAL, swapchainImage,
                    VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region, VK10.VK_FILTER_NEAREST);

            // Swapchain TRANSFER_DST -> PRESENT_SRC_KHR (1000001002).
            VkImageMemoryBarrier2.Buffer toPresent = VkImageMemoryBarrier2.calloc(1, stack).sType$Default();
            toPresent.get(0).srcStageMask(4096L).srcAccessMask(4096L).dstStageMask(65536L).dstAccessMask(0L)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(1000001002)
                    .srcQueueFamilyIndex(-1).dstQueueFamilyIndex(-1).image(swapchainImage);
            toPresent.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VkMemoryBarrier2.Buffer mem2 = VkMemoryBarrier2.calloc(1, stack).sType$Default();
            mem2.get(0).srcStageMask(4096L).srcAccessMask(2048L).dstStageMask(65536L).dstAccessMask(98304L);
            VkDependencyInfo dep2 = VkDependencyInfo.calloc(stack).sType$Default().pImageMemoryBarriers(toPresent).pMemoryBarriers(mem2);
            KHRSynchronization2.vkCmdPipelineBarrier2KHR(cmd, dep2);

            if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("vkEndCommandBuffer(sdr present) failed");
            }
            enc.waitSemaphore(acquireSem, 0L, 65536L);
            enc.execute(cmd);
            enc.signalSemaphore(presentSem, 0L, 4096L);
        }
        return true;
    }

    /**
     * Linear-filtered blit of the full render-res image into the full display-res image. Used as the
     * non-RR / fallback upscale so display mapping always sees a display-res RT image; a no-op stretch when
     * the two are the same size (RR disabled -> render == display).
     */
    private static void blitUpscale(VkCommandBuffer cmd, MemoryStack stack, RtImage src, RtImage dst) {
        VkImageBlit.Buffer region = VkImageBlit.calloc(1, stack);
        region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.get(0).srcOffsets(1).set(src.width, src.height, 1); // srcOffsets[0] zeroed by calloc
        region.get(0).dstOffsets(1).set(dst.width, dst.height, 1);
        VK10.vkCmdBlitImage(cmd, src.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                dst.image, VK10.VK_IMAGE_LAYOUT_GENERAL, region, VK10.VK_FILTER_LINEAR);
    }

    /**
     * DLSS Frame Generation quality: capture a copy of {@code main} (the main render target) into
     * {@link #fgHudlessImage} for {@link #fgInterpolate} to feed DLSSG as the "hudless" resource. Call from
     * {@code GameRendererMixin} right after {@code GuiRenderer.render()} but BEFORE
     * {@link RtUiOverlay#compositeIfUsed()} — at that point, when the UI overlay redirect is active, {@code
     * main} still has no combined UI baked in (world overlays, hand/screen effects and GUI went to the
     * overlay target instead). No-op (and {@link #fgInterpolate} passes 0/0/0 for hudless, same as always)
     * unless both FG and the UI overlay redirect are active — capturing this without the redirect would just
     * copy the ALREADY-composited backbuffer, which is useless as a distinct hudless input.
     */

    public void captureFgHudless(RenderTarget main) {
        if (!RtDlssFg.enabled() || !RtFramePresenter.hudlessInputWanted() || !RtUiOverlay.enabled()
                || main == null || main.getColorTexture() == null) {
            return;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return;
        }
        long srcImage;
        try {
            srcImage = vkImage(main.getColorTexture());
        } catch (IllegalStateException e) {
            return; // not a Vulkan-backed texture (shouldn't happen on this backend)
        }
        if (fgHudlessImage == null || fgHudlessImage.width != main.width || fgHudlessImage.height != main.height) {
            if (fgHudlessImage != null) {
                fgHudlessImage.destroy();
            }
            fgHudlessImage = ctx.createStorageImage(main.width, main.height, VK10.VK_FORMAT_R8G8B8A8_UNORM,
                    "FG hudless capture " + main.width + "x" + main.height);
        }
        var encoder = (VulkanCommandEncoder) ((CommandEncoderAccessor) RenderSystem.getDevice().createCommandEncoder()).caustica$getBackend();
        VkCommandBuffer cmd = encoder.allocateAndBeginTransientCommandBuffer();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Make writes into `main` visible to the copy (the combined UI has not touched `main` yet this
            // frame — it went to the UI overlay target instead).
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            VK10.vkCmdCopyImage(cmd, srcImage, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    fgHudlessImage.image, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack, main.width, main.height));
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(fg hudless capture) failed");
        }
        encoder.execute(cmd);
    }

    /**
     * HDR counterpart of {@link #captureFgHudless} — copies {@code src} (this frame's {@code hdrDisplayImage},
     * before the combined UI overlay is blended in) into {@link #fgHdrHudlessImage} for {@link
     * #fgInterpolate}'s HDR path to feed DLSSG as the "hudless" resource. A plain copy, not a format
     * conversion: both images are
     * already PQ-encoded (the display-ready EOTF-encoded [0,1] signal DLSS-FG's programming guide requires),
     * so no encode step is needed. Called from {@link #presentHdr} using its already-open {@code cmd}/
     * {@code stack}, right before that method's own combined-UI composite dispatch overwrites
     * {@code hdrDisplayImage} in place — same "capture before the UI gets baked back in" timing as the SDR
     * version, just within a single method instead of split across a mixin hook.
     */
    private void captureFgHdrHudless(VkCommandBuffer cmd, MemoryStack stack, RtImage src) {
        if (!RtFramePresenter.hudlessInputWanted()) {
            return;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return;
        }
        if (fgHdrHudlessImage == null || fgHdrHudlessImage.width != src.width || fgHdrHudlessImage.height != src.height) {
            if (fgHdrHudlessImage != null) {
                fgHdrHudlessImage.destroy();
            }
            fgHdrHudlessImage = ctx.createStorageImage(src.width, src.height, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "FG HDR hudless capture (PQ) " + src.width + "x" + src.height);
        }
        // Make composite()'s writes to hdrDisplayImage (an earlier submit this frame) visible to this copy;
        // the copy's write is then made visible to the UI-composite dispatch that follows (and to DLSSG's
        // read, in a later command buffer) by the same idiom.
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        VK10.vkCmdCopyImage(cmd, src.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                fgHdrHudlessImage.image, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack, src.width, src.height));
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    /**
     * DLSS Frame Generation: record the DLSSG evaluate for generated frame {@code index} of {@code count}
     * (backbuffer = the final frame; HW depth = {@code gDepth}; motion = {@code gMotion}) into Minecraft's
     * command encoder, returning the interpolated output image (backbuffer size) for {@link RtFramePresenter}
     * to blit into a generated swapchain image. On {@code index == 1} it ensures the feature (created in its
     * own synchronous submit), the per-index output images, and the jitter-free reprojection matrices.
     * Returns {@code null} (caller falls back to duplicating the real frame for this one frame, no session
     * impact) when there's simply no captured RT frame to interpolate from right now — routine and expected
     * on menu/loading/transition frames, since {@link RtFramePresenter#isActive} only gates on being in a
     * world, not on RT having actually produced a frame this tick. Throws instead for failures that should
     * never happen once RT is actively producing frames (DLSSG feature creation failing, an out-of-range
     * index, the evaluate itself failing) — the caller treats those as fatal and disables FG for the
     * session, same as any other FG present-record failure, rather than silently degrading to duplicated
     * (non-interpolated) frames forever with no visible sign anything is wrong. Rotation-only matrices;
     * camera translation is carried by the mvecs (cameraMotionIncluded).
     *
     * <p>{@code hdrBackbuffer} selects the HDR path. Per the DLSS-FG programming guide's HDR section, scRGB is
     * explicitly unsupported as a DLSS-FG input ("not suitable as inputs to DLSS-FG" — it wants a
     * display-ready, EOTF-encoded [0,1] signal, recommending HDR10/ST.2084) — since the renderer's whole HDR
     * pipeline is natively PQ-encoded, every image fed to {@code RtDlssFg.evaluate} in HDR mode is already in
     * that format with no extra conversion needed: the backbuffer is the raw {@code backbufferView}/
     * {@code backbufferImage} the caller passed in ({@link #hdrBackbufferView()}, already PQ + UI-composited
     * by {@link #presentHdr}); the hudless resource is {@link #fgHdrHudlessImage} (copied by {@link
     * #presentHdr} <em>before</em> its own UI composite ran, mirroring {@link #captureFgHudless}'s pre-UI
     * timing); and DLSSG's own (also PQ-encoded) output is returned as-is, since the swapchain itself is
     * PQ-native and can blit it directly. The UI resource itself needs no HDR-specific handling — it's the
     * same combined {@link RtUiOverlay} texture used by both present paths (only the *compositing* math that
     * consumes it differs, done separately by {@code presentHdr}/{@code RtUiOverlay}, not here).
     */
    public RtImage fgInterpolate(VulkanCommandEncoder enc, long backbufferView, long backbufferImage,
            int swapW, int swapH, int index, int count, boolean hdrBackbuffer) {
        if (failed || gDepth == null || gMotion == null || !frameCaptured) {
            return null;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return null;
        }
        final int fmt = hdrBackbuffer ? VK10.VK_FORMAT_R16G16B16A16_SFLOAT : VK10.VK_FORMAT_R8G8B8A8_UNORM;
        if (index == 1) {
            if (!ensureFgFeature(ctx, swapW, swapH, renderW, renderH, fmt)) {
                throw new IllegalStateException("DLSSG feature not ready (ensureFgFeature failed)");
            }
            ensureFgInterp(ctx, count, swapW, swapH, fmt);
            // clipToPrevClip = prevVP * inverse(curVP); prevClipToClip = curVP * inverse(prevVP). Both from
            // the (rotation-only, camera-relative) MV view-projections, so jitter-free.
            fgMatTmp.set(mvCurProjView).invert();
            fgClipToPrev.set(mvPrevProjView).mul(fgMatTmp);
            fgMatTmp.set(mvPrevProjView).invert();
            fgPrevToClip.set(mvCurProjView).mul(fgMatTmp);
        }
        if (index < 1 || index > fgInterp.length || fgInterp[index - 1] == null) {
            throw new IllegalStateException(
                    "fgInterpolate index " + index + " out of range for fgInterp[" + fgInterp.length + "]");
        }
        RtImage out = fgInterp[index - 1];
        // Only feed hudless/ui when they exist AND match this frame's backbuffer size — a stale or mismatched
        // size (e.g. mid-resize) is worse than skipping, so fall back to 0/0/0 (DLSSG just does without).
        RtImage hudlessSrc = hdrBackbuffer ? fgHdrHudlessImage : fgHudlessImage;
        boolean hudlessReady = hudlessSrc != null && hudlessSrc.width == swapW && hudlessSrc.height == swapH;
        long hudlessView = hudlessReady ? hudlessSrc.view : 0L;
        long hudlessImg = hudlessReady ? hudlessSrc.image : 0L;
        int hudlessFmt = hdrBackbuffer ? VK10.VK_FORMAT_R16G16B16A16_SFLOAT : VK10.VK_FORMAT_R8G8B8A8_UNORM;
        boolean uiReady = RtUiOverlay.overlayWidth() == swapW && RtUiOverlay.overlayHeight() == swapH
                && RtUiOverlay.overlayColorView() != 0L && RtUiOverlay.overlayColorImage() != 0L;
        long uiView = uiReady ? RtUiOverlay.overlayColorView() : 0L;
        long uiImg = uiReady ? RtUiOverlay.overlayColorImage() : 0L;

        VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();
        boolean ok = RtDlssFg.INSTANCE.evaluate(cmd.address(),
                backbufferView, backbufferImage, fmt,
                gDepth.view, gDepth.image, VK10.VK_FORMAT_R32_SFLOAT,
                gMotion.view, gMotion.image, VK10.VK_FORMAT_R16G16_SFLOAT,
                hudlessView, hudlessImg, hudlessReady ? hudlessFmt : 0,
                uiView, uiImg, uiReady ? VK10.VK_FORMAT_R8G8B8A8_UNORM : 0,
                out.view, out.image, fmt,
                swapW, swapH, renderW, renderH, count, index, 1.0f, 1.0f,
                true /* depthInverted (reversed-Z) */, hdrBackbuffer /* colorBuffersHDR */,
                true /* cameraMotionIncluded (in mvecs) */, fgReset,
                fgClipToPrev, fgPrevToClip);
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(fg interpolate) failed");
        }
        fgReset = false;
        if (!ok) {
            throw new IllegalStateException("ngxshim_evaluate_dlssg failed (RtDlssFg.evaluate returned false)");
        }
        enc.execute(cmd);
        return out;
    }

    /**
     * Generate the next frame with AMD FidelityFX. Returns {@code null} when this frame has no
     * traced guides, when the context could not come up, or when the dispatch failed; the caller
     * then duplicates the rendered frame instead.
     *
     * <p>Only image handles cross the shim boundary. ffx-api builds its own image views from the
     * {@code VkImage} and the format, so unlike the DLSS path no view handle is needed here, and
     * every buffer it reads has to be in {@code VK_IMAGE_LAYOUT_GENERAL} - which is what
     * {@link RtContext#createStorageImage} produces.
     *
     * <p>{@code srcW}/{@code srcH} are the extent of {@code backbufferImage} itself and are what the backend
     * is told to treat as the display size. The swapchain extent is not usable for that: the window system
     * can make it a pixel different from the render target (a 2560x1441 swapchain over a 2560x1440 target),
     * and describing a 1440-row image as 1441 rows puts the generated frame on a slightly different
     * vertical scale from the real one. The two are then presented on alternate frames, which reads as the
     * whole screen, user interface included, vibrating up and down.
     */
    public RtImage fgInterpolateFsr(VulkanCommandEncoder enc, long backbufferImage,
            int srcW, int srcH, boolean hdrBackbuffer) {
        if (failed || gDepth == null || gMotion == null || !frameCaptured) {
            return null;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return null;
        }
        final int fmt = hdrBackbuffer ? VK10.VK_FORMAT_R16G16B16A16_SFLOAT : VK10.VK_FORMAT_R8G8B8A8_UNORM;
        if (!RtFsrFg.INSTANCE.ensureContext(srcW, srcH, renderW, renderH, fmt)) {
            return null;
        }
        // The output is allocated at the colour input's extent too, so the backend writes one image whose
        // geometry matches its input; the extra present stretches it to the swapchain the same way
        // Minecraft's own blit stretches the render target.
        ensureFgInterp(ctx, 1, srcW, srcH, fmt);
        RtImage out = fgInterp[0];
        // A hudless target that is missing or the wrong size is worse than none: pass 0 and FSR
        // separates screen-fixed content by itself.
        RtImage hudlessSrc = hdrBackbuffer ? fgHdrHudlessImage : fgHudlessImage;
        // The HUD-less copy is sized by the main render target, and a pixel of disagreement between the two
        // is a resize in progress rather than a reason to drop the input. Without it FSR interpolates the
        // presented image - smearing the UI into the world motion and putting a static HUD into the
        // optical-flow input - so only a real mismatch is worth giving up over; the buffer is described by
        // its own size either way, so nothing is read out of bounds.
        boolean hudlessReady = hudlessSrc != null
                && Math.abs(hudlessSrc.width - srcW) <= 2
                && Math.abs(hudlessSrc.height - srcH) <= 2;
        if (!hudlessReady && fgHudlessMismatch != srcW * 100000 + srcH) {
            fgHudlessMismatch = srcW * 100000 + srcH;
            CausticaMod.LOGGER.warn("FSR-FG has no usable HUD-less source: color input {}x{}, captured hudless {} "
                            + "(the presented image becomes the interpolation source, so the UI is interpolated too)",
                    srcW, srcH, hudlessSrc == null ? "was never captured"
                            : hudlessSrc.width + "x" + hudlessSrc.height);
        }

        VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();
        boolean ok = RtFsrFg.INSTANCE.frame(cmd.address(),
                backbufferImage, fmt,
                hudlessReady ? hudlessSrc.image : 0L,
                hudlessReady ? (hdrBackbuffer ? VK10.VK_FORMAT_R16G16B16A16_SFLOAT
                        : VK10.VK_FORMAT_R8G8B8A8_UNORM) : 0,
                hudlessReady ? hudlessSrc.width : 0,
                hudlessReady ? hudlessSrc.height : 0,
                out.image, fmt,
                gDepth.image, VK10.VK_FORMAT_R32_SFLOAT,
                gMotion.image, VK10.VK_FORMAT_R16G16_SFLOAT,
                fgReset);
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(fsr interpolate) failed");
        }
        fgReset = false;
        if (!ok) {
            return null;
        }
        enc.execute(cmd);
        return out;
    }

    /**
     * Generate the frame halfway between the previous and current displayed frames from the ray-traced
     * motion guide, for {@link RtFramePresenter} to present as its generated frame. Returns {@code null} —
     * the caller then duplicates the real frame — when this frame has no traced motion guide, when the HDR
     * backbuffer is in use, or on the first frame after the history is allocated or resized. The history
     * copy is recorded on every call regardless, so a frame that cannot be interpolated still feeds the next
     * one that can.
     */
    public RtImage fgInterpolateSelf(VulkanCommandEncoder enc, long backbufferView, long backbufferImage,
            int swapW, int swapH, boolean hdrBackbuffer) {
        if (failed || hdrBackbuffer || gMotion == null || !frameCaptured) {
            return null;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return null;
        }
        if (!ensureFgSelfImages(ctx, swapW, swapH)) {
            return null;
        }
        boolean hasHistory = fgSelfHistoryWritten;
        VkCommandBuffer cmd = enc.allocateAndBeginTransientCommandBuffer();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            if (hasHistory) {
                fgInterpPipeline.setImages(fgSelfInterp.view, backbufferView, fgSelfHistory.view, gMotion.view);
                fgInterpPipeline.dispatch(cmd, swapW, swapH, CausticaConfig.Rt.Fg.DEBUG_VIEW.value());
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            VK10.vkCmdCopyImage(cmd, backbufferImage, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    fgSelfHistory.image, VK10.VK_IMAGE_LAYOUT_GENERAL, copyRegion(stack, swapW, swapH));
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        if (VK10.vkEndCommandBuffer(cmd) != VK10.VK_SUCCESS) {
            throw new IllegalStateException("vkEndCommandBuffer(fg self interpolate) failed");
        }
        enc.execute(cmd);
        fgSelfHistoryWritten = true;
        return hasHistory ? fgSelfInterp : null;
    }

    /**
     * Create the interpolated output and the history image at backbuffer size, and report whether they are
     * usable. A fresh pair is not: it has no previous frame in it yet. Resizing discards the history, and
     * the pipeline is rebuilt with the images because its descriptor set refers to them.
     */
    private boolean ensureFgSelfImages(RtContext ctx, int w, int h) {
        final int fmt = VK10.VK_FORMAT_R8G8B8A8_UNORM;
        if (fgSelfInterp != null && fgSelfHistory != null && fgInterpPipeline != null
                && fgSelfInterp.width == w && fgSelfInterp.height == h) {
            return true;
        }
        // Both images may still be read by an interpolation queued for the frame being replaced.
        ctx.waitIdle();
        if (fgInterpPipeline != null) {
            fgInterpPipeline.destroy();
            fgInterpPipeline = null;
        }
        if (fgSelfInterp != null) {
            fgSelfInterp.destroy();
        }
        if (fgSelfHistory != null) {
            fgSelfHistory.destroy();
        }
        fgSelfInterp = ctx.createStorageImage(w, h, fmt, "FG self interp " + w + "x" + h);
        fgSelfHistory = ctx.createStorageImage(w, h, fmt, "FG history " + w + "x" + h);
        fgInterpPipeline = RtFgInterpPipeline.create(ctx);
        fgSelfHistoryWritten = false;
        return false;
    }

    private boolean ensureFgFeature(RtContext ctx, int w, int h, int rw, int rh, int fmt) {
        if (RtDlssFg.INSTANCE.featureReadyFor(w, h, rw, rh, fmt)) {
            return true;
        }
        // Create the feature in its own submit + wait (not folded into MC's frame submit).
        ctx.submitSync(c -> RtDlssFg.INSTANCE.ensureFeature(c.address(), w, h, rw, rh, fmt));
        fgReset = true; // fresh feature has no temporal history
        return RtDlssFg.INSTANCE.featureReadyFor(w, h, rw, rh, fmt);
    }

    private void ensureFgInterp(RtContext ctx, int count, int w, int h, int fmt) {
        if (fgInterp.length == count && fgInterpW == w && fgInterpH == h && fgInterpFormat == fmt
                && (count == 0 || fgInterp[0] != null)) {
            return;
        }
        for (RtImage img : fgInterp) {
            if (img != null) {
                img.destroy();
            }
        }
        fgInterp = new RtImage[count];
        for (int i = 0; i < count; i++) {
            fgInterp[i] = ctx.createStorageImage(w, h, fmt, "FG interp " + i + " " + w + "x" + h);
        }
        fgInterpW = w;
        fgInterpH = h;
        fgInterpFormat = fmt;
    }
}

package dev.comfyfluffy.caustica.rt;

import java.util.Optional;

import org.joml.Vector4f;

import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.PrimitiveTopology;
import com.mojang.blaze3d.pipeline.BlendFunction;
import com.mojang.blaze3d.pipeline.ColorTargetState;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.FilterMode;

import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.BindGroupLayouts;
import net.minecraft.client.renderer.RenderPipelines;

/**
 * Transparent final-UI overlay. World-space overlay features and the vanilla GUI/HUD
 * are routed into one transparent {@code RGBA8} target, then that single image is composited back over the
 * world (from {@code GameRendererMixin}, right after {@code GuiRenderer.render} returns). In SDR this
 * reproduces vanilla; the point is to keep SDR-authored UI out of the world's HDR tonemap once HDR
 * presentation lands (the compositor will then blend this same overlay over the HDR world at paper white
 * rather than over the SDR main target).
 *
 * <p>Composite blend: vanilla GUI pipelines use {@code BlendFunction.TRANSLUCENT} (colour {@code SRC_ALPHA,
 * ONE_MINUS_SRC_ALPHA}; alpha {@code ONE, ONE_MINUS_SRC_ALPHA}), so drawing onto a cleared target
 * accumulates <em>premultiplied</em> colour ({@code rgb = C*A}, {@code a = A}). The composite therefore uses
 * premultiplied-over ({@code TRANSLUCENT_PREMULTIPLIED_ALPHA}); {@code ENTITY_OUTLINE_BLIT} expects straight
 * alpha and would double-darken semi-transparent UI. The pipeline is unregistered, so its shaders are not
 * preloaded — it lazily compiles fine once resources are loaded, but cannot compile during the loading
 * screen; hence the {@link #enabled()} {@code isGameLoadFinished} guard, plus a defensive try/catch.
 *
 * <p>Depth: the overlay clears depth to 0.0 each frame, exactly as {@code GameRenderer.render} clears the
 * main depth right before the GUI. Blur ({@code GameRenderer.processBlurEffect}) still operates on the real
 * main target, so the world behind screens is blurred as usual and the overlay composites over the result.
 */
public final class RtUiOverlay {
    private static final Vector4f TRANSPARENT = new Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

    /** Fullscreen blit that composites the premultiplied overlay over the destination (premultiplied-over). */
    private static final RenderPipeline COMPOSITE_PIPELINE = RenderPipeline.builder(RenderPipelines.GLOBALS_SNIPPET)
            .withLocation("pipeline/caustica_ui_overlay_composite")
            .withVertexShader("core/screenquad")
            .withFragmentShader("core/blit_screen")
            .withBindGroupLayout(BindGroupLayouts.IN_SAMPLER)
            .withColorTargetState(new ColorTargetState(
                    Optional.of(BlendFunction.TRANSLUCENT_PREMULTIPLIED_ALPHA), GpuFormat.RGBA8_UNORM, ColorTargetState.WRITE_COLOR))
            .withPrimitiveTopology(PrimitiveTopology.TRIANGLES)
            .build();

    private static TextureTarget overlay;
    private static boolean usedThisFrame;
    private static boolean compositeFailed;
    // The overlay is cleared once per frame, before the first thing that renders into it (RT world overlays,
    // the hand/screen-effects redirects in HDR mode, or the GUI). Reset at the start of GameRenderer.render
    // via beginFrame().
    private static boolean overlayClearedThisFrame;

    private RtUiOverlay() {
    }

    /**
     * Runs regardless of HDR mode because the GUI redirect and composite-back reproduce vanilla in SDR,
     * while {@code RtWorldOverlay}'s composite point is this same
     * seam and needs it to fire every frame. Active only once the game has finished loading: the composite
     * pipeline lazily compiles its shaders, which are not available during the loading screen (would crash
     * with "Couldn't find source for core/screenquad"). Gating the redirect here keeps the loading-screen
     * GUI on the normal path.
     */
    public static boolean enabled() {
        return !compositeFailed && Minecraft.getInstance().isGameLoadFinished();
    }

    /** Whether the overlay holds this frame's UI (for the HDR present path to composite + consume). */
    public static boolean populatedThisFrame() {
        return usedThisFrame && overlay != null;
    }

    /** Mark the overlay consumed by the HDR present composite (so it isn't reused next frame). */
    public static void markConsumed() {
        usedThisFrame = false;
    }

    public static int overlayWidth() {
        return overlay != null ? overlay.width : 0;
    }

    public static int overlayHeight() {
        return overlay != null ? overlay.height : 0;
    }

    /** The overlay color image view, for the HDR composite compute pass (0 if not available). */
    public static long overlayColorView() {
        if (overlay == null || overlay.getColorTextureView() == null) {
            return 0L;
        }
        if (overlay.getColorTextureView() instanceof com.mojang.blaze3d.vulkan.VulkanGpuTextureView v) {
            return v.vkImageView();
        }
        return 0L;
    }

    /** The overlay color image (0 if not available) — pairs with {@link #overlayColorView()} for callers
     * (e.g. the DLSSG "ui" optional resource) that need both the view and the raw image. */
    public static long overlayColorImage() {
        if (overlay == null || overlay.getColorTexture() == null) {
            return 0L;
        }
        if (overlay.getColorTexture() instanceof com.mojang.blaze3d.vulkan.VulkanGpuTexture t) {
            return t.vkImage();
        }
        return 0L;
    }

    /**
     * Prepare the overlay (sized to {@code main}, cleared transparent with depth cleared to 0.0) and return
     * it so {@code GuiRenderer.draw} renders the GUI into it instead of the main target. Called from the
     * {@code GuiRendererMixin} redirect on the render thread.
     */
    public static RenderTarget beginAndRedirect(RenderTarget main) {
        return prepare(main);
    }

    /**
     * Prepare the shared transparent overlay for non-GUI contributors such as RT world overlays. Call before
     * the GUI redirect so the GUI draws over those contributors in the same image.
     */
    public static RenderTarget beginCompositeLayer(RenderTarget main) {
        return prepare(main);
    }

    /** Reset the per-frame clear latch. Called at the start of {@code GameRenderer.render} (every frame). */
    public static void beginFrame() {
        overlayClearedThisFrame = false;
    }

    /**
     * Ensure the overlay exists, sized to {@code main}, and cleared (transparent + depth 0.0) exactly once
     * this frame, then mark it used. Both the hand redirect and the GUI redirect funnel through here so the
     * overlay is cleared before the hand (which renders first) and not wiped before the GUI.
     */
    private static TextureTarget prepare(RenderTarget main) {
        TextureTarget target = ensureSized(main);
        if (!overlayClearedThisFrame) {
            CommandEncoder enc = RenderSystem.getDevice().createCommandEncoder();
            if (target.useDepth && target.getDepthTexture() != null) {
                enc.clearColorAndDepthTextures(target.getColorTexture(), TRANSPARENT, target.getDepthTexture(), 0.0);
            } else {
                enc.clearColorTexture(target.getColorTexture(), TRANSPARENT);
            }
            overlayClearedThisFrame = true;
        }
        usedThisFrame = true;
        return target;
    }

    /**
     * HDR mode: redirect a world-space overlay render (the held-item/hand, then the fire/underwater/
     * view-blocking screen effects) into the overlay so it composites over the HDR world at paper white,
     * via the render-system output overrides honoured by {@code PreparedRenderType}. Both share the overlay's
     * color+depth (cleared once per frame), matching vanilla where hand and screen effects share the main
     * target's depth without a clear between them. Must be paired with {@link #endOutputRedirect()}.
     */
    public static void beginOutputRedirect(RenderTarget main) {
        TextureTarget target = prepare(main);
        RenderSystem.outputColorTextureOverride = target.getColorTextureView();
        RenderSystem.outputDepthTextureOverride = target.getDepthTextureView();
    }

    public static void endOutputRedirect() {
        RenderSystem.outputColorTextureOverride = null;
        RenderSystem.outputDepthTextureOverride = null;
    }

    /**
     * Composite the overlay over the real main target. Called once per frame from {@code GameRendererMixin}
     * after {@code GuiRenderer.render} returns (the {@code GuiRenderer.draw} TAIL did not fire on in-game HUD
     * frames). A compile/render failure latches the overlay off rather than crashing the frame.
     */
    public static void compositeIfUsed() {
        if (!usedThisFrame || overlay == null) {
            usedThisFrame = false;
            return;
        }
        if (RtComposite.INSTANCE.isHdrPresentActive()) {
            // HDR path composites the overlay over the PQ HDR image at present; leave usedThisFrame set so
            // presentHdr can consume it. Do NOT composite over the SDR main target (it isn't presented).
            return;
        }
        usedThisFrame = false;
        RenderTarget main = Minecraft.getInstance().gameRenderer.mainRenderTarget();
        if (main == null || main.getColorTextureView() == null) {
            return;
        }
        CommandEncoder enc = RenderSystem.getDevice().createCommandEncoder();
        try (RenderPass pass = enc.createRenderPass(() -> "UI overlay composite", main.getColorTextureView(), Optional.empty())) {
            pass.setPipeline(COMPOSITE_PIPELINE);
            RenderSystem.bindDefaultUniforms(pass);
            pass.bindTexture("InSampler", overlay.getColorTextureView(),
                    RenderSystem.getSamplerCache().getClampToEdge(FilterMode.NEAREST));
            pass.draw(3, 1, 0, 0);
        } catch (Throwable t) {
            compositeFailed = true;
            org.slf4j.LoggerFactory.getLogger("Caustica")
                    .error("UI overlay composite failed; disabling overlay", t);
        }
    }

    private static TextureTarget ensureSized(RenderTarget main) {
        if (overlay == null) {
            overlay = new TextureTarget("caustica UI overlay", main.width, main.height, true, GpuFormat.RGBA8_UNORM);
        } else if (overlay.width != main.width || overlay.height != main.height) {
            overlay.resize(main.width, main.height);
        }
        return overlay;
    }

    public static void destroy() {
        RenderSystem.outputColorTextureOverride = null;
        RenderSystem.outputDepthTextureOverride = null;
        usedThisFrame = false;
        overlayClearedThisFrame = false;
        compositeFailed = false;
        if (overlay != null) {
            overlay.destroyBuffers();
            overlay = null;
        }
    }

    /** Clear the composite failure latch so a render-state invalidation (F3+A) retries the overlay. */
    public static void resetFailureLatch() {
        compositeFailed = false;
    }
}

package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float4;
import org.joml.Matrix4fc;

/**
 * The primary pass's thin lens, as the lengths its ray generation reads each frame: the radius of the
 * aperture a pixel samples a point out of, and the distance of the plane the re-aimed ray crosses, both in
 * blocks.
 *
 * <p>The radius is the dial, in blocks, because that is what a lens physically has. A lens of radius r
 * focused at f throws a point at range s off its pinhole ray by r * (1 - s / f) blocks there, which is r / s
 * of the frame's tangent span: the circle closes on the focal plane, opens as the surface comes toward the
 * eye, and settles at r / f out at the horizon. Both ends get the same ceiling: the host applies it to the
 * horizon, which is the one surface whose blur the focus distance alone sets, and the raygen applies it per
 * pixel against the depth that pixel showed last frame — because a foreground disc is what runs away as the
 * focal plane recedes, and no frame can know its own depth before it has traced it.
 *
 * <p>The focus is metered off the trace rather than raycast through the block state. The primary pass is one
 * aperture point per pixel, so a frame cannot know what the centre of the view is standing in front of until
 * the next one, and the depth image it publishes for its own blur ceiling already answers that for every
 * pixel — including the entities, particles, water faces and partial blocks a block-shape probe walks
 * straight through. That measurement reaches the host a frame or two late, which is what
 * {@code dof.rack-time} exists to absorb: the plane glides to the answer instead of strobing between two
 * depths as the view crosses a doorway.
 *
 * <p>What that image holds is the distance to the surface each pixel drew, except where the pixel's picture
 * is a reflection: a plane reflector images its content as far behind itself as that content is in front, so
 * water at arm's length carrying a mountain is showing something much further out than the water, and
 * {@code dof.reflection-focus} makes the pass publish that distance instead. The lens needs no second rule
 * for it — the focal plane is placed on the pinhole ray either way, and a mirror image lies on that ray.
 */
final class RtDepthOfField {
    /**
     * The distance the lens treats as infinity, in blocks. The meter clamps everything past it — so a view
     * of open sky focuses here rather than at a range no blur disc can tell apart from it — and it is where
     * the plane sits until the first measurement arrives.
     */
    static final float FAR_BLOCKS = 64.0f;

    /**
     * The blur cap, in pixels of the finished frame, or 0 when {@code dof.blur-cap} is off. The dial is a
     * physical lens radius, so nothing bounds the discs it asks for: a wall half a block from the eye
     * wants a disc a hundred pixels wide, and one aperture point per frame cannot fill that, so the
     * denoiser hands back a mosaic instead of a blur. The host applies the cap to the horizon (the one
     * disc the focus distance alone sets) and the raygen applies it per pixel against the depth that
     * pixel showed last frame, unless {@code dof.per-pixel-cap} retires that half. Zero is the shader's
     * own gate, so an uncapped lens costs no clamp.
     */
    private static float blurCapPixels() {
        return CausticaConfig.Rt.DepthOfField.BLUR_CAP.value()
                ? CausticaConfig.Rt.DepthOfField.MAX_BLUR_PIXELS.value() : 0.0f;
    }

    // The plane the last frame showed, 0 until a frame has established one.
    private float focusBlocks;

    /**
     * The focal plane the most recent {@link #push} carried, in blocks; 0 before the first push. The
     * post-process gather reads it for its circle-of-confusion math, since the push it defocuses
     * against is the one this frame's trace ran with.
     */
    float focusBlocks() {
        return focusBlocks;
    }

    /**
     * This frame's {@code WorldPush.dof} lanes: the lens radius and the focal distance in blocks, the
     * per-pixel blur ceiling, and whether the guides are held on the pinhole ray. A zero radius is the
     * host's half of the raygen's gate back to a pinhole camera, and it is what the frame gets while
     * the mode is off — the lens is then never sampled and the traced ray is the one this feature did
     * not exist to build.
     *
     * <p>Mode 1 (traced) fills all four lanes and the raygen samples the lens. Mode 2 (post) fills
     * only the radius, and the smallest legal one at that: the raygen's lens branch needs the radius
     * AND the focus distance, so a lone radius keeps the camera pinhole while the depth publication in
     * the guide pass — gated on the radius alone — keeps running. That published depth is the whole
     * point: it is both the autofocus meter's input and the per-pixel circle of confusion the
     * post-process gather defocuses with. The floor exists because a zero radius would silently retire
     * both readers, and because an aperture dial of 0 means "no blur", which the gather produces on
     * its own.
     */
    Float4 push(float frameSeconds, int displayHeight, Matrix4fc projection, float meteredFocusBlocks) {
        int mode = CausticaConfig.Rt.DepthOfField.MODE.value();
        if (mode == 0) {
            focusBlocks = 0.0f;
            return new Float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float target = targetFocusBlocks(meteredFocusBlocks);
        // A lens with no history starts at its target: racking in from nothing would show a swing the
        // player never asked for, and on the frame the feature is switched on there is nothing to rack
        // from.
        focusBlocks = focusBlocks > 0.0f
                ? rackFocus(focusBlocks, target, frameSeconds,
                        CausticaConfig.Rt.DepthOfField.RACK_SECONDS.value())
                : target;
        float aperture = CausticaConfig.Rt.DepthOfField.APERTURE.value();
        if (mode == 2) {
            return new Float4(Math.max(aperture, 1.0e-4f), 0.0f, 0.0f, 0.0f);
        }
        // The cap as a share of the frame's tangent span, and the radius that share allows at this
        // frame's focus: the far field softens at radius over focus, so the horizon is the bound the host
        // can apply on its own. The raygen applies the same cap per pixel against the depth that pixel
        // showed last frame, which is the bound no frame can know before it traces — unless that read-back
        // is switched off, and then the lane it reads goes zero and only the horizon stays held.
        float cap = blurCapPixels() * tangentPerDisplayPixel(displayHeight, projection);
        float perPixelCap = CausticaConfig.Rt.DepthOfField.PER_PIXEL_CAP.value() ? cap : 0.0f;
        return new Float4(cap > 0.0f ? Math.min(aperture, cap * focusBlocks) : aperture,
                focusBlocks, perPixelCap,
                CausticaConfig.Rt.DepthOfField.PINHOLE_GUIDES.value() ? 1.0f : 0.0f);
    }

    /**
     * Where the plane is asked to sit: the dial, or what the middle of the frame was actually showing.
     *
     * <p>A meter that has not completed a round trip yet answers 0, and the far stop is the honest answer
     * for that case rather than a guess at a distance: the first frames of a lens-on history have no
     * measurement, and a plane aimed at the horizon walks in from there as soon as one arrives.
     */
    private float targetFocusBlocks(float meteredFocusBlocks) {
        if (!CausticaConfig.Rt.DepthOfField.AUTO_FOCUS.value()) {
            return CausticaConfig.Rt.DepthOfField.FOCUS_BLOCKS.value();
        }
        return meteredFocusBlocks > 0.0f
                ? Math.min(meteredFocusBlocks, FAR_BLOCKS) : FAR_BLOCKS;
    }

    /**
     * One frame of the rack: exponential approach, so the plane covers the same fraction of its remaining
     * distance per second whatever the frame rate is, and a hitch cannot jump it further than a steady
     * frame would. {@code rackSeconds} is the time constant the settings dial names.
     */
    static float rackFocus(float current, float target, float frameSeconds, float rackSeconds) {
        float approach = 1.0f - (float) Math.exp(-Math.max(0.0f, frameSeconds) / rackSeconds);
        return current + (target - current) * approach;
    }

    /**
     * The share of the frame's tangent span one display pixel covers, which is the unit a surface's blur
     * offset is measured in (offset over depth). {@code m11} is the projection's cotangent of the vertical
     * half field of view, so the frame spans {@code 2 / |m11|} of it and one pixel spans that over its
     * height. Pixels are what a player judges a blur by, and the projection is the only truth about how
     * wide the frame is — a zoom, a spectator camera and a field-of-view mod all change it. An unusable
     * projection answers zero, which pins the ceiling shut and leaves the frame on the pinhole camera.
     */
    static float tangentPerDisplayPixel(int displayHeight, Matrix4fc projection) {
        float m11 = projection.m11();
        if (!Float.isFinite(m11) || Math.abs(m11) < 1.0e-6f || displayHeight <= 0) {
            return 0.0f;
        }
        return 2.0f / (Math.abs(m11) * displayHeight);
    }
}

package dev.comfyfluffy.caustica.rt;

import org.joml.Matrix4f;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The lens host's two per-frame numbers: the aperture radius after its horizon ceiling, and the focal plane
 * after one frame of racking toward its target.
 *
 * <p>The tangent-per-pixel helper is checked against the projection's own tangent span rather than against
 * an angle, because that is the unit the raygen's circle of confusion is measured in: a surface's offset is
 * offset-over-depth, and a pixel covers {@code 2 / (|m11| * height)} of it. Reading it through the field of
 * view's angle instead is the small-angle approximation, which is 15% off at the default 70-degree field
 * and 49% off at 110.
 */
final class DepthOfFieldTest {
    private static final float TAN_HALF_FOV_70 = (float) Math.tan(Math.toRadians(35.0));
    private static final float RACK_SECONDS = 0.2f;

    private static Matrix4f projectionWithCotangent(float cotHalfFovY) {
        return new Matrix4f().m11(cotHalfFovY);
    }

    @Test
    void onePixelIsItsTangentSpan() {
        Matrix4f projection = projectionWithCotangent(1.0f / TAN_HALF_FOV_70);
        assertEquals(2.0 * TAN_HALF_FOV_70 / 1080.0,
                RtDepthOfField.tangentPerDisplayPixel(1080, projection), 1.0e-9);
    }

    @Test
    void theSpanFallsWithTheFrameHeight() {
        Matrix4f projection = projectionWithCotangent(1.0f / TAN_HALF_FOV_70);
        float base = RtDepthOfField.tangentPerDisplayPixel(1080, projection);
        assertEquals(0.5 * base, RtDepthOfField.tangentPerDisplayPixel(2160, projection), 1.0e-12);
        assertEquals(0.0f, RtDepthOfField.tangentPerDisplayPixel(0, projection));
    }

    @Test
    void anUnusableProjectionIsAPlannedCamera() {
        // Zero radius is the raygen's gate back to the pinhole, so a projection the host cannot read has
        // to close the lens rather than invent a span for it. The ceiling is the only thing the projection
        // is used for, so this pins the horizon blur to zero and the lens follows it shut.
        assertEquals(0.0f, RtDepthOfField.tangentPerDisplayPixel(1080, projectionWithCotangent(0.0f)));
    }

    @Test
    void theCeilingTracksTheFocusDistance() {
        // The horizon blurs at radius over focus, so holding it at a fixed pixel count means a ceiling on
        // the radius that grows with the focus: twice the focus, twice the lens.
        Matrix4f projection = projectionWithCotangent(1.0f / TAN_HALF_FOV_70);
        float perPixel = RtDepthOfField.tangentPerDisplayPixel(1080, projection);
        assertEquals(2.0 * 14.0 * perPixel * 8.0, 14.0 * perPixel * 16.0, 1.0e-9);
    }

    @Test
    void theRackApproachesItsTargetWithoutOvershooting() {
        float current = 8.0f;
        float previous = current;
        for (int frame = 0; frame < 120; frame++) {
            current = RtDepthOfField.rackFocus(current, 2.0f, 1.0f / 60.0f, RACK_SECONDS);
            assertTrue(current <= previous && current >= 2.0f,
                    "the plane moved away from its target: " + previous + " -> " + current);
            previous = current;
        }
        assertEquals(2.0f, current, 0.02f);
    }

    @Test
    void theRackIsTheSameDistancePerSecondAtAnyFrameRate() {
        float at120 = 8.0f;
        for (int frame = 0; frame < 240; frame++) {
            at120 = RtDepthOfField.rackFocus(at120, 2.0f, 1.0f / 120.0f, RACK_SECONDS);
        }
        float at60 = 8.0f;
        for (int frame = 0; frame < 120; frame++) {
            at60 = RtDepthOfField.rackFocus(at60, 2.0f, 1.0f / 60.0f, RACK_SECONDS);
        }
        assertEquals(at60, at120, 1.0e-5);
    }

    @Test
    void aFrameWithNoTimeMovesThePlaneNoFurther() {
        assertEquals(5.0f, RtDepthOfField.rackFocus(5.0f, 30.0f, 0.0f, RACK_SECONDS));
        // A hitch clamps its step rather than jumping the plane across the scene.
        assertEquals(5.0f, RtDepthOfField.rackFocus(5.0f, 30.0f, -1.0f, RACK_SECONDS), 1.0e-6);
    }

    @Test
    void theDialIsTheTimeThePlaneArrives() {
        // The dial names a duration, so one dial of time has to leave the plane on target rather than a
        // time constant's 63% of the way: half of it is already nine tenths of the defocus, and the whole
        // of it is within a percent — which is what a number a player reads in seconds has to mean.
        float dial = 0.2f;
        float half = RtDepthOfField.rackFocus(8.0f, 2.0f, dial * 0.5f, dial);
        assertTrue(half < 2.2f, "half the dial left " + half + " blocks to go");
        assertEquals(2.0f, RtDepthOfField.rackFocus(8.0f, 2.0f, dial, dial), 0.02f);
        float quick = RtDepthOfField.rackFocus(8.0f, 2.0f, 1.0f / 60.0f, 0.05f);
        float slow = RtDepthOfField.rackFocus(8.0f, 2.0f, 1.0f / 60.0f, 0.8f);
        assertTrue(slow > quick && quick > 2.0f,
                "a slower dial moved the plane further: " + quick + " vs " + slow);
    }

    @Test
    void theRackOnlyEverSlowsDown() {
        // What a pull cannot do is hang at the far end, rush through the middle and crawl into the near
        // end: the defocus it covers per frame has to fall from the first frame on, whatever the range.
        float current = 64.0f;
        float previous = Float.MAX_VALUE;
        for (int frame = 0; frame < 90; frame++) {
            float next = RtDepthOfField.rackFocus(current, 3.0f, 1.0f / 60.0f, RACK_SECONDS);
            float step = Math.abs(1.0f / next - 1.0f / current);
            assertTrue(step <= previous + 1.0e-6f, "the rack sped up on frame " + frame);
            previous = step;
            current = next;
        }
    }

    @Test
    void theRackTakesTheSameTimeInBothDirections() {
        // Dioptres are symmetric where blocks are not: a dial that answered in a hundredth of a second
        // racking out and a second racking in would be the same pull read two ways.
        assertEquals(framesToCover(64.0f, 3.0f), framesToCover(3.0f, 64.0f),
                "the rack is not symmetric between the two directions");
    }

    /** The frame on which the plane has covered 95% of the move in dioptres, or -1 if it never does. */
    private static int framesToCover(float from, float to) {
        float start = 1.0f / from;
        float goal = Math.abs(1.0f / to - start);
        float current = from;
        for (int frame = 1; frame <= 600; frame++) {
            current = RtDepthOfField.rackFocus(current, to, 1.0f / 60.0f, RACK_SECONDS);
            if (Math.abs(1.0f / current - start) >= 0.95f * goal) {
                return frame;
            }
        }
        return -1;
    }
}

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
    void aLongerFocusTimeEasesThePlaneIn() {
        // The dial is the glide's time constant: after it elapses the plane has covered about 63% of the
        // distance, and a slower dial must arrive later, never earlier.
        float tau = RtDepthOfField.rackFocus(8.0f, 2.0f, 0.2f, 0.2f);
        assertEquals(8.0f - 6.0f * (1.0f - (float) Math.exp(-1.0)), tau, 1.0e-5);
        float quick = RtDepthOfField.rackFocus(8.0f, 2.0f, 1.0f / 60.0f, 0.05f);
        float slow = RtDepthOfField.rackFocus(8.0f, 2.0f, 1.0f / 60.0f, 0.8f);
        assertTrue(slow > quick && quick > 2.0f,
                "a slower dial moved the plane further: " + quick + " vs " + slow);
    }
}

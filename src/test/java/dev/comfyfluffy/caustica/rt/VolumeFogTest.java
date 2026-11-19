package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Invariants of the analytic height fog.
 *
 * Two kinds of check, mirroring {@code RisMergeTest}'s split:
 *
 * - The arithmetic tests run against a Java port of volume.slang's closed forms. They compare the
 *   closed-form optical depth against a brute-force numeric integration of the density profile, pin the
 *   boundary behaviour the clamp at the fog base produces, and verify the algebraic identity the
 *   closed-form in-scatter stands on (the integral of transmittance times extinction is 1 - T). They
 *   cannot catch a divergence inside the shader itself — the SPIR-V compile and a fixed-pose A/B cover
 *   that — but they are the cheap place to notice that a formula edit broke a property the estimator
 *   depends on.
 * - The source-pairing tests read the real shader sources and the Java that fills the push block, and
 *   assert the two agree on the flag-bit positions and on the gating rules that keep a fog-off frame
 *   bit-identical to the pre-volume renderer.
 */
final class VolumeFogTest {
    private static final Path WORLD_COMMON = Path.of("shaders/pipelines/world/world_common.slang");
    private static final Path VOLUME = Path.of("shaders/pipelines/world/volume.slang");
    private static final Path MEDIUM = Path.of("shaders/pipelines/world/medium.slang");
    private static final Path COMPOSITE =
            Path.of("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");
    private static final Path PIPELINE =
            Path.of("src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtPipeline.java");
    private static final Path CONFIG =
            Path.of("src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java");
    private static final Path SETTINGS =
            Path.of("src/main/java/dev/comfyfluffy/caustica/client/CausticaSettings.java");

    // ---- Java port of volume.slang's closed forms ----------------------------------------------------

    private static float fogDensity(float sigma0, float baseY, float heightH, float y) {
        if (sigma0 <= 0.0f) {
            return 0.0f;
        }
        return sigma0 * (float) Math.exp(-Math.max(y - baseY, 0.0f) / Math.max(heightH, 1.0e-4f));
    }

    private static float fogExpStretchDepth(float sigma0, float heightH, float above, float dirY,
                                            float len) {
        return sigma0 * (float) Math.exp(-Math.max(above, 0.0f) / heightH) * (heightH / dirY)
                * (float) (1.0 - Math.exp(-dirY * len / heightH));
    }

    private static float fogOpticalDepth(float sigma0, float baseY, float heightH, float y0,
                                         float dirY, float t) {
        if (t <= 0.0f) {
            return 0.0f;
        }
        float above = y0 - baseY;
        if (Math.abs(dirY) < 1.0e-6f) {
            return sigma0 * (float) Math.exp(-Math.max(above, 0.0f) / heightH) * t;
        }
        if (dirY > 0.0f) {
            if (above >= 0.0f) {
                return fogExpStretchDepth(sigma0, heightH, above, dirY, t);
            }
            float sBase = -above / dirY;
            if (t <= sBase) {
                return sigma0 * t;
            }
            return sigma0 * sBase + fogExpStretchDepth(sigma0, heightH, 0.0f, dirY, t - sBase);
        }
        if (above <= 0.0f) {
            return sigma0 * t;
        }
        float sBase = above / (-dirY);
        if (t <= sBase) {
            return fogExpStretchDepth(sigma0, heightH, above, dirY, t);
        }
        return fogExpStretchDepth(sigma0, heightH, above, dirY, sBase) + sigma0 * (t - sBase);
    }

    /** Midpoint-rule optical depth of the same profile, no closed form involved. */
    private static double numericOpticalDepth(float sigma0, float baseY, float heightH, float y0,
                                              float dirY, float t, int steps) {
        double h = (double) t / steps;
        double sum = 0.0;
        for (int i = 0; i < steps; i++) {
            double s = (i + 0.5) * h;
            sum += fogDensity(sigma0, baseY, heightH, (float) (y0 + dirY * s)) * h;
        }
        return sum;
    }

    /**
     * Java port of fogSegmentInScatterMarched's transmittance quadrature with the noise amplitude at
     * zero, the shadow switch off (vis = 1 everywhere) and the jitter at zero (every sample at its
     * exact step midpoint, volumeParams.y = 0): the exact configuration the march-to-analytic
     * convergence claim is made against.
     */
    private static float marchedOpticalDepth(float sigma0, float baseY, float heightH, float y0,
                                             float dirY, float t, int steps) {
        float dt = t / steps;
        float tau = 0.0f;
        for (int i = 0; i < steps; i++) {
            float s = (i + 0.5f) * dt;
            tau += fogDensity(sigma0, baseY, heightH, y0 + dirY * s) * dt;
        }
        return tau;
    }

    private static void assertClose(String message, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        assertTrue(Math.abs(expected - actual) <= tolerance * scale,
                message + ": expected " + expected + ", got " + actual);
    }

    // ---- Arithmetic ----------------------------------------------------------------------------------

    @Test
    void closedFormMatchesNumericIntegration() {
        float sigma0 = 0.02f;
        float baseY = 62f;
        float heightH = 24f;
        int steps = 1_000_000;
        // Up, down, and near-horizontal rays from above the base, below it, and exactly at it; short
        // and long distances. The downward cases cross the clamp at the base, which is where the
        // closed form is piecewise and where a sign slip would show up as a negative or wildly wrong
        // optical depth.
        float[] originsY = {baseY + 40f, baseY + 5f, baseY, baseY - 10f};
        float[] dirY = {0.8f, 0.05f, 1.0e-7f, -0.05f, -0.8f};
        float[] distances = {1f, 37f, 500f};
        for (float y0 : originsY) {
            for (float dy : dirY) {
                for (float t : distances) {
                    float closed = fogOpticalDepth(sigma0, baseY, heightH, y0, dy, t);
                    double numeric = numericOpticalDepth(sigma0, baseY, heightH, y0, dy, t, steps);
                    assertClose("optical depth y0=" + y0 + " dy=" + dy + " t=" + t,
                            numeric, closed, 1.0e-5);
                }
            }
        }
    }

    @Test
    void opticalDepthIsZeroAtZeroAndMonotonicInDistance() {
        float sigma0 = 0.03f;
        float baseY = 62f;
        float heightH = 16f;
        assertEquals(0.0f, fogOpticalDepth(sigma0, baseY, heightH, baseY + 30f, 0.4f, 0.0f), 0.0f);
        float previous = 0.0f;
        for (float t = 1f; t <= 2000f; t += 7f) {
            float depth = fogOpticalDepth(sigma0, baseY, heightH, baseY + 30f, 0.2f, t);
            assertTrue(depth >= previous, "optical depth must not decrease along the ray at t=" + t);
            previous = depth;
        }
    }

    @Test
    void densityProfileBoundaries() {
        float sigma0 = 0.02f;
        float baseY = 62f;
        float heightH = 24f;
        // Below and at the base the density clamps to sigma0...
        assertEquals(sigma0, fogDensity(sigma0, baseY, heightH, baseY - 100f), 0.0f);
        assertEquals(sigma0, fogDensity(sigma0, baseY, heightH, baseY), 1.0e-7f);
        // ...and decays exponentially above it, reaching effectively nothing a few scales up.
        assertEquals(sigma0 * (float) Math.exp(-1.0), fogDensity(sigma0, baseY, heightH, baseY + heightH),
                1.0e-7f);
        assertTrue(fogDensity(sigma0, baseY, heightH, baseY + 8f * heightH) < 1.0e-3f * sigma0);
        // Zero density means no fog anywhere, whatever the geometry.
        assertEquals(0.0f, fogDensity(0.0f, baseY, heightH, baseY - 50f), 0.0f);
    }

    /**
     * The closed-form in-scatter stands on integral[T(s) * sigma_t(s) ds] = 1 - T(t) (transmittance's
     * own derivative). For the fog's constant albedo that makes albedo * (1 - T) the exact
     * transmittance-weighted scattering integral; this pins the identity numerically.
     */
    @Test
    void inScatterIdentityMatchesNumericalIntegration() {
        float sigma0 = 0.05f;
        float baseY = 62f;
        float heightH = 24f;
        int steps = 200_000;
        float[] originsY = {baseY + 30f, baseY, baseY - 15f};
        float[] dirY = {0.5f, 1.0e-7f, -0.5f};
        float[] distances = {5f, 80f, 1200f};
        for (float y0 : originsY) {
            for (float dy : dirY) {
                for (float t : distances) {
                    double h = (double) t / steps;
                    double tau = 0.0;
                    double weighted = 0.0;
                    for (int i = 0; i < steps; i++) {
                        double s = (i + 0.5) * h;
                        double sigma = fogDensity(sigma0, baseY, heightH, (float) (y0 + dy * s));
                        tau += sigma * h;
                        weighted += Math.exp(-tau) * sigma * h;
                    }
                    double closed = 1.0 - Math.exp(-tau);
                    assertClose("in-scatter identity y0=" + y0 + " dy=" + dy + " t=" + t,
                            weighted, closed, 1.0e-3);
                }
            }
        }
    }

    /**
     * The border haze is one flat addition to the fog's extinction, so everything the closed forms stand on
     * has to survive it: the identity integral[T(s)·sigma_t(s) ds] = 1 - T(t) holds over the SUM as long as
     * the albedo stays the same ratio for both terms, and a frame with no height fog at all has to degenerate
     * to plain Beer–Lambert over the segment rather than to nothing.
     */
    @Test
    void boundaryHazeJoinsTheFogMediumWithoutBreakingTheClosedForm() {
        float sigmaB = 0.015f;
        float baseY = 62f;
        float heightH = 24f;
        int steps = 200_000;
        // Fog plus haze, thin fog with haze, and haze alone — the last one is the frame a player gets by
        // switching on Boundary Fog and leaving Volumetric Fog off.
        float[] sigmas = {0.05f, 0.02f, 0.0f};
        float[] originsY = {baseY + 30f, baseY, baseY - 15f};
        float[] dirY = {0.5f, 1.0e-7f, -0.5f};
        float[] distances = {5f, 80f, 1200f};
        for (float sigma0 : sigmas) {
            for (float y0 : originsY) {
                for (float dy : dirY) {
                    for (float t : distances) {
                        double h = (double) t / steps;
                        double tau = 0.0;
                        double weighted = 0.0;
                        for (int i = 0; i < steps; i++) {
                            double s = (i + 0.5) * h;
                            double sigma = fogDensity(sigma0, baseY, heightH, (float) (y0 + dy * s))
                                    + sigmaB;
                            tau += sigma * h;
                            weighted += Math.exp(-tau) * sigma * h;
                        }
                        // volume.slang's airOpticalDepth: the height profile's closed form plus sigmaB * t.
                        float closed = fogOpticalDepth(sigma0, baseY, heightH, y0, dy, t) + sigmaB * t;
                        assertClose("air optical depth sigma0=" + sigma0 + " y0=" + y0 + " dy=" + dy
                                + " t=" + t, tau, closed, 1.0e-5);
                        assertClose("in-scatter identity sigma0=" + sigma0 + " y0=" + y0 + " dy=" + dy
                                + " t=" + t, weighted, 1.0 - Math.exp(-tau), 1.0e-3);
                    }
                }
            }
        }
        // With no height fog in the profile, the flat lane is the whole medium and prices the segment exactly.
        assertEquals(sigmaB * 40f, fogOpticalDepth(0.0f, baseY, heightH, baseY + 12f, 0.3f, 40f)
                + sigmaB * 40f, 0.0f);
        // A segment that starts 300 blocks above the fog base has effectively no height fog left in it —
        // the exponential is down at 1e-5 of its base — while the haze is flat and prices the whole stretch.
        // This is the property the marched tier's span clamp has to stand down for.
        float high = baseY + 300f;
        float hazeOnly = sigmaB * 200f;
        float profileAtHeight = fogOpticalDepth(0.02f, baseY, heightH, high, 0.0f, 200f);
        assertTrue(profileAtHeight < 1.0e-4f * hazeOnly,
                "the height profile must be negligible up there, or the comparison below proves nothing");
        assertEquals(hazeOnly, profileAtHeight + hazeOnly, 1.0e-4f * hazeOnly);
    }

    /**
     * The wiring the border haze needs to be a fog rather than a second layer: it joins the density the
     * segments integrate, on both tiers, and it raises the fog's own flag so every consumer that asks
     * "is there air on this segment" sees it. The leg toward the sun is the one place it must NOT appear:
     * a medium with no falloff has infinite optical depth that way, so including it would delete the sun
     * term, and with it every light shaft, at haze levels too thin to see on the view ray.
     */
    @Test
    void boundaryHazeJoinsTheFogMediumInSource() throws IOException {
        String volume = Files.readString(VOLUME);
        String worldCommon = Files.readString(WORLD_COMMON);
        String composite = Files.readString(COMPOSITE);
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));

        assertTrue(worldCommon.contains("public float4   boundaryFog;"),
                "WorldPush must carry the border haze's lane");
        assertTrue(volume.contains("return push.boundaryFog.x;"),
                "the shader must read the haze straight off the pushed lane, with no second gate");
        assertTrue(volume.contains("+ boundarySigma(worldPush) * t"),
                "the closed form must price the flat term over the segment it is integrating");
        assertTrue(volume.contains("float3 sigmaT = density * FOG_RAYLEIGH_COEFF + boundarySigma(push);"),
                "sampleVolume must hand consumers the same medium the segments integrate");
        assertTrue(volume.contains("float3 sigmaT = density * FOG_RAYLEIGH_COEFF + sigmaB;"),
                "the marched tier must add the haze after its own noise modulation, so the billows "
                        + "never scale the flat term");
        assertTrue(volume.contains("if (density0 + sigmaB <= 0.0 || dot(albedo"),
                "both tiers must run for a frame that carries only the haze, where the height profile "
                        + "is empty");
        assertTrue(volume.contains("float yTop = sigmaB > 0.0 ? 1.0e30"),
                "the marched span clamp must stand down for a flat medium: it trims the stretch above "
                        + "the fog's significant band, and a flat term has no such band to stop at");
        assertTrue(volume.contains("* fogT_toSunRo(ro, celestialLight.dir, density0)"),
                "the unshadowed sun leg must carry the height fog's own density — the flat term is left "
                        + "out on purpose");
        assertTrue(volume.contains("return fogTransmittanceRGB(density0, worldPush.fog0.z, worldPush.fog0.y,"),
                "fogT_toSunRo must stay on the height profile alone, not the combined air depth");
        // The flag the haze rides, and where it is raised.
        assertTrue(composite.contains("boolean boundaryOn = CausticaConfig.Rt.Volume"
                        + ".BOUNDARY_FOG_ENABLED.value() && skyLit;"),
                "the haze must share the fog's sky-light gate: its ambient term reads the sky basis, "
                        + "which is only baked where the sky lights the world");
        assertTrue(composite.contains("if (fogOn || boundaryOn) {"),
                "the fog flag must be up for either air medium, because every air read gates on that one "
                        + "bit; the edge fade deliberately does not raise it, since it is not a medium");
        assertTrue(composite.contains("new Float4(boundarySigma, 0f, 0f, 0f)"),
                "the frame must push the haze it computed, and nothing else in its lane");
        assertTrue(composite.contains("level.getWorldBorder().getDistanceToBorder(camX, camZ)"),
                "the ramp is measured from the camera to the border the server sent, in world blocks");
        // The guide blend prices the same air and the same dissolve the ray does, or the reconstruct reads
        // the fog's dither as material detail on the surface behind it.
        assertTrue(primary.contains("float awayFromSurface = 1.0 - airT * (1.0 - coverage);"),
                "the fog-dominated guide blend must combine the air transmittance with the edge fade's "
                        + "coverage, or a dissolving pixel keeps the surface's albedo guide");
    }

    /**
     * The Rayleigh coefficient is the whole term: Photon's per-channel air, from_srgb(0.31, 0.67, 1.00)
     * times the 0.5 its airmass integral folds in, carried through the push's Rec.709 -> ACEScg matrix.
     * The shader literal and the Java conversion here are separate transcriptions of that one spec, so a
     * drift in either would quietly change the hue the aerial perspective turns. The last check is what
     * the term is FOR: at the default density the display-space e-folding distance is 4 km for blue and
     * 51 km for red, which is why distance blues a landscape out instead of washing it grey.
     */
    @Test
    void rayleighCoefficientCarriesThePhotonSpec() throws IOException {
        String volume = Files.readString(VOLUME);
        float[] coeff = shaderFloat3(volume, "FOG_RAYLEIGH_COEFF");
        float peak = shaderFloat(volume, "FOG_RAYLEIGH_PEAK");
        float mean = shaderFloat(volume, "FOG_RAYLEIGH_MEAN");

        double[] srgb = {0.31, 0.67, 1.00};
        double[] linear = new double[3];
        for (int i = 0; i < 3; i++) {
            linear[i] = (srgb[i] <= 0.04045 ? srgb[i] / 12.92
                    : Math.pow((srgb[i] + 0.055) / 1.055, 2.4)) * 0.5;
        }
        double[][] toAcescg = {
                {0.61309743, 0.33952314, 0.04737945},
                {0.07019372, 0.91635388, 0.01345240},
                {0.02061559, 0.10956977, 0.86981463},
        };
        double[] expected = new double[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                expected[i] += toAcescg[i][j] * linear[j];
            }
            assertEquals(expected[i], coeff[i], 1.0e-5,
                    "channel " + i + " of the Rayleigh coefficient must be the Photon filter in ACEScg");
        }
        assertEquals(Math.max(expected[0], Math.max(expected[1], expected[2])), peak, 1.0e-5,
                "the peak must be the largest channel: it bounds the marched tier's significant band");
        assertEquals(0.2722287 * coeff[0] + 0.6740818 * coeff[1] + 0.0536895 * coeff[2], mean, 1.0e-5,
                "the scalar proxy must be the coefficient's ACEScg luminance");

        // Back to display space: the AP1 -> Rec.709 matrix, which is what the eye's channels integrate.
        double[][] toRec709 = {
                {1.70505, -0.62179, -0.08326},
                {-0.13026, 1.14080, -0.01055},
                {-0.02400, -0.12897, 1.15297},
        };
        double density = 0.0005;
        double[] eFold = {51100.0, 9840.0, 4000.0};
        for (int i = 0; i < 3; i++) {
            double display = 0.0;
            for (int j = 0; j < 3; j++) {
                display += toRec709[i][j] * coeff[j];
            }
            assertEquals(linear[i], display, 1.0e-3,
                    "channel " + i + " must come back to the Photon filter in display space");
            assertEquals(eFold[i], 1.0 / (density * display), eFold[i] * 0.01,
                    "the default density must buy the specified e-folding distance, channel " + i);
        }
    }

    private static float[] shaderFloat3(String source, String name) {
        Matcher m = Pattern.compile("float3 " + name + " = float3\\(([^)]*)\\);").matcher(source);
        assertTrue(m.find(), name + " must be a float3 literal in volume.slang");
        String[] parts = m.group(1).split(",");
        assertEquals(3, parts.length, name + " must carry three channels");
        float[] out = new float[3];
        for (int i = 0; i < 3; i++) {
            out[i] = Float.parseFloat(parts[i].trim().replaceAll("f$", ""));
        }
        return out;
    }

    private static float shaderFloat(String source, String name) {
        Matcher m = Pattern.compile("float " + name + " = ([0-9.eE+-]+)f?;").matcher(source);
        assertTrue(m.find(), name + " must be a float literal in volume.slang");
        return Float.parseFloat(m.group(1));
    }

    /**
     * The fade's coverage is a blend weight, so its shape is the whole feature: exactly zero at the start
     * distance, exactly the opacity dial at the edge, monotone in between. A shell of density had a step at
     * the start — the inner wall that read as a ball of fog around the camera — and a smoothstep cannot.
     */
    @Test
    void edgeFadeCoverageRampsSmoothlyToTheOpacityDial() {
        for (float opacity : new float[] {0.25f, 0.5f, 0.9f, 1.0f}) {
            for (float startFraction : new float[] {0.0f, 0.5f, 0.95f}) {
                double start = 192.0 * startFraction;
                double edge = Math.max(192.0, start + 1.0);
                assertEquals(0.0, coverage(start, opacity, start, edge), 0.0,
                        "the fade must not begin before its start distance");
                assertEquals(0.0, coverage(start - 64.0, opacity, start, edge), 0.0,
                        "nothing in front of the band may be faded at all");
                assertEquals(opacity, coverage(edge, opacity, start, edge), 1.0e-6,
                        "the edge must dissolve exactly as much of the surface as the dial asks");
                assertEquals(opacity, coverage(edge * 4.0, opacity, start, edge), 1.0e-6,
                        "past the edge the coverage saturates: it is a blend, not an integral");
                float previous = -1.0f;
                for (double d = start - 8.0; d <= edge + 8.0; d += 2.0) {
                    float c = coverage(d, opacity, start, edge);
                    assertTrue(c >= previous, "coverage must never fall with distance at d=" + d);
                    previous = c;
                }
            }
        }
        // The off switch: a zero lane fades nothing, whatever the two distances say.
        assertEquals(0.0, coverage(1024.0, 0.0f, 96.0, 192.0), 0.0);
    }

    /** Java port of volume.slang's edgeFadeCoverage, with the pushed lanes passed in. */
    private static float coverage(double distance, float peak, double begin, double edge) {
        if (peak <= 0.0) {
            return 0.0f;
        }
        double t = Math.max(0.0, Math.min(1.0, (distance - begin) / (edge - begin)));
        return (float) (peak * t * t * (3.0 - 2.0 * t)); // smoothstep
    }

    /**
     * The fade is a blend of the finished segment toward the sky, not a medium. Nothing about it may reach
     * the air's optical depth: an extinction lane makes it a veil that lights the view up instead of taking
     * the terrain away, which is the failure the coverage weighting exists to rule out. And it has to be a
     * function of the EYE, so it reaches the segment integrators and nothing that asks a point what the air
     * is worth there — a point query has no view ray to measure a coverage from.
     */
    @Test
    void edgeFadeIsABlendAndNotAMedium() throws IOException {
        String volume = Files.readString(VOLUME);
        String worldCommon = Files.readString(WORLD_COMMON);
        String composite = Files.readString(COMPOSITE);
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));

        assertTrue(worldCommon.contains("public float4   edgeFade;"),
                "WorldPush must carry the fade's lanes");
        assertTrue(worldCommon.contains("public static const uint WORLD_FLAG_EDGE_FADE = 1u << 16u;"),
                "the fade needs its own flag: riding the fog's would tie a blend of the finished pixel to "
                        + "whether the air is being integrated at all");
        assertTrue(volume.contains("float3 eyeRelative = ro + rd * t - worldPush.camOffset;"),
                "the coverage must be measured from the EYE. The trace runs in the terrain's rebased "
                        + "space, where the eye sits at camOffset — reading the hit point's own length "
                        + "measures distance from the terrain origin, which follows the player's "
                        + "coordinates instead of the view");
        assertTrue(volume.contains("return peak * smoothstep(begin, edge, length(eyeRelative));"),
                "the coverage must be a smooth function of that eye distance — a flat density inside a "
                        + "radius is a ball of fog with a visible inner wall");
        assertTrue(volume.contains("public float3 edgeFadeSky(float3 rd)"),
                "the fade needs the sky behind the ray: that is what it dissolves the surface into");
        assertTrue(volume.contains("return fogSkyRadiance(rd);"),
                "the fill must ride the fog's own sky lookup, so a segment the fade has fully dissolved and "
                        + "a segment fog has made fully opaque land on the colour the miss shader would have "
                        + "returned for the ray — never on a second, differently-priced sky");
        assertTrue(volume.contains("skyDiffuseRadiance(skyViewLut, float3(rd.x, abs(rd.y), rd.z), state)"),
                "fogSkyRadiance must mirror the ray's elevation: the view LUT's lower half is the space "
                        + "floor, and terrain dissolving past the edge is haze, not space");
        // Every piece of the old shell must be gone. One that survived in the fog's depth would keep
        // lighting what the fade removes, and would keep attenuating the sun legs.
        for (String gone : new String[]{"farOpticalDepth", "farSigmaOver", "farDensityAt",
                "FAR_ESCAPE_T", "distanceHaze"}) {
            assertTrue(!volume.contains(gone),
                    gone + " belonged to the render-edge shell; the fade carries no optical depth at all");
            assertTrue(!worldCommon.contains(gone),
                    gone + " belonged to the render-edge shell; the fade carries no optical depth at all");
        }

        int sampler = volume.indexOf("public VolumeSample sampleVolume");
        int samplerEnd = volume.indexOf("\n}", sampler);
        assertTrue(sampler > 0 && samplerEnd > sampler, "sampleVolume must be findable");
        assertTrue(!volume.substring(sampler, samplerEnd).contains("edgeFade"),
                "the point sampler must not read the fade: it has no view ray to measure a coverage from");

        // The blend itself: the surviving fraction scales throughput, the sky fills the rest, and the shared
        // bounce-0 segment's fill rides the same once-per-sample prefix the fog's in-scatter does.
        assertTrue(indirect.contains("if (CAUSTICA_VOLUME_PATH && edgeFadeOn && segmentIsAir) {"),
                "the indirect pass must fade every air segment, so a reflection of the far world loses the "
                        + "same edge the direct view does");
        assertTrue(indirect.contains("throughput *= 1.0 - coverage;"),
                "the fade must scale what the segment returns, never add to it");
        assertTrue(indirect.contains("if (firstSegment) prefixL += fill; else L += fill;"),
                "the shared bounce-0 segment must evaluate its fill once, or spp>1 counts it per sample");
        assertTrue(primary.contains("float coverage = edgeFadeOn && (payload.flags & PAYLOAD_LOD) == 0u")
                        && primary.contains("edgeFadeCoverage(ro, rd, payload.hitT)"),
                "the guide blend must price the fade, or RR keeps the surface's albedo under a pixel that "
                        + "is mostly sky");

        assertTrue(composite.contains("getEffectiveRenderDistance()"),
                "the fade's outer face is the renderer's own geometry window, which is the render "
                        + "distance in chunks");
        assertTrue(composite.contains(
                        "new Float4(edgeFadeOn ? CausticaConfig.Rt.Volume.EDGE_FADE_OPACITY.value() : 0f,"),
                "the frame must push the coverage it fades to, together with the band it computed");
        assertTrue(composite.contains("WORLD_FLAG_EDGE_FADE"),
                "the host must raise the flag, and must keep the frame off the volume-free raygen records "
                        + "while it is up");
    }

    /**
     * Fog and the render-edge fill sample the view LUT from raygen, while the visible dome samples it
     * from the miss shader. The host descriptor must expose both stages or the raygen fill is undefined.
     */
    @Test
    void skyViewLutDescriptorCoversRaygenAndMiss() throws IOException {
        String pipeline = Files.readString(PIPELINE);
        int start = pipeline.indexOf("binds.get(WORLD_SKY_VIEW)");
        int end = pipeline.indexOf("binds.get(WORLD_TRANSMITTANCE)", start);
        assertTrue(start >= 0 && end > start, "the sky-view descriptor declaration must be findable");
        String declaration = pipeline.substring(start, end);
        assertTrue(declaration.contains("VK_SHADER_STAGE_MISS_BIT_KHR"),
                "the visible sky miss shader must retain access to the view LUT");
        assertTrue(declaration.contains("VK_SHADER_STAGE_RAYGEN_BIT_KHR"),
                "fog and render-edge sky fills sample the view LUT from raygen; a miss-only descriptor "
                        + "leaves those reads outside the layout and their colour undefined");
    }

    /**
     * Phase 2's hardest acceptance line: the marched tier's integral must converge monotonically to
     * the Phase 1 closed form as the step count rises (8 → 16 → 32 → 64), because with the noise at
     * zero both integrate the same analytic profile.
     */
    @Test
    void marchedTierConvergesMonotonicallyToClosedForm() {
        float sigma0 = 0.02f;
        float baseY = 62f;
        float heightH = 24f;
        int[] stepsList = {8, 16, 32, 64};
        float[] originsY = {baseY + 40f, baseY, baseY - 10f};
        float[] dirY = {0.8f, 0.05f, -0.6f};
        float[] distances = {80f, 500f};
        for (float y0 : originsY) {
            for (float dy : dirY) {
                for (float t : distances) {
                    float closed = fogOpticalDepth(sigma0, baseY, heightH, y0, dy, t);
                    double previousError = Double.MAX_VALUE;
                    // A ray crossing the fog base integrates past the profile's kink, where the
                    // midpoint rule's local error depends on where the kink lands inside a step: the
                    // error stays small (well under 2% here) but is NOT monotone in the step count,
                    // and a coarse step grid can get lucky with kink placement. Smooth rays claim
                    // strict monotone convergence; kink-crossing rays claim the same 2% bound only.
                    boolean crossesBase = (y0 > baseY && dy < 0.0f && (y0 - baseY) / -dy < t)
                            || (y0 < baseY && dy > 0.0f && (baseY - y0) / dy < t);
                    for (int steps : stepsList) {
                        double marched = marchedOpticalDepth(sigma0, baseY, heightH, y0, dy, t, steps);
                        double error = Math.abs(marched - closed) / Math.max(1.0, closed);
                        if (!crossesBase && previousError > 1.0e-6) {
                            // Below the floor the closed form is already hit exactly and the residual
                            // is float accumulation order, which carries no monotonicity meaning.
                            assertTrue(error <= previousError,
                                    "march error must not grow with the step count at y0=" + y0
                                            + " dy=" + dy + " t=" + t + " steps=" + steps);
                        }
                        previousError = error;
                        if (steps == 64) {
                            assertTrue(error < 2.0e-2,
                                    "64-step march must sit within 2% of the closed form at y0=" + y0
                                            + " dy=" + dy + " t=" + t + ": error=" + error);
                        }
                    }
                }
            }
        }
    }

    // ---- Water-volume arithmetic (Java port of medium.slang's water branch + the water in-scatter) ----

    /** Port of medium.slang's waterExtinction: Photon's coefficient bent by a unit-mean biome hue. */
    private static float[] waterExtinction(float tintR, float tintG, float tintB, float opacity) {
        float[] absorption = {0.151676f, 0.087885f, 0.047682f};
        float[] tint = {tintR, tintG, tintB};
        float[] lift = new float[3];
        float meanLift = 0.0f;
        for (int i = 0; i < 3; i++) {
            lift[i] = 1.0f - Math.clamp(tint[i], 0.0f, 1.0f);
            meanLift += lift[i] / 3.0f;
        }
        float[] out = new float[3];
        for (int i = 0; i < 3; i++) {
            float hue = 1.0f + (lift[i] / Math.max(meanLift, 1.0e-3f) - 1.0f) * 0.6f;
            out[i] = absorption[i] * hue * (Math.max(opacity, 0.0f) / 3.0f);
        }
        return out;
    }

    @Test
    void waterAbsorptionSplitsChannelsLikePhoton() throws IOException {
        String shader = Files.readString(MEDIUM);
        assertTrue(shader.contains("public static const float3 WATER_ABSORPTION = "
                + "float3(0.151676, 0.087885, 0.047682);"), "the water coefficient must stay authored");
        // The default tint (a neutral blue body) is the coefficient itself: the hue weight is unit-mean,
        // so a white line only shrinks each channel by the same amount and the ratio survives.
        float[] d = waterExtinction(0.5f, 0.5f, 0.5f, 3.0f);
        assertEquals(0.151676f, d[0], 1e-5f);
        assertEquals(0.087885f, d[1], 1e-5f);
        assertEquals(0.047682f, d[2], 1e-5f);
        // Red is the fastest channel by roughly 3x, the property that makes depth read teal.
        assertTrue(d[0] / d[2] > 2.8f && d[0] / d[2] < 3.6f, "red/blue split: " + (d[0] / d[2]));
        // Halving the opacity halves every channel; zero opacity is clear water.
        float[] half = waterExtinction(0.5f, 0.5f, 0.5f, 1.5f);
        assertEquals(d[0] * 0.5f, half[0], 1e-6f);
        float[] clear = waterExtinction(0.5f, 0.5f, 0.5f, 0.0f);
        assertEquals(0.0f, clear[0] + clear[1] + clear[2], 0.0f);
        // A brown swamp tint (little blue) leans the absorption onto blue, so the body browns instead of
        // turning teal.
        float[] swamp = waterExtinction(0.9f, 0.5f, 0.2f, 3.0f);
        assertTrue(swamp[0] / swamp[2] < d[0] / d[2], "a brown tint must brown the absorption");
        assertTrue(swamp[2] > d[2], "the tint must raise the channel it lacks");
        assertTrue(swamp[0] > 0.0f && swamp[1] > 0.0f && swamp[2] > 0.0f, "no channel may go negative");
    }

    /** Bit position of a `WORLD_FLAG_*` constant in the shader source. */
    private static int shaderFlagBit(String source, String name) {
        Matcher m = Pattern.compile("public static const uint " + name + " = 1u << (\\d+)u;")
                .matcher(source);
        assertTrue(m.find(), "shader must define " + name);
        return Integer.parseInt(m.group(1));
    }

    /** Bit position of the same constant on the Java side of RtComposite. */
    private static int javaFlagBit(String source, String name) {
        Matcher m = Pattern.compile("private static final int " + name + " = 1 << (\\d+);")
                .matcher(source);
        assertTrue(m.find(), "RtComposite must define " + name);
        return Integer.parseInt(m.group(1));
    }

    @Test
    void volumeFlagBitsMatchShader() throws IOException {
        String shader = Files.readString(WORLD_COMMON);
        String java = Files.readString(COMPOSITE);
        for (String flag : new String[]{"WORLD_FLAG_FOG", "WORLD_FLAG_CLOUD", "WORLD_FLAG_WATER_FOG",
                "WORLD_FLAG_VOLUME_SHADOW", "WORLD_FLAG_VOLUME_MARCH", "WORLD_FLAG_VOLUME_DEBUG",
                "WORLD_FLAG_CLOUD_VOLUMETRIC", "WORLD_FLAG_SHAFTS",
                "WORLD_FLAG_EDGE_FADE", "WORLD_FLAG_DOF_REFLECTION", "WORLD_FLAG_CLOUD_OUTSCATTER",
                "WORLD_FLAG_RR_DISABLED", "WORLD_FLAG_CLOUD_SHADOW"}) {
            assertEquals(shaderFlagBit(shader, flag), javaFlagBit(java, flag),
                    flag + " must sit on the same bit in world_common.slang and RtComposite");
        }
    }

    /**
     * The rules that keep a fog-off (or underwater) frame bit-identical: the unified sampler returns an
     * empty sample for water and when the fog flag is clear, and the path integrator's fog term is
     * gated on the same flag. These are source pairings because the behaviour lives in the shader.
     */
    @Test
    void volumeGatingGuardsStayInPlace() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains("if (inWater || (push.flags & WORLD_FLAG_FOG) == 0u)"),
                "sampleVolume must return an empty sample underwater and when fog is disabled");
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        // Both in-scatter insertion points (hit and miss) gate on the fog flag and on the segment being
        // either the trace's first or the air segment directly behind a dielectric crossing (what keeps
        // aerial perspective alive through glass). The fog flag caps everything else. The latched
        // first-air-segment flag is what stops one crossing from licensing both media twice. The first
        // segment's probes are seed-free and shared across spp (evaluated once, into the prefix), but
        // the behind-the-glass probes belong to each sample's own Fresnel outcome and must not be
        // skipped — the airProbeSite split below is what keeps the two apart.
        assertTrue(indirect.contains("bool firstAirSegment = firstSegment || fogAfterInterface;"),
                "the air terms must run on the first segment and behind dielectric crossings only");
        assertTrue(indirect.contains(
                        "bool airProbeSite = (firstSegment && firstSample) || fogAfterInterface;"),
                "shared first-segment probes must be evaluated once while behind-the-glass probes "
                        + "stay per-sample");
        assertTrue(indirect.contains(
                        "if (CAUSTICA_VOLUME_PATH && firstAirSegment && fogOn && segmentIsAir) {"),
                "the fog in-scatter must be gated on the fog flag and on an air segment, and its "
                        + "transmittance must reach every resample of that segment");
        assertTrue(indirect.contains("if (CAUSTICA_VOLUME_PATH && airProbeSite && shaftOn && segmentIsAir)"),
                "the shaft in-scatter must ride the same two gates from its own flag");
        assertTrue(!indirect.contains("(bounce == seg.bounce || fogAfterInterface) && (!fogOn || shaftOn)"),
                "the shafts must not widen the fog's own gate");
        assertTrue(indirect.contains("WORLD_FLAG_SHAFTS) != 0u"),
                "the shaft in-scatter must be gated on its own flag");
        // The air shafts are geometry-only on purpose: the sky LUT already carries the atmosphere's own
        // in-scatter, so a sky miss must not add airlight on top of it. The water column is the
        // exception — nothing else carries it — and it integrates only inside the water-medium branch,
        // before the path stops. The miss branch ends where the diffuse-sky comment picks up.
        String missBranch = indirect.substring(indirect.indexOf("if (payload.hitT < 0.0)"),
                indirect.indexOf("// Only a diffuse lobe may claim the analytic sky term"));
        assertTrue(!missBranch.contains("worldPush.shaft.x"),
                "the air shaft must not run on a sky miss");
        int waterMiss = missBranch.indexOf("if (medium.current.water) {");
        assertTrue(waterMiss >= 0 && missBranch.substring(waterMiss).contains("worldPush.waterShaft.x"),
                "a water miss must still integrate its own column before the path stops");
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        assertTrue(primary.contains("if (!any(medium.current.extinction > 0.0)) {\n"
                        + "                if (CAUSTICA_VOLUME_PATH && fogOn) {"),
                "the pre-interface fog attenuation must apply only while the stack medium is air");
        assertTrue(primary.contains("prefixRadiance += shaftInScatter(ro, rd, payload.hitT, pix,"),
                "the pre-interface shafts must ride the same prefix write as the fog");
        // The pre-interface fog GLOW is radiance common to both Fresnel branches: the primary pass must
        // write it into outImage and flag its records, and the indirect pass must accumulate on top of
        // the stored value rather than replacing it.
        String segment = Files.readString(Path.of("shaders/pipelines/world/segment.slang"));
        assertTrue(segment.contains("public static const uint PATH_PREFIX_IN_IMAGE = 1u << 11u;"),
                "PATH_PREFIX_IN_IMAGE must keep bit 11, the only free pathFlags lane");
        assertTrue(primary.contains("packedTerminal.pathFlags |= PATH_PREFIX_IN_IMAGE"),
                "the primary pass must flag records whose pre-interface fog radiance it wrote");
        assertTrue(indirect.contains("stored += outImage[pix].rgb"),
                "the indirect pass must accumulate the flagged prefix radiance, not overwrite it");
    }

    /**
     * The marched tier's wiring: the shader dispatches it inside fogSegmentInScatter on the march flag
     * plus a sane step count, and Java sets the flag and fills the fog1/volumeParams lanes only when
     * the tier is actually on — an analytic frame must push an inert block, exactly like the fog-off
     * case above.
     */
    @Test
    void marchedTierDispatchAndGating() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains(
                        "(worldPush.flags & WORLD_FLAG_VOLUME_MARCH) != 0u && worldPush.volumeParams.x >= 1.0"),
                "the marched tier must dispatch inside fogSegmentInScatter on the march flag");
        assertTrue(volume.contains("if (inWater || (push.flags & WORLD_FLAG_FOG) == 0u)"),
                "the march-tier density must stay behind sampleVolume's water/fog gate");
        String composite = Files.readString(COMPOSITE);
        assertTrue(composite.contains("boolean fogOn = CausticaConfig.Rt.Volume.FOG_ENABLED.value() && skyLit;"),
                "the fog's own gate is the switch AND the sky-light gate, in one place both tiers read");
        assertTrue(composite.contains("boolean fogMarchOn = fogOn && CausticaConfig.Rt.Volume.FOG_MARCH.value()"),
                "the march flag must sit behind the fog switch and the sky-light gate, through fogOn");
        assertTrue(composite.contains("fogMarchOn ? CausticaConfig.Rt.Volume.FOG_NOISE.value() : 0f"),
                "the fog1 noise lane must be zero while the march tier is off");
        assertTrue(composite.contains("(float) fogDriftX,\n"
                        + "                            (float) fogDriftZ,"),
                "the fog wind drift must be accumulated Java-side per axis (pre-wrapped in whole "
                        + "4096-block periods) instead of the shader-side speed x clock product, "
                        + "which snapped the whole billow field at the wave clock's hourly wrap");
        assertTrue(composite.contains("fogMarchOn ? (float) CausticaConfig.Rt.Volume.FOG_MARCH_STEPS.value() : 0f"),
                "the volumeParams step lane must be zero while the march tier is off");
    }

    /**
     * The shadow floor no longer thins the medium. A visibility-scaled sigma made the fog's opacity a
     * per-pixel lottery a temporal reconstruct cannot accumulate, and it deleted the medium wherever the
     * sun is occluded (the air over the sea reading as a dark band while the sky-facing segment, which
     * traces no probes, kept its full density). The floor now rides the SUN TERMS of both tiers as
     * max(vis, floor): it holds a minimum share of direct light and can never amplify.
     */
    @Test
    void shadowFloorRidesTheSunTerms() throws IOException {
        String volume = Files.readString(VOLUME);
        String composite = Files.readString(COMPOSITE);
        assertTrue(volume.contains("public float fogShadowFloor()"),
                "the floor must be a function reading the pushed lane");
        assertTrue(volume.contains("clamp(worldPush.fog1.w, 0.0, 0.95)"),
                "fogShadowFloor must clamp the fog1.w lane into the safe range");
        // The medium is the unthinned profile in both tiers.
        assertTrue(volume.contains("float3 transmittance = airTransmittance(ro, rd, t);"),
                "the analytic tier must integrate the unthinned profile");
        assertTrue(volume.contains("float3 sigmaT = density * FOG_RAYLEIGH_COEFF + sigmaB;"),
                "the march tier must integrate the unthinned profile");
        // The floor rides the sun legs, as a max() with the measured visibility.
        assertTrue(volume.contains("float sunVis = max(vis, fogShadowFloor());"),
                "the marched tier's sun leg must carry the floor");
        assertTrue(volume.contains("float sunVis = max(vis[k], fogShadowFloor());"),
                "the analytic tier's stratum sun legs must carry the floor");
        // The floor must ride the fog1.w lane specifically — the v13 regression pushed it into the
        // look-alike volumeParams.w lane, where nothing read it and the slider did nothing.
        assertTrue(composite.contains("fogMarchOn ? CausticaConfig.Rt.Volume.FOG_NOISE.value() : 0f,\n"
                + "                            CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR.value())"),
                "Java must push the fog-shadow-floor config into fog1.w (after the noise lane)");
        assertTrue(!composite.contains("0f, CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR.value()"),
                "the floor must NOT ride volumeParams.w, where no shader reads it");

        // max() semantics: the floor holds a minimum share of direct light and can never amplify.
        for (float floor : new float[]{0.0f, 0.10f, 0.35f, 0.90f}) {
            for (float vis = 0.0f; vis <= 1.0f + 1.0e-6f; vis += 0.01f) {
                float factor = Math.max(vis, floor);
                assertTrue(factor <= 1.0f + 1.0e-6f, "floor must not exceed 1 at vis=" + vis);
                assertTrue(factor >= floor - 1.0e-6f, "floor must hold at vis=" + vis);
            }
        }
    }

    /**
     * The water fog's inertness: the shader's water in-scatter returns zero while any of its three
     * coefficients is zero, and Java only raises the flag when the switch is on and the fraction is
     * non-zero — a water-fog-off frame pushes an inert water1 block, exactly the volume block's
     * zero-when-off doctrine.
     */
    @Test
    void waterFogGating() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains(
                "if (sigmaT <= 0.0 || sigmaS <= 0.0 || t <= RAY_TMIN)"),
                "waterSegmentInScatter must return zero without a medium to integrate");
        String composite = Files.readString(COMPOSITE);
        assertTrue(composite.contains(
                "boolean waterFogOn = CausticaConfig.Rt.Composite.WATER_FOG.value()"),
                "the water fog flag must sit behind its own switch and a non-zero fraction");
        assertTrue(composite.contains(
                "new Float4(waterFogOn ? CausticaConfig.Rt.Composite.WATER_FOG_DENSITY.value() : 0f,"),
                "the water1.x lane must be zero while the water fog is off");
        assertTrue(composite.contains("| WORLD_FLAG_WATER_FOG | WORLD_FLAG_VOLUME_DEBUG"),
                "the water fog must route a frame to the raygen records that integrate it");
    }

    /**
     * Numeric contracts of the procedural cloud stack, ported from volume.slang: the coverage mask is
     * a bounded isosurface (never a smooth wash), the vertical profile closes the body at both its
     * flat base and its domed crown, the shadow transmittance sits in (0, 1] and stands down for a
     * horizon-grazing body, and the powder weight is monotone in the sample's own density. Every
     * consumer of the stack — the walk, the planar crossings, the NEE sun legs, the fog shafts — leans
     * on these properties without re-deriving them.
     */
    @Test
    void cloudStackInvariants() {
        final double bias = 0.10;
        final double fineWeight = 1.30;
        final double largeWeight = 0.60;
        final double softness = 0.20;
        for (double big = 0.0; big <= 1.0 + 1.0e-9; big += 0.25) {
            for (double fine = 0.0; fine <= 1.0 + 1.0e-9; fine += 0.25) {
                double mask = largeWeight * big + fineWeight * fine + bias;
                double previous = Double.NEGATIVE_INFINITY;
                for (double cov = 0.2; cov <= 2.0 + 1.0e-9; cov += 0.2) {
                    double q = Math.min(1.0, Math.max(0.0,
                            ((cov - mask) / Math.sqrt(Math.max(cov, 1.0e-3))) / softness));
                    double shape = q * q * (3.0 - 2.0 * q);
                    assertTrue(shape >= 0.0 && shape <= 1.0,
                            "the shape must be a bounded isosurface at cov=" + cov + " mask=" + mask);
                    if (mask >= cov) {
                        assertClose("a mask past the coverage clears the sky", 0.0, shape, 1.0e-9);
                    }
                    assertTrue(shape >= previous - 1.0e-9,
                            "raising coverage must never remove a body");
                    previous = shape;
                }
            }
        }

        // Java port of the cumulus vertical profile (shape = 1 inside the body).
        final double tallness = 232.0 * 0.56;
        final double baseRound = 26.0;
        final double topPower = 2.2;
        double previousInterior = -1.0;
        for (double above = 0.0; above <= tallness + 1.0e-9; above += tallness / 20.0) {
            double below = tallness - above;
            double bottom = 1.0 - Math.pow(1.0 - Math.min(Math.max(below / baseRound, 0.0), 1.0), 5.0);
            double top = 1.0 - Math.pow(1.0 - Math.min(Math.max(above / tallness, 0.0), 1.0), topPower);
            double interior = Math.max(1.0 - 1.0 + top * bottom, 0.0);
            assertTrue(interior >= 0.0 && interior <= 1.0,
                    "the profile must be a bounded vertical weight at above=" + above);
            assertClose("the crown must close on itself", 0.0, interiorAt(tallness, tallness, baseRound, topPower), 1.0e-9);
            assertClose("the flat base must close on itself", 0.0, interiorAt(0.0, tallness, baseRound, topPower), 1.0e-9);
            previousInterior = interior;
        }
        assertTrue(previousInterior >= 0.0, "the crown's profile must be bounded");
        assertTrue(interiorAt(tallness * 0.5, tallness, baseRound, topPower) > 0.5,
                "the interior of the deck must be solid");

        // Java port of cloudShadowTransmittance's slant-thickened column and its elevation gate.
        for (double density = 0.0; density <= 1.0 + 1.0e-9; density += 0.1) {
            for (double sunY = 0.01; sunY <= 1.0 + 1.0e-9; sunY += 0.1) {
                double denom = Math.max(sunY, 0.02);
                double tau = density * 0.5 * tallness / denom;
                double gate = smoothstep(0.0, 0.06, sunY);
                double t = 1.0 + (Math.exp(-tau) - 1.0) * gate;
                assertTrue(t >= 0.0 && t <= 1.0 + 1.0e-9,
                        "the shadow transmittance must be a transmittance at density=" + density);
                if (density == 0.0) {
                    assertClose("a clear sky casts no shadow", 1.0, t, 1.0e-9);
                }
            }
        }
        assertClose("the horizon gate must stand the shadow field down", 1.0,
                1.0 + (Math.exp(-1.0) - 1.0) * smoothstep(0.0, 0.06, 0.0), 1.0e-9);

        // Java port of cloudPowder: the thin rim keeps the strong forward lobe, the solid interior
        // does not, and the weight saturates rather than growing without bound.
        double previousPowder = -1.0;
        for (double shape = 0.0; shape <= 1.0 + 1.0e-9; shape += 0.05) {
            double powder = Math.min(Math.exp(-4.0 * Math.exp(-4.0 * shape)) * 3.5, 1.0);
            assertTrue(powder > 0.0 && powder <= 1.0,
                    "the powder weight must be bounded at shape=" + shape);
            assertTrue(powder >= previousPowder - 1.0e-9,
                    "a denser sample must not lose the powder weight");
            previousPowder = powder;
        }
        assertClose("the powder weight must saturate inside the body", 1.0, previousPowder, 1.0e-9);
    }

    /** cloudCumulusDensity's interior density at a height `above` the deck's top. */
    private static double interiorAt(double above, double tallness, double baseRound, double topPower) {
        double below = tallness - above;
        double bottom = 1.0 - Math.pow(1.0 - Math.min(Math.max(below / baseRound, 0.0), 1.0), 5.0);
        double top = 1.0 - Math.pow(1.0 - Math.min(Math.max(above / tallness, 0.0), 1.0), topPower);
        return Math.max(top * bottom, 0.0);
    }

    private static double smoothstep(double a, double b, double x) {
        double q = Math.min(1.0, Math.max(0.0, (x - a) / (b - a)));
        return q * q * (3.0 - 2.0 * q);
    }

    /**
     * The cloud stack's wiring, end to end: Java raises WORLD_FLAG_CLOUD behind the sky-light gate and
     * fills the four cloud lanes, the world pipeline binds the wrapping noise atlas, the shader builds
     * FOUR decks from one shape field, and the six public consumers keep the signatures the raygen,
     * the miss shader and the debug pass call them by. The fog sun legs and every NEE site still
     * multiply the
     * stack's shadow field, so a clouded sun dims identically everywhere.
     */
    @Test
    void cloudStackGatingAndConsumers() throws IOException {
        String volume = Files.readString(VOLUME);
        String skyMiss = Files.readString(Path.of("shaders/pipelines/world/sky.rmiss.slang"));
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        String composite = Files.readString(COMPOSITE);
        String bindings = Files.readString(Path.of("shaders/pipelines/world/bindings.slang"));
        String worldCommon = Files.readString(WORLD_COMMON);
        String pipeline = Files.readString(PIPELINE);

        // The host's flag assembly and the four pushed lanes.
        assertTrue(composite.contains(
                        "boolean cloudOn = CausticaConfig.Rt.Volume.CLOUD_ENABLED.value() && skyLit"),
                "the cloud flag must sit behind the sky-light gate");
        assertTrue(composite.contains("flags |= WORLD_FLAG_CLOUD")
                        && composite.contains("flags |= WORLD_FLAG_CLOUD_SHADOW"),
                "Java must raise both the visible-layer and the shadow-field flags");
        assertTrue(composite.contains(
                        "boolean cloudShadowActive = CausticaConfig.Rt.Volume.CLOUD_SHADOW.value()\n"
                                + "                    && cloudFieldUsable"
                                + " && (cloudVisible || (cameraLooksDown() && shadowReachOk));"),
                "the shadow field must follow its own switch");
        assertTrue(composite.contains("cloudOn = cloudVisible || cloudShadowActive;"),
                "turning shadows off must not take the visible deck with it");
        assertTrue(composite.contains(
                        "new Float4(cloudOn ? CausticaConfig.Rt.Volume.CLOUD_COVERAGE.value() : 0f,\n"
                                + "                            CausticaConfig.Rt.Volume.CLOUD_ALTITUDE"
                                + ".value() - terrain.blockY,"),
                "the cloud0 lanes must push the slider's coverage first and the rebased cumulus base second");
        String configSource = Files.readString(CONFIG);
        assertTrue(configSource.contains(
                        "bool(\"caustica.rt.volume.cloudShadow\", \"volume.cloud-shadow\", true)"),
                "cloud shadows must be a switch of their own, on by default");
        String settingsSource = Files.readString(SETTINGS);
        assertTrue(settingsSource.contains("\"volume.cloud-shadow\"")
                        && settingsSource.contains("CausticaConfig.Rt.Volume.CLOUD_SHADOW"),
                "the switch must be on the Sky page");
        assertTrue(composite.contains("cloudOn ? cloudCoverageBoost : 0f,")
                        && configSource.contains("public static float rainCoverage(float storm, float boost)"),
                "the weather's coverage lift must ride cloud1.y, not be folded into the slider's lane");
        assertTrue(composite.contains("cloudOn ? cloudFlowClock : 0f, 1f)")
                        && composite.contains("CLOUD_SIZE_REFERENCE"),
                "the size scale and the flow phase must ride cloudColor.x and cloudColor.z");
        // The three quality dials ride the block's spare lanes, each gated on the consumer that reads
        // it: the sun-leg count on the volumetric tier, the detail dial on the active field, the
        // shadow count on the shadow field. Zero is the shader's clamp-to-1 inert default, so an off
        // consumer pushes nothing and the default frame is the shipped single-tap, detail-on one.
        assertTrue(composite.contains("volumetricCloudOn\n"
                        + "                                    ? (float) CausticaConfig.Rt.Volume.CLOUD_LIGHT_STEPS.value() : 0f,")
                        && volume.contains("int steps = int(clamp(push.cloud1.x, 1.0, 8.0));"),
                "the sun-leg sample count must ride cloud1.x, zero unless the volumetric tier runs");
        assertTrue(composite.contains("cloudOn ? (CausticaConfig.Rt.Volume.CLOUD_DETAIL.value() ? 1f : 0f) : 0f,")
                        && volume.contains("bool detailOn = worldPush.cloudColor.y > 0.5;")
                        && volume.contains("float2 curl = cloudCurl(fineAddr / l.detailTile) * CLOUD_FINE_CURL;")
                        && volume.contains("if (!eroded || !detailOn) {"),
                "the detail dial must ride cloudColor.y and cut only the erosion pass: the cumulus shape "
                        + "field's curl stays unconditional, so the shadow the deck casts does not move "
                        + "under a close-view switch");
        assertTrue(composite.contains("cloudShadowActive\n"
                        + "                                    ? (float) CausticaConfig.Rt.Volume.CLOUD_SHADOW_STEPS.value() : 0f,")
                        && volume.contains("int steps = int(clamp(worldPush.volumeParams.z, 1.0, 4.0));")
                        && volume.contains("tau += (densitySum / float(steps)) * l.density * l.tallness / denom;"),
                "the shadow sample count must ride volumeParams.z and average the leg's profile, "
                        + "1 keeping the shipped mid-plane estimate's exact arithmetic");
        assertTrue(configSource.contains(
                        "clampedInt(\"caustica.rt.volume.cloudLightSteps\", \"volume.cloud-light-steps\", 1, 1, 8)")
                        && configSource.contains("bool(\"caustica.rt.volume.cloudDetail\", \"volume.cloud-detail\", true)")
                        && configSource.contains(
                        "clampedInt(\"caustica.rt.volume.cloudShadowSteps\", \"volume.cloud-shadow-steps\", 1, 1, 4)"),
                "the three quality dials must default to the shipped single-tap, detail-on frame");
        assertTrue(settingsSource.contains("\"volume.cloud-light-steps\"")
                        && settingsSource.contains("\"volume.cloud-detail\"")
                        && settingsSource.contains("\"volume.cloud-shadow-steps\""),
                "the three quality dials must sit on the Sky page");
        assertTrue(composite.contains(
                        "new Float4((float) terrain.blockX, (float) terrain.blockZ,\n"
                                + "                            (float) cloudDriftX, (float) cloudDriftZ),"),
                "the field must anchor on the EXACT terrain origin plus the Java-accumulated drift");
        assertTrue(!composite.contains("CLOUD_TYPE") && !composite.contains("CLOUD_RANDOM_STYLE")
                        && !composite.contains("cloudVolumes"),
                "the baked-shape slider, the random-style flag and the atlas must be gone");

        // The binding and the generated constant describe ONE wrapping 2D atlas.
        assertTrue(bindings.contains("[[vk::binding(19, 0)]] public Sampler2D cloudNoise;"),
                "the world set must bind one 2D noise atlas at slot 19");
        assertTrue(pipeline.contains("WORLD_CLOUD_NOISE")
                        && pipeline.contains("setCloudNoiseTexture"),
                "the pipeline must expose the noise atlas's binding under its new name");
        assertTrue(!worldCommon.contains("WORLD_FLAG_CLOUD_RANDOM")
                        && !worldCommon.contains("atlas's LAST shape index"),
                "the push block must not describe retired lanes or flags");

        // Four decks, one shape language, no instances and no baked volumes.
        assertTrue(volume.contains("public static const int CLOUD_LAYER_COUNT = 4;")
                        && volume.contains("public static const int CLOUD_CIRRUS = 3;"),
                "the stack must be four decks");
        assertTrue(volume.contains("CloudLayer cloudLayerOf(WorldPush push, int index) {")
                        && volume.contains("float coverage = clamp(push.cloud0.x + push.cloud1.y, 0.0, 2.5);")
                        && volume.contains("float sizeScale = clamp(push.cloudColor.x, 0.25, 4.0);"),
                "every deck must resolve from the same coverage, weather and size lanes");
        assertTrue(volume.contains("float cloudCoverageMask(float big, float fine, float cov, float patchiness) {")
                        && volume.contains("CLOUD_MASK_BIAS")
                        && volume.contains("CLOUD_SHAPE_SOFTNESS"),
                "every deck must clear the one biased, narrow isosurface");
        assertTrue(volume.contains("float cloudCumulusDensity(CloudLayer l, float3 p, float2 wind, float lod, bool eroded,")
                        && volume.contains("float cloudSheetDensity(CloudLayer l, float3 p, float2 wind, float lod) {")
                        && volume.contains("float2 cloudCurl(float2 p) {"),
                "the two cumulus decks and the two sheets must be implemented, cirrus with a curl warp");
        assertTrue(volume.contains("float2 uv = (p.xz + p.y * float2(0.0, 193.0)) * (1.0 / 512.0);"),
                "the erosion field's vertical offset must be continuous, not a per-cell row jump");
        assertFalseStaleCloud(volume);

        // The six public consumers keep the call signatures every stage was written against.
        assertTrue(volume.contains("public float cloudShadowTransmittance(float3 pos, float3 sunDir) {")
                        && volume.contains("public float cloudCoverageWorld(float2 rebasedXZ) {")
                        && volume.contains("public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,")
                        && volume.contains("public float3 cloudSegmentOutScatter(WorldPush push, float3 ro, float3 rd, SkyState state,")
                        && volume.contains("public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,")
                        && volume.contains("public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {")
                        && volume.contains("public float cloudBoxMid(WorldPush push) {"),
                "the stack's public surface must keep the signatures the other stages call");
        assertTrue(volume.contains("for (int index = CLOUD_LOW; index <= CLOUD_ALTO; index++)"),
                "the ground shadow field must price the cumulus and altostratus decks, not the cirrus");
        assertTrue(volume.contains("float shadowGate = smoothstep(0.0, 0.06, sunDir.y);"),
                "the shadow field must fade out with the sun's elevation rather than switching off");
        assertTrue(volume.contains("if ((worldPush.flags & WORLD_FLAG_CLOUD_SHADOW) == 0u || worldPush.cloud0.x <= 0.0) {"),
                "the shadow field must return exactly 1 while it is off");

        // The consumers outside volume.slang are unchanged.
        assertTrue(skyMiss.contains("color = cloudL + cloudT * color;"),
                "the sky miss shader must composite the stack over the dome");
        assertTrue(indirect.contains("vis *= cloudShadowTransmittance(p, lightDir);")
                        && indirect.contains("vis *= cloudShadowTransmittance(shadowOrigin, lightDir);"),
                "both celestial NEE sites must pick up the stack's shadow");
        assertTrue(volume.contains("* cloudShadowTransmittance(ro, celestialLight.dir));")
                        && volume.contains("cloudShadow = cloudShadowTransmittance(ro + rd * midDepth, celestialLight.dir);")
                        && volume.contains("cloudShadowTransmittance(ro + rd * (0.5 * (t0 + t1)), sunDir)")
                        && volume.contains("shaftDepthAtWeight(sigma, reach, SHAFT_STRATA / 2)")
                        && volume.contains("* (vis * legT * cloudShadowTransmittance(p, celestialLight.dir));"),
                "every sun leg must read the same shadow field, once per segment rather than per stratum");
        assertTrue(primary.contains("cloudDensity = cloudCoverageWorld(ro.xz + rd.xz * tPlane);")
                        && primary.contains("cloudShadow = cloudShadowTransmittance(ro, celestialLight.dir);"),
                "the volume debug image must fill both cloud lanes");
    }

    /** No identifier from the retired baked-volume field may survive in the new implementation. */
    private static void assertFalseStaleCloud(String volume) {
        for (String stale : new String[]{"cloudInstanceDensity(", "cloudAtlasSample(", "cloudVolumes",
                "CLOUD_VOLUME_RES", "cloudGridHash(", "cloudDistrictShape(", "cloudSelfShadow(",
                "cloudLightTau(", "CLOUD_COLUMN_DEPTH", "CLOUD_BOX_SPAN"}) {
            assertTrue(!volume.contains(stale),
                    "the retired baked-volume field must not leave " + stale + " behind");
        }
    }

    /**
     * The volumetric push block is a POSITIONAL record: world_common.slang's field order generates
     * WorldPushData's constructor, and RtComposite's call site passes bare Float4 arguments into it.
     * A slip in that order is silent — it moves 16 bytes of lanes — and it has bitten twice now
     * (a floor pushed into a look-alike lane, then cloud2 and cloudColor swapped, which froze the
     * clouds and fed a growing drift into the sheet's ambient weight). This pins the order on both
     * sides of the boundary so the next added lane has to keep them in step.
     */
    @Test
    void volumeLaneOrderMatchesGeneratedRecord() throws IOException {
        String shader = Files.readString(WORLD_COMMON);
        String composite = Files.readString(COMPOSITE);
        Path generated = Path.of(
                "build/generated/sources/shaderRecords/dev/comfyfluffy/caustica/rt/gen/WorldPushData.java");
        assertTrue(Files.exists(generated),
                "the generated WorldPushData must exist; run generateShaderRecords first");
        String record = Files.readString(generated);

        int cloud2Field = shader.indexOf("public float4   cloud2;");
        int cloudColorField = shader.indexOf("public float4   cloudColor;");
        assertTrue(cloud2Field > 0 && cloudColorField > 0 && cloud2Field < cloudColorField,
                "world_common.slang must declare cloud2 before cloudColor");

        int cloud2Arg = record.indexOf("Float4 cloud2,");
        int cloudColorArg = record.indexOf("Float4 cloudColor,");
        assertTrue(cloud2Arg > 0 && cloudColorArg > 0 && cloud2Arg < cloudColorArg,
                "the generated record must keep cloud2 before cloudColor");

        int cloud2Call = composite.indexOf("new Float4((float) terrain.blockX, (float) terrain.blockZ,");
        int cloudColorCall = composite.indexOf("new Float4(Math.clamp(CausticaConfig.Rt.Volume.CLOUD_SIZE.value() / CLOUD_SIZE_REFERENCE,");
        assertTrue(cloud2Call > 0 && cloudColorCall > 0,
                "both cloud lanes must appear at the call site");
        assertTrue(cloud2Call < cloudColorCall,
                "the call site must pass the cloud anchor/drift block BEFORE the cloud colour lane, "
                        + "matching the generated record order");

        // The border haze's lane, added between the march parameters and the water-volume block: the same
        // three-way check, because the argument that feeds it is a bare Float4 like every other one here.
        int volumeParamsField = shader.indexOf("public float4   volumeParams;");
        int hazeField = shader.indexOf("public float4   boundaryFog;");
        assertTrue(volumeParamsField > 0 && hazeField > volumeParamsField,
                "world_common.slang must declare boundaryFog after volumeParams");
        int volumeParamsArg = record.indexOf("Float4 volumeParams,");
        int hazeArg = record.indexOf("Float4 boundaryFog,");
        assertTrue(volumeParamsArg > 0 && hazeArg > volumeParamsArg,
                "the generated record must keep boundaryFog after volumeParams");
        int hazeCall = composite.indexOf("new Float4(boundarySigma, 0f, 0f, 0f)");
        int waterVolumeCall = composite.indexOf(
                "new Float4(waterFogOn ? CausticaConfig.Rt.Composite.WATER_FOG_DENSITY.value() : 0f,");
        assertTrue(hazeCall > 0 && waterVolumeCall > hazeCall,
                "the call site must pass the haze lane before the water-volume block, matching the "
                        + "generated record order");
        // The edge fade's lane sits right behind the border's, and both before the water-volume block.
        int fadeField = shader.indexOf("public float4   edgeFade;");
        assertTrue(fadeField > hazeField,
                "world_common.slang must declare edgeFade after boundaryFog");
        int fadeArg = record.indexOf("Float4 edgeFade,");
        assertTrue(fadeArg > hazeArg, "the generated record must keep edgeFade after boundaryFog");
        int fadeCall = composite.indexOf("new Float4(edgeFadeOn ?");
        assertTrue(fadeCall > hazeCall && fadeCall < waterVolumeCall,
                "the call site must pass the fade lane between the border's and the water-volume block");
    }

    /**
     * The beacon beam pays a dedicated fraction of the unlit emission baseline. It shares that baseline
     * (anchored on a glowstone face's exitance) with the lightning bolt and the end portal, which are
     * meant to read as bright — so the fix has to be the beam's own scale, keyed on its fragment shader,
     * and the collector has to actually apply it.
     */
    @Test
    void beaconBeamEmissionIsScaled() throws IOException {
        String unlit = Files.readString(Path.of(
                "src/main/java/dev/comfyfluffy/caustica/rt/material/RtUnlitPipelines.java"));
        String collector = Files.readString(Path.of(
                "src/main/java/dev/comfyfluffy/caustica/rt/entity/RtEntityCollector.java"));
        assertTrue(unlit.contains("BEACON_BEAM_FRAGMENT_SHADER = \"core/rendertype_beacon_beam\""),
                "the beam classifier must key on the beacon-beam fragment shader");
        assertTrue(unlit.contains("public static boolean beaconBeam(RenderPipeline pipeline)"),
                "RtUnlitPipelines must expose the beam classifier");
        assertTrue(collector.contains("BEACON_BEAM_EMISSION_SCALE"),
                "the collector must declare the beam's emission scale");
        assertTrue(collector.contains(
                        "* (RtUnlitPipelines.beaconBeam(submission) ? BEACON_BEAM_EMISSION_SCALE : 1f)"),
                "the beam's scale must multiply the unlit baseline at the emission assignment");
    }

    /**
     * The volumetric tier walks every deck the ray meets, front to back, rather than pricing one slab.
     * What has to hold is the SAME field as the planar tier (so the two degrade into each other), the
     * fade dissolving the layer's OPACITY and not just its glow (the v17 black-band lesson), the
     * cheap column gate ahead of the expensive fetches, the light leg reading the same shape, and the
     * walk being reserved for the primary ray while continuation rays get the mean extinction.
     */
    @Test
    void volumetricCloudTierInvariants() throws IOException {
        String volume = Files.readString(VOLUME);
        String skyMiss = Files.readString(Path.of("shaders/pipelines/world/sky.rmiss.slang"));

        assertTrue(volume.contains("bool cloudLayerSpan(CloudLayer l, float3 ro, float3 rd, float tMax, out float t0, out float t1) {")
                        && volume.contains("if (!cloudLayerSpan(l, ro, rd, fadeDist, t0, t1)) {"),
                "the walk must bracket each deck's crossing before spending a step on it");
        assertTrue(volume.contains("bool upward = rd.y >= 0.0;")
                        && volume.contains("int index = upward ? n : CLOUD_LAYER_COUNT - 1 - n;"),
                "the stack must composite front to back in whichever order the ray meets the decks");
        assertTrue(volume.contains("float ideal = ceil(spans[n] / CLOUD_MARCH_STEP);")
                        && volume.contains("float share = budget * spans[n] / max(spanSum, 1.0e-3);")
                        && volume.contains("steps = clamp(min(ideal, share), 4.0, budget);"),
                "the two cumulus decks must share the step budget in proportion to the span each catches; "
                        + "a five-block sheet needs a crossing");
        assertTrue(volume.contains("void cloudMarchLayer(WorldPush push, int index, float3 ro, float3 rd, float t0, float t1,")
                        && volume.contains("float3 cloudPlanarStack(WorldPush push, float3 ro, float3 rd, SkyState state, float tMax,"),
                "the walk and the crossings must both be served by the stack's own helpers");
        assertTrue(volume.contains("if (cloudMaskLarge(worldPush.cloud1.z) * big + CLOUD_MASK_BIAS >= l.coverage) {")
                        && volume.contains("float cloudCumulusMaskBig(CloudLayer l, float3 p, float2 wind, float lod, out float2 warp) {")
                        && volume.contains("cloudCumulusDensity(l, p, wind, lod, true, warp, big)")
                        && volume.contains("float cloudMaskLarge(float patchiness) {")
                        && volume.contains("float cloudMaskFine(float patchiness) {"),
                "one cheap column fetch must reject empty sky before the profile and the light leg, and "
                        + "the fetch the rejection made must be the one the density reads");
        assertTrue(volume.contains("float absorbed = (1.0 - exp(-tauCell)) * fade;")
                        && volume.contains("L += T * absorbed * radiance;")
                        && volume.contains("T *= 1.0 - absorbed;"),
                "dissipation must dissolve the layer's opacity, radiance and transmittance together");
        assertTrue(volume.contains("float cloudSunTau(WorldPush push, int index, CloudLayer l, float3 p, float2 wind, float3 ro,")
                        && volume.contains("int steps = int(clamp(push.cloud1.x, 1.0, 8.0));")
                        && volume.contains("float tap = cloudLayerDensity(push, index, lp, wind, cloudLod(length(lp - ro), l.detailTile),\n"
                        + "                false) * l.density * dt;")
                        && volume.contains("float3 lp = p + sunDir * ((float(i) + 0.5) * dt);"),
                "the light leg must read the same shape field the view ray walked, clipped to the slab "
                        + "and sampled at each sub-segment's midpoint — the Cloud Light Steps dial, "
                        + "1 being the single midpoint tap the tier shipped with");
        assertTrue(volume.contains("float cloudPhase4(float cosTheta, float powder) {")
                        && volume.contains("float powder = cloudPowder(shape);")
                        && volume.contains("float sunMask = 1.0 - exp(-tauSun / CLOUD_LIGHT_DEPTH_SCALE);")
                        && volume.contains("float beer = 6.28 * exp(-5.0 * sunMask);")
                        && volume.contains("CLOUD_LIGHT_DEPTH_SCALE"),
                "the four-lobe powder phase and the normalised multi-scatter gain must light every sample");
        assertTrue(volume.contains("if (push.volumeParams.w < 1.0) {")
                        && volume.contains("return cloudPlanarStack(push, ro, rd, state, 1.0e12, cloudT);"),
                "an off tier must fall through to the planar crossings");
        assertTrue(volume.contains("float fadeDist = max(push.cloud1.w, 64.0) * (l.kind == 0 ? 1.0 : CLOUD_SHEET_FADE_MULT);"),
                "the high sheets must ride a longer fade than the cumulus decks");

        // Only the primary ray pays for a walk; reflections and bounces get the mean extinction.
        assertTrue(skyMiss.contains("cloudL = volumetricCloudInScatter(worldPush, WorldRayOrigin(), dir, state,")
                        && skyMiss.contains("cloudT = volumetricCloudTransmittanceFast(worldPush, WorldRayOrigin(), dir);"),
                "only the primary ray may pay for the walk; continuation rays get the mean extinction");
        assertTrue(volume.contains("public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {"),
                "the extinction-only path must exist for continuation rays");
        assertFalseStaleCloud(volume);
    }

    /**
     * The volumetric quality preset. It is the one setting whose value means "write these other ones",
     * so what has to hold is that the write is complete and monotone — and, above all, that tier 0 turns
     * every last volumetric term off. That tier is the whole system's regression claim: with it selected
     * the renderer has to produce the frame it produced before any of this existed, which it cannot do
     * while one switch is still standing.
     */
    @Test
    void volumeQualityPresetWritesEveryVolumetricSwitch() throws IOException {
        String config = Files.readString(CONFIG);
        String settings = Files.readString(SETTINGS);

        // The tier table, Java ported: (cloud, cloudVolumetric, fog, fogShadow, fogMarch, waterFog) for
        // tiers 0..4.
        boolean[][] on = new boolean[5][];
        int[] steps = new int[5];
        for (int t = 0; t <= 4; t++) {
            on[t] = new boolean[]{t >= 1, t >= 2, t >= 3, t >= 2, t >= 4, t >= 1};
            steps[t] = t == 2 ? 12 : (t == 4 ? 40 : 24);
        }
        // Off clears everything: clouds, the volumetric tier, the fog layer and its shadow, and the water
        // body's in-scatter.
        for (boolean b : on[0]) {
            assertTrue(!b, "tier 0 must clear every volumetric switch");
        }
        // Monotone: a higher tier never takes an effect away, and the last one has everything on.
        for (int t = 1; t <= 4; t++) {
            for (int k = 0; k < on[t].length; k++) {
                assertTrue(!on[t - 1][k] || on[t][k],
                        "tier " + t + " must not take switch " + k + " away from tier " + (t - 1));
            }
        }
        for (boolean b : on[4]) {
            assertTrue(b, "the top tier must have every switch on");
        }
        assertTrue(steps[2] < steps[3] && steps[3] < steps[4],
                "the march budgets must climb with the tier");
        assertTrue(on[0][5] == false && on[1][5],
                "the water body's in-scatter arrives with the first tier above Off");

        // The table is data in the config, and the write is a total function of it rather than a list of
        // spells at each call site.
        assertTrue(config.contains("public record QualityTier(boolean cloud, boolean cloudVolumetric, int cloudSteps,")
                        && config.contains("public static QualityTier qualityTier(int tier) {")
                        && config.contains("public static void applyQuality(int tier) {"),
                "the tier table must be data, with the write derived from it");
        for (String member : new String[]{"CLOUD_ENABLED.set(", "CLOUD_VOLUMETRIC.set(", "CLOUD_STEPS.set(",
                "FOG_ENABLED.set(", "FOG_SHADOW.set(", "FOG_MARCH.set(", "FOG_MARCH_STEPS.set(",
                "CausticaConfig.Rt.Composite.WATER_FOG.set("}) {
            assertTrue(config.contains(member), "applyQuality must write " + member);
        }
        // The choice list rejects a hand-edited config rather than letting an out-of-range tier through.
        assertTrue(config.contains("intChoice(\"caustica.rt.volume.quality\", \"volume.quality\", 1, QUALITY_TIERS)"),
                "the tier must be a choice list, so a hand-edited value cannot land outside the table");

        // The side effect belongs to the option, not to whichever screen is open: both hosts funnel every
        // committed write through afterChange, which is where the preset applies itself.
        assertTrue(settings.contains("CausticaConfig.Rt.Volume.QUALITY.tomlPath().equals(option.path())")
                        && settings.contains("CausticaConfig.Rt.Volume.applyQuality(option.integer().get());"),
                "selecting the preset must write the other settings through the shared afterChange funnel");
        // Declared as a count whose caption is a readback of the switches, so the name is what the page
        // currently spells rather than what was last clicked.
        assertTrue(settings.contains("countWithReadout(out, Page.VOLUME, \"preset\", \"volume.quality\",")
                        && settings.contains("CausticaConfig.Rt.Volume.currentTier()"),
                "the preset row must read its label back from the individual dials");
        assertTrue(config.contains("public static int currentTier() {")
                        && config.contains("return -1;"),
                "a mix that spells no tier must be reportable as Custom");
    }

    /**
     * Weather and field evolution on a procedural stack: rain closes the sky by ADDING to the shape
     * threshold (the bodies grow into each other), the density lift stays Java-side so every consumer
     * reads one number, and the wind, the flow phase and the size scale ride three independent lanes.
     * The numeric ports pin the monotone/identity behaviour; the source assertions pin the wiring on
     * both sides of the push.
     */
    @Test
    void cloudWeatherAndFlow() throws IOException {
        String config = Files.readString(CONFIG);
        String composite = Files.readString(COMPOSITE);
        String volume = Files.readString(VOLUME);

        // Java ports of stormLevel, rainCoverage and rainDensity.
        float[] boosts = {0.0f, 0.35f, 0.6f, 1.0f};
        for (float boost : boosts) {
            for (int rainPct = 0; rainPct <= 10; rainPct++) {
                for (int thunderPct = 0; thunderPct <= 10; thunderPct++) {
                    float rain = rainPct / 10f;
                    float thunder = thunderPct / 10f;
                    float storm = Math.clamp(rain + 0.35f * thunder, 0.0f, 1.0f);
                    float coverageLift = 0.4f * Math.clamp(boost * storm, 0.0f, 1.0f);
                    float density = 1.0f * (1.0f + 1.6f * boost * storm);
                    assertTrue(coverageLift >= 0.0f && coverageLift <= 0.4f,
                            "the coverage lift must be bounded");
                    assertTrue(density >= 1.0f, "weather must never thin the deck");
                    if (rain == 0f && thunder == 0f) {
                        assertClose("no weather must leave the coverage alone", 0.0f, coverageLift, 1.0e-6f);
                        assertClose("no weather must leave the density alone", 1.0f, density, 1.0e-6f);
                    }
                    float stormNoThunder = Math.clamp(rain, 0.0f, 1.0f);
                    assertTrue(coverageLift >= 0.4f * Math.clamp(boost * stormNoThunder, 0.0f, 1.0f) - 1.0e-6f,
                            "thunder must not reduce the coverage lift");
                }
            }
        }
        assertClose("boost 0 must hold the coverage to the slider", 0.0f,
                0.4f * Math.clamp(0.0f * 1.0f, 0.0f, 1.0f), 1.0e-6f);
        assertClose("a full storm at full boost multiplies the density by 2.6", 2.6f,
                1.0f * (1.0f + 1.6f * 1.0f * 1.0f), 1.0e-6f);

        assertTrue(config.contains("public static float stormLevel(float rain, float thunder)")
                        && config.contains("public static float rainCoverage(float storm, float boost)")
                        && config.contains("public static float rainDensity(float density, float storm, float boost)"),
                "the weather formulas must be pure functions in the config");
        assertTrue(composite.contains("CausticaConfig.Rt.Volume.rainCoverage(storm, cloudRainBoost)")
                        && composite.contains("CausticaConfig.Rt.Volume.rainDensity(")
                        && composite.contains("CausticaConfig.Rt.Volume.stormLevel(rainLevel, thunderLevel)"),
                "the push must carry the coverage lift and the lifted density");
        assertTrue(composite.contains("new Float4(cloudOn ? CausticaConfig.Rt.Volume.CLOUD_COVERAGE.value() : 0f,")
                        && volume.contains("float coverage = clamp(push.cloud0.x + push.cloud1.y, 0.0, 2.5);"),
                "the slider must stay in cloud0.x and the shader must add the weather's share once");
        assertTrue(!config.contains("rainSwelling") && !volume.contains("swellFull"),
                "the old footprint swell must be gone with the instance grid");

        assertTrue(volume.contains("float2 flow = float2(worldPush.cloudColor.z * 1.7, -worldPush.cloudColor.z * 1.1);")
                        && volume.contains("float2 fineWind = flow - wind * 0.6;"),
                "the flow phase must scroll the fine detail, never the body's position");
        assertTrue(composite.contains("cloudFlowClock = (cloudFlowClock + waveClockStep")
                        && composite.contains("cloudOn ? cloudFlowClock : 0f, 1f),"),
                "the flow phase must accumulate on the clamped wave clock and ride cloudColor.z");
        assertTrue(volume.contains("float sizeScale = clamp(push.cloudColor.x, 0.25, 4.0);")
                        && volume.contains("CLOUD_LOW_FIELD_TILE * sizeScale"),
                "cloudColor.x must scale every deck's tiles, and nothing else");
        assertTrue(volume.contains("return (push.cloud2.xy + push.cloud2.zw) * l.windScale;"),
                "the wind must be one world-space drift, sheared per deck");
        assertTrue(!composite.contains("CLOUD_TYPE") && !composite.contains("CLOUD_RANDOM_STYLE")
                        && !composite.contains("cloudVolumes.shapeCount()")
                        && !volume.contains("CLOUD_VOLUME_RES"),
                "the baked-shape slider, the random-style flag and the atlas must have no readers");
    }

    // ---- Sun shafts (air and water) -------------------------------------------------------------------

    /** Java port of volume.slang's fogTransmittance over the same closed form. */
    private static float fogTransmittance(float sigma0, float baseY, float heightH, float y0,
                                          float dirY, float t) {
        return (float) Math.exp(-fogOpticalDepth(sigma0, baseY, heightH, y0, dirY, t));
    }

    /** Java port of shaftDepthAtWeight: the depth where the cumulative weight reaches k/strata. */
    private static double shaftDepthAtWeight(double sigma, double reach, int k, int strata) {
        if (k <= 0) {
            return 0.0;
        }
        if (k >= strata) {
            return reach;
        }
        double total = 1.0 - Math.exp(-sigma * reach);
        double weight = (double) k * total / strata;
        return -Math.log(Math.max(1.0 - weight, 1.0e-6)) / sigma;
    }

    /**
     * The strata are one integral split into equal weights, so their boundaries are the inverse of the
     * medium's transmittance. Two consequences the estimator leans on: the depths stay monotone and
     * pinned to [0, reach] whatever the density, and every stratum carries exactly 1/strata of the
     * total. The dense case is why the spacing is optical rather than geometric: at water's extinction
     * even spacing puts seven probes where the light is already gone.
     */
    @Test
    void shaftStrataSplitTheStretchIntoEqualWeights() {
        for (double sigma : new double[]{0.0005, 0.01, 0.05, 0.3, 1.0}) {
            for (double reach : new double[]{16.0, 64.0, 192.0, 512.0}) {
                double total = 1.0 - Math.exp(-sigma * reach);
                assertEquals(0.0, shaftDepthAtWeight(sigma, reach, 0, 8), 1.0e-9);
                assertEquals(reach, shaftDepthAtWeight(sigma, reach, 8, 8), 1.0e-9);
                double previous = 0.0;
                for (int k = 0; k < 8; k++) {
                    double a = shaftDepthAtWeight(sigma, reach, k, 8);
                    double b = shaftDepthAtWeight(sigma, reach, k + 1, 8);
                    assertTrue(b > a, "stratum " + k + " must have extent, got [" + a + ", " + b + "]");
                    assertEquals(total / 8.0, Math.exp(-sigma * a) - Math.exp(-sigma * b), 1.0e-6,
                            "sigma=" + sigma + " reach=" + reach + " stratum=" + k);
                    previous = b;
                }
                assertEquals(reach, previous, 1.0e-9);
            }
        }
        // A murky column's first stratum sits inside its own e-folding, not at an eighth of the reach.
        assertTrue(shaftDepthAtWeight(0.3, 64.0, 1, 8) < 1.0,
                "a dense medium's first stratum must land within its e-folding");
    }

    /**
     * A lit beam is bounded by the medium's own scatter weight: with every probe seeing the sun the
     * term is phase * illuminance * strength * (1 - exp(-sigma*reach)), which rises monotonically with
     * the reach and saturates instead of growing without bound. The shipped defaults must land in the
     * visible-but-unsaturated band, or the reach control would read as a wall.
     */
    @Test
    void shaftTermIsBoundedMonotoneAndSaturating() {
        // The shipped defaults (volume.shaft-density / volume.shaft-distance).
        double sigma = 0.010;
        double previous = -1.0;
        for (double reach = 1.0; reach <= 2048.0; reach *= 2.0) {
            double weight = 1.0 - Math.exp(-sigma * reach);
            assertTrue(weight > previous, "the scatter weight must rise with the reach at " + reach);
            assertTrue(weight < 1.0 + 1.0e-6, "the scatter weight must stay below 1 at " + reach);
            previous = weight;
        }
        // Near foliage — a canopy a few blocks overhead — has to carry enough of the medium to read at
        // all; that was the whole defect of the first set of defaults (0.0025 put a 30-block beam at
        // 7%, which vanished against shaded grass).
        double nearBeam = 1.0 - Math.exp(-sigma * 30.0);
        assertTrue(nearBeam > 0.20, "a 30-block beam must carry real weight, got " + nearBeam);
        // And the cap must not sit in the saturated tail, or the reach slider would do nothing visible.
        double shipped = 1.0 - Math.exp(-sigma * 192.0);
        assertTrue(shipped > 0.70 && shipped < 0.92,
                "the default density/reach must stay short of saturation, got " + shipped);
    }

/**
     * The water column carries two terms, and they stay separable: the body's own in-scatter (its glow)
     * and the structure-gated shaft layer (its beams). The in-scatter is a closed form over a constant
     * medium, so it needs no march; the shafts run on a medium of their own, so a player can want beams
     * without the milky water. Losing either one is what this pins.
     */
    @Test
    void waterColumnCarriesInScatterAndShafts() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains("public float3 waterSegmentInScatter("),
                "the body's own in-scatter must be integrated");
        assertTrue(volume.contains("public static const float WATER_AMBIENT_SURFACE"),
                "the surface's transmission for downward sky light is the ambient term's entry factor");
        assertTrue(volume.contains("sigmaS / sigmaT"),
                "the sun term must scale by the scattering fraction, so it cannot invent energy");
        assertTrue(volume.contains("float sigmaS = (medium.scattering.r + medium.scattering.g"),
                "the coefficients come from the medium, not from a second derivation");
        // The lobe is capped and the gate keeps a floor: without the cap a grazing look at the surface
        // goes white, without the floor the gate's product is a hairline band.
        assertTrue(volume.contains("phase = min(phase, worldPush.waterShaft.y);"),
                "the underwater lobe must be capped by the pushed ceiling");
        assertTrue(volume.contains("structure = lerp(WATER_SHAFT_STRUCTURE_FLOOR, 1.0, structure);"),
                "the underwater gate must keep a floor");
        assertTrue(volume.contains(
                        "float fogT = underwater ? 1.0 : luminance(airTransmittance(ro, rd, reach));"),
                "the air fog must not attenuate the underwater term, and on the air side the veil that "
                        + "swallows a beam is the whole medium - height profile and border haze together");
        assertTrue(volume.contains("vis *= exp(-sigma * probe.waterHitT);"),
                "every underwater probe must carry its own Beer-Lambert up to the surface");
        assertTrue(volume.contains("caustic = waterWaves"),
                "the surface focus must ride the wave field while it is on");
        // Both call sites run the two terms on the water medium: the shared prefix on the primary pass,
        // the first-segment prefix on the indirect one.
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        for (String source : new String[]{indirect, primary}) {
            assertTrue(source.contains("medium.current.water"),
                    "the water terms must gate on the water medium");
            assertTrue(source.contains("waterSegmentInScatter(ro, rd,"),
                    "the body's in-scatter must be called on the water segment");
            assertTrue(source.contains("worldPush.waterShaft.x"),
                    "the shaft layer must integrate the dedicated water medium");
            assertTrue(source.contains("if (waterFogOn) {"),
                    "the in-scatter must gate on its own flag");
        }
        String config = Files.readString(CONFIG);
        assertTrue(!config.contains("WATER_TURBIDITY"),
                "the retired turbidity setting must not come back");
        assertTrue(config.contains("clampedFloat(\"caustica.rt.waterFogDensity\", "
                        + "\"composite.water-fog-density\","),
                "the in-scatter's scattering fraction needs its own setting");
        assertTrue(config.contains("clampedFloat(\"caustica.rt.volume.waterShaftDensity\", "
                        + "\"volume.water-shaft-density\","),
                "the dedicated water medium needs its own density setting");
        assertTrue(config.contains("clampedFloat(\"caustica.rt.volume.waterShaftCeiling\", "
                        + "\"volume.water-shaft-ceiling\","),
                "the lobe ceiling needs its own setting");
    }

    /**
     * The gating that keeps a shaft-off frame identical to one without the feature: the integrator
     * early-outs on a zero density and a zero strength (both pushed as zero while the switch is off),
     * Java pushes those zeros, and both call sites sit behind the flag. Also the independence the
     * feature is for: nothing inside the shaft block reads a fog switch.
     */
    @Test
    void shaftGatingAndIndependence() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains("if (sigma <= 0.0 || strength <= 0.0 || t <= RAY_TMIN)"),
                "the shaft integrator must early-out on zero density and zero strength");
        assertTrue(volume.contains("float reach = min(t, max(worldPush.shaft.z, 1.0));"),
                "the shaft reach must clamp the segment to the pushed distance");
        assertTrue(volume.contains("float shaftWeight = total * fogT;"),
                "the shaft weight must be the stretch's total in-scatter times the fog's transmittance");
        assertTrue(volume.contains("if (shaftWeight <= 1.0e-4)"),
                "a hit close enough that the stretch holds no air must skip the probes entirely");
        // The structure gate is what separates a Tyndall beam from a veil: a path that never leaves the
        // light scatters uniformly along its whole length, which reads as mist over every sunlit wall.
        // It must stay LINEAR in the lit fraction: a smoothstep multiplies the probe quantisation by
        // its slope, and at a forward-scattering phase that renders as contour arcs across the wall.
        assertTrue(volume.contains("float structure = clamp(1.0 - lit, 0.0, 1.0);"),
                "the shaft term must fade out on fully-lit paths, linearly in the lit fraction");
        // ... and it must not ride the albedo guide: a structured volumetric term in the guide is
        // material as far as the reconstruct is concerned, so its grain survives instead of averaging.
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        assertTrue(!primary.contains("shaftOpacity"),
                "the shaft term must not be blended into the RR albedo guide");
        assertTrue(volume.contains("lit += dot(vis, float3(0.3333333, 0.3333333, 0.3333333));"),
                "the gate must read the grey lit fraction of the same probes");
        assertTrue(volume.contains("strength * fogT * structure"),
                "the gate must multiply the finished in-scatter");
        assertTrue(volume.contains("public static const int SHAFT_STRATA = 8;"),
                "the stratum count must stay high enough for the gate to place a shadow boundary");
        assertTrue(!volume.substring(volume.indexOf("public float3 shaftInScatter"))
                        .contains("WORLD_FLAG_FOG"),
                "the shaft integrator must not depend on a fog switch");
        String composite = Files.readString(COMPOSITE);
        assertTrue(composite.contains(
                        "boolean shaftOn = CausticaConfig.Rt.Volume.SHAFT_ENABLED.value() && skyLit"),
                "the shaft flag must sit behind its own switch and the sky-light gate");
        assertTrue(composite.contains(
                        "new Float4(shaftOn ? CausticaConfig.Rt.Volume.SHAFT_DENSITY.value() : 0f,\n"
                                + "                            CausticaConfig.Rt.Volume.SHAFT_ANISOTROPY.value(),\n"
                                + "                            CausticaConfig.Rt.Volume.SHAFT_DISTANCE.value(),\n"
                                + "                            shaftOn ? CausticaConfig.Rt.Volume.SHAFT_INTENSITY.value() : 0f),"),
                "Java must push the shaft block with density and strength zeroed while it is off");
        assertTrue(composite.contains(
                        "new Float4(shaftOn ? CausticaConfig.Rt.Volume.WATER_SHAFT_DENSITY.value() : 0f,"),
                "Java must push the underwater medium's density, zeroed while the feature is off");
        String config = Files.readString(CONFIG);
        assertTrue(config.contains("bool(\"caustica.rt.volume.shafts\", \"volume.shafts\", false)"),
                "the shaft master switch ships off: its per-pixel probes are a frame budget a fresh "
                        + "install keeps for the rest");
        String settings = Files.readString(SETTINGS);
        assertTrue(settings.contains("\"volume.shafts\", RT + \"shafts\""),
                "the shaft switch must be reachable from the settings surface");
    }
}

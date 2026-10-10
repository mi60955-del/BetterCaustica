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
    private static final Path COMPOSITE =
            Path.of("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");
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

    /** Port of medium.slang's waterExtinction. */
    private static float waterExtinction(float tint, float opacity) {
        float floorR = 0.015f;
        float floorG = 0.010f;
        float floorB = 0.008f;
        float density = 0.1f;
        float baseR = floorR + density * (1.0f - Math.clamp(tint, 0.0f, 1.0f));
        float baseG = floorG + density * (1.0f - Math.clamp(tint, 0.0f, 1.0f));
        float baseB = floorB + density * (1.0f - Math.clamp(tint, 0.0f, 1.0f));
        float mean = (baseR + baseG + baseB) / 3.0f;
        float chromaScale = Math.min(opacity, 1.0f);
        return mean * opacity; // the port tracks the achromatic mean; the chroma split cancels here
    }

    /**
     * Phase 3's degradation and energy lines: with waterOpacity 0 the water medium carries no
     * extinction and therefore no scattering — the pre-water-fog behaviour, bit for bit — and for
     * any turbidity in [0, 1] the scattering never exceeds the extinction (sigmaS <= sigmaT per
     * construction), while the extinction itself stays independent of the turbidity so the
     * transmittance a turbidity change produces is exactly zero.
     */
    @Test
    void waterScatteringInvariants() {
        float[] tints = {0.0f, 0.2f, 0.5f, 0.8f, 1.0f};
        float[] opacities = {0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
        float[] turbidities = {0.0f, 0.35f, 0.7f, 1.0f};
        for (float tint : tints) {
            for (float opacity : opacities) {
                float extinction = waterExtinction(tint, opacity);
                if (opacity == 0.0f) {
                    assertEquals(0.0f, extinction, 0.0f,
                            "opacity 0 must leave no residual extinction at tint=" + tint);
                }
                for (float turbidity : turbidities) {
                    float scattering = turbidity > 0.0f ? turbidity * extinction : 0.0f;
                    assertTrue(scattering <= extinction + 1.0e-9f,
                            "sigmaS must not exceed sigmaT at tint=" + tint + " opacity=" + opacity
                                    + " turbidity=" + turbidity);
                    // The grey-mean single-scatter weight the underwater in-scatter uses.
                    if (extinction > 0.0f) {
                        float weight = scattering / extinction;
                        assertTrue(weight >= 0.0f && weight <= 1.0f + 1.0e-6f,
                                "single-scatter fraction must stay in [0, 1] at turbidity="
                                        + turbidity);
                    }
                }
            }
        }
    }

    /**
     * Java port of waterSegmentInScatter's depth-attenuated ambient integral: the ambient light field
     * inside water decays as exp(-sigmaT·depth) below the surface plane, so the in-scatter integral
     * becomes sigmaS·A·exp(-sigmaT·d0)·(1 - exp(-sigmaT·t·(1-rd.y))) / (sigmaT·(1-rd.y)) with the
     * straight-up limit t.
     */
    private static double waterAmbientIntegral(float sigmaT, float t, float dirY, float d0) {
        float upCos = 1.0f - dirY;
        if (upCos < 1.0e-4f) {
            return t * Math.exp(-sigmaT * d0);
        }
        return Math.exp(-sigmaT * d0) * (1.0 - Math.exp(-sigmaT * t * upCos)) / (sigmaT * upCos);
    }

    /**
     * The "white surface from below" regression: the underwater ambient in-scatter must collapse
     * with depth (deep water goes dark, not milk), match a brute-force numeric integration of the
     * decaying field, and reduce to the un-attenuated closed form right at the surface (d0 = 0).
     * Deliberately NOT monotone in depth overall: near the surface the path to the plane is short,
     * so the integral first grows with the available path length, then the decay wins.
     */
    @Test
    void waterAmbientDecaysWithDepth() {
        float sigmaT = waterExtinction(0.5f, 4.0f); // default murky water, ~0.25/block
        float sigmaS = 0.35f * sigmaT;              // default turbidity
        float ambient = 1500.0f;                    // arbitrary sky radiance — ratios are what matter
        for (float dirY : new float[]{0.95f, 0.5f, -0.5f}) {
            for (float d0 : new float[]{1f, 5f, 20f, 60f}) {
                // Upward rays end at the surface plane (t = d0/dirY, depth stays >= 0 along the
                // segment); downward rays run the full 120 blocks away from it.
                float t = dirY > 0.0f ? d0 / dirY : 120.0f;
                double closed = sigmaS * ambient * waterAmbientIntegral(sigmaT, t, dirY, d0);
                // Brute force: 200k midpoint samples of T(s)·sigmaS·A·exp(-sigmaT·depth(s)).
                double h = t / 200_000.0;
                double numeric = 0.0;
                for (int i = 0; i < 200_000; i++) {
                    double s = (i + 0.5) * h;
                    double depth = Math.max(d0 - dirY * s, 0.0);
                    numeric += Math.exp(-sigmaT * s) * Math.exp(-sigmaT * depth) * h;
                }
                numeric *= sigmaS * ambient;
                assertClose("ambient integral dirY=" + dirY + " d0=" + d0, numeric, closed, 1.0e-3);
            }
        }
        // At the surface plane with a horizontal view the integral reduces to the un-attenuated
        // closed form weight/sigmaT, i.e. the pre-fix term.
        double atSurface = sigmaS * ambient * waterAmbientIntegral(sigmaT, 50.0f, 0.0f, 0.0f);
        double legacy = (sigmaS / sigmaT) * (1.0 - Math.exp(-sigmaT * 50.0)) * ambient;
        assertClose("surface-consistency with the un-attenuated form", legacy, atSurface, 1.0e-6);
        // Deep water collapses to darkness: 60 blocks down, the glow must be gone against the
        // saturated pre-fix value a long horizontal view used to hold everywhere.
        double deep = sigmaS * ambient * waterAmbientIntegral(sigmaT, 120.0f, 0.0f, 60.0f);
        assertTrue(deep < legacy * 1.0e-4,
                "60 blocks down the ambient glow must be gone, got " + deep + " vs " + legacy);
    }

    /**
     * The underwater in-scatter's closed form over one segment: scatter weight = 1 - exp(-sigmaT·t)
     * sits in [0, 1] for every geometry, and the whole term is bounded by the turbidity — the
     * firefly-safety claim in waterSegmentInScatter's header, pinned numerically over the parameter
     * box the push block can produce (turbidity <= 1, opacity <= 4).
     */
    @Test
    void waterInScatterStaysBounded() {
        float sigmaT = waterExtinction(0.3f, 4.0f); // the murkiest legal water
        float turbidity = 1.0f;
        for (float t = 1.0f; t <= 500.0f; t += 3.0f) {
            float scatterWeight = 1.0f - (float) Math.exp(-sigmaT * t);
            assertTrue(scatterWeight >= 0.0f && scatterWeight <= 1.0f + 1.0e-6f,
                    "scatter weight must stay in [0, 1] at t=" + t);
            // L = turbidity * scatterWeight * (phase·illuminance·vis·legT + ambient): every factor
            // <= 1 except the clamped caustic (water.slang's CAUSTIC_MAX, source-paired below), so
            // the term is bounded by turbidity * illuminance — no firefly channel exists.
            float bound = turbidity * scatterWeight;
            assertTrue(bound <= 1.0f + 1.0e-6f, "the grey in-scatter bound must hold at t=" + t);
        }
    }

    // ---- Source pairing ------------------------------------------------------------------------------

    /** Bit position of a `WORLD_FLAG_*` constant in the shader source. */
    private static int shaderFlagBit(String source, String name) {
        Matcher m = Pattern.compile("public static const uint " + name + " = 1u << (\\d+)u;")
                .matcher(source);
        assertTrue(m.find(), "shader must define " + name);
        return Integer.parseInt(m.group(1));
    }

    /** Java port of volume.slang's fogTransmittance over the same closed form. */
    private static float fogTransmittance(float sigma0, float baseY, float heightH, float y0,
                                          float dirY, float t) {
        return (float) Math.exp(-fogOpticalDepth(sigma0, baseY, heightH, y0, dirY, t));
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
        for (String flag : new String[]{"WORLD_FLAG_FOG", "WORLD_FLAG_CLOUD", "WORLD_FLAG_SPRAY",
                "WORLD_FLAG_VOLUME_SHADOW", "WORLD_FLAG_VOLUME_MARCH", "WORLD_FLAG_VOLUME_DEBUG",
                "WORLD_FLAG_CLOUD_VOLUMETRIC"}) {
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
        // aerial perspective alive through glass). The fog flag caps everything else.
        assertTrue(indirect.contains("(bounce == seg.bounce || fogAfterInterface) && fogOn"),
                "the fog in-scatter must run on the first segment and behind dielectric crossings only");
        assertTrue(indirect.contains("WORLD_FLAG_FOG) != 0u"),
                "the fog in-scatter must be gated on the fog flag");
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        assertTrue(primary.contains("fogOn && !any(medium.current.extinction > 0.0)"),
                "the pre-interface fog attenuation must apply only while the stack medium is air");
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
        assertTrue(composite.contains("boolean fogMarchOn = CausticaConfig.Rt.Volume.FOG_ENABLED.value() && skyLit"),
                "the march flag must sit behind the fog switch and the sky-light gate");
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
     * The shadow cull's core promise: fully-shadowed fog is never denser than the outdoor fog. Both
     * tiers thin the medium by a factor max(vis, floor) with vis the measured sun visibility in
     * [0, 1], so the promise holds for every floor in [0, 1). The floor is now the live slider
     * volume.fog-shadow-floor riding fog1.w (no longer a compile-time constant), so this pins the
     * whole chain across the slider's range: the shader reads fog1.w at every culling site in both
     * tiers, Java pushes the config value into that lane, and the resulting density and transmittance
     * never cross the unthinned outdoor values over a full visibility sweep per floor setting.
     */
    @Test
    void shadowCulledFogNeverExceedsOutdoorFog() throws IOException {
        String volume = Files.readString(VOLUME);
        String composite = Files.readString(COMPOSITE);
        // The constant is gone; the floor is the fog1.w lane, read through fogShadowFloor() at every
        // culling site in both tiers.
        assertTrue(volume.contains("public float fogShadowFloor()"),
                "the floor must be a function reading the pushed lane");
        assertTrue(volume.contains("clamp(worldPush.fog1.w, 0.0, 0.95)"),
                "fogShadowFloor must clamp the fog1.w lane into the safe range");
        assertTrue(volume.contains("sigma = sigma0 * max(visBar, fogShadowFloor())"),
                "the analytic tier must thin by max(visBar, fogShadowFloor())");
        assertTrue(volume.contains("float sigmaEff = sigma * max(vis, fogShadowFloor())"),
                "the march tier must thin by max(vis, fogShadowFloor())");
        // The floor must ride the fog1.w lane specifically — the v13 regression pushed it into the
        // look-alike volumeParams.w lane, where nothing read it and the slider did nothing.
        assertTrue(composite.contains("fogMarchOn ? CausticaConfig.Rt.Volume.FOG_NOISE.value() : 0f,\n"
                + "                            CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR.value())"),
                "Java must push the fog-shadow-floor config into fog1.w (after the noise lane)");
        assertTrue(!composite.contains("0f, CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR.value()"),
                "the floor must NOT ride volumeParams.w, where no shader reads it");
        // The sun-leg transmittance of the marched tier rides the same floored sigma.
        assertTrue(volume.contains("sigma0 * max(vis, fogShadowFloor())"),
                "the march tier's toward-sun transmittance must use the floored sigma");

        // Sweep a representative set of slider positions (0 = hard cull, default 0.10, dense 0.90):
        // culled density never exceeds the outdoor density, and culled transmittance never drops
        // below the outdoor transmittance (thinner fog is more transparent), along every ray
        // geometry the closed form handles.
        float sigma0 = 0.02f;
        float baseY = 62f;
        float heightH = 24f;
        for (float floor : new float[]{0.0f, 0.10f, 0.35f, 0.90f}) {
            for (float vis = 0.0f; vis <= 1.0f + 1.0e-6f; vis += 0.01f) {
                float factor = Math.max(vis, floor);
                assertTrue(factor <= 1.0f + 1.0e-6f, "cull factor must not exceed 1 at vis=" + vis);
                assertTrue(factor >= floor - 1.0e-6f, "cull factor must hold the floor at vis=" + vis);
                float sigmaCulled = sigma0 * factor;
                assertTrue(sigmaCulled <= sigma0 + 1.0e-9f,
                        "culled density must not exceed outdoor density at floor=" + floor
                                + " vis=" + vis);
                for (float y0 : new float[]{baseY + 30f, baseY, baseY - 10f}) {
                    for (float dy : new float[]{0.6f, 0.05f, -0.5f}) {
                        float tOutdoor = fogTransmittance(sigma0, baseY, heightH, y0, dy, 120f);
                        float tCulled = fogTransmittance(sigmaCulled, baseY, heightH, y0, dy, 120f);
                        assertTrue(tCulled >= tOutdoor - 1.0e-7f,
                                "culled transmittance must not fall below outdoor transmittance at floor="
                                        + floor + " vis=" + vis + " y0=" + y0 + " dy=" + dy);
                    }
                }
            }
        }
    }

    /**
     * Phase 3's wiring: the underwater in-scatter fires only inside a water medium with a non-zero
     * turbidity (zero turbidity must keep the pre-water-fog frame untouched), both call sites gate on
     * the same predicate, the firefly bound is the caustic clamp, and a resumed packed record
     * re-derives the scattering instead of expecting a lane the 48-byte record never carried.
     */
    @Test
    void waterScatteringGatingAndFireflyBound() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains("if (sigmaT <= 0.0 || sigmaS <= 0.0 || t <= RAY_TMIN)"),
                "waterSegmentInScatter must early-out on a zero medium (the zero-when-off invariant)");
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        for (String source : new String[]{indirect, primary}) {
            assertTrue(source.contains("medium.current.water && worldPush.water1.x > 0.0"),
                    "the underwater in-scatter must gate on the water medium AND a non-zero turbidity");
        }
        // Probe budget: the indirect pass fires the sun probe on the walk's FIRST segment only (the
        // primary prefix owns the camera's own stretch) — a probe per TIR bounce cost a third of the
        // frame for second-order radiance.
        assertTrue(indirect.contains("worldPush.water1.x > 0.0\n                && bounce == seg.bounce"),
                "the water sun probe must be gated to the first indirect segment");
        // The fog and water in-scatters must never co-fire on one segment: fog stands down while any
        // extinction is present, which is exactly the water/glass case.
        assertTrue(indirect.contains("!any(medium.current.extinction > 0.0)"),
                "the fog in-scatter must stay behind the no-extinction gate next to the water term");
        // Firefly bound: the caustic is clamped at the source, and the in-scatter multiplies the
        // clamped value (never a raw focus).
        String water = Files.readString(Path.of("shaders/pipelines/world/water.slang"));
        assertTrue(water.contains("lerp(1.0, min(focus, CAUSTIC_MAX), fade)"),
                "waterCaustic must clamp the focusing factor at CAUSTIC_MAX");
        assertTrue(volume.contains("vis *= waterCaustic("),
                "the underwater sun leg must consume the caustic through the clamped call");
        // Resumed records re-derive scattering from the packed extinction plus the turbidity global;
        // the derivation itself must be the turbidity product, not a free lane.
        String medium = Files.readString(Path.of("shaders/pipelines/world/medium.slang"));
        assertTrue(medium.contains("turbidity > 0.0 ? turbidity * extinction : float3(0.0, 0.0, 0.0)"),
                "waterScattering must be the turbidity fraction of the extinction");
        String segment = Files.readString(Path.of("shaders/pipelines/world/segment.slang"));
        assertTrue(segment.contains("current.water ? waterScattering(current.extinction)"),
                "unpackPathSegment must re-derive the water scattering for resumed records");
    }

    /**
     * The spray band's inertness: the shader's spray term returns zero whenever the spray flag is
     * clear, and Java only raises that flag when the marched fog tier is running and the strength is
     * non-zero — an analytic or spray-off frame pushes an inert weather.z, exactly the volume block's
     * zero-when-off doctrine.
     */
    @Test
    void sprayBandGating() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(volume.contains("if ((push.flags & WORLD_FLAG_SPRAY) == 0u)"),
                "sprayDensity must return zero while the spray flag is clear");
        String composite = Files.readString(COMPOSITE);
        assertTrue(composite.contains(
                "boolean sprayOn = fogMarchOn && CausticaConfig.Rt.Composite.WATER_SPRAY.value() > 0f"),
                "the spray flag must sit behind the marched fog tier and a non-zero strength");
        assertTrue(composite.contains("sprayOn ? CausticaConfig.Rt.Composite.WATER_SPRAY.value()"),
                "the weather.z spray lane must be zero while the spray flag is down");
    }

    /**
     * The cloud layer's numeric contracts, ported from volume.slang: the coverage remap never leaves
     * [0, 1], stays empty at coverage 0 and saturates toward overcast at 1, and is monotone in the
     * underlying noise; the shadow transmittance sits in (0, 1], is monotone non-increasing in both
     * coverage and density, equals exactly 1 at zero coverage, and stands down for a
     * horizon-grazing body. These are the invariants every consumer (sheet, NEE, fog sun legs)
     * silently leans on.
     */
    @Test
    void cloudLayerInvariants() {
        // Java port of cloudCoverageField's remap (the noise n is treated as its full 0..1 domain).
        for (float c = 0.0f; c <= 1.0f + 1.0e-6f; c += 0.1f) {
            float previous = -1.0f;
            for (float n = 0.0f; n <= 1.0f + 1.0e-6f; n += 0.01f) {
                // Java port of the live rule: threshold 1 - 1.15*c^0.7, width 0.08 + 0.32c (the
                // calibrated constants). c=0 still yields (n - 1)/0.08 <= 0, the empty sky the
                // zero-coverage invariant promises.
                float cov = Math.clamp(
                        (n - (1.0f - 1.15f * (float) Math.pow(c, 0.7f)))
                                / Math.max(0.08f + 0.32f * c, 1.0e-3f),
                        0.0f, 1.0f);
                assertTrue(cov >= 0.0f && cov <= 1.0f,
                        "coverage must stay in [0, 1] at c=" + c + " n=" + n);
                if (c <= 1.0e-6f) {
                    assertEquals(0.0f, cov, 1.0e-9f,
                            "coverage 0 must produce an empty sky at every noise value");
                }
                assertTrue(cov >= previous - 1.0e-9f,
                        "coverage must be monotone in the noise at c=" + c);
                previous = cov;
            }
        }

        // Java port of cloudShadowTransmittance's tau = cov * density * 0.22 * thickness / sunY.
        final float extinction = 0.22f;
        for (float thickness : new float[]{0.5f, 12.0f, 64.0f}) {
            for (float density : new float[]{0.0f, 0.5f, 1.0f, 3.0f}) {
                float previous = Float.MAX_VALUE;
                for (float cov = 0.0f; cov <= 1.0f + 1.0e-6f; cov += 0.01f) {
                    float tau = cov * density * extinction * thickness / 0.6f;
                    float t = (float) Math.exp(-tau);
                    assertTrue(t > 0.0f && t <= 1.0f + 1.0e-6f,
                            "shadow transmittance must sit in (0, 1] at cov=" + cov
                                    + " density=" + density + " thickness=" + thickness);
                    assertTrue(t <= previous + 1.0e-9f,
                            "shadow transmittance must not grow with coverage at cov=" + cov);
                    previous = t;
                    if (cov <= 1.0e-6f) {
                        assertEquals(1.0f, t, 1.0e-9f,
                                "zero coverage must cast no shadow at all");
                    }
                }
            }
        }
        // Below the grazing gate the shadow stands down entirely.
        assertEquals(1.0f, (float) Math.exp(-0.0f), 0.0f);
    }

    /**
     * The cloud tier's wiring, end to end: Java raises WORLD_FLAG_CLOUD behind the sky-light gate
     * and fills the cloud0/cloud1 lanes (lane-anchored — the v15 regression pushed a floor into a
     * look-alike lane nothing read), the sky miss shader renders the sheet and blends it over the
     * sky, every celestial NEE site multiplies the cloud shadow, and all three sun legs (analytic
     * fog, marched fog, underwater) read the same field. The fog-void fill rides the same miss
     * shader, gated on a fog that is actually rendering.
     */
    @Test
    void cloudTierGatingAndConsumers() throws IOException {
        String volume = Files.readString(VOLUME);
        String skyMiss = Files.readString(Path.of("shaders/pipelines/world/sky.rmiss.slang"));
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        String composite = Files.readString(COMPOSITE);

        // Flag assembly: behind the sky-light gate, like the fog but independent of every fog switch.
        assertTrue(composite.contains(
                        "boolean cloudOn = CausticaConfig.Rt.Volume.CLOUD_ENABLED.value() && skyLit"),
                "the cloud flag must sit behind the sky-light gate");
        assertTrue(composite.contains("flags |= WORLD_FLAG_CLOUD"),
                "Java must raise WORLD_FLAG_CLOUD when the cloud layer is enabled");

        // Lane pushes, anchored by their neighbours so a look-alike Float4 cannot swallow them.
        assertTrue(composite.contains(
                        "new Float4(cloudOn ? CausticaConfig.Rt.Volume.rainCoverage(\n"
                                + "                            CausticaConfig.Rt.Volume.CLOUD_COVERAGE.value(), storm, cloudRainBoost) : 0f,\n"
                                + "                            CausticaConfig.Rt.Volume.CLOUD_ALTITUDE"
                                + ".value() - terrain.blockY,"),
                "the cloud0 lanes must be pushed behind the cloudOn gate (coverage first, rebased base second)");
        // The streak frame's direction rides cloud1.y, the localized-coverage strength cloud1.z,
        // the fade cloud1.w; the drift itself is Java-accumulated into cloud2.zw (the v15 lesson:
        // anchor the lane by its neighbours).
        assertTrue(composite.contains(
                        "new Float4(volumetricCloudOn\n"
                                + "                                    ? CausticaConfig.Rt.Volume"
                                + ".CLOUD_TYPE.value() : 0f,\n"
                                + "                            cloudOn ? (float) Math.toRadians(\n"
                                + "                                    CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE.value()) : 0f,\n"
                                + "                            cloudOn ? CausticaConfig.Rt.Volume"
                                + ".CLOUD_PATCHINESS.value() : 0f,\n"
                                + "                            cloudOn ? CausticaConfig.Rt.Volume"
                                + ".CLOUD_FADE_DISTANCE.value() : 0f),"),
                "cloud1 must carry the profile mix, the streak azimuth, patchiness and fade, with the "
                        + "drift gone to cloud2");
        // The volumetric tier's two own lanes: the step count rides volumeParams.w beside the fog
        // march's lanes, and the type mix rides cloud1.x. Both are zero while the tier is off.
        assertTrue(composite.contains(
                        "                            0f,\n"
                                + "                            volumetricCloudOn\n"
                                + "                                    ? (float) CausticaConfig.Rt.Volume"
                                + ".CLOUD_STEPS.value() : 0f),"),
                "the volumetric step count must ride volumeParams.w, gated on the tier's own switch");
        assertTrue(!composite.contains("cloudOn ? CausticaConfig.Rt.Volume.CLOUD_WIND_SPEED.value()"),
                "the cloud wind speed must feed the Java drift accumulator, not a shader lane");
        assertTrue(composite.contains(
                        "new Float4((float) terrain.blockX, (float) terrain.blockZ,\n"
                                + "                            (float) cloudDriftX, (float) cloudDriftZ),"),
                "the cloud field must anchor on the EXACT terrain origin (no mod-4096 wrap: the "
                        + "streak lattice has no period) plus the Java-accumulated drift");
        assertTrue(volume.contains("float2 p = rebasedXZ + push.cloud2.xy + push.cloud2.zw;"),
                "the coverage field must sample through the exact anchor and the drift");
        // The sheet's light must not survive its body's sunset: the pushed illuminance is a look
        // constant, so only the elevation gate keeps noon light off a midnight sky.
        assertTrue(volume.contains("bool sunUp = state.sunIlluminance > 0.0 && state.sunDir.y > 0.0;")
                        && volume.contains("(moonUp ? state.moonIlluminance : 0.0);"),
                "the sheet's sun/moon term must gate on the body being above the horizon");
        assertTrue(volume.contains("clamp(push.cloudColor.w, 0.0, 2.0);"),
                "the sheet's ambient weight must be clamped so a lane slip cannot relight the sky");
        // The slab's self-shadow runs the multi-scatter octave stack (energy-normalised, so tau = 0
        // still gives exactly 1): a single-scatter deck reads blue-grey by day and black at night.
        assertTrue(volume.contains("float selfT = cloudSelfShadow(tauSelf);")
                        && volume.contains("public float cloudSelfShadow(float tau)")
                        && volume.contains("sum += weight * exp(-tau * exp2(-float(i)));")
                        && volume.contains("CLOUD_ALBEDO * opacity * (phase * lightIll * selfT + ambient)"),
                "the sheet's self-shadow must use the energy-normalised multi-scatter octaves");
        // Localised coverage: the multi-scale macro field gates the coverage into patches and
        // fades out toward full cover.
        // The gate REDISTRIBUTES instead of scaling: a zero-mean modulation of the LOCAL coverage keeps
        // the slider's total honest. The multiplicative form capped the delivered coverage at the gate's
        // own mean (~0.44 at patchiness 1) and cleared the sky to threads whenever the macro dipped.
        assertTrue(volume.contains("f.cLocal = saturate(c * (1.0 + f.gateMix * (smoothstep(0.28, 0.78, macro)")
                        && volume.contains("- CLOUD_GATE_MEAN)));")
                        && volume.contains("f.gateMix = clamp(push.cloud1.z, 0.0, 1.0);")
                        && volume.contains("return saturate((n - f.threshold) / f.width);"),
                "the macro field must redistribute the local coverage around its mean, riding cloud1.z");
        // The anti-repetition structure (all previewed offline before shipping): a wandering local
        // wind direction, a streak breadth that breathes with the macro, and a slider response
        // whose usable range is not a sliver.
        assertTrue(volume.contains("* (2.0 * CLOUD_DIR_JITTER);"),
                "the local wind direction must wander instead of running dead parallel");
        assertTrue(volume.contains("float crossEff = lerp(CLOUD_CELL_CROSS, CLOUD_CELL_CROSS_MAX, macro);")
                        && volume.contains("f.cellCross = lerp(crossEff, cumulusCell, f.cumulus);")
                        && volume.contains("dot(s, f.perp) / f.cellCross"),
                "the streak breadth must breathe with the macro field, and close up into cumulus cells");
        assertTrue(volume.contains("f.threshold = 1.0 - CLOUD_REMAP_K * pow(f.cLocal, CLOUD_REMAP_GAMMA);")
                        && volume.contains("f.width = max(CLOUD_REMAP_W0 + CLOUD_REMAP_W1 * f.cLocal, 1.0e-3);")
                        && volume.contains("return saturate((n - f.threshold) / f.width);")
                        && volume.contains("public static const float CLOUD_REMAP_GAMMA = 0.7;"),
                "the coverage remap must keep the calibrated constants (they are what makes the slider "
                        + "honest after the cumulus rework changed the field's distribution)");
        assertTrue(volume.contains("float n = cloudFbm(q, 3) - 0.07 * cloudValueNoise(q * 0.25 + 7.0 + f.evolve * 1.7)"),
                "the streak field must run three octaves (a fourth was sub-pixel speckle)");

        // The sheet-only dissipation: the fade reads cloud1.w, clamps to a sane minimum, and the
        // shadow field deliberately never reads it.
        assertTrue(volume.contains("float fadeDist = max(push.cloud1.w, 64.0);"),
                "the sheet's fade distance must ride the cloud1.w lane with a sane floor");
        assertTrue(volume.contains("float opacity = (1.0 - exp(-tau)) * fade;")
                        && volume.contains("cloudT = 1.0 - opacity;"),
                "the fade must dissolve the sheet's opacity (radiance and occlusion together) — "
                        + "fading the radiance alone left a black band where grazing tau had "
                        + "saturated the transmittance to 0");
        assertTrue(volume.contains("1.0 - smoothstep(fadeDist * 0.6, fadeDist, t)"),
                "the sheet must fade over the last 40% of the dissipation range");
        assertTrue(volume.contains(
                "float tau = cov * worldPush.cloud0.w * CLOUD_EXTINCTION * worldPush.cloud0.z / sunDir.y;"),
                "cloudShadowTransmittance's optical depth must stay free of the fade distance");

        // The sheet: the miss shader imports the volume module, renders the planar in-scatter, and
        // blends it over everything behind it — in-scatter on top, transmittance over the sky.
        assertTrue(skyMiss.contains("import volume;"),
                "the sky miss shader must import the cloud field's module");
        assertTrue(skyMiss.contains("planarCloudSkyInScatter(worldPush, WorldRayOrigin(), dir, state,"),
                "the sky miss shader must render the planar sheet from the pushed plane");
        assertTrue(skyMiss.contains("color = cloudL + cloudT * color;"),
                "the sheet must attenuate the sky behind it, not just add on top");
        assertTrue(skyMiss.contains("if (!endSky) {"),
                "the sheet must not render over an end-sky dome");

        // The fog-void fill: mirrored dome only while the fog actually renders.
        assertTrue(skyMiss.contains(
                        "(worldPush.flags & WORLD_FLAG_FOG) != 0u && worldPush.fog0.x > 0.0"),
                "the below-horizon void fill must gate on a fog that is actually rendering");
        assertTrue(skyMiss.contains("float3 domeDir = fogVoidFill && dir.y < 0.0"),
                "the void fill must mirror below-horizon dome lookups only");

        // Consumers: both NEE sites and all three sun legs multiply the same shadow field.
        assertTrue(indirect.contains("vis *= cloudShadowTransmittance(p, lightDir);"),
                "the terrain celestial NEE must pick up the cloud shadow");
        assertTrue(indirect.contains("vis *= cloudShadowTransmittance(shadowOrigin, lightDir);"),
                "the particle celestial NEE must pick up the cloud shadow");
        assertTrue(volume.contains("* cloudShadowTransmittance(ro, celestialLight.dir));"),
                "the analytic fog's unshadowed sun leg must pick up the cloud shadow");
        assertTrue(volume.contains("cloudShadowTransmittance(probe, celestialLight.dir));"),
                "the analytic fog's per-stratum sun legs must pick up the cloud shadow");
        assertTrue(volume.contains("* cloudShadowTransmittance(p, sunDir));"),
                "the marched fog's per-sample sun leg must pick up the cloud shadow");
        assertTrue(volume.contains(
                        "sunPart = phase * celestialLight.illuminance\n"
                                + "                    * (vis * legT * cloudShadowTransmittance(p, celestialLight.dir));"),
                "the underwater sun leg must pick up the cloud shadow");

        // The shadow field's own gate: exactly 1 while clouds are off, so every multiply above
        // keeps the cloud-off frame bit-identical.
        assertTrue(volume.contains(
                        "if ((worldPush.flags & WORLD_FLAG_CLOUD) == 0u || worldPush.cloud0.x <= 0.0) {\n"
                                + "        return 1.0;"),
                "cloudShadowTransmittance must return exactly 1 while the cloud layer is off");

        // The debug image's cloud lanes (b coverage, a shadow) live.
        assertTrue(primary.contains("cloudDensity = cloudCoverageWorld(ro.xz + rd.xz * tPlane);"),
                "the volume debug image must fill the cloud coverage lane");
        assertTrue(primary.contains("cloudShadow = cloudShadowTransmittance(ro, celestialLight.dir);"),
                "the volume debug image must fill the cloud shadow lane");
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
        int cloudColorCall = composite.indexOf("new Float4((float) cloudEvolve, 0f, 0f, 1f),");
        assertTrue(cloud2Call > 0 && cloudColorCall > 0,
                "both cloud lanes must appear at the call site");
        assertTrue(cloud2Call < cloudColorCall,
                "the call site must pass the cloud anchor/drift block BEFORE the cloud colour lane, "
                        + "matching the generated record order");
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
     * The volumetric cloud tier's numeric contracts and wiring. The slab is the SAME layer the planar
     * tier crosses — same base, same thickness, same coverage field — so the invariants that matter are
     * that the two vertical profiles carry the same column depth (turning the type knob must not change
     * how much cloud is overhead), that the walk's fade dissolves its OPACITY rather than its glow (the
     * v17 black-band lesson, which a walk reaches just as easily as a single crossing), and that the
     * expensive path is reachable only from a primary ray while continuation rays get the cheap mean.
     */
    @Test
    void volumetricCloudTierInvariants() throws IOException {
        String volume = Files.readString(VOLUME);
        String skyMiss = Files.readString(Path.of("shaders/pipelines/world/sky.rmiss.slang"));
        String composite = Files.readString(COMPOSITE);

        // The slab samples the shared coverage field: one field, three consumers, so the deck the player
        // sees casts the shadow they see.
        assertTrue(volume.contains("public float cloudSlabDensity(WorldPush push, CloudFieldFrame f, float3 pos)")
                        && volume.contains("float cov = cloudCoverageAt(f, pos.xz + push.cloud2.xy + push.cloud2.zw);"),
                "the volumetric slab must be built from the same coverage field the NEE and fog legs read");

        // One field, one implementation, two call patterns. The plane-crossing consumers build a frame at
        // the sample itself; the slab walk builds ONE at its crossing's midpoint and advances the
        // per-sample octaves from it. Both go through cloudCoverageAt, so the deck a player sees cannot
        // drift away from the shadow it casts.
        assertTrue(volume.contains("public CloudFieldFrame cloudFieldFrameAt(WorldPush push, float2 p)"),
                "the macro-derived field invariants must have a single builder");
        assertTrue(volume.contains("return cloudCoverageAt(cloudFieldFrameAt(push, p), p);"),
                "the plane-crossing consumers must build a frame at the sample and share the per-sample core");
        int frameBuilds = 0;
        int idx = 0;
        while ((idx = volume.indexOf("cloudFieldFrameAt(push,", idx)) >= 0) {
            frameBuilds++;
            idx += 1;
        }
        assertEquals(2, frameBuilds,
                "the frame builder must be called exactly twice and never per sample: once by "
                        + "cloudCoverageField and once by the slab walk");
        int walk = volume.indexOf("public float3 volumetricCloudInScatter");
        int loop = volume.indexOf("for (int i = 0; i < int(steps); i++)", walk);
        int build = volume.indexOf("CloudFieldFrame frame = cloudFieldFrameAt(push,", walk);
        assertTrue(build > walk && build < loop,
                "the slab walk must build its frame before the sample loop, not inside it");
        assertTrue(volume.contains("float density = cloudSlabDensity(push, frame, p);"),
                "the walk's samples must ride the one frame built for the crossing");
        // The frame's sample point is CLAMPED along the ray. Unbounded, a horizon ray reads its wind
        // direction and gate kilometres out while an overhead ray reads them nearby, and the sky fans
        // into wedges centred on the camera as the streak orientation snaps between jitter cells.
        assertTrue(volume.contains("ro.xz + rd.xz * min(0.5 * (t0 + t1), CLOUD_FRAME_DIST)")
                        && volume.contains("public static const float CLOUD_FRAME_DIST = 1536.0;"),
                "the frame must be sampled at a bounded distance along the ray");
        assertTrue(!volume.substring(walk, volume.indexOf("public float volumetricCloudTransmittanceFast", walk))
                        .contains("cloudCoverageField("),
                "no per-sample path may re-derive the macro octave: that is the whole point of the frame");
        // A shut gate closes coverage over a whole region, which is far wider than one crossing, so a ray
        // whose frame is shut is skipped outright instead of paying the field once per step to find out.
        // A redistributed coverage this close to zero leaves a hair of cloud at most anywhere in the
        // crossing (the threshold it implies is past the field's reach), so the walk is skipped rather
        // than paying the per-sample field to discover that 24 times.
        assertTrue(volume.contains("if (frame.cLocal <= 0.02) {"),
                "a ray whose redistributed coverage is negligible must skip the walk entirely");
        assertTrue(volume.contains("if (h < 0.0 || h > 1.0) {\n        return 0.0;"),
                "the slab must be exactly zero outside its base..top band");

        // Profile depths: the stratus board and the cumulus heap must integrate to the same column, so
        // the type dial changes the shape the light falls on and never how much cloud is overhead. Flat
        // is the 0.6-weighted rim softening; heap is the smoothstep base/crown pair normalised by
        // CLOUD_CUMULUS_NORM.
        double flatMean = 0.0;
        double heapMean = 0.0;
        int samples = 20000;
        for (int i = 0; i < samples; i++) {
            float h = (i + 0.5f) / samples;
            float flat = (float) (1.0 - 0.6 * Math.pow(Math.abs(2.0 * h - 1.0), 12.0));
            float t = Math.clamp(h / 0.09f, 0.0f, 1.0f);
            float base = t * t * (3.0f - 2.0f * t);
            float u = Math.clamp((h - 0.60f) / 0.40f, 0.0f, 1.0f);
            float crown = 1.0f - u * u * (3.0f - 2.0f * u);
            float heap = (float) (base * crown * 1.3245);
            assertTrue(flat >= 0.0f && heap >= 0.0f, "both profiles must be non-negative at h=" + h);
            flatMean += flat;
            heapMean += heap;
        }
        flatMean /= samples;
        heapMean /= samples;
        assertClose("the cumulus heap must integrate to exactly the column depth", 1.0, heapMean, 4.0e-3);
        assertTrue(Math.abs(flatMean - heapMean) < 0.06,
                "both profiles must carry the same column depth, so the type knob changes the shape "
                        + "the light falls on and not how much cloud is overhead (flat=" + flatMean
                        + " heap=" + heapMean + ")");
        // The heap is a heap: thin at the base, fat at the shoulder, rounded at the crown.
        assertTrue(1.0f * 1.0f * 1.3245f > 1.0f,
                "the cumulus shoulder must be denser at mid-slab than the stratus board");
        assertTrue(0.09f < 0.60f, "the base ramp must finish well below the crown ramp");

        // The 3D shape field, and why it exists. The old density was a product of a 2D function of xz and
        // a 1D function of height: separable, hence a stack of identical horizontal sheets whose
        // brightness stepped with height (layer cake from the side) with the same streak lanes repeated at
        // every level (maze walls). The regression guard is that the density's noise must depend on
        // height — the lobe coordinate must carry pos.y.
        assertTrue(volume.contains("public float cloudFbm3(float3 p, int octaves)"),
                "the slab's shape must come from a 3D field");
        assertTrue(volume.contains("float3 q = float3(dot(w, f.dir), pos.y, dot(w, f.perp)) / cell;"),
                "the lobe coordinate must carry height: a density separable into (2D of xz) * (1D of y) "
                        + "is a layer cake no matter how finely it is walked");
        assertTrue(volume.contains("float cell = max(thickness * CLOUD_LOBE_RATIO, CLOUD_LOBE_MIN);"),
                "the lobe cell must scale with the slab: a fixed cell is either too coarse to vary "
                        + "vertically in a thin deck or too fine to hold a tower in a thick one");
        assertTrue(volume.contains("float lobe = cloudFbm3(q, 2);")
                        && volume.contains("+ float3(f.evolve, 0.0) * 0.55, 1);")
                        && volume.contains("return saturate(cov * contrast) * cloudHeightProfile(h, f.cumulus)")
                        && volume.contains("push.cloud0.w;"),
                "the density must be coverage x 3D contrast x height profile x density multiplier");
        assertTrue(volume.contains("public float cloudHeightProfile(float h, float typeMix)"),
                "the height profile must be its own named function so the type dial's normalisation is "
                        + "visible in one place");

        // Both modulations must be MEAN-PRESERVING, which is what keeps the slab's column optical depth
        // equal to the coverage field's tau — and therefore keeps the deck a player sees casting the
        // shadow they see, and the planar tier a fair reference for how thick the cloud is.
        double contrastMean = 0.0;
        double erosionMean = 0.0;
        for (int i = 0; i < samples; i++) {
            float u = (i + 0.5f) / samples;
            contrastMean += 1.0 + 1.8 * (u * 2.0 - 1.0);
            erosionMean += 1.0 - 0.35 + 2.0 * 0.35 * u;
        }
        contrastMean /= samples;
        erosionMean /= samples;
        assertClose("the 3D lobe contrast must preserve the column", 1.0, contrastMean, 1.0e-6);
        assertClose("the rim erosion must preserve the column", 1.0, erosionMean, 1.0e-6);

        // Cumulus samples the same lattice coarser, and the dilation has to live inside the field so the
        // shadow probes read it too — otherwise the clouds a player sees would not be the clouds that
        // cast the shadows they see.
        assertTrue(volume.contains("f.cellAlong = lerp(CLOUD_CELL_ALONG, cumulusCell, f.cumulus);")
                        && volume.contains("float cumulusCell = max(push.cloud0.z * CLOUD_LOBE_RATIO, CLOUD_LOBE_MIN);"),
                "the cumulus planform must be isotropic and sized by the cloud's own size (the slab "
                        + "thickness), not by the stratus lattice");
        assertTrue(volume.contains("f.broaden = (push.flags & WORLD_FLAG_CLOUD_VOLUMETRIC) != 0u")
                        && volume.contains("? lerp(1.0, CLOUD_BROADEN, f.cumulus) : 1.0;")
                        && volume.contains("float2 s = p * f.broaden;"),
                "the cumulus planform must dilate the shared coverage field itself, gated on the tier");

        // The walk: steps come from volumeParams.w behind the shader's own floor, and the fade dissolves
        // opacity with the radiance following it at the same scale (the v17 lesson).
        assertTrue(volume.contains("float steps = clamp(push.volumeParams.w, CLOUD_STEP_MIN, 96.0);"),
                "the walk must read its step count from the pushed lane and honour the alias floor");
        assertTrue(volume.contains("float opacity = (1.0 - T) * fade;")
                        && volume.contains("cloudT = 1.0 - opacity;")
                        && volume.contains("return L * fade;"),
                "the walk's fade must dissolve the slab's opacity, radiance and occlusion together");
        assertTrue(volume.contains("float jitter = frac(interleaveGradientNoise(pix)")
                        && volume.contains("+ float(push.frameIndex) * 0.6180339887);"),
                "the walk's dither must be a pixel-fixed pattern advanced one golden-ratio stride per "
                        + "frame — a per-frame re-roll is a signal the denoiser cannot accumulate");
        // The local-density self-shadow: no extra field tap, and the multi-scatter stack stays on top.
        assertTrue(volume.contains("float tauSun = tauUp / max(lightDir.y, 0.05);")
                        && volume.contains("float3 sunT = cloudSelfShadow(tauSun);")
                        && volume.contains("float ambT = cloudSelfShadow(tauUp * 0.25);"),
                "each sample's light must come from its own density and the path left to the slab top, "
                        + "with the multi-scatter octave stack on top");
        assertTrue(volume.contains("float phase = hg(cosTheta, CLOUD_G) * 0.7 * (2.0 / (1.0 + exp(-2.0 * cosTheta)));"),
                "the samples must carry the plan's Beer-Powder phase term");

        // Tier routing: the flag is separate from the base cloud bit, only a primary ray walks the slab,
        // and a continuation ray pays the cheap mean extinction instead.
        assertTrue(skyMiss.contains("if ((worldPush.flags & WORLD_FLAG_CLOUD_VOLUMETRIC) != 0u) {")
                        && skyMiss.contains("cloudL = volumetricCloudInScatter(worldPush, WorldRayOrigin(), dir, state,")
                        && skyMiss.contains("cloudT = volumetricCloudTransmittanceFast(worldPush, WorldRayOrigin(), dir);"),
                "the miss shader must route between the volumetric walk and the cheap extinction, keyed "
                        + "on the tier's own flag");
        assertTrue(skyMiss.contains("if (showCelestial) {"),
                "only the primary ray may pay for the walk; continuation rays get the mean extinction");
        assertTrue(volume.contains("public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd)"),
                "the extinction-only path must exist for continuation rays");
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

        // The tier table, Java ported: (cloud, cloudVolumetric, cloudSteps, fog, fogShadow, fogMarch,
        // fogMarchSteps, turbidity, spray) for tiers 0..4.
        boolean[][] on = new boolean[5][];
        int[] steps = new int[5];
        float[] turb = new float[5];
        float[] spray = new float[5];
        for (int t = 0; t <= 4; t++) {
            on[t] = new boolean[]{t >= 1, t >= 2, t >= 3, t >= 3, t >= 2, t >= 4, t >= 1, t >= 4};
            steps[t] = t == 2 ? 12 : (t == 4 ? 40 : 24);
            turb[t] = t >= 1 ? 0.35f : 0.0f;
            spray[t] = t >= 4 ? 0.5f : 0.0f;
        }
        // Off clears everything: clouds, the volumetric tier, the fog layer and its shadow, and the water
        // medium's scattering (which is what the underwater in-scatter is gated on).
        for (boolean b : on[0]) {
            assertTrue(!b, "tier 0 must clear every volumetric switch");
        }
        assertEquals(0.0f, turb[0], 0.0f, "tier 0 must clear the water medium's scattering");
        assertEquals(0.0f, spray[0], 0.0f, "tier 0 must clear the surface spray band");
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
        assertTrue(on[2][5] == false && on[4][5],
                "the sprayed fog band belongs to the marched tier, not to the analytic one");

        // The table is data in the config, and the write is a total function of it rather than a list of
        // spells at each call site.
        assertTrue(config.contains("public record QualityTier(boolean cloud, boolean cloudVolumetric, int cloudSteps,")
                        && config.contains("public static QualityTier qualityTier(int tier) {")
                        && config.contains("public static void applyQuality(int tier) {"),
                "the tier table must be data, with the write derived from it");
        for (String member : new String[]{"CLOUD_ENABLED.set(", "CLOUD_VOLUMETRIC.set(", "CLOUD_STEPS.set(",
                "FOG_ENABLED.set(", "FOG_SHADOW.set(", "FOG_MARCH.set(", "FOG_MARCH_STEPS.set(",
                "CausticaConfig.Rt.Composite.WATER_TURBIDITY.set(", "CausticaConfig.Rt.Composite.WATER_SPRAY.set("}) {
            assertTrue(config.contains(member), "applyQuality must write " + member);
        }
        // The choice list rejects a hand-edited config rather than letting an out-of-range tier through.
        assertTrue(config.contains("intChoice(\"caustica.rt.volume.quality\", \"volume.quality\", 0, QUALITY_TIERS)"),
                "the tier must be a choice list, so a hand-edited value cannot land outside the table");

        // The side effect belongs to the option, not to whichever screen is open: both hosts funnel every
        // committed write through afterChange, which is where the preset applies itself.
        assertTrue(settings.contains("CausticaConfig.Rt.Volume.QUALITY.tomlPath().equals(option.path())")
                        && settings.contains("CausticaConfig.Rt.Volume.applyQuality(option.integer().get());"),
                "selecting the preset must write the other settings through the shared afterChange funnel");
        // Declared as a count with value keys, which is what makes the tiers read as words on both hosts.
        assertTrue(settings.contains("\"volume.quality\", RT + \"volumeQuality\",")
                        && settings.contains("RT + \"volumeQuality.\""),
                "the tiers must be a named integer choice");
    }

    /**
     * Two behaviours that make the deck a weather system rather than a decal: the field EVOLVES (its
     * noise layers crawl against each other, so the shapes reorganise while the whole thing advects),
     * and rain thickens it. The thickening is computed Java-side and pushed, which is what keeps every
     * consumer - sheet, walk, shadow field, NEE, fog legs - on the same lifted numbers; the test ports
     * those formulas and pins the wiring on both sides of the push.
     */
    @Test
    void cloudWeatherThickeningAndFieldEvolution() throws IOException {
        String config = Files.readString(CONFIG);
        String composite = Files.readString(COMPOSITE);
        String volume = Files.readString(VOLUME);

        // Java port of the pushed formulas. Identity at no weather and at boost 0 - the two cases that
        // have to reproduce the pre-feature sky bit for bit - monotone in weather, clamped at overcast,
        // and thunder stacking on rain rather than replacing it.
        float[] boosts = {0.0f, 0.35f, 0.6f, 1.0f};
        for (float boost : boosts) {
            for (int rainPct = 0; rainPct <= 10; rainPct++) {
                for (int thunderPct = 0; thunderPct <= 10; thunderPct++) {
                    float rain = rainPct / 10f;
                    float thunder = thunderPct / 10f;
                    float storm = Math.clamp(rain + 0.35f * thunder, 0.0f, 1.0f);
                    float cov = Math.clamp(0.5f + 0.45f * boost * storm, 0.0f, 1.0f);
                    float den = 0.06f * (1.0f + 1.6f * boost * storm);
                    if (rain == 0f && thunder == 0f) {
                        assertClose("no weather must leave the coverage alone", 0.5f, cov, 1.0e-6f);
                        assertClose("no weather must leave the density alone", 0.06f, den, 1.0e-6f);
                    }
                    assertTrue(cov >= 0.5f && den >= 0.06f,
                            "weather must never thin the deck (cov=" + cov + " den=" + den + ")");
                    assertTrue(cov <= 1.0f, "the coverage must clamp at solid overcast");
                    // thunder only ever adds: a thunderstorm is at least as thick as the same rain alone
                    float stormNoThunder = Math.clamp(rain, 0.0f, 1.0f);
                    float covNoThunder = Math.clamp(0.5f + 0.45f * boost * stormNoThunder, 0.0f, 1.0f);
                    assertTrue(cov >= covNoThunder - 1.0e-6f,
                            "thunder must not reduce the coverage");
                }
            }
        }
        // boost 0 is the second identity: any weather, no thickening.
        assertClose("boost 0 must hold the coverage to the slider", 0.5f,
                Math.clamp(0.5f + 0.45f * 0.0f * 1.0f, 0.0f, 1.0f), 1.0e-6f);
        assertClose("boost 0 must hold the density to the slider", 0.06f,
                0.06f * (1.0f + 1.6f * 0.0f * 1.0f), 1.0e-6f);
        // a full storm at full boost decks over completely and near-triples the shadow strength
        assertClose("a full storm at full boost lifts 50% to 95%", 0.95f,
                Math.clamp(0.5f + 0.45f * 1.0f * 1.0f, 0.0f, 1.0f), 1.0e-6f);
        assertClose("a full storm at full boost multiplies the density by 2.6", 0.06f * 2.6f,
                0.06f * (1.0f + 1.6f * 1.0f * 1.0f), 1.0e-6f);

        // The formulas live in the config as pure functions and the push reads them, so no consumer can
        // re-derive a different answer.
        assertTrue(config.contains("public static float stormLevel(float rain, float thunder)")
                        && config.contains("public static float rainCoverage(float coverage, float storm, float boost)")
                        && config.contains("public static float rainDensity(float density, float storm, float boost)"),
                "the weather formulas must be pure functions in the config");
        assertTrue(composite.contains("CausticaConfig.Rt.Volume.rainCoverage(")
                        && composite.contains("CausticaConfig.Rt.Volume.rainDensity(")
                        && composite.contains("CausticaConfig.Rt.Volume.stormLevel(rainLevel, thunderLevel)"),
                "the push must carry the weather-lifted coverage and density");

        // Evolution: the phase rides cloudColor.x, is accumulated Java-side without a wrap (a wrap would
        // jump every cloud in the sky), and is pushed unconditionally like the drift so a tier toggle
        // cannot rewind the sky. The shader offsets each layer by a different multiple of it.
        assertTrue(volume.contains("float2 evolve;     // the field's evolution phase")
                        && volume.contains("f.evolve = float2(push.cloudColor.x, push.cloudColor.x * 0.618034);"),
                "the field frame must carry the evolution phase from the pushed lane");
        assertTrue(volume.contains("+ f.evolve * 0.06, 4);")
                        && volume.contains("q * 0.35 + 17.0 + f.evolve)")
                        && volume.contains("q * 0.25 + 7.0 + f.evolve * 1.7)")
                        && volume.contains("+ 31.0 + f.evolve * 2.3);"),
                "each field layer must crawl at its own multiple of the phase, large scale slowest");
        assertTrue(volume.contains("q.xz += f.evolve * 0.25;")
                        && volume.contains("+ float3(f.evolve, 0.0) * 0.55, 1);"),
                "the 3D lobes must churn inside the coverage envelope");
        assertTrue(composite.contains("cloudEvolve += CLOUD_EVOLVE_SPEED * waveClockStep;")
                        && composite.contains("new Float4((float) cloudEvolve, 0f, 0f, 1f),"),
                "the phase must be accumulated on the wave clock and pushed unconditionally");
        assertTrue(!composite.contains("cloudEvolve %") && !composite.contains("cloudEvolve -= "),
                "the phase must never wrap: the lattices it offsets are not periodic");
    }
}

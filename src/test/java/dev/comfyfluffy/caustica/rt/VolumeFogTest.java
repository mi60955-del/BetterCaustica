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
        for (String flag : new String[]{"WORLD_FLAG_FOG", "WORLD_FLAG_CLOUD", "WORLD_FLAG_SPRAY",
                "WORLD_FLAG_VOLUME_SHADOW", "WORLD_FLAG_VOLUME_MARCH", "WORLD_FLAG_VOLUME_DEBUG",
                "WORLD_FLAG_CLOUD_VOLUMETRIC", "WORLD_FLAG_SHAFTS"}) {
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
        assertTrue(indirect.contains("if (CAUSTICA_VOLUME_PATH && airProbeSite && fogOn && segmentIsAir)"),
                "the fog in-scatter must be gated on the fog flag and on an air segment");
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
        assertTrue(volume.contains("float sigma = sigma0;"),
                "the analytic tier must integrate the unthinned profile");
        assertTrue(volume.contains("float sigmaEff = sigma;"),
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
     * The cloud layer's numeric contracts, ported from volume.slang's instance-grid model: the
     * occupancy gate never leaves [0, 1], stays empty at coverage 0 and saturates to a full grid
     * toward overcast; the shadow transmittance sits in (0, 1], is monotone non-increasing in both
     * sampled density and density multiplier, equals exactly 1 at zero density, and stands down for a
     * horizon-grazing body; the multi-scatter octave stack returns exactly 1 at tau 0 and decays
     * monotonically; the atlas layer index stays inside the shape's own layer band. These are the
     * invariants every consumer (sheet, NEE, fog sun legs) silently leans on.
     */
    @Test
    void cloudLayerInvariants() {
        // Java port of the occupancy gate: roll < saturate(c * (1 + patchiness * (cluster*2-1))),
        // pinned open at coverage >= 0.95. The grid must be empty at coverage 0, never hold more
        // than the coverage asks for, and be entirely full near 1.
        for (float c = 0.0f; c <= 1.0f + 1.0e-6f; c += 0.1f) {
            for (float patch = 0.0f; patch <= 1.0f + 1.0e-6f; patch += 0.25f) {
                for (float cluster = 0.0f; cluster <= 1.0f + 1.0e-6f; cluster += 0.125f) {
                    float gate = c >= 0.95f
                            ? 2.0f
                            : Math.clamp(c * (1.0f + patch * (cluster * 2.0f - 1.0f)), 0.0f, 1.0f);
                    assertTrue(gate >= 0.0f && gate <= 2.0f,
                            "the occupancy gate must stay bounded at c=" + c);
                    if (c <= 1.0e-6f) {
                        assertEquals(0.0f, gate, 1.0e-9f,
                                "coverage 0 must leave every grid cell empty");
                    }
                    if (c >= 0.95f) {
                        assertEquals(2.0f, gate, 1.0e-9f,
                                "near-full coverage must fill the whole grid");
                    } else {
                        assertTrue(gate <= c * (1.0f + patch) + 1.0e-6f,
                                "the gate may not exceed the coverage times its cluster swing");
                    }
                }
            }
        }

        // Java port of cloudShadowTransmittance's tau = d * density * 0.22 * thickness / sunY.
        final float extinction = 0.22f;
        for (float thickness : new float[]{0.5f, 12.0f, 64.0f}) {
            for (float densityMult : new float[]{0.0f, 0.5f, 1.0f, 3.0f}) {
                float previous = Float.MAX_VALUE;
                for (float d = 0.0f; d <= 1.0f + 1.0e-6f; d += 0.01f) {
                    float tau = d * densityMult * extinction * thickness / 0.6f;
                    float t = (float) Math.exp(-tau);
                    assertTrue(t > 0.0f && t <= 1.0f + 1.0e-6f,
                            "shadow transmittance must sit in (0, 1] at d=" + d
                                    + " density=" + densityMult + " thickness=" + thickness);
                    assertTrue(t <= previous + 1.0e-9f,
                            "shadow transmittance must not grow with density at d=" + d);
                    previous = t;
                    if (d <= 1.0e-6f) {
                        assertEquals(1.0f, t, 1.0e-9f,
                                "zero sampled density must cast no shadow at all");
                    }
                }
            }
        }

        // Java port of cloudSelfShadow: energy-normalised octaves, so tau = 0 returns exactly 1.
        for (float tau = 0.0f; tau <= 12.0f; tau += 0.25f) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            float weight = 1.0f;
            for (int i = 0; i < 3; i++) {
                sum += weight * (float) Math.exp(-tau * Math.pow(2.0, -i));
                weightSum += weight;
                weight *= 0.5f;
            }
            float self = sum / weightSum;
            assertTrue(self > 0.0f && self <= 1.0f + 1.0e-6f,
                    "the multi-scatter stack must sit in (0, 1] at tau=" + tau);
            if (tau <= 1.0e-6f) {
                assertEquals(1.0f, self, 1.0e-6f, "tau 0 must be exactly unshadowed");
            }
        }

        // The atlas layer index: layerF is clamped to RES-1.001 so the hand-made trilinear's second
        // tap stays inside the shape's own layer band (never a neighbour shape's first layer). The
        // shader's own vertical01 < 0 || > 1 gate means the arithmetic only ever sees [0, 1].
        final float res = 128.0f;
        for (float v = 0.0f; v <= 1.0f + 1.0e-6f; v += 0.01f) {
            float layerF = Math.min(v * (res - 1.0f), res - 1.001f);
            float l0 = (float) Math.floor(layerF);
            assertTrue(l0 >= 0.0f && l0 + 1.0f <= res - 1.0f + 1.0e-3f,
                    "the layer pair must stay inside the shape's band at v=" + v);
        }
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
        // The baked shape's index rides cloud1.x (mapped from the 0..1 slider across the atlas's
        // shape count), the wind azimuth cloud1.y, the instance jitter strength cloud1.z, the fade
        // cloud1.w; the drift itself is Java-accumulated into cloud2.zw (the v15 lesson: anchor the
        // lane by its neighbours).
        assertTrue(composite.contains(
                        "new Float4(cloudOn\n"
                                + "                                    ? (float) Math.round(CausticaConfig.Rt.Volume"
                                + ".CLOUD_TYPE.value()\n"
                                + "                                            * (cloudVolumes != null ? cloudVolumes.shapeCount() - 1 : 0))\n"
                                + "                                    : 0f,\n"
                                + "                            cloudOn ? (float) Math.toRadians(\n"
                                + "                                    CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE.value()) : 0f,\n"
                                + "                            cloudOn ? CausticaConfig.Rt.Volume"
                                + ".CLOUD_PATCHINESS.value() : 0f,\n"
                                + "                            cloudOn ? CausticaConfig.Rt.Volume"
                                + ".CLOUD_FADE_DISTANCE.value() : 0f),"),
                "cloud1 must carry the shape index, the drift azimuth, the jitter strength and the "
                        + "fade, with the drift gone to cloud2");
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
                        + "instance grid has no period) plus the Java-accumulated drift");
        // The baked-volume field: one hash-grid instance lookup, the atlas fetch (hand-lerped across
        // layers, because 2D-array filtering is nearest along the layer axis), and the shape index
        // read straight out of cloud1.x into the atlas layer base.
        assertTrue(volume.contains("float cloudInstanceDensity(WorldPush push, float2 driftedXZ, float vertical01)"),
                "every cloud consumer must read the one instance-grid density");
        assertTrue(volume.contains("cloudVolumes.SampleLevel(")
                        && volume.contains("float shape = clamp(floor(push.cloud1.x)")
                        && volume.contains("0.0, max(push.cloudColor.y, 0.0));")
                        && volume.contains("float layer = shape * CLOUD_VOLUME_RES + l0;"),
                "the density must fetch the pushed shape's atlas layers");
        assertTrue(volume.contains("float2 cellId = floor(driftedXZ / cell);")
                        && volume.contains("float2 drifted = pos.xz + sunDir.xz * t + worldPush.cloud2.xy + worldPush.cloud2.zw;"),
                "the instance grid must sample through the exact anchor and the drift");
        // The deck's light must be the derived celestial for body AND colour, with the level kept on the
        // pushed look constant: a literal physical level at a one-degree sun is a rounding error, which
        // left the deck black instead of burning. The colour is a FRACTION of the derived chromaticity —
        // at full strength a one-degree sunset colour is an impossible light at that level, and the ACES
        // highlight roll-off turned the deck khaki.
        assertTrue(volume.contains("CelestialLight light = dominantCelestialLight(state, transmittanceLut);")
                        && volume.contains("float3 lightIll = lightLevel * cloudLightTint(light.illuminance);")
                        && volume.contains("public static const float CLOUD_LIGHT_TINT = 0.5;")
                        && volume.contains("return lerp(float3(1.0, 1.0, 1.0), chroma, CLOUD_LIGHT_TINT);"),
                "the cloud tiers must take the light's colour from the derived body, trimmed, and its "
                        + "level from the pushed constant");
        assertTrue(volume.contains("skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0)) * INV_PI")
                        && volume.contains("clamp(push.cloudColor.w, 0.0, 2.0) * CLOUD_AMBIENT_GAIN;")
                        && volume.contains("max(cloudSelfShadow(tauSun * CLOUD_AMBIENT_OCCLUSION),")
                        && volume.contains("CLOUD_AMBIENT_FLOOR);"),
                "the deck's ambient must be the sky's hemisphere irradiance, with the weight clamped so "
                        + "a lane slip cannot relight the sky");
        // The slab's self-shadow runs the multi-scatter octave stack (energy-normalised, so tau = 0
        // still gives exactly 1): a single-scatter deck reads blue-grey by day and black at night.
        assertTrue(volume.contains("float selfT = cloudSelfShadow(tauSelf);")
                        && volume.contains("public float cloudSelfShadow(float tau)")
                        && volume.contains("sum += weight * exp(-tau * exp2(-float(i)));")
                        && volume.contains("CLOUD_ALBEDO * opacity * (phase * lightIll * selfT + ambient)"),
                "the sheet's self-shadow must use the energy-normalised multi-scatter octaves");
        // The instance grid's anti-repetition and clustering structure: quarter-turn + mirror poses
        // out of one texture, and a parent-cell hash that gathers instances into weather systems.
        assertTrue(volume.contains("float turn = floor(pose * 3.999);")
                        && volume.contains("if (frac(pose * 7.31) < 0.5) {"),
                "instances must rotate and mirror their volume so one texture dresses the sky");
        assertTrue(volume.contains("float cluster = lerp(")
                        && volume.contains("cloudGridHash(pf, 83.0), cloudGridHash(pf + float2(1.0, 0.0), 83.0), pu.x)")
                        && volume.contains("push.cloud0.x >= 0.95"),
                "the patchiness must cluster instances via bilinearly smoothed parent-cell hashes "
                        + "(a binary per-block hash drew straight grid-aligned fronts across the sky), "
                        + "saturating near full cover");
        // The volumetric tier's light march: the baked shape's own occlusion toward the body, feeding
        // the same multi-scatter stack the sheet's column estimate uses.
        assertTrue(volume.contains("float cloudLightTau(WorldPush push, float3 p, float3 lightDir, float boxH)")
                        && volume.contains("float tauSun = cloudLightTau(push, p, lightDir, boxH) * push.cloud0.w;"),
                "the march tier must light its samples through a secondary march of the baked field");

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
                "float tau = density * worldPush.cloud0.w * CLOUD_EXTINCTION * CLOUD_COLUMN_DEPTH / sunDir.y;"),
                "cloudShadowTransmittance must price the column every tier prices, free of the fade "
                        + "distance and of the thickness dial");

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
        assertTrue(volume.contains("stratumCloud = cloudShadowTransmittance(p, sunDir);")
                        && volume.contains("* (sunVis * sunFogT * stratumCloud);"),
                "the marched fog's stratum probe must measure the cloud shadow once per stratum and "
                        + "its sun leg must consume that measurement");
        // The underwater sun leg is the shaft layer now, and it carries the cloud shadow per probe —
        // the same field the terrain NEE reads, so a clouded sun thins the water shafts too.
        assertTrue(volume.contains(
                        "sun += vis * (weight * cloudShadowTransmittance(p, celestialLight.dir));"),
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
        int cloudColorCall = composite.indexOf("new Float4(CausticaConfig.Rt.Volume.CLOUD_SIZE.value(),");
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
     * The volumetric cloud tier's numeric contracts and wiring. The layer is the SAME body the planar
     * tier prices — same base, same instance field, and the column's optical depth still reads off
     * cloud0.z — so the invariants that matter are
     * that the walk's fade dissolves its OPACITY rather than its glow (the v17 black-band lesson, which
     * a walk reaches just as easily as a single crossing), that the expensive path is reachable only
     * from a primary ray while continuation rays get the cheap mean, and that the baked volume's shape
     * reaches the walk through one clamped, hand-lerped atlas fetch per sample.
     */
    @Test
    void volumetricCloudTierInvariants() throws IOException {
        String volume = Files.readString(VOLUME);
        String skyMiss = Files.readString(Path.of("shaders/pipelines/world/sky.rmiss.slang"));
        String composite = Files.readString(COMPOSITE);

        // The walk samples the shared instance field: one field, three consumers, so the deck the
        // player sees casts the shadow they see. Its vertical coordinate rides the instance box
        // (standing on the cloud base), and the density is exactly zero outside any instance's window.
        boolean hasWalkV = volume.contains("float v = (p.y - boxBottom) / boxH;");
        boolean hasWalkDensity = volume.contains("float density = cloudInstanceDensity(push,\n"
                + "                p.xz + push.cloud2.xy + push.cloud2.zw, v);");
        assertTrue(hasWalkV && hasWalkDensity,
                "the volumetric slab must be built from the same instance field the NEE and fog legs read"
                        + " (v=" + hasWalkV + ", density=" + hasWalkDensity
                        + ", length=" + volume.length() + ")");
        // The instance boxes give a cloud a body rather than a sheet: CLOUD_BOX_ASPECT of its
        // footprint tall, standing on the base so the baked flat bottom is the layer's base. The
        // walk's extinction is scaled by CLOUD_TAU_ASPECT over the live height, so a taller box
        // changes the shape without changing the column optical depth cloud0.z prices.
        assertTrue(volume.contains("float cloudSpan(WorldPush push) {\n"
                        + "    return clamp(push.cloudColor.x, 32.0, 2048.0);")
                        && volume.contains("return clamp(push.cloud0.z, 8.0, 1024.0);")
                        && volume.contains("float cloudBoxBottom(WorldPush push) {\n    return push.cloud0.y;")
                        && volume.contains("CLOUD_COLUMN_DEPTH / cloudBoxHeight(push)")
                        && volume.contains("float boxBottom = cloudBoxBottom(push);")
                        && volume.contains("steps = clamp(ceil((t1 - t0) / max(boxH / CLOUD_STEPS_PER_THICKNESS, 1.0)),"),
                "the walk must intersect an instance box sized only by the Cloud Size lane, with the "
                        + "slab's thickness entering the column through the extinction scale");
        assertTrue(volume.contains("float radius = pow(cc.x * cc.x * cc.x + cc.y * cc.y * cc.y, 1.0 / 3.0);")
                        && volume.contains("float window = 1.0 - smoothstep(0.96, 1.12, radius);")
                        && volume.contains("if (window <= 0.0) {"),
                "an instance's density must be exactly zero outside its rounded window (a plain box "
                        + "window read as square cards)");
        // The atlas fetch lerps two adjacent layers by hand (2D-array filtering is nearest along the
        // layer axis) and clamps the pair inside the shape's own band, so a top-of-slab sample can
        // never read the next shape's first layer.
        assertTrue(volume.contains(
                        "float layerF = min(vertical01 * (CLOUD_VOLUME_RES - 1.0), CLOUD_VOLUME_RES - 1.001);")
                        && volume.contains("float a = cloudVolumes.SampleLevel(float3(local.x, local.y, layer), 0.0).r;")
                        && volume.contains("return lerp(a, b, layerF - l0) * window;"),
                "the density fetch must hand-lerp two clamped atlas layers");
        // The grid cell grows with the slab so the footprint always fits: a thick slab means bigger
        // clouds AND proportionally bigger gaps, from the same field and the same constants.
        assertTrue(volume.contains("float cloudGridCell(WorldPush push) {")
                        && volume.contains("return max(CLOUD_GRID_CELL, cloudSpan(push) / CLOUD_BOX_SPAN);"),
                "the instance cell must grow with the slab, holding the thickness-proportional footprint");

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
        // The light march: the baked shape's own occlusion toward the body, graded through the
        // multi-scatter octaves; the ambient leg rides the same tau softened. It is a bounded few
        // taps with an early-out, not an unbounded walk — a grazing sun would otherwise march the
        // whole sky per in-cloud sample.
        assertTrue(volume.contains("float tauSun = cloudLightTau(push, p, lightDir, boxH) * push.cloud0.w;")
                        && volume.contains("float3 sunT = cloudSelfShadow(tauSun);")
                        && volume.contains("float ambT = max(cloudSelfShadow(tauSun * CLOUD_AMBIENT_OCCLUSION),")
                        && volume.contains("public static const float CLOUD_AMBIENT_OCCLUSION = 0.20;"),
                "each sample's light must come from a march through the baked field, with the "
                        + "multi-scatter octave stack on top and the ambient leg softened");
        assertTrue(volume.contains("for (int s = 0; s < CLOUD_LIGHT_STEPS; s++) {")
                        && volume.contains("if (tau > 4.0) {"),
                "the light march must be a bounded few taps that early-outs once the stack saturates");
        // The phase must carry the Beer-Powder term AND the multi-scatter share that keeps the lobe's own
        // two orders of magnitude from drawing one deck white on one side of the sky and black on the other.
        assertTrue(volume.contains("hg(cosTheta, CLOUD_G) * 0.7 * (2.0 / (1.0 + exp(-2.0 * cosTheta)));")
                        && volume.contains("float phase = CLOUD_SCATTER_FLOOR + CLOUD_SCATTER_LOBE")
                        && volume.contains("public static const float CLOUD_SCATTER_FLOOR = 0.15;")
                        && volume.contains("public static const float CLOUD_SCATTER_LOBE = 0.85;"),
                "the samples must carry the plan's Beer-Powder phase term, compressed onto the isotropic "
                        + "multi-scatter share");

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

        // The tier table, Java ported: (cloud, cloudVolumetric, fog, fogShadow, fogMarch, spray) for
        // tiers 0..4.
        boolean[][] on = new boolean[5][];
        int[] steps = new int[5];
        float[] spray = new float[5];
        for (int t = 0; t <= 4; t++) {
            on[t] = new boolean[]{t >= 1, t >= 2, t >= 3, t >= 3, t >= 2, t >= 4, t >= 1, t >= 4};
            steps[t] = t == 2 ? 12 : (t == 4 ? 40 : 24);
            spray[t] = t >= 4 ? 0.5f : 0.0f;
        }
        // Off clears everything: clouds, the volumetric tier, the fog layer and its shadow, and the water
        // medium's scattering (which is what the underwater in-scatter is gated on).
        for (boolean b : on[0]) {
            assertTrue(!b, "tier 0 must clear every volumetric switch");
        }
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
                "CausticaConfig.Rt.Composite.WATER_SPRAY.set("}) {
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
     * Two behaviours that make the deck a weather system rather than a decal: rain thickens it, and
     * the layer reads one baked shape whose index the type slider selects. The thickening is computed
     * Java-side and pushed, which is what keeps every consumer - sheet, walk, shadow field, NEE, fog
     * legs - on the same lifted numbers; the test ports those formulas and pins the wiring on both
     * sides of the push, and pins the shape index's path from the slider to the atlas layer base.
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

        // The baked-volume layer has no evolving procedural field left — the shape selection replaced
        // the evolution phase in cloudColor.x — so the old accumulator and its lane must be gone, and
        // the shape index must be what rides cloud1.x instead.
        assertTrue(!volume.contains("f.evolve")
                        && volume.contains("clamp(push.cloudColor.x, 32.0, 2048.0)"),
                "cloudColor.x carries the Cloud Size lane, not an evolution phase: the shader must read "
                        + "it as the instance footprint and nothing else");
        assertTrue(!composite.contains("cloudEvolve") && !composite.contains("CLOUD_EVOLVE_SPEED"),
                "the evolution accumulator must be gone with the procedural field");
        assertTrue(composite.contains("(RtCloudVolumes.SHAPE_COUNT - 1))")
                        || composite.contains("cloudVolumes.shapeCount() - 1"),
                "the shape index must be mapped across the atlas's shape count");
        assertTrue(volume.contains("float layer = shape * CLOUD_VOLUME_RES + l0;")
                        && composite.contains("new Float4(CausticaConfig.Rt.Volume.CLOUD_SIZE.value(),\n"
                                + "                            cloudVolumes != null ? cloudVolumes.shapeCount() - 1 : 0f, 0f, 1f),"),
                "the shape index must reach the atlas as the layer base, with the atlas's last shape "
                        + "index riding cloudColor.y for the variety clamp");
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
     * The water column's only volumetric term is the shaft layer, on a medium of its own. The old
     * turbidity-driven pair — a sky-lit ambient and a single-probe sun leg — is gone, and with it the
     * milky water and the scattering fraction that had to be on for the water to have any body.
     */
    @Test
    void underwaterColumnIsTheShaftLayer() throws IOException {
        String volume = Files.readString(VOLUME);
        assertTrue(!volume.contains("waterSegmentInScatter"),
                "the old water volume function must be gone, not left as dead code");
        assertTrue(!volume.contains("WATER_AMBIENT_SURFACE"),
                "the water ambient constant must be gone with the ambient term");
        assertTrue(!volume.contains("sigmaS / sigmaT"),
                "nothing may still scale a water term by a scattering fraction");
        // The lobe is capped and the gate keeps a floor: without the cap a grazing look at the surface
        // goes white, without the floor the gate's product is a hairline band.
        assertTrue(volume.contains("phase = min(phase, worldPush.waterShaft.y);"),
                "the underwater lobe must be capped by the pushed ceiling");
        assertTrue(volume.contains("structure = lerp(WATER_SHAFT_STRUCTURE_FLOOR, 1.0, structure);"),
                "the underwater gate must keep a floor");
        assertTrue(volume.contains("float fogT = underwater ? 1.0 : fogTransmittance("),
                "the air fog must not attenuate the underwater term");
        assertTrue(volume.contains("vis *= exp(-sigma * probe.waterHitT);"),
                "every underwater probe must carry its own Beer-Lambert up to the surface");
        assertTrue(volume.contains("caustic = waterWaves"),
                "the surface focus must ride the wave field while it is on");
        // Both call sites run the layer directly on the water medium, with no scattering fraction.
        String indirect = Files.readString(Path.of("shaders/pipelines/world/indirect.rgen.slang"));
        String primary = Files.readString(Path.of("shaders/pipelines/world/primary.rgen.slang"));
        for (String source : new String[]{indirect, primary}) {
            assertTrue(source.contains("medium.current.water && shaftOn")
                            || source.contains("medium.current.water && firstSegment && shaftOn && firstSample"),
                    "the water term must gate on the water medium and the shaft switch only");
            assertTrue(source.contains("worldPush.waterShaft.x"),
                    "the water term must integrate the dedicated water medium");
        }
        String config = Files.readString(CONFIG);
        assertTrue(!config.contains("WATER_TURBIDITY"),
                "the scattering setting is gone with the effect it drove");
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
        assertTrue(config.contains("bool(\"caustica.rt.volume.shafts\", \"volume.shafts\", true)"),
                "the shaft master switch ships on: a clear-air effect nobody can see is not shipped");
        String settings = Files.readString(SETTINGS);
        assertTrue(settings.contains("\"volume.shafts\", RT + \"shafts\""),
                "the shaft switch must be reachable from the settings surface");
    }
}

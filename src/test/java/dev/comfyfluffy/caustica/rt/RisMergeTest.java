package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Invariants of the bounce-0 RIS reuse merge.
 *
 * Two kinds of check live here, and they fail in different ways:
 *
 * - The source-pairing tests read the real shader and the Java that allocates its images, and assert the
 *   two agree. Those contracts cross a language boundary, so nothing else in the build can see them break;
 *   a reservoir texel whose declared format disagrees with the allocated one makes every lane access alias
 *   its neighbour, which reads as noise no compiler can attribute.
 * - The arithmetic tests pin the algebra the merge is specified to implement. They run against a Java port,
 *   so they cannot detect a divergence inside lighting.slang on their own — the SPIR-V compile, the binding
 *   generator and a fixed-pose in-game A/B cover that. What they do catch is a change to the rules
 *   themselves that breaks an invariant the estimator depends on, and they are the cheap place to discover
 *   that before writing shader code.
 */
final class RisMergeTest {
    private static final Path SHADER = Path.of("shaders/pipelines/world/lighting.slang");
    private static final Path BINDINGS = Path.of("shaders/pipelines/world/bindings.slang");
    private static final Path COMPOSITE =
            Path.of("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");

    /** Slang image format -> the Vulkan format that allocates exactly those bytes per texel. */
    private static String expectedVkFormat(String slangFormat) {
        switch (slangFormat) {
            case "rgba32f": return "R32G32B32A32_SFLOAT";
            case "rgb32f":  return "R32G32B32_SFLOAT";
            case "rg32f":   return "R32G32_SFLOAT";
            case "r32f":    return "R32_SFLOAT";
            case "rgba16f": return "R16G16B16A16_SFLOAT";
            case "rg16f":   return "R16G16_SFLOAT";
            case "r16f":    return "R16_SFLOAT";
            default: throw new IllegalArgumentException("unmapped image format " + slangFormat);
        }
    }

    /** Slang image format -> the vector width its RWTexture2D declaration must use. */
    private static int expectedLanes(String slangFormat) {
        if (slangFormat.startsWith("rgba")) return 4;
        if (slangFormat.startsWith("rgb")) return 3;
        if (slangFormat.startsWith("rg")) return 2;
        return 1;
    }

    private static String read(Path path) throws IOException {
        assertTrue(Files.exists(path),
                () -> path + " not found: tests run from the project directory");
        return Files.readString(path);
    }

    @Test
    void reservoirImageFormatsMatchBetweenShaderAndAllocation() throws IOException {
        String bindings = read(BINDINGS);
        String composite = read(COMPOSITE);
        Matcher declared = Pattern
                .compile("\\[format\\(\"(\\w+)\"\\)]\\s*public\\s+RWTexture2D<float(\\d)>\\s+(reservoir\\w*);")
                .matcher(bindings);
        int checked = 0;
        while (declared.find()) {
            String format = declared.group(1);
            int lanes = Integer.parseInt(declared.group(2));
            String variable = declared.group(3);
            assertEquals(expectedLanes(format), lanes,
                    () -> variable + " declares " + format + " but a float" + lanes + " texel");
            // The allocation site names each image with a label starting "reservoir", and the sample pair
            // with "reservoir sample", so the label picks which declaration the format must agree with.
            String labelPrefix = variable.startsWith("reservoirSample") ? "reservoir sample " : "reservoir ";
            String vk = firstVkFormatForLabel(composite, labelPrefix);
            assertEquals(expectedVkFormat(format), vk,
                    () -> variable + " is declared " + format + " in the shader but allocated as " + vk);
            checked++;
        }
        assertEquals(4, checked, "expected prev/cur plus the sample pair to be declared");
    }

    private static String firstVkFormatForLabel(String composite, String labelPrefix) {
        Matcher m = Pattern.compile("VK_FORMAT_(\\w+),\\s*\\n\\s*\"" + Pattern.quote(labelPrefix) + "[AB]")
                .matcher(composite);
        assertTrue(m.find(), "no allocation labelled \"" + labelPrefix.trim() + " A/B\" found");
        return m.group(1);
    }

    @Test
    void persistedSecondLaneIsTheWeightTheSurvivorWasSelectedUnder() throws IOException {
        String shader = read(SHADER);
        // The merge reads lane 1 as a W and multiplies it by p-hat and an admitted count. Storing the raw
        // weight sum there instead would silently double-normalise, so the writer and the reader have to be
        // pinned together.
        assertTrue(Pattern.compile("stream\\.W\\s*=\\s*stored\\.y;").matcher(shader).find(),
                "lane 1 is read as the weight the survivor was selected under; expected stream.W = stored.y");
        assertTrue(Pattern.compile("reservoirCur\\[pix]\\s*=\\s*float4\\(float\\(\\w+\\.lightIndex\\),\\s*"
                        + "\\w+\\.W,\\s*\\w+\\.M,\\s*depth\\);")
                        .matcher(shader).find(),
                "lane 1 must be written from the reservoir's W, not its wSum");
    }

    // ---------------------------------------------------------------------------------------------
    // The merge algebra. Mirrors reuseInto() and the two admitted-count rules in lighting.slang.

    private static final class Res {
        float m;       // effective sample count
        float w;       // normalisation the survivor was selected under
        float wSum;    // accumulated stream weight
        float phat;    // survivor's target function at the shading point that selected it

        Res(float m, float wSum, float phat) {
            this.m = m;
            this.wSum = wSum;
            this.phat = phat;
            this.w = phat > 0f ? wSum / (m * phat) : 0f;
        }
    }

    /** Admits {@code stream} into {@code best}, drawing the keep decision from {@code draw}. */
    private static void reuseInto(Res best, Res stream, float admitted, Random draw) {
        if (admitted <= 0f) {
            return;
        }
        float w = stream.w * stream.phat * admitted;
        if (w <= 0f) {
            return;
        }
        float wSum = best.wSum + w;
        float m = best.m + admitted;
        if (draw.nextFloat() * wSum < w) {
            best.m = m;
            best.wSum = wSum;
            best.phat = stream.phat;
            best.w = wSum / (m * best.phat);
        } else {
            best.m = m;
            best.wSum = wSum;
            best.w = wSum / (m * best.phat);
        }
    }

    private static float relativeError(float actual, float expected) {
        return Math.abs(actual - expected) / Math.max(Math.abs(expected), Float.MIN_NORMAL);
    }

    @Test
    void weightProductIsTheWeightSumWhenTheTwoTargetFunctionsAgree() {
        Random r = new Random(20261001);
        for (int i = 0; i < 200_000; i++) {
            float m = (float) (8 * Math.pow(2, r.nextDouble() * 4));
            float wSum = (float) Math.pow(10, (r.nextDouble() - 0.5) * 12);
            float phat = (float) Math.pow(10, (r.nextDouble() - 0.5) * 12);
            Res stream = new Res(m, wSum, phat);
            // W * p-hat * admitted must equal the plain weight-sum merge for the same admitted count.
            float admitted = m * r.nextFloat();
            float product = stream.w * stream.phat * admitted;
            float legacy = wSum / m * admitted;
            assertTrue(relativeError(product, legacy) < 1e-5f,
                    () -> "W*p-hat*M diverges from the weight sum: " + product + " vs " + legacy);
        }
    }

    @Test
    void mergeScalesByTheRatioOfCurrentToOriginTargetFunction() {
        Random r = new Random(7);
        for (int i = 0; i < 50_000; i++) {
            float m = 8f + 8f * r.nextInt(8);
            float wSum = (float) Math.pow(10, (r.nextDouble() - 0.5) * 8);
            float originPhat = (float) Math.pow(10, (r.nextDouble() - 0.5) * 8);
            float ratio = (float) Math.pow(10, (r.nextDouble() - 0.5) * 6);
            Res asOrigin = new Res(m, wSum, originPhat);
            Res asCurrent = new Res(m, wSum, originPhat);
            asCurrent.phat = originPhat * ratio;   // the survivor re-evaluated at a different shading point
            float admitted = m;
            float wAtOrigin = asOrigin.w * asOrigin.phat * admitted;
            float wAtCurrent = asCurrent.w * asCurrent.phat * admitted;
            assertTrue(relativeError(wAtCurrent, wAtOrigin * ratio) < 1e-5f,
                    () -> "reweighting did not track the p-hat ratio: " + wAtCurrent + " vs "
                            + (wAtOrigin * ratio));
        }
    }

    @Test
    void ceilingAtTheCandidateCountAdmitsNothingAndLeavesTheReservoirUntouched() {
        Random r = new Random(11);
        for (int i = 0; i < 20_000; i++) {
            float curM = 8f;
            Res best = new Res(curM, (float) Math.pow(10, (r.nextDouble() - 0.5) * 8),
                    (float) Math.pow(10, (r.nextDouble() - 0.5) * 8));
            float beforeW = best.w, beforeSum = best.wSum, beforeM = best.m;
            Res history = new Res(64f, 1e4f, 1e3f);
            // The dial's low end is the self-check: a ceiling equal to this frame's own candidate count must
            // degenerate exactly to single-frame shading.
            float ceiling = Math.max(8f, curM);
            reuseInto(best, history, Math.min(history.m, ceiling - curM), r);
            assertEquals(beforeM, best.m);
            assertEquals(beforeSum, best.wSum);
            assertEquals(beforeW, best.w);
        }
    }

    @Test
    void effectiveCountStopsAtTheCeilingAndTheWeightStopsGrowing() {
        Random r = new Random(3);
        float ceiling = 64f;
        float curM = 8f;
        Res history = new Res(curM, 500f, 40f);
        float maxSeen = 0f;
        for (int frame = 0; frame < 400; frame++) {
            // Each frame starts from its own M candidates and admits at most (ceiling - M) of the history,
            // so the accumulated count is bounded no matter how long a pixel stands still.
            Res current = new Res(curM, 500f, 40f);
            reuseInto(current, new Res(history.m, history.wSum, history.phat),
                    Math.min(history.m, Math.max(ceiling, curM) - curM), r);
            history = current;
            maxSeen = Math.max(maxSeen, current.m);
            int at = frame;
            assertTrue(current.m <= ceiling + 1e-4f,
                    () -> "effective count escaped the ceiling at frame " + at + ": " + current.m);
            assertTrue(current.w < 1e6f, () -> "weight ran away at frame " + at + ": " + current.w);
        }
        assertEquals(ceiling, maxSeen, 1e-3f, "the ceiling should be reached, not merely respected");
    }

    @Test
    void keepDecisionIsProportionalToTheAdmittedWeight() {
        Random r = new Random(101);
        int draws = 200_000;
        int taken = 0;
        for (int i = 0; i < draws; i++) {
            Res best = new Res(8f, 30f, 5f);
            Res stream = new Res(16f, 20f, 4f);
            reuseInto(best, stream, 16f, r);
            if (best.phat == stream.phat) {
                taken++;
            }
        }
        float expected = (streamWeight(20f, 16f, 4f)) / (30f + streamWeight(20f, 16f, 4f));
        float observed = taken / (float) draws;
        assertEquals(expected, observed, 0.005f,
                "the survivor should be taken with probability w_stream / (w_best + w_stream)");
    }

    private static float streamWeight(float wSum, float m, float phat) {
        return (wSum / (m * phat)) * phat * m;
    }

    @Test
    void eachSpatialNeighbourIsCappedAtOneFramesWorthOfCandidates() {
        Random r = new Random(13);
        for (int k = 1; k <= 6; k++) {
            float curM = 8f;
            Res best = new Res(curM, 40f, 6f);
            for (int j = 0; j < k; j++) {
                Res neighbour = new Res(64f, 9e3f, 5f);   // a neighbour with a long history
                reuseInto(best, neighbour, Math.min(neighbour.m, curM), r);
            }
            // Every neighbour is capped at this frame's own candidate count however long its own history
            // is, so no single one can dominate the survivor and the total stays a multiple of M.
            int neighbours = k;
            assertEquals(curM * (neighbours + 1), best.m, 1e-3f,
                    () -> neighbours + " neighbours should lift the effective count to "
                            + (neighbours + 1) + " frames' worth");
            assertTrue(best.w > 0f && !Float.isInfinite(best.w));
        }
    }

    @Test
    void weightTimesEffectiveCountTimesTargetFunctionRecoversTheWeightSum() {
        Random r = new Random(29);
        List<Float> errors = new ArrayList<>();
        for (int i = 0; i < 50_000; i++) {
            Res best = new Res(8f, (float) Math.pow(10, (r.nextDouble() - 0.5) * 6),
                    (float) Math.pow(10, (r.nextDouble() - 0.5) * 6));
            Res stream = new Res((float) (4 + r.nextInt(60)), (float) Math.pow(10, (r.nextDouble() - 0.5) * 6),
                    (float) Math.pow(10, (r.nextDouble() - 0.5) * 6));
            reuseInto(best, stream, (float) (1 + r.nextInt(30)), r);
            // shadeReservoir multiplies the contribution by W alone, so W has to carry the whole 1/M
            // normalisation: W * M * p-hat == wSum is the closure the estimator depends on.
            errors.add(relativeError(best.w * best.m * best.phat, best.wSum));
        }
        errors.sort(null);
        assertTrue(errors.get(errors.size() - 1) < 1e-3f,
                () -> "closure W*M*p-hat == wSum broken, worst relative error " + errors.get(errors.size() - 1));
    }
}

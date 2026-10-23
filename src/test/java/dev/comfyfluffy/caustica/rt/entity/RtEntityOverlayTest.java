package dev.comfyfluffy.caustica.rt.entity;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The hurt / spawn-flash overlay survives two hand-offs: vanilla's packed OverlayTexture coordinates become
 * prim flag bits in {@link RtEntityOverlay}, and {@code world_common.slang}'s {@code entityOverlayRgba}
 * turns those bits back into a texel. These tests pin both ends.
 *
 * <p>Two kinds of check, mirroring {@code VolumeFogTest}'s split:
 *
 * <ul>
 *   <li>The arithmetic tests rebuild OverlayTexture's own fill — a 16x16 dynamic texture whose hurt row is
 *       {@code 0xB2FF0000} at every column and whose other rows are
 *       {@code ARGB.white((int) ((1 - x / 15 * 0.75) * 255))} — and compare it against the shader's closed
 *       form for every column. They fail if the ramp or the hurt alpha drifts away from vanilla's, which
 *       the shader can only express analytically.
 *   <li>The source-pairing test reads the shader and asserts its flag bit positions are the ones this class
 *       writes, and that they avoid the bits the pipeline already owns.
 * </ul>
 */
final class RtEntityOverlayTest {
    private static final Path WORLD_COMMON = Path.of("shaders/pipelines/world/world_common.slang");

    /** {@code OverlayTexture.SIZE - 1}: the highest column {@code OverlayTexture.u(float)} can produce. */
    private static final int LAST_COLUMN = 15;
    /** {@code OverlayTexture.RED_OVERLAY_V} — sampled by every submission of a hurt or dying entity. */
    private static final int HURT_ROW = 3;
    /** {@code OverlayTexture.WHITE_OVERLAY_V} — sampled by everything else. */
    private static final int WHITE_ROW = 10;
    /** {@code OverlayTexture.NO_OVERLAY} = {@code pack(NO_WHITE_U, WHITE_OVERLAY_V)}. */
    private static final int NO_OVERLAY = pack(0, WHITE_ROW);
    /** {@code OverlayTexture}'s hurt fill, as an ARGB int. */
    private static final int HURT_TEXEL = 0xB2FF0000;

    @Test
    void everyHurtColumnResolvesToTheHurtFill() {
        for (int column = 0; column <= LAST_COLUMN; column++) {
            float[] overlay = shaderOverlay(RtEntityOverlay.encode(pack(column, HURT_ROW)));
            assertEquals(1f, overlay[0], "red channel, column " + column);
            assertEquals(0f, overlay[1], "green channel, column " + column);
            assertEquals(0f, overlay[2], "blue channel, column " + column);
            assertEquals(channel(HURT_TEXEL, 24), overlay[3], 1f / 255f, "alpha, column " + column);
        }
    }

    @Test
    void whiteRowAlphaMatchesVanillasRamp() {
        for (int column = 0; column <= LAST_COLUMN; column++) {
            float[] overlay = shaderOverlay(RtEntityOverlay.encode(pack(column, WHITE_ROW)));
            assertEquals(1f, overlay[0], "white rgb is column-independent");
            assertEquals(1f, overlay[1], "white rgb is column-independent");
            assertEquals(1f, overlay[2], "white rgb is column-independent");
            assertEquals(channel(vanillaTexel(column, WHITE_ROW), 24), overlay[3], 1f / 255f,
                    "alpha, column " + column);
        }
    }

    @Test
    void columnZeroIsVanillasNoOp() {
        assertEquals(0, RtEntityOverlay.encode(NO_OVERLAY), "NO_OVERLAY must encode to no flag bits");
        assertEquals(0, RtEntityOverlay.encode(pack(0, WHITE_ROW)));
        float[] overlay = shaderOverlay(0);
        assertEquals(1f, overlay[3], "the no-op entry has to leave mix() at the tinted colour");
    }

    @Test
    void columnIsClippedInsteadOfWrapping() {
        // A progress above 1 must saturate at the strongest white, not alias back to a weaker overlay.
        assertEquals(RtEntityOverlay.encode(pack(LAST_COLUMN, WHITE_ROW)),
                RtEntityOverlay.encode(pack(LAST_COLUMN + 1, WHITE_ROW)));
        assertEquals(RtEntityOverlay.encode(pack(LAST_COLUMN, HURT_ROW)),
                RtEntityOverlay.encode(pack(99, HURT_ROW)));
    }

    @Test
    void overlayBitsAvoidThePipelineFlags() {
        // world_common.slang's ENTITY_PRIM_UNLIT (bit 0) and ENTITY_PRIM_END_PORTAL (bit 1) are assigned by
        // the collector's own pipelines; the overlay must not be able to overwrite either.
        assertEquals(0, RtEntityOverlay.MASK & 0b11, "overlay bits must start above the pipeline flags");
        assertTrue(RtEntityOverlay.MASK != 0, "the overlay has to occupy at least one bit");
    }

    @Test
    void shaderDecodesTheBitsThisClassWrites() throws IOException {
        String shader = Files.readString(WORLD_COMMON);
        assertEquals(RtEntityOverlay.U_SHIFT, shaderConstant(shader, "ENTITY_PRIM_OVERLAY_U_SHIFT"));
        assertEquals(RtEntityOverlay.U_MASK, shaderConstant(shader, "ENTITY_PRIM_OVERLAY_U_MASK"));
        assertEquals(RtEntityOverlay.HURT, shaderConstant(shader, "ENTITY_PRIM_OVERLAY_HURT"));
        assertEquals(RtEntityOverlay.MASK,
                (RtEntityOverlay.U_MASK << RtEntityOverlay.U_SHIFT) | RtEntityOverlay.HURT);
        // The bits the overlay sits beside, and the shader that has to consume it.
        assertEquals(1, shaderConstant(shader, "ENTITY_PRIM_UNLIT"));
        assertEquals(2, shaderConstant(shader, "ENTITY_PRIM_END_PORTAL"));
        assertTrue(shader.contains("public float4 entityOverlayRgba(uint flags)"),
                "world_common.slang no longer decodes the overlay");
    }

    /** {@code OverlayTexture.pack}: column in the low half, row in the high half, as UV1 carries them. */
    private static int pack(int column, int row) {
        return column | (row << 16);
    }

    /** OverlayTexture's constructor fill: the texel a submission sampling (column, row) would receive. */
    private static int vanillaTexel(int column, int row) {
        if (row < 8) {
            return HURT_TEXEL;
        }
        return ((int) ((1f - column / (float) LAST_COLUMN * 0.75f) * 255f) << 24) | 0xFFFFFF;
    }

    /** Mirrors world_common.slang's {@code entityOverlayRgba}: {rgb, alpha} for a prim flag word. */
    private static float[] shaderOverlay(int flags) {
        if ((flags & RtEntityOverlay.HURT) != 0) {
            return new float[] {1f, 0f, 0f, 178f / 255f};
        }
        int column = (flags >>> RtEntityOverlay.U_SHIFT) & RtEntityOverlay.U_MASK;
        return new float[] {1f, 1f, 1f, 1f - 0.05f * column};
    }

    private static float channel(int argb, int shift) {
        return ((argb >>> shift) & 0xFF) / 255f;
    }

    /** Read one top-level {@code public static const uint} initializer out of a shader source file. */
    private static int shaderConstant(String source, String name) {
        Matcher matcher = Pattern.compile(
                "public static const uint " + name + "\\s*=\\s*(0[xX][0-9a-fA-F]+|[0-9]+)[uU]?\\s*;")
                .matcher(source);
        assertTrue(matcher.find(), name + " is not declared as a uint in world_common.slang");
        String value = matcher.group(1);
        return value.startsWith("0x") || value.startsWith("0X")
                ? Integer.parseInt(value.substring(2), 16)
                : Integer.parseInt(value);
    }
}

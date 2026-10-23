package dev.comfyfluffy.caustica.rt.entity;

/**
 * The entity hurt / spawn-flash overlay, packed into a {@code Prim}'s flag word for the hit shader.
 *
 * <p>Vanilla hands every entity submission the {@code OverlayTexture} texel it must sample as UV1 (the
 * {@code overlayCoords} argument of {@code SubmitNodeCollector.submitModel}/{@code submitItem}/{@code
 * submitBlockModel}) and {@code entity.fsh} mixes that texel over the vertex-tinted colour:
 *
 * <pre>color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);</pre>
 *
 * <p>The capture shades from the entity texture directly, so the texel has to travel with the geometry
 * instead of being sampled. It rides the prim — not the entity record — because it is a per-SUBMISSION
 * property: the renderers that pass {@code OverlayTexture.NO_OVERLAY} (capes, eyes, block decorations,
 * item frames, …) stay untinted in vanilla even while their entity is hurt.
 *
 * <p>{@code OverlayTexture} is a 16x16 dynamic texture; {@code pack(u(white), v(hurt))} picks its column
 * and row, and the fill makes the meaning of the two reachable rows a closed form:
 *
 * <ul>
 *   <li>row {@code 3} (hurt / dying): every column is ARGB {@code 0xB2FF0000} — red at alpha 178/255.
 *   <li>row {@code 10} (not hurt): column {@code u} is {@code ARGB.white((int) ((1 - u / 15 * 0.75) * 255))}
 *       — white whose alpha ramps 1 → 0.25 with the spawn-flash progress. {@code u = 0} is therefore a
 *       no-op, which is exactly what {@code NO_OVERLAY = pack(0, 10)} resolves to, so anything that is not
 *       the hurt row only needs its column.
 * </ul>
 *
 * <p>Two fields — the column and "is this the hurt row" — fit in the flag word's spare bits beside {@code
 * ENTITY_PRIM_UNLIT}/{@code ENTITY_PRIM_END_PORTAL}. {@code world_common.slang} owns the decoder
 * ({@code entityOverlayRgba}); {@code RtEntityOverlayTest} reads that shader and pins the two
 * representations to the same bits.
 */
final class RtEntityOverlay {
    /** Bit 2 of the flag word: low bit of the {@code OverlayTexture} column the submission samples. */
    static final int U_SHIFT = 2;
    /** Four bits hold the column, which vanilla clamps to {@code OverlayTexture.SIZE - 1} = 15. */
    static final int U_MASK = 0xF;
    /** Bit 6: the submission samples the hurt row instead of the white one. */
    static final int HURT = 1 << 6;
    /** Every bit of the flag word this class owns; the rest belongs to the submission's pipeline. */
    static final int MASK = (U_MASK << U_SHIFT) | HURT;

    /** {@code OverlayTexture.v(true)} — the row a hurt or dying entity's submissions sample. */
    private static final int HURT_ROW = 3;

    private RtEntityOverlay() {
    }

    /**
     * Pack vanilla's {@code overlayCoords} (UV1 = {@code column | row << 16}) into the prim flag bits.
     * Vanilla's {@code u(float white)} is {@code (int) (white * 15)}, so an in-range column always fits in
     * {@link #U_MASK} bits; the clamp only bounds a caller that passes a progress above 1.
     */
    static int encode(int overlayCoords) {
        int column = overlayCoords & 0xFFFF;
        int row = (overlayCoords >>> 16) & 0xFFFF;
        return (Math.min(column, U_MASK) << U_SHIFT) | (row == HURT_ROW ? HURT : 0);
    }
}

package dev.comfyfluffy.caustica.rt.dh;

/**
 * Turns decoded DH LOD columns into flat-colour triangle soup for the ray tracer.
 *
 * <p>A region is a uniform grid of {@code cells × cells} cells (each {@code 2^dl} blocks). Every
 * non-air cell contributes its TOP face — the surface a distant view actually sees — plus side bands
 * where a lower neighbour exposes it, plus a downward "skirt" on the region's outer border so the
 * seams between regions of different cell sizes (rings coarsen outward) never read as cracks.
 *
 * <p>Greedy rectangle merging collapses runs of identical cells (same top height, colour, water
 * flag, light level) into single quads; real terrain merges aggressively, which is what keeps a
 * whole LOD field inside a modest triangle budget. Colours are compared quantised so neighbouring
 * grass/foliage cells — one biome tint step apart — still merge; at these view distances the
 * difference is invisible and the merged quad keeps the first cell's exact colour.
 *
 * <p>Output arrays are byte-compatible with the traced-terrain pipeline ({@code RtTerrainMesher
 * .PackedSection}): positions are region-local float3s, indices are uint triangles, and each
 * triangle carries one 12-float {@code TerrainPrim} record whose {@code materialId} is the LOD
 * sentinel and whose {@code tint.w} marks water. UVs are empty — the LOD hit shader never samples
 * the atlas — but a 2-float stub keeps the buffer allocation non-degenerate.
 */
public final class DhLodMesher {
    /** TerrainPrim.materialId sentinel, mirroring MATERIAL_ID_LOD in world_common.slang. */
    public static final int LOD_MATERIAL_ID_BITS = Float.floatToIntBits(Float.MAX_VALUE);
    /** TerrainPrim.flags bit 1, mirroring TERRAIN_PRIM_THIN_DIELECTRIC (unused for LOD today). */
    public static final int TERRAIN_PRIM_THIN_DIELECTRIC = 2;
    /** Colour quantisation: high bits kept for the merge key, full colour kept for the vertex. */
    private static final int COLOR_MERGE_SHIFT = 3;

    /** One merged mesh, ready for {@code RtSectionBuilder.prepare}. */
    public static final class Mesh {
        public final float[] positions;
        public final int[] indices;
        public final float[] uvs;
        public final float[] material;
        public final int[] bucketTris;
        public final int[] triBase;
        public final int triangleCount;

        Mesh(float[] positions, int[] indices, float[] material, int triangleCount) {
            this.positions = positions;
            this.indices = indices;
            this.uvs = new float[2];
            this.material = material;
            this.bucketTris = new int[]{triangleCount};
            this.triBase = new int[]{0, 0, 0, 0};
            this.triangleCount = triangleCount;
        }

        public boolean isEmpty() {
            return triangleCount == 0;
        }
    }

    private DhLodMesher() {
    }

    /**
     * @param columns    row-major {@code cells × cells} grid of decoded columns (from DhBridge)
     * @param cells      cells per region side (equals {@code 2^dl} of the source query)
     * @param cellSize   blocks per cell ({@code 2^dl})
     * @param baseY      world Y mapped to local 0 (the region's vertical base)
     * @param emission   whether block-light levels become heuristic emission
     * @param originX    region's world block origin X (for mask-rect coordinates)
     * @param originZ    region's world block origin Z
     * @param maskRects  flat world-space rects {minX, minZ, maxX, maxZ, …}; cells overlapping any of
     *                   them are dropped (finer LOD rings own that ground). Null = nothing masked.
     */
    public static Mesh mesh(DhBridge.Column[] columns, int cells, int cellSize, int baseY, boolean emission,
                            int originX, int originZ, long[] maskRects) {
        // Merge keys: a cell merges with its neighbour only when everything the shader reads matches.
        long[] keys = new long[cells * cells];
        boolean hasMask = maskRects != null && maskRects.length >= 4;
        for (int i = 0; i < cells * cells; i++) {
            DhBridge.Column c = columns[i];
            if (c.air) {
                keys[i] = -1L;
                continue;
            }
            if (hasMask) {
                int cx0 = originX + (i % cells) * cellSize;
                int cz0 = originZ + (i / cells) * cellSize;
                int cx1 = cx0 + cellSize - 1;
                int cz1 = cz0 + cellSize - 1;
                if (cellMasked(maskRects, cx0, cz0, cx1, cz1)) {
                    keys[i] = -1L;
                    continue;
                }
            }
            keys[i] = mergeKey(c);
        }

        Counters out = new Counters();
        // ---- top faces: greedy 2D rectangle decomposition over equal-key runs ----------------------
        // Standard scan: each row runs of equal keys, then vertical extension while the run repeats.
        boolean[] consumed = new boolean[cells * cells];
        for (int z = 0; z < cells; z++) {
            for (int x = 0; x < cells; ) {
                int idx = z * cells + x;
                long key = keys[idx];
                if (key < 0 || consumed[idx]) {
                    x++;
                    continue;
                }
                int runEnd = x;
                while (runEnd < cells && keys[z * cells + runEnd] == key && !consumed[z * cells + runEnd]) {
                    runEnd++;
                }
                // Extend upward while every row matches the full run.
                int zEnd = z + 1;
                extend:
                while (zEnd < cells) {
                    for (int xx = x; xx < runEnd; xx++) {
                        int nIdx = zEnd * cells + xx;
                        if (keys[nIdx] != key || consumed[nIdx]) {
                            break extend;
                        }
                    }
                    zEnd++;
                }
                for (int zz = z; zz < zEnd; zz++) {
                    for (int xx = x; xx < runEnd; xx++) {
                        consumed[zz * cells + xx] = true;
                    }
                }
                emitTop(out, columns[z * cells + x], x, z, runEnd - x, zEnd - z, cellSize, baseY, emission);
                x = runEnd;
            }
        }

        // ---- side bands + skirts --------------------------------------------------------------------
        // A side quad appears where a cell's top stands above its neighbour's top. The halo-free
        // variant: neighbours outside the grid count as "same height" (the neighbouring region owns
        // that face — its own skirt covers the seam).
        for (int z = 0; z < cells; z++) {
            for (int x = 0; x < cells; x++) {
                int idx = z * cells + x;
                if (keys[idx] < 0) {
                    continue;
                }
                DhBridge.Column c = columns[idx];
                // Water cells draw their surface only: a distant water sheet is a plane, and water
                // walls would shade as vertical dielectric filters.
                if (c.water) {
                    continue;
                }
                int topY = c.topY + 1; // surface height in world Y
                int bottomLimit = skirtBottom(c, cellSize);

                // Interior sides: band from neighbour top to this top.
                if (x > 0) {
                    DhBridge.Column n = columns[idx - 1];
                    if (!n.air && n.topY + 1 < topY) {
                        emitSide(out, c, x, z, cellSize, baseY, n.topY + 1, topY, Direction.WEST, emission);
                    }
                }
                if (x < cells - 1) {
                    DhBridge.Column n = columns[idx + 1];
                    if (!n.air && n.topY + 1 < topY) {
                        emitSide(out, c, x, z, cellSize, baseY, n.topY + 1, topY, Direction.EAST, emission);
                    }
                }
                if (z > 0) {
                    DhBridge.Column n = columns[idx - cells];
                    if (!n.air && n.topY + 1 < topY) {
                        emitSide(out, c, x, z, cellSize, baseY, n.topY + 1, topY, Direction.NORTH, emission);
                    }
                }
                if (z < cells - 1) {
                    DhBridge.Column n = columns[idx + cells];
                    if (!n.air && n.topY + 1 < topY) {
                        emitSide(out, c, x, z, cellSize, baseY, n.topY + 1, topY, Direction.SOUTH, emission);
                    }
                }

                // Region-border skirts: always present so a neighbouring region built from different
                // cell sizes (or with no data yet) can never open a crack at the seam.
                if (x == 0) {
                    emitSide(out, c, x, z, cellSize, baseY, bottomLimit, topY, Direction.WEST, emission);
                }
                if (x == cells - 1) {
                    emitSide(out, c, x, z, cellSize, baseY, bottomLimit, topY, Direction.EAST, emission);
                }
                if (z == 0) {
                    emitSide(out, c, x, z, cellSize, baseY, bottomLimit, topY, Direction.NORTH, emission);
                }
                if (z == cells - 1) {
                    emitSide(out, c, x, z, cellSize, baseY, bottomLimit, topY, Direction.SOUTH, emission);
                }
            }
        }

        return new Mesh(out.positions(), out.indices(), out.material(), out.triangles);
    }

    /** True when the cell rect overlaps any mask rect (finer-ring ground). */
    private static boolean cellMasked(long[] rects, int cx0, int cz0, int cx1, int cz1) {
        for (int i = 0; i + 3 < rects.length; i += 4) {
            if (cx0 <= rects[i + 2] && cx1 >= rects[i] && cz0 <= rects[i + 3] && cz1 >= rects[i + 1]) {
                return true;
            }
        }
        return false;
    }

    private static long mergeKey(DhBridge.Column c) {
        int r = (c.colorArgb >>> 16) & 0xFF, g = (c.colorArgb >>> 8) & 0xFF, b = c.colorArgb & 0xFF;
        long key = c.topY & 0xFFFFF;
        key |= (long) (r >> COLOR_MERGE_SHIFT) << 20;
        key |= (long) (g >> COLOR_MERGE_SHIFT) << 25;
        key |= (long) (b >> COLOR_MERGE_SHIFT) << 30;
        key |= (c.water ? 1L : 0L) << 35;
        key |= (long) c.light << 36;
        return key;
    }

    /**
     * Where a border cell's skirt bottoms out: a shallow band below the surface, just deep enough to
     * cover the height step against a neighbouring ring built at another cell size. It is a seam hider,
     * not a cliff — a cell-sized drop would stand in the skyline as a wall.
     */
    private static int skirtBottom(DhBridge.Column c, int cellSize) {
        return c.topY + 1 - Math.min(48, Math.max(16, cellSize / 4));
    }

    private enum Direction {
        WEST, EAST, NORTH, SOUTH
    }

    /** Reusable grow-on-demand arrays; a fresh instance per region keeps ownership simple. */
    private static final class Counters {
        private float[] positions = new float[1024];
        private int vertexFloats;
        private int[] indices = new int[512];
        private int indexCount;
        private float[] material = new float[512];
        private int materialFloats;
        int triangles;

        float[] positions() {
            float[] out = new float[vertexFloats];
            System.arraycopy(positions, 0, out, 0, vertexFloats);
            return out;
        }

        int[] indices() {
            int[] out = new int[indexCount];
            System.arraycopy(indices, 0, out, 0, indexCount);
            return out;
        }

        float[] material() {
            float[] out = new float[materialFloats];
            System.arraycopy(material, 0, out, 0, materialFloats);
            return out;
        }
    }

    private static void ensureQuads(Counters out, int extraQuads) {
        int needFloats = out.vertexFloats + extraQuads * 4 * 3;
        if (needFloats > out.positions.length) {
            out.positions = java.util.Arrays.copyOf(out.positions, Math.max(needFloats, out.positions.length * 2));
        }
        int needIndices = out.indexCount + extraQuads * 6;
        if (needIndices > out.indices.length) {
            out.indices = java.util.Arrays.copyOf(out.indices, Math.max(needIndices, out.indices.length * 2));
        }
        int needMaterial = out.materialFloats + extraQuads * 2 * 12;
        if (needMaterial > out.material.length) {
            out.material = java.util.Arrays.copyOf(out.material, Math.max(needMaterial, out.material.length * 2));
        }
    }

    /** Emit one top quad covering {@code w × h} cells starting at cell ({@code x}, {@code z}). */
    private static void emitTop(Counters out, DhBridge.Column c, int x, int z, int w, int h,
                                int cellSize, int baseY, boolean emission) {
        ensureQuads(out, 1);
        float y = (c.topY + 1 - baseY);
        float x0 = x * (float) cellSize;
        float z0 = z * (float) cellSize;
        float x1 = x0 + w * (float) cellSize;
        float z1 = z0 + h * (float) cellSize;
        int water = c.water ? 1 : 0;
        emitQuad(out, x0, y, z0, x1, y, z1,
                0, 1, 0,
                c.colorArgb, water, emission ? c.light : 0);
    }

    /**
     * Emit one vertical side band. {@code yLow}/{@code yHigh} are world Y; the quad spans the cell's
     * full face on the given side, whatever the band height.
     */
    private static void emitSide(Counters out, DhBridge.Column c, int x, int z, int cellSize, int baseY,
                                 int yLow, int yHigh, Direction dir, boolean emission) {
        if (yHigh <= yLow) {
            return;
        }
        ensureQuads(out, 1);
        float x0 = x * (float) cellSize;
        float z0 = z * (float) cellSize;
        float x1 = x0 + cellSize;
        float z1 = z0 + cellSize;
        float y0 = yLow - baseY;
        float y1 = yHigh - baseY;
        int water = c.water ? 1 : 0;
        switch (dir) {
            case WEST -> emitQuad(out, x0, y0, z0, x0, y1, z1, -1, 0, 0,
                    c.colorArgb, water, emission ? c.light : 0);
            case EAST -> emitQuad(out, x1, y0, z0, x1, y1, z1, 1, 0, 0,
                    c.colorArgb, water, emission ? c.light : 0);
            case NORTH -> emitQuad(out, x0, y0, z0, x1, y1, z0, 0, 0, -1,
                    c.colorArgb, water, emission ? c.light : 0);
            case SOUTH -> emitQuad(out, x0, y0, z1, x1, y1, z1, 0, 0, 1,
                    c.colorArgb, water, emission ? c.light : 0);
        }
    }

    /**
     * One quad = two triangles + two identical 12-float prim records (per-triangle records, matching
     * the traced-terrain ABI). The hit shader flips the normal toward the viewer, so winding only
     * feeds the BLAS; the normal vector is the face's canonical outward direction. {@code a} and
     * {@code b} are two opposite corners of the axis-aligned rectangle.
     */
    private static void emitQuad(Counters out, float ax, float ay, float az,
                                 float bx, float by, float bz,
                                 float nx, float ny, float nz,
                                 int colorArgb, int water, int light) {
        int base = out.vertexFloats / 3;
        float[] corners = quadCorners(ax, ay, az, bx, by, bz, nx, ny, nz);

        float[] p = out.positions;
        int vi = out.vertexFloats;
        System.arraycopy(corners, 0, p, vi, 12);
        out.vertexFloats += 12;

        int[] idx = out.indices;
        idx[out.indexCount] = base;
        idx[out.indexCount + 1] = base + 1;
        idx[out.indexCount + 2] = base + 2;
        idx[out.indexCount + 3] = base;
        idx[out.indexCount + 4] = base + 2;
        idx[out.indexCount + 5] = base + 3;
        out.indexCount += 6;

        float cr = ((colorArgb >>> 16) & 0xFF) / 255f;
        float cg = ((colorArgb >>> 8) & 0xFF) / 255f;
        float cb = (colorArgb & 0xFF) / 255f;
        float emissionLevel = light / 15f;
        for (int t = 0; t < 2; t++) {
            out.material[out.materialFloats] = nx;
            out.material[out.materialFloats + 1] = ny;
            out.material[out.materialFloats + 2] = nz;
            out.material[out.materialFloats + 3] = emissionLevel;
            out.material[out.materialFloats + 4] = cr;
            out.material[out.materialFloats + 5] = cg;
            out.material[out.materialFloats + 6] = cb;
            out.material[out.materialFloats + 7] = water;
            out.material[out.materialFloats + 8] = Float.intBitsToFloat(LOD_MATERIAL_ID_BITS);
            out.material[out.materialFloats + 9] = Float.intBitsToFloat(0);
            out.material[out.materialFloats + 10] = 0f;
            out.material[out.materialFloats + 11] = 0f;
            out.materialFloats += 12;
        }
        out.triangles += 2;
    }

    /**
     * The four corners of an axis-aligned rectangle given two opposite corners {@code a}/{@code b}
     * and the face normal. The normal's own axis stays constant across the quad; the other two axes
     * span between the corner pair. Ordering is irrelevant to the ray tracer (no backface culling).
     */
    private static float[] quadCorners(float ax, float ay, float az, float bx, float by, float bz,
                                       float nx, float ny, float nz) {
        float[] c = new float[12];
        if (ny != 0) {
            // Horizontal (y constant): walk x then z.
            setQuad(c, ax, ay, az, bx, ay, az, bx, ay, bz, ax, ay, bz);
        } else if (nx != 0) {
            // X-facing wall (x constant): walk y then z.
            setQuad(c, ax, ay, az, ax, ay, bz, ax, by, bz, ax, by, az);
        } else {
            // Z-facing wall (z constant): walk x then y.
            setQuad(c, ax, ay, az, bx, ay, az, bx, by, az, ax, by, az);
        }
        return c;
    }

    private static void setQuad(float[] c, float ax, float ay, float az,
                                float bx, float by, float bz,
                                float cx, float cy, float cz,
                                float dx, float dy, float dz) {
        c[0] = ax; c[1] = ay; c[2] = az;
        c[3] = bx; c[4] = by; c[5] = bz;
        c[6] = cx; c[7] = cy; c[8] = cz;
        c[9] = dx; c[10] = dy; c[11] = dz;
    }
}

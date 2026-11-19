package dev.comfyfluffy.caustica.rt.terrain;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtAccel;
import dev.comfyfluffy.caustica.rt.dh.DhBridge;
import dev.comfyfluffy.caustica.rt.dh.DhLodMesher;
import dev.comfyfluffy.caustica.rt.dh.DhPresence;
import dev.comfyfluffy.caustica.rt.terrain.RtSectionBuilder.PreparedSection;
import dev.comfyfluffy.caustica.rt.terrain.RtSectionTable.Generation;
import dev.comfyfluffy.caustica.rt.terrain.RtSectionTable.SectionGeom;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrainMesher.PackedSection;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Distant Horizons LOD residency: meshes DH's far-terrain columns as flat-colour boxes into the
 * traced scene beyond the vanilla render-distance window, so the traced world stops where DH's data
 * starts instead of where vanilla's chunks end.
 *
 * <p>Ring layout. Regions tile the plane on three nested power-of-two grids 鈥?256-block regions of
 * 16-block cells (dl 4), 1024-block regions of 32-block cells (dl 5), and 4096-block regions of
 * 64-block cells (dl 6) 鈥?because a DH bulk query at detail level {@code dl} returns exactly
 * {@code 2^dl 脳 2^dl} columns covering a {@code 2^(2路dl)}-block square: one API call per region, no
 * stitching. Ring membership is by the gap between the region's rectangle and the vanilla window
 * rectangle (never by player distance alone, or the window would be double-covered): near band dl 4,
 * then dl 5, then dl 6 out to the max distance. A coarse region's cells whose rectangles fall inside
 * any finer region's rectangle are masked out at mesh time, so rings never double-draw where a
 * coarse region straddles a band edge; the residual sub-cell sliver at those edges is hidden by the
 * border skirts DhLodMesher emits.
 *
 * <p>Residency reuses the terrain pipeline wholesale: mesh output is a {@link PackedSection}
 * (positions region-local, one 12-float {@code TerrainPrim} per triangle with the LOD material
 * sentinel, empty UVs), workers run {@link RtSectionBuilder#prepare} + the BLAS build on
 * {@link RtWorkerPool}, and published geometry enters the SAME {@link RtSectionTable} the traced
 * sections use 鈥?one slot map, one instance list, one push-constant table address. The hit shader
 * recognizes LOD triangles by the sentinel material id (see MATERIAL_ID_LOD), so no instance bits,
 * SBT records, or shader tables change.
 *
 * <p>Scheduling. DH's bulk read is synchronous on the inside: it parks the calling thread on a
 * data-source future that DH only completes from work the client does while rendering, so it runs on a
 * dedicated query thread ({@code caustica-dh-lod-query}) fed from the render thread and hands finished
 * column grids back through a queue. Nothing is read before the traced world is on screen (see
 * {@link #worldOnScreen}), and {@code FETCH_BUDGET_MICROS} caps how many reads may be outstanding at
 * once. Workers own meshing and GPU uploads; publication runs on the render thread between frame
 * stages. DH's chunk-modified events turn into region rebuild requests; a staleness rescan catches
 * data that appears without events.
 */
public final class RtDhLodScene {
    private static final RtDhLodScene INSTANCE = new RtDhLodScene();

    /**
     * Detail levels of the three rings. DH's API detail level {@code dl} covers cells of
     * {@code 2^(dl+4)} blocks, {@code 2^dl} of them per side, so a region is {@code 2^(2*dl+4)}
     * blocks wide and its cells are 32, 64 and 128 blocks for the three rings. The near ring sits
     * over the traced window and is masked cell by cell, so its cells stay small enough that the
     * masked edge follows the traced footprint closely.
     */
    private static final int RING_DL_NEAR = 1;
    private static final int RING_DL_MID = 2;
    private static final int RING_DL_FAR = 3;
    /** Default far-field extent when DH's own chunk render distance is unreadable, in blocks. */
    private static final int DEFAULT_MAX_DISTANCE = 8192;
    /** Rect-gap thresholds between rings, as multiples of the vanilla render distance. */
    private static final float MID_RING_GAP = 1.0f;
    private static final float FAR_RING_GAP = 4.0f;
    /** Desired-window recompute cadence: player movement that triggers a re-enumeration, in blocks. */
    private static final int RESYNC_MOVE_BLOCKS = 192;
    /** Staleness rescan: regions are re-checked against DH data this often, in nanoseconds. */
    private static final long STALE_RESCAN_NANOS = 10_000_000_000L;
    /** Regions with a read or a build in progress: the pipeline depth, reads included. */
    private static final int MAX_REGION_INFLIGHT = 8;
    /** An outstanding read older than this is reported once (DH's data source may be wedged). */
    private static final long READ_STALL_WARN_NANOS = 15_000_000_000L;

    /**
     * How long a region whose read came back empty is left alone. DH answers a whole region at once and
     * fails it entirely when any one cell lacks data, so the regions at the edge of DH's generated area
     * fail every time; retrying them each frame spends the whole read budget on them.
     */
    private static final long NO_DATA_BACKOFF_NANOS = 30_000_000_000L;

    /**
     * Minimum spacing between near-ring re-reads. The tracer publishes many sections per second while
     * a window streams in; re-reading the ring on every one of those would spend the read budget on
     * regions whose masks changed by a single cell.
     */
    private static final long MASK_REFRESH_NANOS = 250_000_000L;

    private static final float[] EMPTY_LIGHTS = new float[0];

    private final Long2ObjectOpenHashMap<SectionGeom> resident = new Long2ObjectOpenHashMap<>();
    private final LongOpenHashSet published = new LongOpenHashSet();
    private final LongOpenHashSet desired = new LongOpenHashSet();
    private final Long2IntOpenHashMap desiredDl = new Long2IntOpenHashMap();
    private final LongArrayList missing = new LongArrayList();
    private final LongOpenHashSet queuedMissing = new LongOpenHashSet();
    /** Regions DH had no data for, mapped to the nanoTime before which they are not read again. */
    private final Long2LongOpenHashMap readBackoffUntil = new Long2LongOpenHashMap();
    private final Long2LongOpenHashMap inFlight = new Long2LongOpenHashMap();
    private final ConcurrentLinkedQueue<LodResult> completedBuilds = new ConcurrentLinkedQueue<>();
    /** Region reads queued for the query thread, and their results (render thread polls the latter). */
    private final LinkedBlockingQueue<Fetch> fetchQueue = new LinkedBlockingQueue<>();
    private final ConcurrentLinkedQueue<FetchResult> completedFetches = new ConcurrentLinkedQueue<>();
    private volatile Thread fetchThread;
    private volatile boolean fetchThreadStop;
    /** Reads queued or running on the query thread. */
    private int readsOutstanding;
    /** When the oldest outstanding read was dispatched, for the stall warning; 0 = none outstanding. */
    private long readsOutstandingSinceNanos;
    private boolean readStallWarned;
    /** Rolling average of one bulk read's wall time in microseconds; 0 until the first read returns. */
    private long readMicrosAvg;
    private final List<PreparedSection> preparedAcc = new ArrayList<>();
    private final List<SectionGeom> removedAcc = new ArrayList<>();
    /**
     * Every section this scene currently has in the shared instance list. The table is shared with the
     * traced terrain and resets its registries on a world teardown, which can strand an instance that no
     * residency map names any more; clearing through this list is what guarantees the LOD field goes away
     * when the feature is turned off.
     */
    private final List<SectionGeom> ownedGeoms = new ArrayList<>();
    private final Object activeTaskLock = new Object();
    private int activeTasks;
    private volatile long epoch = 1L;
    private long buildToken;
    private volatile boolean fullClearRequested;
    private boolean clearedWithoutWorld;
    /** Latched once the traced world has been on screen in this level; reset by {@link #clearAll}. */
    private boolean worldOnScreen;
    /** Bridge init has settled (ready, absent, or capped) 鈥?stop re-attempting each frame. */
    private boolean dhInitSettled;
    private static final int DH_LINK_UNKNOWN = 0;
    private static final int DH_LINK_READY = 1;
    private static final int DH_LINK_FAILED = 2;
    /** Cached outcome of the one-time {@link DhBridge} link probe (see {@link #dhLinked()}). */
    private static int dhLinkState = DH_LINK_UNKNOWN;

    // Rects of resident finer-ring regions, used to mask coarse cells out of coarse regions. Kept as
    // parallel arrays of (minX, minZ, maxX, maxZ) per ring.
    private final List<long[]> finerRectsNear = new ArrayList<>();
    private final List<long[]> finerRectsMid = new ArrayList<>();

    private boolean windowValid;
    private int windowPx;
    private int windowPz;
    private int windowRd;
    private int windowMaxDist;
    /** Traced window in world blocks; the near ring masks its cells against this rect. */
    private int windowMinX;
    private int windowMinZ;
    private int windowMaxX;
    private int windowMaxZ;
    private long lastResyncPlayerNanos;
    // What the near-ring meshes were masked against: the traced generation and the window rect that
    // were current when those meshes were read. A mismatch means their baked masks are stale.
    private int maskTracedGeneration = -1;
    private long maskRefreshNanos;
    private int maskWindowMinX;
    private int maskWindowMinZ;
    private int maskWindowMaxX;
    private int maskWindowMaxZ;
    /** Baked near-ring masks by region key, so a refresh only rebuilds the regions whose mask moved. */
    private final Long2IntOpenHashMap nearMaskSignature = new Long2IntOpenHashMap();
    /**
     * Regions whose last build produced no geometry at all (every cell masked, or DH had nothing to
     * draw). They hold no residency entry, so without this they would be re-read on every sync; the
     * entry is dropped as soon as the mask or DH's data changes.
     */
    private final LongOpenHashSet emptyRegions = new LongOpenHashSet();

    private RtDhLodScene() {
        inFlight.defaultReturnValue(Long.MIN_VALUE);
        desiredDl.defaultReturnValue(-1);
    }

    public static RtDhLodScene instance() {
        return INSTANCE;
    }

    /** Whether any LOD geometry is published (RtTerrain consults this before its empty-table reset). */
    public static boolean hasResidentGeometry() {
        return !INSTANCE.resident.isEmpty();
    }

    public static void requestFullClear() {
        INSTANCE.fullClearRequested = true;
    }

    /** Render-thread full clear (dimension change / world exit), also re-resolving the DH level. */
    public static void clearNow(RtContext ctx) {
        INSTANCE.fullClearRequested = false;
        INSTANCE.clearAll(ctx);
        if (dhLinked()) {
            DhBridge.invalidateLevel();
        }
    }

    /**
     * Device-teardown path ({@code RtTerrain.shutdown}): join outstanding LOD worker/GPU tasks and
     * free all LOD-owned resources before the shared table and the device go away. Retired-graphics
     * frees flush during the caller's subsequent {@code waitIdle}.
     */
    public static void shutdownNow(RtContext ctx) {
        INSTANCE.clearAll(ctx);
        INSTANCE.stopFetchThread();
        INSTANCE.awaitActiveTasks();
        Throwable failure = null;
        LodResult result;
        while ((result = INSTANCE.completedBuilds.poll()) != null) {
            if (result.prepared() != null) {
                RtSectionBuilder.destroy(result.prepared());
            }
            if (result.failure() != null && failure == null) {
                failure = result.failure();
            }
        }
        INSTANCE.inFlight.clear();
        if (failure != null) {
            throw new RuntimeException("DH LOD worker/build failed during teardown", failure);
        }
    }

    private void awaitActiveTasks() {
        synchronized (activeTaskLock) {
            while (activeTasks > 0) {
                try {
                    activeTaskLock.wait(5);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private static boolean enabled() {
        return CausticaConfig.Rt.Dh.LOD_ENABLED.value() && dhLinked() && DhBridge.available();
    }

    /**
     * Whether {@link DhBridge} can be linked in this session. {@code DhBridge} names DH API types in
     * its own descriptors, so with DH absent the JVM rejects the class before its {@code available()}
     * guard can run 鈥?probing here, behind {@link DhPresence} and a catch, is what keeps a session
     * without DH (or with an API jar that does not match the compiled surface) from taking the
     * first tick down. The outcome is cached: the mod list cannot change mid-session, and a link
     * failure is permanent for the class loader.
     */
    private static boolean dhLinked() {
        if (dhLinkState == DH_LINK_UNKNOWN) {
            if (!DhPresence.present()) {
                CausticaMod.LOGGER.info("Distant Horizons not installed; LOD integration stays inert");
                dhLinkState = DH_LINK_FAILED;
            } else {
                dhLinkState = probeDhBridge() ? DH_LINK_READY : DH_LINK_FAILED;
            }
        }
        return dhLinkState == DH_LINK_READY;
    }

    private static boolean probeDhBridge() {
        try {
            DhBridge.available(); // forces DhBridge to load and link; the value is read later
            return true;
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("Distant Horizons API cannot be linked; LOD integration stays inert", t);
            return false;
        }
    }

    /**
     * Attempts bridge init + event wiring. Returns true when the bridge is ready (listeners bound);
     * false means "not yet / not ever" 鈥?the caller retries until {@code DhBridge.settled()} says
     * stop. Binding is idempotent, so a double call is harmless.
     */
    public static boolean init() {
        if (!dhLinked() || !DhBridge.init()) {
            return false;
        }
        DhBridge.setChunkModifiedListener(chunks -> {
            // DH events fire on DH threads: only enqueue, every mutation happens on the render thread.
            for (int i = 0; i + 1 < chunks.length; i += 2) {
                INSTANCE.pendingDirtyChunks.offer(new long[]{chunks[i], chunks[i + 1]});
            }
        });
        DhBridge.setLevelLoadListener(() -> {
            DhBridge.disableSelfRendering();
            INSTANCE.requestFullClear();
        });
        DhBridge.setLevelUnloadListener(RtDhLodScene::requestFullClear);
        return true;
    }

    public static void frame(RtContext ctx) {
        INSTANCE.frameInner(ctx);
    }

    private void frameInner(RtContext ctx) {
        // Lazy retryable init: Fabric initializes mods in name order, so DH's API singletons may not
        // exist when our client init runs. Each frame re-attempts until the bridge settles.
        if (!dhInitSettled) {
            dhInitSettled = !dhLinked() || DhBridge.settled() || RtDhLodScene.init();
        }
        Minecraft mc = Minecraft.getInstance();
        ClientLevel level = mc.level;
        if (fullClearRequested) {
            fullClearRequested = false;
            clearAll(ctx);
            if (dhLinked()) {
                DhBridge.invalidateLevel();
            }
        }
        if (!enabled()) {
            if (!ownedGeoms.isEmpty() || !resident.isEmpty()) {
                clearAll(ctx);
            }
            return;
        }
        if (level == null || mc.player == null) {
            if (!clearedWithoutWorld) {
                clearAll(ctx);
                clearedWithoutWorld = true;
            }
            return;
        }
        clearedWithoutWorld = false;
        if (dhLinked()) {
            // Every dimension of the save loads together, so reads must name the level being shown.
            DhBridge.setClientDimension(level.dimension().identifier().getPath());
        }

        int px = mc.player.getBlockX();
        int pz = mc.player.getBlockZ();
        int rd = mc.options.getEffectiveRenderDistance();
        int rdBlocks = Math.max(rd << 4, 256);

        // Publish finished builds first 鈥?visible fill progress, and it frees table slots for evictions.
        drainCompletedBuilds(ctx);
        drainDirtyChunks();
        publish(ctx);
        syncDesired(px, pz, rdBlocks);
        evictOutOfRange(px, pz, rdBlocks);
        refreshNearMasks();
        // DH only finishes a LOD section once its GPU upload has run on this thread, and a stalled
        // upload eventually starves DH's task picker, which is what our own reads wait on.
        DhBridge.drainRenderThreadTasks();
        drainCompletedFetches(ctx);
        if (worldOnScreen(mc)) {
            dispatchReads(level, px, pz);
        }
        publish(ctx);
    }

    // ---- desired window ---------------------------------------------------------------------------------

    private static int maxDistanceBlocks(int rdBlocks) {
        int configured = CausticaConfig.Rt.Dh.MAX_DISTANCE.value();
        if (configured > 0) {
            return configured;
        }
        int dhChunks = dhLinked() ? DhBridge.farDistanceChunks() : 0;
        int far = dhChunks > 0 ? dhChunks << 4 : DEFAULT_MAX_DISTANCE;
        return Math.min(Math.max(far, rdBlocks * 8), 32768);
    }

    private static int ringDetailForGap(int gap, int rdBlocks) {
        if (gap <= (int) (rdBlocks * MID_RING_GAP)) {
            return RING_DL_NEAR;
        }
        if (gap <= (int) (rdBlocks * FAR_RING_GAP)) {
            return RING_DL_MID;
        }
        return RING_DL_FAR;
    }

    /** Blocks per cell at a DH detail level: a level-{@code dl} column covers {@code 2^(dl+4)} blocks. */
    private static int cellBlocks(int dl) {
        return 16 << dl;
    }

    /** Region width in blocks: {@code 2^dl} cells of {@code 2^(dl+4)} blocks each. */
    private static int regionSideBlocks(int dl) {
        return 16 << (2 * dl);
    }

    private static int cellsPerRegion(int dl) {
        return 1 << dl;
    }

    private static long regionKey(int dl, int rx, int rz) {
        return ((long) dl << 44) | ((long) (rx + 0x200000) << 22) | (rz + 0x200000);
    }

    private static int regionX(long key) {
        return ((int) (key >>> 22) & 0x3FFFFF) - 0x200000;
    }

    private static int regionZ(long key) {
        return ((int) key & 0x3FFFFF) - 0x200000;
    }

    private static int regionDetail(long key) {
        return (int) (key >>> 44);
    }

    private static int regionOriginX(long key) {
        return regionX(key) * regionSideBlocks(regionDetail(key));
    }

    private static int regionOriginZ(long key) {
        return regionZ(key) * regionSideBlocks(regionDetail(key));
    }

    /** Gap between two rects (0 = touching, negative = overlap). Chebyshev-style, in blocks. */
    private static int rectGap(int ax0, int az0, int ax1, int az1, int bx0, int bz0, int bx1, int bz1) {
        int dx = Math.max(bx0 - ax1, ax0 - bx1);
        int dz = Math.max(bz0 - az1, az0 - bz1);
        return Math.max(dx, dz);
    }

    private void syncDesired(int px, int pz, int rdBlocks) {
        long now = System.nanoTime();
        boolean shapeChanged = !windowValid || windowRd != rdBlocks
                || windowMaxDist != maxDistanceBlocks(rdBlocks);
        boolean moved = Math.abs(px - windowPx) >= RESYNC_MOVE_BLOCKS
                || Math.abs(pz - windowPz) >= RESYNC_MOVE_BLOCKS;
        if (!shapeChanged && !moved && now - lastResyncPlayerNanos < STALE_RESCAN_NANOS) {
            return;
        }
        lastResyncPlayerNanos = now;
        windowValid = true;
        windowPx = px;
        windowPz = pz;
        windowRd = rdBlocks;
        windowMaxDist = maxDistanceBlocks(rdBlocks);

        int maxDist = windowMaxDist;
        // Vanilla window rect: the traced terrain covers everything the loaded chunks do. The radius
        // arrives in blocks and the coordinates below are chunk indices, so it converts once here.
        int rdChunks = Math.max(1, windowRd >> 4);
        int wx0 = (px >> 4) - rdChunks, wx1 = (px >> 4) + rdChunks;
        int wz0 = (pz >> 4) - rdChunks, wz1 = (pz >> 4) + rdChunks;
        int wx0b = wx0 << 4, wx1b = ((wx1 + 1) << 4) - 1;
        int wz0b = wz0 << 4, wz1b = ((wz1 + 1) << 4) - 1;
        windowMinX = wx0b;
        windowMinZ = wz0b;
        windowMaxX = wx1b;
        windowMaxZ = wz1b;

        LongOpenHashSet newDesired = new LongOpenHashSet();
        Long2IntOpenHashMap newDl = new Long2IntOpenHashMap();
        newDl.defaultReturnValue(-1);
        finerRectsNear.clear();
        finerRectsMid.clear();

        // Enumerate each ring's grid over the bounding square of the max distance. Regions intersecting
        // the vanilla window are skipped (traced terrain owns them); the rest join by their gap band.
        int[] rings = {RING_DL_NEAR, RING_DL_MID, RING_DL_FAR};
        for (int dl : rings) {
            int side = regionSideBlocks(dl);
            int bound = Math.min(maxDist + side, 32768 + side);
            int rx0 = floorDiv(px - bound, side), rx1 = floorDiv(px + bound, side);
            int rz0 = floorDiv(pz - bound, side), rz1 = floorDiv(pz + bound, side);
            for (int rx = rx0; rx <= rx1; rx++) {
                int x0 = rx * side, x1 = x0 + side - 1;
                for (int rz = rz0; rz <= rz1; rz++) {
                    int z0 = rz * side, z1 = z0 + side - 1;
                    int gap = rectGap(x0, z0, x1, z1, wx0b, wz0b, wx1b, wz1b);
                    // The near ring may overlap the traced window 鈥?its cells are masked out at build
                    // time, so the LOD field meets the window edge instead of a whole region short of it.
                    if (gap < 0 && dl != RING_DL_NEAR) {
                        continue; // another ring's ground; those must not reach into the traced window
                    }
                    if (gap > maxDist) {
                        continue;
                    }
                    int ringDl = ringDetailForGap(gap, rdBlocks);
                    if (ringDl != dl) {
                        continue; // another grid's band owns this gap
                    }
                    long key = regionKey(dl, rx, rz);
                    newDesired.add(key);
                    newDl.put(key, dl);
                    // Record rects for coarse-cell masking (a coarse region must not draw where a
                    // finer region already stands).
                    if (dl == RING_DL_NEAR) {
                        finerRectsNear.add(new long[]{x0, z0, x1, z1});
                    } else if (dl == RING_DL_MID) {
                        finerRectsMid.add(new long[]{x0, z0, x1, z1});
                    }
                }
            }
        }

        // Diff into residency: evict no-longer-desired, queue the new ones.
        for (var it = resident.long2ObjectEntrySet().fastIterator(); it.hasNext(); ) {
            var entry = it.next();
            long key = entry.getKey();
            if (!newDesired.contains(key) && !inFlight.containsKey(key)) {
                removedAcc.add(entry.getValue());
            }
        }
        for (var it = newDesired.iterator(); it.hasNext(); ) {
            long key = it.nextLong();
            if (!resident.containsKey(key) && !inFlight.containsKey(key) && !queuedMissing.contains(key)
                    && !emptyRegions.contains(key)) {
                enqueueMissing(key);
            }
        }
        desired.clear();
        desired.addAll(newDesired);
        desiredDl.clear();
        desiredDl.putAll(newDl);
        RtFrameStats.FRAME.count("dhLodRegionsDesired", newDesired.size());
    }

    private static int floorDiv(int a, int b) {
        return java.lang.Math.floorDiv(a, b);
    }

    /**
     * Re-reads the near ring when the traced footprint or the window rect moved under it.
     *
     * <p>The near ring overlaps the traced window, and its cells are dropped where the tracer has
     * ground. That decision is baked into the mesh at read time, and the tracer fills its window over
     * the first seconds of a level — a mask computed before then leaves LOD standing on top of the
     * real terrain, which reads as z-fighting and as coarse slabs cutting through the world. Both a
     * new traced generation and a moved window invalidate the masks, so the ring is rebuilt then.
     */
    private void refreshNearMasks() {
        if (!windowValid) {
            return;
        }
        boolean windowMoved = windowMinX != maskWindowMinX || windowMinZ != maskWindowMinZ
                || windowMaxX != maskWindowMaxX || windowMaxZ != maskWindowMaxZ;
        int traced = RtTerrain.tracedGeneration();
        if (traced == maskTracedGeneration && !windowMoved) {
            return;
        }
        long now = System.nanoTime();
        if (now - maskRefreshNanos < MASK_REFRESH_NANOS) {
            return; // rebuild at most this often; the tracer publishes many sections per frame at first
        }
        maskRefreshNanos = now;
        maskTracedGeneration = traced;
        maskWindowMinX = windowMinX;
        maskWindowMinZ = windowMinZ;
        maskWindowMaxX = windowMaxX;
        maskWindowMaxZ = windowMaxZ;
        int dl = RING_DL_NEAR;
        int cellSize = cellBlocks(dl);
        for (var it = resident.long2ObjectEntrySet().fastIterator(); it.hasNext(); ) {
            var entry = it.next();
            long key = entry.getKey();
            if (regionDetail(key) != dl || inFlight.containsKey(key)) {
                continue;
            }
            int bits = nearMaskBits(dl, regionOriginX(key), regionOriginZ(key), cellSize);
            if (nearMaskSignature.get(key) == bits) {
                continue; // the baked mask still matches the traced footprint
            }
            emptyRegions.remove(key);
            enqueueMissing(key);
        }
    }

    /**
     * Bit {@code z * cells + x} is set for every cell of the region that carries traced ground, which
     * is the set of cells the near-ring mesh must drop. The test samples a 3x3 grid inside the cell
     * because the traced footprint is a circle: a cell whose centre is clear can still hold loaded
     * ground in a corner, and that corner is where the LOD would stand on the real terrain and z-fight
     * with it.
     */
    private static int nearMaskBits(int dl, int originX, int originZ, int cellSize) {
        int cells = cellsPerRegion(dl);
        int bits = 0;
        for (int z = 0; z < cells; z++) {
            for (int x = 0; x < cells; x++) {
                if (cellHasTraced(originX + x * cellSize, originZ + z * cellSize, cellSize)) {
                    bits |= 1 << (z * cells + x);
                }
            }
        }
        return bits;
    }

    private static boolean cellHasTraced(int cx0, int cz0, int cellSize) {
        int cx1 = cx0 + cellSize - 1;
        int cz1 = cz0 + cellSize - 1;
        int cxm = cx0 + (cellSize >> 1);
        int czm = cz0 + (cellSize >> 1);
        for (int i = 0; i < 3; i++) {
            int sx = i == 0 ? cx0 : i == 1 ? cxm : cx1;
            for (int j = 0; j < 3; j++) {
                int sz = j == 0 ? cz0 : j == 1 ? czm : cz1;
                if (RtTerrain.hasTracedColumn(sx, sz)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void enqueueMissing(long key) {
        if (readBackoffUntil.get(key) > System.nanoTime()) {
            return; // DH had nothing here; a chunk-modified event or the backoff expiry re-queues it
        }
        if (queuedMissing.add(key)) {
            // Nearest-first is approximated by insertion order after the per-ring near鈫抐ar enumeration;
            // a proper heap is not worth it while dispatch is budget-bound anyway.
            missing.add(key);
        }
    }

    /** Chunk-modified offers from DH threads, drained on the render thread (see frameInner). */
    private final ConcurrentLinkedQueue<long[]> pendingDirtyChunks = new ConcurrentLinkedQueue<>();

    /** DH reported chunk data changed: rebuild every resident region that contains any chunk. */
    private void drainDirtyChunks() {
        if (pendingDirtyChunks.isEmpty()) {
            return;
        }
        long[] chunk;
        while ((chunk = pendingDirtyChunks.poll()) != null) {
            int cx = (int) chunk[0];
            int cz = (int) chunk[1];
            int bx = cx << 4, bz = cz << 4;
            for (int dl = RING_DL_FAR; dl >= RING_DL_NEAR; dl--) {
                int side = regionSideBlocks(dl);
                long key = regionKey(dl, floorDiv(bx, side), floorDiv(bz, side));
                readBackoffUntil.remove(key); // data changed here, so a refused read deserves a retry
                emptyRegions.remove(key);
                SectionGeom g = resident.remove(key);
                if (g != null) {
                    removedAcc.add(g);
                    published.remove(key);
                    enqueueMissing(key);
                } else if (desired.contains(key) && !inFlight.containsKey(key)) {
                    // Data appeared for a region that previously had none.
                    enqueueMissing(key);
                }
            }
        }
    }

    private void evictOutOfRange(int px, int pz, int rdBlocks) {
        int maxDist = windowMaxDist;
        int rdChunks = Math.max(1, windowRd >> 4);
        int wx0b = ((px >> 4) - rdChunks) << 4, wx1b = (((px >> 4) + rdChunks + 1) << 4) - 1;
        int wz0b = ((pz >> 4) - rdChunks) << 4, wz1b = (((pz >> 4) + rdChunks + 1) << 4) - 1;
        // Collect first: fastutil's fast iterator is invalidated by a structural change, so the map is
        // only modified once the scan has finished.
        LongArrayList drop = null;
        for (var it = resident.long2ObjectEntrySet().fastIterator(); it.hasNext(); ) {
            var entry = it.next();
            long key = entry.getKey();
            if (!desired.contains(key)) {
                continue; // syncDesired already collected the eviction
            }
            int dl = regionDetail(key);
            int side = regionSideBlocks(dl);
            int x0 = regionOriginX(key), z0 = regionOriginZ(key);
            int gap = rectGap(x0, z0, x0 + side - 1, z0 + side - 1, wx0b, wz0b, wx1b, wz1b);
            if (gap > maxDist || ringDetailForGap(gap, rdBlocks) != dl) {
                if (drop == null) {
                    drop = new LongArrayList();
                }
                drop.add(key);
            }
        }
        if (drop == null) {
            return;
        }
        for (int i = 0; i < drop.size(); i++) {
            long key = drop.getLong(i);
            SectionGeom g = resident.remove(key);
            if (g != null) {
                removedAcc.add(g);
                published.remove(key);
                enqueueMissing(key);
            }
        }
    }

    // ---- data fetch + dispatch --------------------------------------------------------------------------

    /**
     * True once the traced world has been on screen in this level. DH's bulk read parks the calling
     * thread on a data-source future that DH completes from work the client does while rendering, so a
     * read issued during the world-loading screen 鈥?before the RT player section is resident 鈥?never
     * returns and the loading screen never finishes. Nothing is read until that section has been
     * resident once; the latch then stays set for the level, so a section that is briefly missing later
     * (fast travel, chunk reload) cannot stall the LOD field.
     */
    private boolean worldOnScreen(Minecraft mc) {
        if (!worldOnScreen) {
            worldOnScreen = RtTerrain.isSectionReady(mc.player.blockPosition());
        }
        return worldOnScreen;
    }

    /**
     * Hands the nearest {@code missing} regions to the query thread, inside this frame's read budget.
     * Each region's task token is reserved here, before the read starts, so a level change during the
     * read leaves a result that no longer matches and gets dropped on arrival.
     */
    private void dispatchReads(ClientLevel level, int px, int pz) {
        ensureFetchThread();
        int maxReads = maxOutstandingReads();
        while (inFlight.size() < MAX_REGION_INFLIGHT && readsOutstanding < maxReads && !missing.isEmpty()) {
            // Nearest-first by rect gap; the list is small (hundreds), a linear scan is fine.
            int bestIdx = -1;
            long bestKey = 0;
            int bestGap = Integer.MAX_VALUE;
            for (int i = 0; i < missing.size(); i++) {
                long key = missing.getLong(i);
                int dl = regionDetail(key);
                int side = regionSideBlocks(dl);
                int x0 = regionOriginX(key), z0 = regionOriginZ(key);
                int gap = rectGap(x0, z0, x0 + side - 1, z0 + side - 1,
                        ((px >> 4) - windowRd) << 4, ((pz >> 4) - windowRd) << 4,
                        (((px >> 4) + windowRd + 1) << 4) - 1, (((pz >> 4) + windowRd + 1) << 4) - 1);
                if (gap < bestGap) {
                    bestGap = gap;
                    bestIdx = i;
                    bestKey = key;
                }
            }
            if (bestIdx < 0) {
                break;
            }
            missing.removeLong(bestIdx);
            if (!desired.contains(bestKey) || inFlight.containsKey(bestKey)) {
                queuedMissing.remove(bestKey);
                continue;
            }
            queuedMissing.remove(bestKey);

            int dl = regionDetail(bestKey);
            int cells = cellsPerRegion(dl);
            int cellSize = cellBlocks(dl);
            int originX = regionOriginX(bestKey);
            int originZ = regionOriginZ(bestKey);
            // DH covers a region as 2^dl cells of 2^(dl+4) blocks, so the query coordinate is the
            // region origin shifted down by the full region width.
            int queryX = originX >> (2 * dl + 4);
            int queryZ = originZ >> (2 * dl + 4);
            DhBridge.Column[] columns = new DhBridge.Column[cells * cells];
            for (int i = 0; i < columns.length; i++) {
                columns[i] = new DhBridge.Column();
            }
            long token = ++buildToken;
            LodTask task = new LodTask(bestKey, token, epoch, originX, level.getMinY(), originZ);
            inFlight.put(bestKey, token);
            fetchQueue.add(new Fetch(task, dl, cells, cellSize, originX, originZ, queryX, queryZ, columns));
            readsOutstanding++;
            if (readsOutstandingSinceNanos == 0L) {
                readsOutstandingSinceNanos = System.nanoTime();
            }
        }
    }

    /**
     * Reads allowed outstanding at once: the per-frame read budget divided by one read's recent cost,
     * so a slow level (cold SQLite, world-gen backlog) keeps fewer reads in flight and the work that
     * empties them stays inside the budget. One read at a time until the first one has been measured.
     */
    private int maxOutstandingReads() {
        if (readMicrosAvg <= 0) {
            return 1;
        }
        int budgetMicros = CausticaConfig.Rt.Dh.FETCH_BUDGET_MICROS.value();
        int reads = (int) (budgetMicros / readMicrosAvg);
        return Math.max(1, Math.min(reads, MAX_REGION_INFLIGHT));
    }

    /** Applies finished reads on the render thread: meshing is dispatched, empty/stale results dropped. */
    private void drainCompletedFetches(RtContext ctx) {
        FetchResult result;
        boolean any = false;
        while ((result = completedFetches.poll()) != null) {
            any = true;
            readsOutstanding = Math.max(0, readsOutstanding - 1);
            Fetch fetch = result.fetch();
            LodTask task = fetch.task();
            readMicrosAvg = readMicrosAvg <= 0
                    ? result.micros()
                    : (readMicrosAvg * 7 + result.micros()) / 8;
            RtFrameStats.FRAME.count("dhLodQueries", 1);
            RtFrameStats.FRAME.count("dhLodQueryMicros", (int) Math.min(result.micros(), Integer.MAX_VALUE));
            if (!isTaskCurrent(task)) {
                // Cleared while the read ran (level change, dimension change, disabled): not wanted.
                if (inFlight.get(task.key()) == task.token()) {
                    inFlight.remove(task.key());
                }
                continue;
            }
            if (!result.hasData()) {
                // No DH data (unexplored or outside DH's generated area): nothing to draw, and a retry
                // would fail the same way. A chunk-modified event or the backoff expiry re-queues it.
                inFlight.remove(task.key());
                readBackoffUntil.put(task.key(), System.nanoTime() + NO_DATA_BACKOFF_NANOS);
                continue;
            }
            dispatchMesh(ctx, fetch);
        }
        if (readsOutstanding == 0) {
            readsOutstandingSinceNanos = 0L;
            readStallWarned = false;
        } else if (!any && !readStallWarned
                && System.nanoTime() - readsOutstandingSinceNanos > READ_STALL_WARN_NANOS) {
            readStallWarned = true;
            CausticaMod.LOGGER.warn("Distant Horizons LOD read has been pending for over {} ms; "
                            + "the LOD field stays empty until DH's data source answers",
                    READ_STALL_WARN_NANOS / 1_000_000L);
        }
    }

    /**
     * Starts the read thread on first use. Reads are serialized there 鈥?DH's repository takes its own
     * lock and waits on a data-source future 鈥?which keeps DH's internal scheduling intact while the
     * render thread stays free to drive the frames that DH's own pipeline depends on.
     */
    private void ensureFetchThread() {
        if (fetchThread != null) {
            return;
        }
        fetchThreadStop = false;
        Thread thread = new Thread(this::runFetchLoop, "caustica-dh-lod-query");
        thread.setDaemon(true);
        thread.setPriority(Thread.NORM_PRIORITY - 1);
        thread.start();
        fetchThread = thread;
    }

    private void runFetchLoop() {
        while (!fetchThreadStop) {
            Fetch fetch;
            try {
                fetch = fetchQueue.poll(200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                if (fetchThreadStop) {
                    return;
                }
                continue;
            }
            if (fetch == null) {
                continue;
            }
            long start = System.nanoTime();
            boolean hasData = false;
            try {
                hasData = DhBridge.queryBulkColumns(fetch.dl(), fetch.queryX(), fetch.queryZ(),
                        fetch.columns());
            } catch (Throwable t) {
                CausticaMod.LOGGER.error("Distant Horizons LOD read failed at dl {} pos ({}, {})",
                        fetch.dl(), fetch.queryX(), fetch.queryZ(), t);
            }
            completedFetches.add(new FetchResult(fetch, hasData, (System.nanoTime() - start) / 1000));
        }
    }

    /**
     * Stops the read thread and drops every queued read. A read already inside DH cannot be recalled: it
     * finishes on its own and its result is discarded when the token no longer matches.
     */
    private void stopFetchThread() {
        fetchThreadStop = true;
        Thread thread = fetchThread;
        fetchThread = null;
        if (thread != null) {
            thread.interrupt();
            try {
                thread.join(2000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        fetchQueue.clear();
        completedFetches.clear();
        readsOutstanding = 0;
        readsOutstandingSinceNanos = 0L;
        readStallWarned = false;
    }

    private void dispatchMesh(RtContext ctx, Fetch fetch) {
        LodTask task = fetch.task();
        long key = task.key();
        int dl = fetch.dl();
        int cells = fetch.cells();
        int cellSize = fetch.cellSize();
        int originX = fetch.originX();
        int originZ = fetch.originZ();
        DhBridge.Column[] columns = fetch.columns();
        long token = task.token();
        int baseY = fetch.task().originY();
        boolean emission = CausticaConfig.Rt.Dh.EMISSIVE.value();
        // Coarse rings drop cells that fall inside a finer resident ring's rect, so a ring boundary
        // never double-draws. Only rects actually overlapping this region are passed down.
        List<long[]> finer = dl == RING_DL_MID ? finerRectsNear
                : dl == RING_DL_FAR ? finerRectsMid : null;
        long[] maskRects = null;
        if (dl == RING_DL_NEAR) {
            // The near ring reaches into the traced window, and only the cells that actually carry
            // traced geometry are masked: the render-distance square is larger than the ground the
            // tracer built, so masking the square as a whole would cut a hole wherever the loaded
            // chunks hold nothing. The test covers the cell's corners and centre because a cell whose
            // centre is clear can still hold traced ground in one corner — that is the strip where the
            // LOD would otherwise stand on top of the real terrain and z-fight with it.
            LongArrayList tracedCells = new LongArrayList();
            int bits = nearMaskBits(dl, originX, originZ, cellSize);
            int nearCells = cellsPerRegion(dl);
            for (int z = 0; z < nearCells; z++) {
                for (int x = 0; x < nearCells; x++) {
                    if ((bits & (1 << (z * nearCells + x))) == 0) {
                        continue;
                    }
                    int cx0 = originX + x * cellSize;
                    int cz0 = originZ + z * cellSize;
                    tracedCells.add(cx0);
                    tracedCells.add(cz0);
                    tracedCells.add(cx0 + cellSize - 1);
                    tracedCells.add(cz0 + cellSize - 1);
                }
            }
            nearMaskSignature.put(key, bits);
        } else if (finer != null) {
            int side = regionSideBlocks(dl);
            LongArrayList overlapping = new LongArrayList();
            for (long[] r : finer) {
                if (r[0] <= originX + side && originX <= r[2]
                        && r[1] <= originZ + side && originZ <= r[3]) {
                    overlapping.add(r[0]);
                    overlapping.add(r[1]);
                    overlapping.add(r[2]);
                    overlapping.add(r[3]);
                }
            }
            maskRects = overlapping.toArray(new long[0]);
        }
        boolean compact = CausticaConfig.Rt.Terrain.BLAS_COMPACTION.value();
        // The mesh build runs on a worker; it takes the mask by value so the task cannot observe a later
        // reassignment of the local above.
        final long[] meshMask = maskRects;

        beginActiveTask();
        try {
            RtWorkerPool.INSTANCE.submit(() -> {
                try {
                    DhLodMesher.Mesh mesh = DhLodMesher.mesh(columns, cells, cellSize, baseY, emission,
                            originX, originZ, meshMask);
                    if (!isTaskCurrent(task) || mesh.isEmpty()) {
                        completeTask(task, null, null, null);
                        return;
                    }
                    PackedSection packed = new PackedSection(mesh.positions, mesh.indices, mesh.uvs,
                            mesh.material, mesh.bucketTris, mesh.triBase, EMPTY_LIGHTS);
                    PreparedSection prepared = RtSectionBuilder.prepare(ctx, packed, null, compact,
                            key, task.originX, task.originY, task.originZ);
                    if (!isTaskCurrent(task)) {
                        RtSectionBuilder.destroy(prepared);
                        completeTask(task, null, null, null);
                        return;
                    }
                    ctx.gpuExecutor().submit(
                            () -> !isTaskCurrent(task),
                            cmd -> {
                                RtSectionBuilder.recordUpload(cmd, prepared);
                                RtAccel.recordBlasBuilds(ctx, cmd, List.of(prepared.blas()));
                            },
                            () -> {
                                RtAccel.freeBlasScratch(List.of(prepared.blas()));
                                prepared.releaseUpload();
                            },
                            (build, failure) -> completeTask(task, prepared, build, failure));
                } catch (Throwable t) {
                    completeTask(task, null, null, t);
                    throw t;
                }
            });
        } catch (Throwable t) {
            finishActiveTask();
            // The read's reservation must not outlive the dispatch that failed to start.
            if (inFlight.get(key) == token) {
                inFlight.remove(key);
            }
            throw t;
        }
    }

    // ---- completion + publication -----------------------------------------------------------------------

    private record LodTask(long key, long token, long epoch, int originX, int originY, int originZ) {
    }

    /**
     * One region's DH read, queued for the query thread. It carries everything the mesh stage needs, so
     * the query thread never touches client state and the render thread only has to look at the result.
     */
    private record Fetch(LodTask task, int dl, int cells, int cellSize, int originX, int originZ,
                         int queryX, int queryZ, DhBridge.Column[] columns) {
    }

    /** A finished read: the {@link Fetch} it came from, whether DH had data, and how long it took. */
    private record FetchResult(Fetch fetch, boolean hasData, long micros) {
    }

    private record LodResult(LodTask task, PreparedSection prepared,
                             RtGpuExecutor.Build build, Throwable failure) {
    }

    private boolean isTaskCurrent(LodTask task) {
        return task.epoch == epoch && inFlight.get(task.key) == task.token;
    }

    private void beginActiveTask() {
        synchronized (activeTaskLock) {
            activeTasks++;
            activeTaskLock.notifyAll();
        }
    }

    private void finishActiveTask() {
        synchronized (activeTaskLock) {
            activeTasks--;
            activeTaskLock.notifyAll();
        }
    }

    private void completeTask(LodTask task, PreparedSection prepared,
                              RtGpuExecutor.Build build, Throwable failure) {
        finishActiveTask();
        completedBuilds.add(new LodResult(task, prepared, build, failure));
    }

    private void drainCompletedBuilds(RtContext ctx) {
        LodResult result;
        while ((result = completedBuilds.poll()) != null) {
            LodTask task = result.task();
            boolean valid = isTaskCurrent(task);
            // Only release the slot this task owns: a clear may already have handed the region to a newer
            // read, and that one has to keep the pipeline depth it was counted against.
            if (inFlight.get(task.key()) == task.token()) {
                inFlight.remove(task.key());
            }
            if (!valid) {
                destroyResult(result);
                continue;
            }
            if (result.failure() != null) {
                if (result.prepared() != null) {
                    RtSectionBuilder.destroy(result.prepared());
                }
                CausticaMod.LOGGER.error("DH LOD region build failed at {},{} dl {}",
                        regionX(task.key), regionZ(task.key), regionDetail(task.key), result.failure());
                continue;
            }
            if (result.prepared() != null) {
                try {
                    ctx.gpuExecutor().markPublished(result.build());
                    RtFrameStats.FRAME.count("dhLodBuildsCompleted", 1);
                } catch (Throwable t) {
                    RtSectionBuilder.destroy(result.prepared());
                    CausticaMod.LOGGER.error("DH LOD GPU build failed", t);
                    continue;
                }
                preparedAcc.add(result.prepared());
            } else {
                // Legitimately empty (all air, no data, or every cell masked). A region that was built
                // before its cells became masked has geometry standing here, so the replacement has to
                // take it down: an empty result is the new state of the region, not a no-op.
                published.add(task.key);
                SectionGeom prev = resident.get(task.key);
                if (prev != null) {
                    removedAcc.add(prev);
                }
                emptyRegions.add(task.key);
            }
        }
    }

    private void destroyResult(LodResult result) {
        if (result.prepared() != null) {
            RtSectionBuilder.destroy(result.prepared());
        }
    }

    private void publish(RtContext ctx) {
        if (preparedAcc.isEmpty() && removedAcc.isEmpty()) {
            return;
        }
        RtGpuExecutor.GraphicsUse lastGraphicsUse = ctx.gpuExecutor().latestGraphicsUse();
        for (SectionGeom g : removedAcc) {
            table.removePublished(resident, published, g);
            ownedGeoms.remove(g);
            nearMaskSignature.remove(g.key);
            emptyRegions.remove(g.key);
        }
        retire(ctx, lastGraphicsUse, removedAcc);
        removedAcc.clear();

        if (!preparedAcc.isEmpty()) {
            int needed = table.nextSlot + preparedAcc.size();
            Generation oldGeneration = table.beginWriteGeneration(ctx, needed);
            if (oldGeneration != null) {
                retireGeneration(ctx, lastGraphicsUse, oldGeneration);
            }
        }
        int publishedCount = 0;
        int rebaseX = RtTerrain.instanceBlockX();
        int rebaseY = RtTerrain.instanceBlockY();
        int rebaseZ = RtTerrain.instanceBlockZ();
        for (PreparedSection ps : preparedAcc) {
            if (!desired.contains(ps.key())) {
                ctx.gpuExecutor().retireAfterGraphics(lastGraphicsUse,
                        () -> RtSectionBuilder.destroy(ps));
                continue;
            }
            SectionGeom g = new SectionGeom(ps.key(), ps.uvs(), ps.material(),
                    ps.blas().accel, ps.triBase(), ps.sx(), ps.sy(), ps.sz(), EMPTY_LIGHTS);
            SectionGeom prev = resident.get(ps.key());
            if (prev != null && prev.slot >= 0) {
                g.slot = prev.slot;
                g.instanceIndex = prev.instanceIndex;
                table.slots.set(g.slot, g);
                resident.put(ps.key(), g);
                table.write(g);
                table.instanceList.set(g.instanceIndex, table.instanceFor(g, rebaseX, rebaseY, rebaseZ));
                ownedGeoms.remove(prev);
                ownedGeoms.add(g);
                retire(ctx, lastGraphicsUse, List.of(prev));
            } else {
                g.slot = table.allocateSlot();
                g.instanceIndex = table.instanceList.size();
                table.slots.set(g.slot, g);
                resident.put(ps.key(), g);
                table.write(g);
                table.instanceList.add(table.instanceFor(g, rebaseX, rebaseY, rebaseZ));
                ownedGeoms.add(g);
            }
            published.add(ps.key());
            publishedCount++;
        }
        preparedAcc.clear();
        table.flushWrites();
        if (publishedCount > 0) {
            RtFrameStats.FRAME.count("dhLodPublished", publishedCount);
        }
    }

    /** Queue old GPU resources until the last graphics submission that could reference them completes. */
    private void retire(RtContext ctx, RtGpuExecutor.GraphicsUse lastGraphicsUse, List<SectionGeom> geoms) {
        for (SectionGeom g : geoms) {
            ctx.gpuExecutor().retireAfterGraphics(lastGraphicsUse, g::destroy);
        }
    }

    private void retireGeneration(RtContext ctx, RtGpuExecutor.GraphicsUse lastGraphicsUse,
                                  Generation generation) {
        ctx.gpuExecutor().retireAfterGraphics(lastGraphicsUse,
                () -> table.recycleGeneration(generation));
    }

    // ---- teardown ---------------------------------------------------------------------------------------

    private void clearAll(RtContext ctx) {
        epoch++;
        // Everything this scene put into the shared instance list, not just what residency still names:
        // a table reset can drop a region from the map while its instance is still drawn.
        for (SectionGeom g : ownedGeoms) {
            removedAcc.add(g);
        }
        ownedGeoms.clear();
        resident.clear();
        published.clear();
        desired.clear();
        desiredDl.clear();
        missing.clear();
        queuedMissing.clear();
        readBackoffUntil.clear();
        nearMaskSignature.clear();
        emptyRegions.clear();
        windowValid = false;
        worldOnScreen = false;
        // Reads that have not started are dropped; one already inside DH keeps its slot until it comes
        // back, because DH's repository is blocked until then and the region is stale anyway.
        Fetch dropped;
        while ((dropped = fetchQueue.poll()) != null) {
            readsOutstanding = Math.max(0, readsOutstanding - 1);
            LodTask task = dropped.task();
            if (inFlight.get(task.key()) == task.token()) {
                inFlight.remove(task.key());
            }
        }
        if (readsOutstanding == 0) {
            readsOutstandingSinceNanos = 0L;
        }
        // In-flight tasks carry the old epoch and destroy their own results on completion; the removals
        // above retire published geometry against graphics completion.
        publish(ctx);
    }

    /** The section table RtDhLodScene publishes into 鈥?shared with the traced sections. */
    private final RtSectionTable table = RtTerrain.accessSectionTable();
}

package dev.comfyfluffy.caustica.rt.dh;

import com.seibel.distanthorizons.api.enums.rendering.EDhApiRendererMode;
import com.seibel.distanthorizons.api.interfaces.config.IDhApiConfigValue;
import com.seibel.distanthorizons.api.interfaces.data.IDhApiTerrainDataCache;
import com.seibel.distanthorizons.api.interfaces.data.IDhApiTerrainDataRepo;
import com.seibel.distanthorizons.api.interfaces.world.IDhApiLevelWrapper;
import com.seibel.distanthorizons.api.methods.events.DhApiEventRegister;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiChunkModifiedEvent;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiLevelLoadEvent;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiLevelUnloadEvent;
import com.seibel.distanthorizons.api.methods.events.sharedParameterObjects.DhApiEventParam;
import com.seibel.distanthorizons.api.objects.DhApiResult;
import com.seibel.distanthorizons.api.objects.data.DhApiTerrainDataPoint;
import dev.comfyfluffy.caustica.CausticaMod;
import net.fabricmc.loader.api.FabricLoader;

import java.awt.Color;
import java.lang.reflect.Method;
import java.util.function.Consumer;

/**
 * The runtime bridge to Distant Horizons. Everything DH-specific lives here so the rest of the
 * renderer sees only plain Java data.
 *
 * <p>This class cannot even be loaded unless Distant Horizons is installed: its fields and method
 * descriptors name DH API types, so the JVM rejects the class during verification with a
 * {@code NoClassDefFoundError} — a {@code LinkageError} raised before any statement of this class
 * runs, which no guard inside it can intercept. The DH jar is {@code compileOnly}, so every caller
 * must consult {@link DhPresence#present()} first and skip the call entirely when it is false
 * (RtDhLodScene probes once and caches the outcome).
 *
 * <p>Once linkable, every call into DH is defensive: a missing method, a changed signature, or a
 * failed query degrades to "no LOD data this frame" and never fails a render. A failed hook must
 * never take the frame down (same contract as {@code XaeroMinimapHook}).
 *
 * <p>Threading: a read must never run on the render thread. DH's bulk read is synchronous on the
 * inside — it blocks the caller on a data-source future that DH completes from work the client only
 * does while it renders, so a read issued from the render thread parks that thread inside DH and DH
 * with it (on the world-loading screen the client then never finishes loading). RtDhLodScene issues
 * every read from its own query thread; {@link #queryLock} keeps exactly one read inside DH at a time,
 * and {@code levelWrapper}/{@code queryCache} are volatile because a level change on the render thread
 * races a read in progress.
 *
 * <p>DH's data pipeline only advances while its render-thread work runs: LOD sections build their
 * buffers on DH's render-loader threads, then hand the GPU upload to the Minecraft render thread.
 * When vanilla world rendering is cancelled that upload never happens, the loader tasks never return,
 * and DH's task picker — which counts running tasks against a global thread budget — stops dispatching
 * anything, data reads included. {@link #drainRenderThreadTasks()} runs that queue from our render
 * thread so the pipeline keeps draining while DH's own rasterizer stays off.
 *
 * <p>Coordinate systems, decoded from the shipped implementation (DhApiTerrainDataRepo /
 * DhSectionPos): a DH detail level {@code dl} has {@code 2^dl} blocks per cell (dl 0 = one block, dl 4
 * = one chunk, dl 6 = 64 blocks). {@code getAllTerrainDataAtDetailLevelAndPos(level, dl, x, z, cache)}
 * returns {@code DhApiTerrainDataPoint[2^dl][2^dl][]}, where {@code [i][j]} is the column stack for the
 * cell at detail-level coordinates {@code (x·2^dl + i, z·2^dl + j)} — block coordinates of that cell's
 * north-west corner are {@code coord · 2^dl}. One call therefore covers a square of {@code 2^(2·dl)}
 * blocks, which is exactly a LOD region footprint (see RtDhLodScene).
 */
public final class DhBridge {
    /** Frames a still-initializing DH gets before the bridge declares it inert (≈ 30 s at 20 fps worst case). */
    private static final int INIT_ATTEMPT_CAP = 600;
    private static int attempts;
    /** Set when DH init found the mod absent or ran out of attempts — no further init calls. */
    private static volatile boolean dhAbsent;
    private static volatile boolean available;

    private static IDhApiTerrainDataRepo repo;
    private static volatile IDhApiTerrainDataCache queryCache;
    private static volatile IDhApiLevelWrapper levelWrapper;
    /** Dimension path the client is rendering, e.g. {@code "overworld"}: selects the level to read. */
    private static volatile String dimensionName;
    private static IDhApiConfigValue<Integer> dhChunkRenderDistance;
    private static IDhApiConfigValue<EDhApiRendererMode> dhRenderingMode;

    /** Serializes reads: one region is inside DH's repository at a time, on the LOD query thread. */
    private static final Object queryLock = new Object();

    // DH's render-thread task queue, resolved once through reflection (see drainRenderThreadTasks).
    private static boolean renderTaskDrainResolved;
    private static boolean renderTaskDrainBroken;
    private static Object renderTaskHandler;
    private static Method renderTaskDrain;

    // Listener wiring. DH's event injector can hold listeners weakly, so the objects are kept in
    // statics for as long as the bridge lives.
    private static volatile Consumer<long[]> chunkModifiedListener; // long[] {chunkX, chunkZ}
    private static volatile Runnable levelLoadListener;
    private static volatile Runnable levelUnloadListener;
    private static DhApiChunkModifiedEvent chunkModifiedEvent;
    private static DhApiLevelLoadEvent levelLoadEvent;
    private static DhApiLevelUnloadEvent levelUnloadEvent;

    private DhBridge() {
    }

    /**
     * Detects DH and resolves its API. Called once per render frame until it settles: Fabric
     * initializes mods in name order, so bettercaustica's client init runs BEFORE DH's, and DH's
     * API singletons only exist after its own init — a first-attempt failure is a timing artifact,
     * not a verdict. Retries are capped; a mod that is genuinely absent or permanently broken
     * settles to "inert" quickly either way.
     */
    public static synchronized boolean init() {
        if (dhAbsent || available || attempts >= INIT_ATTEMPT_CAP) {
            return available;
        }
        attempts++;
        if (!FabricLoader.getInstance().isModLoaded("distanthorizons")) {
            dhAbsent = true;
            CausticaMod.LOGGER.info("Distant Horizons not installed; LOD integration stays inert");
            return false;
        }
        try {
            repo = com.seibel.distanthorizons.core.api.external.methods.data.DhApiTerrainDataRepo.INSTANCE;
            var config = com.seibel.distanthorizons.core.api.external.methods.config.DhApiConfig.INSTANCE;
            dhChunkRenderDistance = config.graphics().chunkRenderDistance();
            dhRenderingMode = config.graphics().renderingMode();
            available = repo != null && dhChunkRenderDistance != null;
            if (!available) {
                return false; // DH still initializing — retry on a later frame
            }
            bindEvents();
            CausticaMod.LOGGER.info("Distant Horizons bridge ready (api {}.{}.{}), DH chunk render distance {}",
                    com.seibel.distanthorizons.api.DhApi.getApiMajorVersion(),
                    com.seibel.distanthorizons.api.DhApi.getApiMinorVersion(),
                    com.seibel.distanthorizons.api.DhApi.getApiPatchVersion(),
                    farDistanceChunks());
        } catch (Throwable t) {
            available = false;
            // Mid-init DH can throw while its singletons settle; the retry cap turns a permanently
            // broken API into a single logged warning per batch of attempts.
            if (attempts >= INIT_ATTEMPT_CAP) {
                CausticaMod.LOGGER.warn("Distant Horizons detected but its API did not load; LOD integration disabled", t);
            }
        }
        return available;
    }

    /**
     * True once DH is present AND its API resolved. All other methods require this.
     */
    public static boolean available() {
        return available;
    }

    /** No further init attempts will change the outcome (ready, absent, or capped out). */
    public static boolean settled() {
        return available || dhAbsent || attempts >= INIT_ATTEMPT_CAP;
    }

    /** DH's own LOD render distance in chunks, or 0 when unreadable. */
    public static int farDistanceChunks() {
        try {
            Integer value = dhChunkRenderDistance.getValue();
            return value != null ? value : 0;
        } catch (Throwable t) {
            return 0;
        }
    }

    /**
     * Stops DH's own LOD rasterization so the ray tracer is the only thing drawing distant terrain.
     * The data pipeline keeps running (that is what we read), and the setting is applied once per
     * session, never fought over: if the player re-enables DH rendering it stays enabled.
     */
    public static void disableSelfRendering() {
        if (!available || dhRenderingMode == null) {
            return;
        }
        try {
            if (dhRenderingMode.getValue() != EDhApiRendererMode.DISABLED) {
                boolean ok = dhRenderingMode.setValue(EDhApiRendererMode.DISABLED,
                        "bettercaustica dh.lod integration");
                CausticaMod.LOGGER.info("{} DH renderer mode DISABLED (bettercaustica dh.lod draws the LOD field)",
                        ok ? "Set" : "Failed to set");
            }
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("Could not switch DH's own renderer off; its output is simply overwritten", t);
        }
    }

    /**
     * Runs the work DH queued for the Minecraft render thread, which is its LOD section GPU upload.
     * Must be called from the render thread; costs whatever DH's queue holds this frame (one upload
     * per finished LOD section), and DH throttles the queue to half a frame internally.
     *
     * <p>Every DH LOD section uploads through this queue before its render-loader task returns, and
     * DH's task picker refuses to start new tasks while the running count is at its thread budget.
     * With vanilla world rendering cancelled nobody drains the queue, so the picker saturates and
     * every later DH task — including the data read this bridge issues — waits forever.
     *
     * <p>Reached reflectively on purpose: the queue lives in a DH core class, and a version that
     * moves it must cost this drain and nothing else instead of failing the whole bridge's linkage.
     */
    public static void drainRenderThreadTasks() {
        if (!available || renderTaskDrainBroken) {
            return;
        }
        if (!renderTaskDrainResolved) {
            resolveRenderTaskDrain();
        }
        if (renderTaskDrain == null) {
            return;
        }
        try {
            renderTaskDrain.invoke(renderTaskHandler);
        } catch (Throwable t) {
            renderTaskDrainBroken = true;
            CausticaMod.LOGGER.warn("Could not run Distant Horizons' render-thread queue; LOD reads may stall", t);
        }
    }

    private static void resolveRenderTaskDrain() {
        renderTaskDrainResolved = true;
        try {
            Class<?> handler = Class.forName("com.seibel.distanthorizons.core.render.RenderThreadTaskHandler");
            Object instance = handler.getField("INSTANCE").get(null);
            Method method = handler.getMethod("runRenderThreadTasks");
            renderTaskHandler = instance;
            renderTaskDrain = method;
        } catch (Throwable t) {
            renderTaskDrain = null;
            CausticaMod.LOGGER.warn("Distant Horizons' render-thread queue is unavailable; LOD reads may stall", t);
        }
    }

    public static void setChunkModifiedListener(Consumer<long[]> listener) {
        chunkModifiedListener = listener;
    }

    public static void setLevelLoadListener(Runnable listener) {
        levelLoadListener = listener;
    }

    public static void setLevelUnloadListener(Runnable listener) {
        levelUnloadListener = listener;
    }

    private static void bindEvents() {
        chunkModifiedEvent = new DhApiChunkModifiedEvent() {
            @Override
            public void onChunkModified(DhApiEventParam<EventParam> param) {
                Consumer<long[]> listener = chunkModifiedListener;
                if (listener != null && param != null && param.value != null) {
                    listener.accept(new long[]{param.value.chunkX, param.value.chunkZ});
                }
            }
        };
        levelLoadEvent = new DhApiLevelLoadEvent() {
            @Override
            public void onLevelLoad(DhApiEventParam<EventParam> param) {
                if (param != null && param.value != null && param.value.levelWrapper != null) {
                    IDhApiLevelWrapper wrapper = param.value.levelWrapper;
                    String wanted = dimensionName;
                    if (wanted == null || matchesDimension(wrapper, wanted)) {
                        levelWrapper = wrapper;
                    }
                }
                Runnable listener = levelLoadListener;
                if (listener != null) {
                    listener.run();
                }
            }
        };
        levelUnloadEvent = new DhApiLevelUnloadEvent() {
            @Override
            public void onLevelUnload(DhApiEventParam<EventParam> param) {
                levelWrapper = null;
                Runnable listener = levelUnloadListener;
                if (listener != null) {
                    listener.run();
                }
            }
        };
        DhApiEventRegister.on(DhApiChunkModifiedEvent.class, chunkModifiedEvent);
        DhApiEventRegister.on(DhApiLevelLoadEvent.class, levelLoadEvent);
        DhApiEventRegister.on(DhApiLevelUnloadEvent.class, levelUnloadEvent);
    }

    /**
     * Current DH level wrapper, resolved against the dimension the client is actually showing.
     * All dimensions load together, so taking whichever wrapper DH happens to hand out first (or the
     * last one to fire a load event) reads another dimension's LOD database and finds nothing.
     */
    private static IDhApiLevelWrapper currentLevel() {
        IDhApiLevelWrapper wrapper = levelWrapper;
        String wanted = dimensionName;
        if (wrapper != null) {
            if (wanted == null || matchesDimension(wrapper, wanted)) {
                return wrapper;
            }
            levelWrapper = null;
        }
        try {
            var proxy = com.seibel.distanthorizons.core.world.DhApiWorldProxy.INSTANCE;
            if (proxy == null || !proxy.worldLoaded()) {
                return null;
            }
            Iterable<IDhApiLevelWrapper> candidates = wanted != null
                    ? proxy.getAllLoadedLevelsWithDimensionNameLike(wanted)
                    : proxy.getAllLoadedLevelWrappers();
            if (candidates != null) {
                for (IDhApiLevelWrapper candidate : candidates) {
                    if (wanted == null || matchesDimension(candidate, wanted)) {
                        levelWrapper = candidate;
                        return candidate;
                    }
                }
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    /** True when DH's level names this dimension, e.g. {@code "minecraft:overworld"} or {@code "overworld"}. */
    private static boolean matchesDimension(IDhApiLevelWrapper wrapper, String dimensionName) {
        try {
            String name = wrapper.getDimensionName();
            return name != null && name.toLowerCase(java.util.Locale.ROOT).endsWith(dimensionName);
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** Tells the bridge which dimension the client renders, so reads target that level's data. */
    public static void setClientDimension(String dimension) {
        String name = dimension == null ? null : dimension.toLowerCase(java.util.Locale.ROOT);
        if (java.util.Objects.equals(dimensionName, name)) {
            return;
        }
        dimensionName = name;
        levelWrapper = null;
        queryCache = null;
    }

    /**
     * Drops the level wrapper and the query cache so a dimension change re-resolves against the new
     * level. Deliberately lock-free: this runs on the render thread, and a read can sit inside DH's
     * data source for seconds, so waiting for {@link #queryLock} here would stall the frame that
     * changes level. The old cache is abandoned instead of cleared; the next read builds a fresh one.
     */
    public static void invalidateLevel() {
        if (!available) {
            return; // no DH types may be touched before init() proved the mod is present
        }
        levelWrapper = null;
        queryCache = null;
    }

    /**
     * Decoded LOD column: everything the mesher needs, no DH types.
     * {@code topY} is the top block's world Y (the mesh surface sits at topY + 1).
     */
    public static final class Column {
        public int topY;
        public int bottomY;
        public int colorArgb = 0xFF808080;
        /** 0..15 block light at the column top (drives the heuristic emission). */
        public int light;
        public boolean water;
        public boolean air = true;
    }

    /**
     * Reads one DH bulk query: {@code 2^dl × 2^dl} columns whose detail-level coordinates start at
     * {@code (x, z)} (the query pos semantics: covered dl coords are {@code [x·2^dl, (x+1)·2^dl)}).
     * {@code out} receives {@code 2^dl·2^dl} columns in row-major [i][j] order. Returns false when DH
     * has no data for the area at all.
     *
     * <p>Blocks until DH's data source answers, and serialized by {@link #queryLock}: call it from the
     * LOD query thread only, never from the render thread.
     */
    public static boolean queryBulkColumns(int dl, int x, int z, Column[] out) {
        if (!available) {
            return false;
        }
        IDhApiLevelWrapper level = currentLevel();
        if (level == null) {
            return false;
        }
        int cells = 1 << dl;
        if (out == null || out.length < cells * cells) {
            return false;
        }
        synchronized (queryLock) {
            try {
                IDhApiTerrainDataCache cache = queryCache;
                if (cache == null) {
                    cache = repo.createSoftCache();
                    queryCache = cache;
                }
                DhApiResult<DhApiTerrainDataPoint[][][]> result =
                        repo.getAllTerrainDataAtDetailLevelAndPos(level, (byte) dl, x, z, cache);
                if (result == null || !result.success || result.payload == null) {
                    return false;
                }
                DhApiTerrainDataPoint[][][] grid = result.payload;
                for (int i = 0; i < cells && i < grid.length; i++) {
                    DhApiTerrainDataPoint[][] columnStacks = grid[i];
                    if (columnStacks == null) {
                        continue;
                    }
                    for (int j = 0; j < cells && j < columnStacks.length; j++) {
                        DhApiTerrainDataPoint[] stack = columnStacks[j];
                        Column column = out[i * cells + j];
                        column.air = true;
                        if (stack == null || stack.length == 0) {
                            continue;
                        }
                        // A stack covers the column vertically: DH emits an air span for the open sky
                        // on top of the terrain, so the highest solid point is what a distant view
                        // sees. Deeper points (overhangs, caves) are future work; order is arbitrary.
                        DhApiTerrainDataPoint top = null;
                        for (DhApiTerrainDataPoint point : stack) {
                            if (point == null || point.blockStateWrapper == null) {
                                continue;
                            }
                            if (point.blockStateWrapper.isAir()) {
                                continue;
                            }
                            if (top == null || point.topYBlockPos > top.topYBlockPos) {
                                top = point;
                            }
                        }
                        if (top == null) {
                            continue; // air-only column
                        }
                        column.air = false;
                        column.topY = top.topYBlockPos;
                        column.bottomY = top.bottomYBlockPos;
                        column.light = clampLight(top.blockLightLevel);
                        column.water = top.blockStateWrapper.isLiquid();
                        column.colorArgb = columnColor(level, top);
                    }
                }
                return true;
            } catch (Throwable t) {
                CausticaMod.LOGGER.warn("DH LOD query failed at dl {} pos ({}, {})", dl, x, z, t);
                return false;
            }
        }
    }

    private static int clampLight(int light) {
        return Math.max(0, Math.min(15, light));
    }

    /**
     * Column colour via DH's own colour resolution (block colour + biome tint — the same values DH's
     * renderer shows). Falls back to a per-material palette when the API answer is unusable, so an
     * API change degrades the far field's palette instead of the frame.
     */
    private static int columnColor(IDhApiLevelWrapper level, DhApiTerrainDataPoint point) {
        try {
            DhApiResult<Color> color = level.getBlockColorPreApi(
                    point.blockStateWrapper, point.biomeWrapper,
                    0, point.topYBlockPos, 0, null);
            if (color != null && color.success && color.payload != null) {
                Color c = color.payload;
                return (0xFF << 24) | ((clamp255(c.getRed())) << 16)
                        | (clamp255(c.getGreen()) << 8) | clamp255(c.getBlue());
            }
        } catch (Throwable ignored) {
        }
        return materialFallbackColor(point);
    }

    private static int clamp255(int v) {
        return Math.max(0, Math.min(255, v));
    }

    /** Palette fallback keyed by DH's own material classification. */
    private static int materialFallbackColor(DhApiTerrainDataPoint point) {
        byte matId = point.blockStateWrapper.getMaterialId();
        com.seibel.distanthorizons.api.enums.rendering.EDhApiBlockMaterial material;
        try {
            material = com.seibel.distanthorizons.api.enums.rendering.EDhApiBlockMaterial.getFromIndex(
                    matId & 0xFF);
        } catch (Throwable t) {
            material = com.seibel.distanthorizons.api.enums.rendering.EDhApiBlockMaterial.UNKNOWN;
        }
        if (material == null) {
            material = com.seibel.distanthorizons.api.enums.rendering.EDhApiBlockMaterial.UNKNOWN;
        }
        return switch (material) {
            case GRASS -> 0xFF5D7C3F;
            case LEAVES -> 0xFF466128;
            case DIRT -> 0xFF79553A;
            case SAND -> 0xFFDBCB94;
            case STONE -> 0xFF7F7F7F;
            case DEEPSLATE -> 0xFF4B4B50;
            case SNOW -> 0xFFF0F6F6;
            case WATER -> 0xFF3357A0;
            case LAVA -> 0xFFC44910;
            case WOOD -> 0xFF66502C;
            case METAL -> 0xFF8F8F8F;
            case TERRACOTTA -> 0xFF985E43;
            case NETHER_STONE -> 0xFF5B303B;
            case ILLUMINATED -> 0xFFFFD98D;
            case AIR -> 0;
            case UNKNOWN -> 0xFF6E6E6E;
        };
    }

    /** Rough probe used by diagnostics: does DH have anything for this single column? */
    public static boolean columnExists(int blockX, int blockZ) {
        if (!available) {
            return false;
        }
        IDhApiLevelWrapper level = currentLevel();
        if (level == null) {
            return false;
        }
        synchronized (queryLock) {
            try {
                IDhApiTerrainDataCache cache = queryCache;
                if (cache == null) {
                    cache = repo.createSoftCache();
                    queryCache = cache;
                }
                DhApiResult<DhApiTerrainDataPoint[]> column =
                        repo.getColumnDataAtBlockPos(level, blockX, blockZ, cache);
                return column != null && column.success && column.payload != null && column.payload.length > 0;
            } catch (Throwable t) {
                return false;
            }
        }
    }
}

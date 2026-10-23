package dev.comfyfluffy.caustica.rt.entity;

import com.mojang.blaze3d.textures.GpuTextureView;
import com.mojang.blaze3d.vulkan.VulkanGpuTextureView;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.mixin.RenderSetupAccessor;
import dev.comfyfluffy.caustica.mixin.RenderTypeAccessor;
import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.material.RtMaterialRegistry;
import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import com.mojang.blaze3d.platform.NativeImage;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.rendertype.PreparedRenderType;
import net.minecraft.client.renderer.rendertype.RenderType;
import net.minecraft.client.renderer.texture.DynamicTexture;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.resources.Identifier;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Supplier;

/**
 * Resolves the texture backing an entity {@link RenderType} to a Vulkan image-view handle, for the
 * bindless entity-texture array. Entities use per-type texture files (zombie.png, …), not the block
 * atlas, so each distinct texture maps to its own bindless slot. Slots are keyed by the resolved image
 * view rather than the {@link RenderType}: some render types are rebuilt every frame with a fresh
 * identity (the charged-creeper energy swirl scrolls its texture transform, so {@code energySwirl()}
 * allocates a new RenderType per frame), and keying by RenderType there would leak a slot per frame
 * until the array exhausts and everything falls back to slot 0.
 *
 * <p>The view is obtained through the <b>public</b> {@code RenderType.prepare()} → {@link
 * PreparedRenderType#textures()} API (a list of {@code Texture(name, GpuTextureView, sampler)}): the
 * primary sampler is {@code "Sampler0"} ({@code "Sampler1"}/{@code "Sampler2"} are the overlay/lightmap
 * the prepared list prepends). Resolution is cached per {@code RenderType} (they are stable singletons),
 * so the prepare() cost is paid once per distinct texture.
 *
 * <p>Slot use is bounded twice. When the array is full, {@link #recycleColdestSlot} steals the
 * least-recently-used slot that was <em>not</em> touched in the current frame: a texture that has scrolled
 * out of the scene must not keep its slot forever, or a world with a few hundred distinct one-shot
 * textures (maps in item frames, paintings, banners) starves every mob. When even that fails — every slot
 * drawn in the same frame — the surface degrades to the reserved white texture and not to slot 0, because
 * slot 0 is the block atlas: sampling it with a mob's 0..1 texture UVs is exactly what turns an entity into
 * coloured atlas blocks, or (where the atlas is empty) into a transparent shell the hit shader discards.
 */
public final class RtEntityTextures {
    /** Bindless array capacity (slot 0 reserved as a fallback texture). {@code -Dcaustica.rt.maxEntityTextures}. */
    public static int maxTextures() {
        return CausticaConfig.Rt.EntityTextures.MAX_TEXTURES.value();
    }

    /** Enable pack-compiled entity materials; disabled entities use the canonical neutral header. */
    public static boolean entityPbr() {
        return CausticaConfig.Rt.EntityTextures.PBR.value();
    }

    public static final RtEntityTextures INSTANCE = new RtEntityTextures();

    // RenderType identity → resolved primary image-view handle. WEAK: some render types are rebuilt
    // every frame with a fresh identity (e.g. the charged-creeper energy-swirl layer, whose scrolling
    // texture transform makes RenderTypes.energySwirl() allocate a new RenderType each frame). A weak map
    // lets those dead identities be collected instead of accumulating; stable singletons (zombie.png, …)
    // stay cached and skip the costly RenderType.prepare().
    private final Map<RenderType, Long> viewCache = new WeakHashMap<>();
    // RenderType → the image view of its SECOND texture (Sampler1). The end portal and the end gateway
    // composite a layer stack from it (rendertype_end_portal.fsh), so a surface that looks like one
    // texture to the primary slot really needs two bound to be reproduced.
    private final Map<RenderType, Long> layerViewCache = new WeakHashMap<>();
    private final Map<RenderType, Identifier> locationCache = new WeakHashMap<>();
    // Resolved image-view handle → bindless slot. The slot identifies a *texture*, not a RenderType, so
    // many render types that differ only by a texture transform we don't replicate (the swirl's scroll)
    // collapse to ONE slot — instead of leaking a slot per frame until the array exhausts and everything
    // falls back to slot 0 (the block atlas). Append-only: a handle's slot never changes once assigned,
    // so update-after-bind writes for new slots never disturb in-flight frames.
    private final Map<Long, Integer> viewSlotCache = new HashMap<>();
    // Atlas-location → bindless slot, for items/blocks (which texture from an atlas, not a per-type
    // file). Seeded with the block atlas = slot 0 (also the fallback). Items use a separate item atlas.
    private final Map<Identifier, Integer> atlasSlotCache = new HashMap<>();
    private final List<Pending> pending = new ArrayList<>(); // albedo slots awaiting descriptor upload
    // Descriptor array capacity of the currently alive world pipeline. A higher config value applies after
    // reset/recreate; a lower value stops allocating new slots immediately without invalidating old ones.
    private int capacity = maxTextures();
    private int nextSlot = 1;
    // LRU bookkeeping for slot recycling: slot → the image view that owns it, and the frame that slot was
    // last asked for. Both are sized to the slot limit on reset and indexed 1..slotLimit()-1.
    private long[] slotView = new long[0];
    private long[] slotLastUsedFrame = new long[0];
    // Slot of the white fallback texture, reserved while the array is still empty so it is guaranteed to
    // exist when an allocation later fails. Never recycled: stealing it would re-texture the one slot that
    // the exhausted-array path depends on. 0 until the reservation succeeds.
    private int reservedWhiteSlot;
    // The 1x1 white DynamicTexture behind reservedWhiteSlot; kept so a resource reload that released its
    // pixels can be detected and re-registered without leaking the previous instance.
    private DynamicTexture whiteTexture;
    private boolean loggedFailure;
    private boolean loggedMaterialFailure;

    // Observability. Every failure mode here — an unresolved view, a full array — silently falls back to
    // slot 0 (the block atlas), and a stale cached handle silently keeps pointing at a destroyed image
    // view. All three render an entity that is either covered in atlas texels or fully transparent, with
    // nothing in the log to tell them apart, so each is counted and reported rather than swallowed.
    // Counters are cumulative across pipeline rebuilds on purpose: a session-wide total is what shows
    // whether slots are being burned by texture churn or simply running out.
    private long slotsAllocated;
    private long evictions;
    private long fullFallbacks;
    private long unresolvedFallbacks;
    private long descriptorWrites;
    private long staleViewDetections;
    private long probeFrame = -1L;
    private int probesThisFrame;
    private long lastSummaryFrame = -1L;
    private boolean loggedFull;
    private boolean loggedUnresolved;

    /** Frames between spot-checks of cached handles against a fresh resolve. */
    private static final long PROBE_INTERVAL_FRAMES = 240L;
    /** Cached handles re-resolved per probe frame. Entity layers use distinct render types, so a handful
     *  covers several surfaces without paying a prepare() for every submission. */
    private static final int PROBES_PER_TICK = 4;
    /** Frames between one-line slot/fallback summaries. */
    private static final long SUMMARY_INTERVAL_FRAMES = 600L;
    /** First N slot allocations are traced with their source, to name whatever eats the array. */
    private static final long ALLOCATION_TRACE_LIMIT = 64L;
    // 1x1 solid-white DynamicTexture: the tint-only albedo for untextured geometry (leash/line ribbons) and
    // the last-resort fallback when no real texture can be bound. See whiteSlot().
    private static final Identifier WHITE_LOCATION = Identifier.fromNamespaceAndPath("caustica", "rt_white");

    // Cached RenderSetup.TextureBinding#location() (the class is package-private, the method public).
    private Method locationMethod;

    private record Pending(int slot, long view) {
    }

    private RtEntityTextures() {
    }

    /**
     * The bindless slot for {@code renderType}'s texture. Keyed by the resolved image view, not the
     * RenderType, so per-frame-allocated render types sharing one texture reuse a single slot. Returns 0
     * (the fallback slot = block atlas) if the texture can't be resolved or the array is full.
     */
    public int slotFor(RenderType renderType) {
        if (renderType == null) {
            return 0;
        }
        Long cached = viewCache.get(renderType);
        if (cached != null) {
            probeCachedView(renderType, cached);
            return slotForView(cached, () -> "render " + renderType);
        }
        return slotForView(resolveView(renderType), () -> "render " + renderType);
    }

    /**
     * {@link #slotFor} for a per-type entity texture (a mob's own sheet, a glyph layer), which must never
     * degrade to slot 0. Slot 0 is the block atlas, and a full-texture UV set sampled against it lands on
     * whatever sprites are stitched there — the coloured-block / transparent-shell failure this was written
     * for. An unresolvable or unallocatable texture degrades to the reserved white slot instead, i.e. to a
     * visibly plain entity rather than a silently wrong one.
     */
    public int slotForEntity(RenderType renderType) {
        int slot = slotFor(renderType);
        return slot != 0 ? slot : whiteSlot();
    }

    /**
     * Spot-check a few cached handles per {@link #PROBE_INTERVAL_FRAMES} against a fresh resolve. A texture
     * rebuild (resource reload, or any path that re-registers the texture without the world pipeline being
     * recreated) hands the render type a new image view while this cache keeps returning the old one; the
     * descriptor written for it then points at a destroyed view. Comparing the two turns that from an
     * unexplained mis-texture into a counted, named event.
     */
    private void probeCachedView(RenderType renderType, long cached) {
        long frame = RtComposite.frameCounter();
        if (frame != probeFrame) {
            probeFrame = frame;
            probesThisFrame = 0;
        }
        if (frame % PROBE_INTERVAL_FRAMES != 0L || probesThisFrame >= PROBES_PER_TICK) {
            return;
        }
        probesThisFrame++;
        long fresh = prepareViewHandle(renderType);
        if (fresh != 0L && fresh != cached) {
            staleViewDetections++;
            if (staleViewDetections <= 8L) {
                CausticaMod.LOGGER.warn("RT entity texture view changed under a cached handle: {} cached=0x{} now=0x{}",
                        renderType, Long.toHexString(cached), Long.toHexString(fresh));
            }
        }
    }

    /**
     * The bindless slot of {@code renderType}'s second texture ({@code Sampler1}), or 0 when it binds
     * none. {@link #slotFor} deliberately resolves only the primary texture, which is right for every
     * surface that is shaded from one image; the end portal's layer stack is the case that needs the
     * other one, and the hit shader reads this slot from the prim's aux1 lane.
     */
    public int layerSlotFor(RenderType renderType) {
        if (renderType == null) {
            return 0;
        }
        Long cached = layerViewCache.get(renderType);
        long view = cached != null ? cached : resolveNamedView(renderType, "Sampler1");
        if (cached == null) {
            layerViewCache.put(renderType, view);
        }
        if (view == 0L) {
            return 0; // this render type binds no second sampler: 0 means "no layer", not a failure
        }
        int slot = slotForView(view, () -> "layer " + renderType);
        return slot != 0 ? slot : whiteSlot();
    }

    /** Canonical material ID for a full entity texture, or the neutral runtime-texture fallback. */
    public int materialIdFor(RenderType renderType, boolean stochasticAlpha) {
        if (!entityPbr()) return RtMaterialRegistry.INSTANCE.entityFallbackId(stochasticAlpha);
        return RtMaterialRegistry.INSTANCE.resolveEntityTexture(textureLocation(renderType), stochasticAlpha);
    }

    /**
     * The bindless slot for a texture atlas (block/item atlas used by item + block-model quads), cached
     * per atlas location. The block atlas is pre-seeded to slot 0; other atlases (the item atlas) get
     * their own slot. Returns 0 (fallback) if unresolvable or the array is full.
     */
    public int slotForAtlas(Identifier atlasLocation) {
        if (atlasLocation == null) {
            return 0;
        }
        Integer cached = atlasSlotCache.get(atlasLocation);
        if (cached != null) {
            touchSlot(cached, RtComposite.frameCounter());
            return cached;
        }
        long view = 0L;
        try {
            GpuTextureView v = Minecraft.getInstance().getTextureManager().getTexture(atlasLocation).getTextureView();
            view = vkImageView(v);
        } catch (Throwable t) {
            if (!loggedFailure) {
                loggedFailure = true;
                CausticaMod.LOGGER.warn("RT atlas texture resolution failed for {}", atlasLocation, t);
            }
        }
        int slot = slotForView(view, () -> "atlas " + atlasLocation);
        if (slot != 0) {
            // Only a resolved slot is worth caching. Caching the 0 fallback would pin this atlas to the
            // block atlas for the rest of the pipeline's life even after it resolves — and every quad that
            // samples it would keep reading atlas texels.
            atlasSlotCache.put(atlasLocation, slot);
        }
        return slot;
    }

    /**
     * {@link #slotForAtlas} for an atlas that is <em>not</em> necessarily the block atlas. Slot 0 is the
     * right answer for the block atlas itself — the terrain <i>s</i>/<i>n</i> atlases and the block-entity
     * sprites are all stitched there — but for a dedicated atlas (items, particles, a block entity's own
     * sheet) a 0 means "unresolved or no room", and its sprite UVs sampled against the block atlas come back
     * as arbitrary block texels. Those degrade to the reserved white slot instead: wrong, but plainly and
     * identifiably wrong.
     */
    public int slotForAtlasSafe(Identifier atlasLocation) {
        if (atlasLocation == null || TextureAtlas.LOCATION_BLOCKS.equals(atlasLocation)) {
            return slotForAtlas(atlasLocation); // slot 0 IS the block atlas
        }
        int slot = slotForAtlas(atlasLocation);
        return slot != 0 ? slot : whiteSlot();
    }

    /**
     * Bindless slot of a solid-white 1x1 texture: the tint-only albedo for untextured geometry (leashes,
     * custom line ribbons) and the fallback for any texture that cannot be bound. Slot 0 (the block atlas)
     * is NOT a substitute for either use: the hit shader multiplies albedo by the sampled texel, and UV
     * (0,0) lands on whatever sprite is stitched at the atlas origin.
     *
     * <p>The texture is registered with the TextureManager on first use, resolved through the ordinary
     * atlas-slot path (so it survives {@link #reset} like any other atlas), and — importantly — its slot is
     * <b>reserved at {@link #reset}</b> while the array is still empty, not lazily on the first allocation
     * that needs it. The old lazy reservation made the fallback unavailable in precisely the case it exists
     * for: an exhausted array has no room left for the white texture either, so the fallback silently became
     * slot 0 again. Returns 0 only if registration and resolution both fail.
     */
    public int whiteSlot() {
        Integer cached = atlasSlotCache.get(WHITE_LOCATION);
        if (cached != null && cached != 0) {
            touchSlot(cached, RtComposite.frameCounter());
            return cached;
        }
        ensureWhiteTexture();
        int slot = slotForAtlas(WHITE_LOCATION);
        if (slot != 0) {
            reservedWhiteSlot = slot;
        }
        return slot;
    }

    /** Register (or re-register, after a reload released its pixels) the 1x1 white DynamicTexture. */
    private void ensureWhiteTexture() {
        if (whiteTexture != null) {
            long view = 0L;
            try {
                view = vkImageView(whiteTexture.getTextureView());
            } catch (Throwable ignored) {
                // the texture was released (a resource reload closed it) → fall through and re-register
            }
            if (view != 0L) {
                return;
            }
            closeWhiteTexture();
        }
        try {
            NativeImage image = new NativeImage(1, 1, false);
            image.setPixel(0, 0, 0xFFFFFFFF);
            whiteTexture = new DynamicTexture(() -> "caustica RT white", image);
            Minecraft.getInstance().getTextureManager().register(WHITE_LOCATION, whiteTexture);
        } catch (Throwable t) {
            closeWhiteTexture();
            if (!loggedFailure) {
                loggedFailure = true;
                CausticaMod.LOGGER.warn("RT white fallback texture registration failed", t);
            }
        }
    }

    private void closeWhiteTexture() {
        if (whiteTexture != null) {
            try {
                whiteTexture.close();
            } catch (Throwable ignored) {
                // already closed / device gone
            }
            whiteTexture = null;
        }
    }

    /**
     * Map a resolved image-view handle to a stable bindless slot, allocating one on first sight (queued
     * for upload via {@link #uploadPending}). Returns 0 (fallback) when the view is unresolved or the array
     * is full and nothing can be recycled. Deduping by handle is what bounds slot use: distinct render types
     * backed by the same texture share a slot instead of each consuming one.
     */
    private int slotForView(long view, Supplier<String> source) {
        if (view == 0L) {
            return 0;
        }
        long frame = RtComposite.frameCounter();
        Integer cached = viewSlotCache.get(view);
        if (cached != null) {
            touchSlot(cached, frame);
            return cached;
        }
        int slot;
        boolean recycled = false;
        if (nextSlot < slotLimit()) {
            slot = nextSlot++;
        } else {
            slot = recycleColdestSlot(frame);
            if (slot == 0) {
                fullFallbacks++;
                if (!loggedFull) {
                    loggedFull = true;
                    CausticaMod.LOGGER.warn("RT entity texture array full at {} slots (configured {}): every slot is "
                                    + "in use in the same frame, so none can be recycled; further textures fall back "
                                    + "to the white slot",
                            slotLimit(), maxTextures());
                }
                return 0;
            }
            recycled = true;
            evictions++;
        }
        slotsAllocated++;
        if (slotsAllocated <= ALLOCATION_TRACE_LIMIT || (recycled && evictions <= 8L)) {
            // Naming the source of the first allocations is what identifies the textures that exhaust the
            // array: a fixed set of entity files cannot reach the configured capacity, so whatever does
            // shows up here by name.
            CausticaMod.LOGGER.info("RT entity texture slot {} <- {} (view=0x{}){}",
                    slot, source.get(), Long.toHexString(view), recycled ? " [recycled]" : "");
        }
        slotView[slot] = view;
        slotLastUsedFrame[slot] = frame;
        viewSlotCache.put(view, slot);
        pending.add(new Pending(slot, view));
        return slot;
    }

    /**
     * Steal the least-recently-used slot that was not already drawn this frame, returning 0 when there is
     * none. Excluding slots touched in the current frame is what keeps recycling safe: prims recorded
     * earlier this frame carry slot numbers, so rewriting a slot they already reference would re-texture
     * geometry that is about to be traced. The white fallback's reserved slot is never stolen either — it is
     * the one slot the exhausted-array path has to be able to count on.
     *
     * <p>Everything cached against the victim is un-mapped, or the next lookup would hand out a slot that
     * now samples a different texture.
     *
     * <p>This deliberately breaks the old "a slot never changes once assigned" append-only rule. That rule
     * was what kept update-after-bind writes off in-flight frames, but it is also what let a few hundred
     * one-shot textures (item-frame maps, paintings) starve every mob permanently. Recycling rewrites the
     * entity-albedo binding only, which is the one binding created with
     * {@code VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT}; the cost is that a frame still in flight at the
     * moment of a rewrite can observe the new texture for the recycled slot for a frame or two — a bounded,
     * transient artifact in exchange for removing an unbounded permanent one.
     */
    private int recycleColdestSlot(long frame) {
        long oldest = Long.MAX_VALUE;
        int victim = 0;
        for (int slot = 1; slot < nextSlot; slot++) {
            if (slot == reservedWhiteSlot) {
                continue;
            }
            long last = slotLastUsedFrame[slot];
            if (last == frame) {
                continue; // drawn this frame → already-submitted prims still point at it
            }
            if (last < oldest) {
                oldest = last;
                victim = slot;
            }
        }
        if (victim == 0) {
            return 0;
        }
        final int recycledSlot = victim;
        long oldView = slotView[recycledSlot];
        slotView[recycledSlot] = 0L;
        if (oldView != 0L) {
            viewSlotCache.remove(oldView);
        }
        atlasSlotCache.entrySet().removeIf(e -> e.getValue() == recycledSlot);
        return recycledSlot;
    }

    /** Mark a slot as used this frame, so recycling cannot steal a slot a live submission depends on. */
    private void touchSlot(int slot, long frame) {
        if (slot > 0 && slot < slotLastUsedFrame.length) {
            slotLastUsedFrame[slot] = frame;
        }
    }

    /** Write any newly-registered entity textures into the pipeline's bindless set (before the trace). */
    public void uploadPending(RtPipeline pipeline, long sampler) {
        logPeriodicSummary();
        if (pending.isEmpty()) {
            return;
        }
        for (Pending p : pending) {
            pipeline.setEntityAlbedoTexture(p.slot(), p.view(), sampler);
            descriptorWrites++;
        }
        pending.clear();
    }

    /**
     * One line per {@link #SUMMARY_INTERVAL_FRAMES} naming what the entity-texture array is doing: how many
     * slots are in use and how often each silent fallback fired since the session started. Without it,
     * "mis-textured because the array is full" and "mis-textured because a cached handle went stale" are
     * indistinguishable from the log alone — both just show entities shaded from atlas texels.
     */
    private void logPeriodicSummary() {
        long frame = RtComposite.frameCounter();
        if (frame == lastSummaryFrame || frame % SUMMARY_INTERVAL_FRAMES != 0L) {
            return;
        }
        lastSummaryFrame = frame;
        CausticaMod.LOGGER.info("RT entity textures: used={}/{} allocs={} writes={} evictions={} fullFallbacks={} "
                        + "unresolvedFallbacks={} staleViews={} pending={} white={} renderTypes={} atlases={} layers={}",
                nextSlot - 1, slotLimit(), slotsAllocated, descriptorWrites, evictions, fullFallbacks,
                unresolvedFallbacks, staleViewDetections, pending.size(), reservedWhiteSlot,
                viewCache.size(), atlasSlotCache.size(), layerViewCache.size());
    }

    /**
     * Drop the registry (call when the world pipeline / bindless set is recreated, or textures reload).
     * Deliberately does NOT re-reserve the white fallback: this overload is also the shutdown path, where
     * touching the TextureManager again is both pointless and unsafe.
     */
    public void reset() {
        resetRegistry(maxTextures(), false);
    }

    /** Drop the registry for a pipeline whose bindless descriptor arrays have this capacity. */
    public void reset(int descriptorCapacity) {
        resetRegistry(descriptorCapacity, true);
    }

    private void resetRegistry(int descriptorCapacity, boolean reserveFallback) {
        capacity = Math.max(1, descriptorCapacity);
        viewCache.clear();
        layerViewCache.clear();
        locationCache.clear();
        viewSlotCache.clear();
        atlasSlotCache.clear();
        atlasSlotCache.put(TextureAtlas.LOCATION_BLOCKS, 0); // block atlas = the slot-0 fallback
        pending.clear();
        nextSlot = 1;
        reservedWhiteSlot = 0;
        int limit = slotLimit();
        if (slotView.length != limit) {
            slotView = new long[limit];
            slotLastUsedFrame = new long[limit];
        } else {
            Arrays.fill(slotView, 0L);
            Arrays.fill(slotLastUsedFrame, 0L);
        }
        if (reserveFallback) {
            reserveWhiteSlot();
        }
    }

    /**
     * Reserve the white fallback's slot now, while the array is still empty. Reserving it lazily — on the
     * first allocation that needs a fallback — is what made the previous fallback useless in the one case it
     * exists for: an exhausted array had no room for the white texture either, so the "fallback" quietly
     * degraded back to slot 0 (the block atlas).
     */
    private void reserveWhiteSlot() {
        reservedWhiteSlot = whiteSlot();
        if (reservedWhiteSlot == 0) {
            CausticaMod.LOGGER.warn("RT entity texture white fallback could not be reserved; an exhausted array "
                    + "keeps degrading to slot 0 (the block atlas)");
        }
    }

    private int slotLimit() {
        return Math.min(capacity, maxTextures());
    }

    /** Recover the resource Identifier of {@code renderType}'s primary texture (Sampler0), or null. The
     *  {@code RenderSetup.TextureBinding} class is package-private, so {@code location()} is reflective. */
    private Identifier textureLocation(RenderType renderType) {
        if (renderType == null) return null;
        if (locationCache.containsKey(renderType)) return locationCache.get(renderType);
        Identifier result = null;
        try {
            // RenderSetup is final, so the accessor cast must go through Object (the interface is only
            // mixed in at runtime); RenderType is non-final so its cast is fine directly.
            Object setup = ((RenderTypeAccessor) renderType).caustica$state();
            Map<String, ?> textures = ((RenderSetupAccessor) setup).caustica$textures();
            Object binding = textures.get("Sampler0");
            if (binding == null) {
                locationCache.put(renderType, null);
                return null;
            }
            if (locationMethod == null) {
                locationMethod = binding.getClass().getMethod("location");
                locationMethod.setAccessible(true);
            }
            result = (Identifier) locationMethod.invoke(binding);
        } catch (Throwable t) {
            warnMaterialOnce("RT entity texture Identifier resolution failed for " + renderType, t);
        }
        locationCache.put(renderType, result);
        return result;
    }

    private void warnMaterialOnce(String msg, Throwable t) {
        if (!loggedMaterialFailure) {
            loggedMaterialFailure = true;
            CausticaMod.LOGGER.warn(msg, t);
        }
    }

    /** The Vulkan image-view handle of {@code renderType}'s primary texture, or 0 if it can't be resolved. */
    public long resolveView(RenderType renderType) {
        if (renderType == null) {
            return 0L;
        }
        Long cached = viewCache.get(renderType);
        if (cached != null) {
            return cached;
        }
        long handle = prepareViewHandle(renderType);
        // A 0 must NOT be cached: caching it pins this render type to the slot-0 fallback (the block atlas)
        // for the rest of the pipeline's life, even after the texture resolves. Re-resolving costs one
        // prepare() per frame and lets the surface recover on its own.
        if (handle != 0L) {
            viewCache.put(renderType, handle);
        } else {
            unresolvedFallbacks++;
            if (!loggedUnresolved) {
                loggedUnresolved = true;
                CausticaMod.LOGGER.warn("RT entity texture view unresolved for {}; sampling the slot-0 fallback "
                        + "until it resolves", renderType);
            }
        }
        return handle;
    }

    /**
     * Uncached resolve: the current primary image-view handle of {@code renderType}, or 0 when the prepared
     * render type carries no bindable sampler. Stores nothing, so a caller can re-check a surface whose
     * backing image may have been rebuilt since the cached handle was taken.
     */
    private long prepareViewHandle(RenderType renderType) {
        try {
            PreparedRenderType prepared = renderType.prepare();
            GpuTextureView chosen = null;
            GpuTextureView firstNonAux = null;
            for (PreparedRenderType.Texture t : prepared.textures()) {
                String name = t.name();
                if ("Sampler0".equals(name)) {
                    chosen = t.textureView();
                    break;
                }
                if (firstNonAux == null && !"Sampler1".equals(name) && !"Sampler2".equals(name)) {
                    firstNonAux = t.textureView();
                }
            }
            if (chosen == null) {
                chosen = firstNonAux;
            }
            return chosen != null ? vkImageView(chosen) : 0L;
        } catch (Throwable t) {
            if (!loggedFailure) {
                loggedFailure = true;
                CausticaMod.LOGGER.warn("RT entity texture resolution failed for {}", renderType, t);
            }
            return 0L;
        }
    }

    /** The Vulkan image-view handle of one named sampler of {@code renderType}, or 0 if it binds none. */
    private long resolveNamedView(RenderType renderType, String samplerName) {
        try {
            for (PreparedRenderType.Texture t : renderType.prepare().textures()) {
                if (samplerName.equals(t.name())) {
                    return vkImageView(t.textureView());
                }
            }
        } catch (Throwable t) {
            if (!loggedFailure) {
                loggedFailure = true;
                CausticaMod.LOGGER.warn("RT {} resolution failed for {}", samplerName, renderType, t);
            }
        }
        return 0L;
    }

    private static long vkImageView(GpuTextureView view) {
        if (view instanceof VulkanGpuTextureView vulkanView) {
            return vulkanView.vkImageView();
        }
        return 0L;
    }
}

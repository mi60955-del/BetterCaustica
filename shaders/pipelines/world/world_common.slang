// Shared definitions for all ray-tracing stages. The generateShaderRecords task generates typed Java
// records and serializers from Slang reflection.
module world_common;

// Read-only device pointer with explicit std430 layout. The layout parameter is LOAD-BEARING: a bare
// Ptr<T> gets Slang's "natural" layout (no vec4 16-byte alignment → e.g. WorldPush.sunDir lands at 200
// instead of the Java writer's 208, silently reading garbage) and additionally requires the
// scalarBlockLayout device feature. Keep std430 unless RtDeviceBringup explicitly enables that feature;
// this alias is the single layout switch and generated Java serialization stays in lock-step.
public typealias ConstPtr<T> = Ptr<T, Access.Read, AddressSpace.Device, Std430DataLayout>;
public typealias DevicePtr<T> = Ptr<T, Access.ReadWrite, AddressSpace.Device, Std430DataLayout>;

// Push constant block. The hot inline lanes avoid dereferencing WorldPush for values that are read in
// hit shaders or used for raygen control flow — every 64-bit device address lives here for exactly that
// reason, rather than behind the worldPushAddr indirection. tableAddr/entityTableAddr/materialTableAddr
// are read in world.rahit/world.rchit, which never load WorldPush at all; the light-buffer addresses are
// only read in world.rgen, which loads WorldPush anyway, but keeping every Addr in one place means there
// is exactly one struct to check when adding a new device buffer, instead of two.
public struct WorldPushConstants {
    public uint64_t worldPushAddr;
    public uint64_t tableAddr;
    public uint64_t entityTableAddr;
    public uint64_t materialTableAddr;
    public uint64_t lightBufAddr;        // RIS emitter-NEE global light buffer (0 = none published)
    public uint64_t lightAliasAddr;      // power-weighted O(1) alias table (0 = uniform fallback)
    public uint64_t lightLocalAliasAddr; // power aliases relative to each section's range
    public uint64_t lightGridCellAddr;   // dense light grid cell headers (0 = global proposals only)
    public uint64_t lightGridSpanAddr;   // packed weighted section spans referenced by cell headers
    // previous-frame light index -> current-frame light index, or -1 when that emitter is gone. Rebuilt
    // with each published generation and uploaded in the same arena, so it always describes the
    // generation the previous frame actually traced with.
    public uint64_t lightTranslationAddr;
    public uint64_t pathQueueAddr;       // packed primary -> indirect continuation records
    public uint frameIndex;
    // Global surface wetness 0..1, applied to every material's authored porosity. It rides the push
    // constant rather than WorldPush because the closest-hit shader evaluates materials and never
    // dereferences WorldPush at all, so a WorldPush field would cost that hot path an extra address
    // load per hit to fetch one frame-constant scalar.
    public float wetness;
    // How far a parallax-displaced texel may sit below its flat position, in sprite-local UV (one
    // sprite spans one block face, so 0.25 is LabPBR's 25% of a block). Scales the whole parallax
    // response; 0 disables the march entirely, which is also the only way to tell a parallax artifact
    // apart from everything else on a surface that has a height field.
    public float parallaxDepth;
    // Layer budget ceiling for that march, in steps. The walk derives its live count from how far the ray
    // has to travel and caps it here, so this is the quality a frame actually pays for; the compile-time
    // loop bound only has to be at least this large.
    public int parallaxLayers;
    // Material feature switches (bits below), read from the push constant for the same reason wetness is:
    // the closest-hit shader evaluates materials without ever dereferencing WorldPush. Every bit is set by
    // default, so a default frame is identical to one rendered before the switches existed.
    public uint materialToggles;
};

// materialToggles bits. Each one gates a material feature the renderer would otherwise apply whenever the
// asset authors it, so a player can isolate one channel against the rest.
public static const uint MATERIAL_TOGGLE_TEXTURE_EMISSION = 1u;  // a pack's own _s emissiveness
public static const uint MATERIAL_TOGGLE_BLOCK_EMISSION = 2u;    // block-state and block-light emission
public static const uint MATERIAL_TOGGLE_SUBSURFACE = 4u;        // labPBR subsurface scattering
public static const uint MATERIAL_TOGGLE_POROSITY = 8u;          // labPBR porosity, which gates wetness absorption
public static const uint MATERIAL_TOGGLE_ENTITY_PARALLAX = 16u;

// worldPush.flags volumetric bits (Java mirrors them in RtComposite). Every bit is clear by default, so a
// frame rendered before the volumetrics existed is bit-identical to one after: the flags gate every
// volume.slang read, the WorldPush volume block rides along inert.
public static const uint WORLD_FLAG_FOG = 1u << 7u;           // analytic/marched height fog
public static const uint WORLD_FLAG_CLOUD = 1u << 8u;         // planar + volumetric cloud layer
public static const uint WORLD_FLAG_SPRAY = 1u << 9u;         // water-surface spray band
public static const uint WORLD_FLAG_VOLUME_SHADOW = 1u << 10u; // fog in-scatter's sun shadow ray
public static const uint WORLD_FLAG_VOLUME_MARCH = 1u << 11u;  // marched tier (clear = analytic)
public static const uint WORLD_FLAG_VOLUME_DEBUG = 1u << 12u;  // write the volume debug image
// Volumetric cloud slab (Phase 5): with this up the cloud layer is walked through instead of crossed
// once. It is a separate bit from WORLD_FLAG_CLOUD on purpose — the planar sheet stays the base tier and
// the fallback, and every consumer that only needs the layer's mean opacity (the NEE probes, the fog sun
// legs, the continuation-ray attenuation) keeps reading it through WORLD_FLAG_CLOUD either way.
public static const uint WORLD_FLAG_CLOUD_VOLUMETRIC = 1u << 13u;

// Minimal push block for the compute passes that only need to dereference the frame's WorldPush: the sky
// LUT bakes read sun/moon/look state from the same BDA slot the trace does, so there is exactly one
// per-frame source of sky state and no way for the LUT and the frame it shades to disagree.
public struct PushAddr {
    public uint64_t worldPushAddr;
};

// Block-breaking overlay entry: xyz = breaking block position (rebased, same space as gl_WorldRay*),
// w = bindless entityAlbedoTex slot of the destroy-stage texture.
public struct BreakEntry {
    public int4 posSlot;
};

// Per-frame world data, read through a 64-bit device address. WorldPushData.java is generated from this
// reflected type, including its serializer, byte size, vector/matrix types, and fixed-array capacities.
public struct WorldPush {
    public float4x4 invViewProj;
    public float3   camOffset;
    public uint     frameIndex;
    public float4x4 prevViewProj;
    public float3   camDelta;
    public uint     spp;
    public float2   jitter;
    public uint     flags;            // bit0 submerged, bit4 waves, bit5 vanilla sun sprite, bit6 vanilla moon sprite;
    // bit7..bit12 volumetric switches (volume.slang), all clear by default so a default frame
    public uint     maxBounces;
    // ---- Sky state. Only what the CPU alone can know: Minecraft's four eased celestial angles (the
    // 26.2 timeline drives them through a cubic-bezier ease and a datapack may replace the track, so
    // they are read from the attribute probe rather than re-derived from the tick), its star brightness,
    // the moon phase, and the look package's immutable constants. Every direction, colour, level and
    // transmittance is derived in sky.slang from these — no atmosphere maths runs on the CPU any more.
    public float4   celestial;        // x sun angle (rad), y moon angle, z star angle, w MC star brightness 0..1
    // x sun illuminance (lux, top of atmosphere), y full-moon illuminance (lux), z night airglow
    // (cd/m²), w star luminance anchor (cd/m²).
    public float4   skyLook0;
    // x noon south tilt (rad), y sun NEE half-angle (rad), z moon NEE half-angle (rad),
    // w moon-phase fixed light fraction.
    public float4   skyLook1;
    // x drawn sun half-angle (rad), y drawn moon half-angle (rad), z viewer altitude (km),
    // w moon phase index (0 full .. 4 new).
    public float4   skyLook2;
    // x ground albedo, y horizon fade (rad), z end flash NEE half-angle (rad), w end sky gradient gain.
    public float4   skyLook3;
    // Dimension sky state, read from Minecraft's environment attributes so a datapack's custom dimension
    // needs no registry here. x is the sky-light factor: 1 where the sky lights the world, 0 in a dimension
    // such as the Nether or the End that has no sky light at all. yzw is the dimension's own sky colour in
    // linear BT.709, used as the dome where that factor is 0.
    public float4   skyEnvironment;
    // Which sky a sky-light-free dimension renders, as DimensionType.Skybox's ordinal (0 none, 1
    // overworld, 2 end). skyLightFactor alone cannot tell the Nether from the End, and the two need
    // opposite skies. y is the end flash's eased intensity (0..1, 0 when the dimension has no flashes or
    // the option hides them), z/w its x/y angle (rad) exactly as EndFlashState reports them.
    public float4   skyboxState;
    // xyz = the dimension's SKY_LIGHT_COLOR (linear BT.709) — the tint vanilla's lightmap gives the end
    // flash and the end's ambient; w = DimensionType.ambientLight (End 0.25, Nether 0.1, overworld 0).
    public float4   dimensionLight;
    // xyz = the dimension's AMBIENT_LIGHT_COLOR (linear BT.709), the second tint of that ambient floor;
    // w reserved.
    public float4   dimensionAmbient;
    // xyz = the dimension's FOG_COLOR (linear BT.709), the horizon end of the sky dome where the
    // dimension supplies no atmosphere; w reserved.
    public float4   dimensionFog;
    // x end flash NEE illuminance (lux at full intensity), y end flash quad peak luminance (cd/m²),
    // z end ambient illuminance (lux at ambientLight 1.0), w end sky box gain.
    public float4   skyLook4;
    public float4   sunUv;            // vanilla sun sprite atlas rect (u0,v0,u1,v1)
    public float4   moonUv;           // vanilla current-moon-phase sprite atlas rect
    public float4   endFlashUv;       // vanilla end_flash sprite atlas rect (u0,v0,u1,v1)
    public float4   waterParams;      // xyz linear ACEScg camera-biome water tint, w wave time (s)
    public float4   waterAnchor;      // xy world-stable wave-domain anchor, z previous wave time
    // Wave-shape controls from the Video Settings water page: x height scale, y wavelength scale,
    // z wave-clock rate. All 1 reproduces the authored spectrum; w reserved.
    public float4   waterWave;
    public float4x4 curViewProj;      // forward camera-relative view-projection
    public uint     breakCount;
    public BreakEntry breaking[8];    // Java capacity and breakCount serialization are generated from this
    public float4   lightRebase;      // xyz hierarchy-to-current rebase, w inverse total light power
    public float4   lightGridOrigin;     // xyz minimum cell origin in rebased blocks, w cell size
    public int4     lightGridDims;       // xyz dense grid dimensions, w reserved
    public uint     lightCount;       // packed Light records in pc.lightBufAddr
    // Entries in pc.lightTranslationAddr, i.e. the previous generation's light count.
    public uint     lightTranslationCount;
    // Non-zero only on a frame whose light generation differs from the last traced frame's. A history
    // index is already in the current generation on every other frame, so translating then would remap
    // it onto an unrelated emitter; identity is the correct lookup whenever this is zero.
    public uint     lightTranslationValid;
    public uint     risCandidates;    // RIS candidate count M per diffuse vertex (0 = emitter NEE off)
    // Bypasses the bounce-0 temporal merge while still storing this frame's reservoir, so switching it
    // back on starts from a full screen of current data instead of a screen of stale texels. Exists so
    // reuse can be A/B-ed in one frame at one position rather than across two scene visits.
    public uint     risTemporalReuse;
    // Effective-sample ceiling the temporal merge may accumulate to. Equal to risCandidates it degenerates
    // to single-frame shading; 0 is treated the same way rather than as an unbounded history.
    public uint     risTemporalHistory;
    // Pre-exposure scalar applied to radiance at the outImage write
    // ONLY, so all light transport above stays in absolute scene units. Keeps stored fp16 values
    // near mid-grey at any scene brightness; the display pass divides it back out, so the two cancel
    // exactly and this cannot change the image. 1.0 disables it.
    public float    preExposure;
    // Multiplier on water's Beer-Lambert extinction (1 = authored look, 0 = fully clear, clamped to
    // [0, 4] on the CPU). Applies wherever waterExtinction builds a medium, so the camera-submerged
    // start medium and every interface crossing scale identically.
    public float    waterOpacity;
    // Minecraft's own day-cycle fraction, ((gameTime % 24000) + partialTick) / 24000 — exactly what the
    // vanilla `GameTime` global uniform holds, because the end portal's layer scroll is authored against
    // it and advances one full cycle per in-game day. Not a wall-clock seconds value: feeding it seconds
    // runs the scroll ~1200x too fast.
    public float    endPortalTime;
    // ---- Volumetric block (see volume.slang). Every field is zero while its feature is off, so a
    // default frame pushes an inert block and the shaders gate on the flags bits before reading it.
    // x fog extinction at the base (1/block), y height scale H (block), z fog base height (REBASED
    // blocks — Java subtracts the terrain origin, the same anchoring waterAnchor uses), w Henyey–
    // Greenstein anisotropy.
    public float4   fog0;
    // x/z the Java-accumulated wind drift (blocks, per axis, each axis pre-wrapped in whole
    // 4096-block periods — the fog lattice's exact period, so a wrap is invisible by
    // construction), z noise amplitude 0..1, w the shadow-cull floor (fogShadowFloor reads it;
    // unlike the drift lanes it is valid on both fog tiers).
    public float4   fog1;
    // xyz single-scatter albedo (ACEScg linear, also the sigmaS/sigmaT ratio), w ambient-term weight.
    public float4   fogColor;
    // x coverage, y cloud base (rebased blocks), z layer thickness (block), w density multiplier.
    public float4   cloud0;
    // x the vertical profile mix 0..1 the volumetric tier's density uses (0 = a flat stratus board,
    // 1 = a rounded cumulus billow; both normalised to the same column depth), y wind azimuth (rad; the
    // streak frame's direction — the drift itself is accumulated Java-side into cloud2.zw), z the
    // localized-coverage (patchiness) strength the field's macro gate reads, w the planar sheet's fade
    // distance (blocks; sheet-only — the shadow field never dissolves).
    public float4   cloud1;
    // xy the EXACT terrain origin in world blocks (fp32; the streak lattice is not
    // periodic, so unlike the wave domain's mod-4096 anchor this one must not wrap), zw the
    // Java-accumulated cloud wind drift per axis (unwrapped; the accumulator is a double and only
    // sheds sub-block precision past millions of blocks of drift).
    public float4   cloud2;
    // x the field's evolution phase (Java-accumulated, no wrap: the noise lattices it offsets are not
    // periodic, and at the rate it advances fp32 keeps sub-cell precision for days of continuous play),
    // yz unused, w ambient-term weight.
    public float4   cloudColor;
    // x rain level 0..1, y thunder level 0..1, z spray strength, w the previous frame's rain level
    // (temporal-stability work interpolates toward it rather than jumping on a weather change).
    public float4   weather;
    // x march step count, y jitter strength 0..1, z max integration distance (block, 0 = ray tmax),
    // w reserved.
    public float4   volumeParams;
    // ---- Water-volume block (the water medium's own scattering + the spray band). x turbidity, the
    // water's single-scatter fraction sigmaS/sigmaT in 0..1 (0 = the feature is off, the same
    // zero-when-inert invariant the rest of the block keeps); y the Henyey–Greenstein g of the
    // water's phase; z spray band height in blocks; w the spray band's base height in REBASED blocks
    // (the sea level minus the terrain origin, the same anchoring fog0.z uses). The spray strength
    // itself rides weather.z, whose lane the plan reserves for it.
    public float4   water1;
};

// 32-byte hot area-light record. Linear ACEScg radiance uses packed R11G11B10; the
// grid-relative owner section uses three unsigned 10-bit coordinates; the geometric normal is
// reconstructed from the half axes.
//
// 32 divides the 64-byte cache line, so a record never straddles one. RIS fetches random records M times
// per shading vertex, making this alignment important. float3 forces 16-byte struct alignment and 32 is
// a multiple of it, so this lands with no tail padding.
//
// The rectangle spans 2U x 2V, so its area is derived as 4*|U x V|. lightGeometricNormal already
// needs that cross product, allowing the compiler to share the length with normalization.
public struct Light {
    public float3 pos;      // rebased world centre, f32 (the RIS target divides by distance to it)
    public uint   le;       // bitcast R11G11B10 radiance
    public uint   halfUxy;  // half2(U.x, U.y)
    public uint   halfUzVx; // half2(U.z, V.x)
    public uint   halfVyz;  // half2(V.y, V.z)
    public uint   section;  // 10:10:10 grid-relative section coord + bit 30 normal-flip
};

// Eight-byte Vose alias-table column. PDFs are reconstructed from the selected Light.
public struct LightAlias {
    public uint  aliasIndex;
    public float accept;
};

// Dense light grid cell entry. Empty coordinates have spanCount=0; populated section cells point into
// an immutable list of weighted section light ranges built and published alongside this grid.
public struct LightGridCell {
    public uint spanOffset;
    public uint spanCount;
    public float invWeightSum;
};

// One 16-byte Vose alias column for a cell's section distribution. The two light counts are uint16s.
public struct LightGridSpan {
    public uint firstLight;
    public uint aliasFirstLight;
    public uint packedLightCounts; // low16 self, high16 alias
    public float accept;
};

// TerrainPrim.flags bit 0: this emissive quad is in the light buffer (RtLightCollector membership).
public static const uint TERRAIN_PRIM_IN_LIGHT_BUFFER = 1u;

// Largest finite half. Producers of fp16 HDR storage clamp to this instead of letting the conversion
// round up to +inf and propagate as NaN — world.rgen's rgba16f outImage store most of all, which is the
// one place the whole path integral's absolute radiance meets a half.
public static const float HALF_MAX = 65504.0;

// Radiance-ray payload (location 0). Member order and types are a cross-stage ABI.
//
// Three-component quantities are packed into half pairs rather than declared as float3: they cost 6 bytes
// instead of 12, and payload storage is reserved per trace call site, so every byte here is paid twice per
// radiance trace (once for TraceRay, once for Invoke) and is preserved across the SER reorder. hitT stays
// f32 — it reaches 10000 blocks and the hit position is reconstructed from it, where half's ~4-block
// spacing at that magnitude would be visible.
public struct Payload {
    // surface0..2 are a hand-packed UNION of the two things a radiance ray can come back with. Reach them
    // only through unpackAlbedo/packAlbedo, unpackNormal/packNormal, unpackSky/packSky — the raw words
    // have no meaning without knowing which case you are in.
    //
    //   hit  — half3 albedo + half3 normal
    //   miss — float3 sky radiance, full fp32
    //
    // The miss case is the entire reason these are hand-packed. Sky is in absolute photometric units and
    // the sun disc alone is ~3.6e5 cd/m², well past half's 65504 ceiling — a
    // half3 could not carry it without first normalising by the frame's exposure, which made a storage
    // format's limitation into a dependency on the exposure controller. A hit has no sky and a miss has
    // neither albedo nor normal, so reinterpreting the same twelve bytes buys the range for free and keeps
    // the payload exactly the size it was.
    //
    // The albedo lane doubles as the shadow ray's accumulated transmittance (see makeShadowPayload).
    public uint   surface0;    // hit: half2(albedo.r, albedo.g) | miss: asuint(sky.r)
    public uint   surface1;    // hit: half2(albedo.b, normal.x) | miss: asuint(sky.g)
    public uint   surface2;    // hit: half2(normal.y, normal.z) | miss: asuint(sky.b)
    public float  hitT;        // >= 0 on hit, < 0 on miss.
    public half3  motionPrev;  // per-vertex world displacement since last frame.
    public half3  f0;          // specular F0.
    public uint   flags;       // bits 0..1 material, bit 2 celestial, bit 3 water-entering, bits 4..6 emission source.
    public uint   roughMetal;  // packHalf2x16(roughness, metalness)
    public uint   emissionSss; // packHalf2x16(emission, sss)
    public uint   iorTransmission; // packHalf2x16(IOR, transmission factor)
    public uint   rayCone;     // packHalf2x16(width, spread)
};

public static const uint PAYLOAD_SHOW_CELESTIAL = 4u;
// Set by world.rchit on a terrain hit whose prim carries TERRAIN_PRIM_IN_LIGHT_BUFFER: this emitter is
// RIS-sampled, so raygen gates its direct-hit emission on diffuse continuation rays (no double count).
public static const uint PAYLOAD_EMITTER_IN_LIST = 128u;
// Set on a hit whose prim carries ENTITY_PRIM_UNLIT: the surface contributes its own emission and no
// diffuse, subsurface or specular response, because what it was authored with is already a final colour.
public static const uint PAYLOAD_UNLIT = 256u;
// Set by world.rchit on any dielectric hit (water or glass/ice): true when the incoming ray travels
// against the prim's outward face normal (entering the volume), false when it exits. Derived from face
// orientation rather than toggled, so a stray or missing face cannot corrupt the medium for the rest of
// the path — each crossing re-derives which side it is on. Bit 2 (PAYLOAD_SHOW_CELESTIAL) is only
// meaningful on a miss and this bit only on a hit, so they never need to be read together, but they get
// distinct bits anyway to keep the two concerns unambiguous.
public static const uint PAYLOAD_DIELECTRIC_ENTERING = 8u;
// Set on a trace whose ONLY consumer is the guide buffers (Pass A's primary trace and every guide
// probe). Alpha coverage then has to resolve deterministically: the diffuse-albedo guide is what
// DLSS-RR demodulates against, and a coverage that flips per pixel per frame is a demodulation signal
// no accumulation can hold, so the sheet's noise survives as a smear instead of averaging to its mean.
// The radiance pass leaves this clear and keeps the dither, which integrates the same composite.
public static const uint PAYLOAD_GUIDE_STABLE = 512u;
public static const uint PAYLOAD_EMISSION_SOURCE_SHIFT = 4u;
public static const uint PAYLOAD_EMISSION_SOURCE_MASK = 7u;
public static const uint EMISSION_SOURCE_NONE = 0u;
public static const uint EMISSION_SOURCE_UNIFORM = 1u;
public static const uint EMISSION_SOURCE_HEURISTIC = 2u;
public static const uint EMISSION_SOURCE_LAB_PBR = 3u;

public static const uint PAYLOAD_MATERIAL_MASK = 3u;
public static const uint MATERIAL_OPAQUE = 0u;
// Both dielectrics are volumes: they refract and push a participating medium. They stay distinct ids
// because water owns behaviour nothing else has — an animated wave surface, the caustic term, biome-tint
// absorption calibrated per block of depth, and a chit path fed by the fluid mesher rather than the
// (inset) translucent terrain layer. MATERIAL_DIELECTRIC is everything else transparent: glass, ice, and
// any solid transparent block a pack defines.
public static const uint MATERIAL_WATER = 1u;
public static const uint MATERIAL_PARTICLE = 2u;
public static const uint MATERIAL_DIELECTRIC = 3u;

// Entity per-triangle record (48 B). The final lane mirrors TerrainPrim's integer material metadata.
public struct Prim {
    public float4 normal; // xyz = geometric normal, w = per-primitive emission strength
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0; // asfloat() = the submission's vertex alpha; vanilla puts some transparency there
    public uint aux1;
};

// The submission came from a vanilla pipeline with no lighting term, so its colour is already final and
// the path tracer must not light it as well as emit it. Assigned by RtEntityCollector from
// RtUnlitPipelines, which matches on the pipeline's fragment shader.
public static const uint ENTITY_PRIM_UNLIT = 1u;

// The submission is vanilla's end portal / end gateway composite: its colour is a base texture plus a
// stack of transformed samples of a second texture, so the hit shader must reproduce that sum instead of
// shading one image. The prim's aux1 lane carries the layer texture's bindless slot in its low 16 bits
// and the layer count in the high 16.
public static const uint ENTITY_PRIM_END_PORTAL = 2u;

// Terrain per-triangle record (48 B). The final 16-byte lane contains integer geometry-local metadata;
// shared surface parameters live once in MaterialHeader instead of being copied into every triangle.
public struct TerrainPrim {
    public float4 normal;
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0;
    public uint aux1;
};

// Compiled material record (80 B). Source-format data has already been decoded into canonical pages;
// texturePage indexes the dedicated canonical page arrays. materialUv maps sprite-local UV into the packed
// page, while albedoUv maps the vanilla atlas rectangle back to sprite-local UV.
public struct MaterialHeader {
    public uint model;
    public uint features;        // low feature bits; bits 24..31 = maximum valid material-page LOD
    public uint texturePage;
    public uint extensionOffset; // zero when no optional extension record is present
    public float4 materialUv; // packed-page uvMin.xy + uvExtent.xy
    public float4 albedoUv;   // vanilla atlas uvMin.xy + inverse uvExtent.xy
    public float4 params;  // roughness, metalness, IOR, transmission
    public float4 average; // whole-sprite linear-input average RGBA for translucent shadow filtering
};

public static const uint MATERIAL_FEATURE_SPEC = 1u;
public static const uint MATERIAL_FEATURE_NORMAL = 2u;
public static const uint MATERIAL_FEATURE_HEURISTIC_EMISSION = 4u;
// The _n alpha carries a real height field — at least one texel strictly between flush and maximum
// depth. Cleared for both degenerate fills, so a pack that never authored the channel, or one whose
// alpha was left at 0, costs the parallax march nothing.
public static const uint MATERIAL_FEATURE_HEIGHT = 8u;
public static const uint MATERIAL_FEATURE_STOCHASTIC_ALPHA = 16u;
// Final HDR emitting-surface luminance (look-package baseline or absolute JSON cd/m² override), baked in Java
// at material-compile time (RtMaterialRegistry) and packed here as a 16-bit fraction of the max — every
// emissive material carries a value (0 for non-emissive), not just resource-pack-overridden ones, so the
// full features word's spare 16 bits (8..23) hold the encoded value.
public static const uint MATERIAL_EMISSION_STRENGTH_SHIFT = 8u;
public static const uint MATERIAL_EMISSION_STRENGTH_MASK = 65535u;
// Mirrors RtMaterialRegistry.MAX_EMISSION_STRENGTH — HALF_MAX, the ceiling emission can survive through
// Payload.emissionSss (half2) and Light.le (R11G11B10). Emission is measured in cd/m².
public static const float MATERIAL_MAX_EMISSION_STRENGTH = 65504.0;
public static const uint MATERIAL_MAX_LOD_SHIFT = 24u;
public static const uint MATERIAL_MAX_LOD_MASK = 255u;

// Per-section geometry record: packed buffer addresses + per-bucket triangle offsets
// (gl_PrimitiveID restarts at 0 per geometry, so hit shaders re-add triBase[geometryIndex]).
public struct Section {
    public uint64_t primAddr;
    public uint64_t uvAddr;
    public uint triBase[4]; // fixed bucket order: solid/cutout/translucent/water
};

// Entity geometry record: packed buffer addresses, motion, and per-geometry triangle bases. Entity BLAS
// geometry order is opaque/any-hit; PrimitiveIndex restarts at zero for each geometry.
public struct EntityGeom {
    public uint64_t primAddr;
    public uint64_t idxAddr;
    public uint64_t uvAddr;
    public uint64_t dispAddr;
    public float4 rigidDisp;
    public uint triBase[2];
    public uint2 reserved;
};

// TLAS instance-custom-index encoding + terrain geometry buckets (shared by rahit/rchit).
public static const uint ENTITY_BIT = 0x800000u;
public static const uint PARTICLE_BIT = 0x400000u; // particles share the entity cutout path
public static const uint IDX_MASK = 0x3FFFFFu;     // low 22 bits = geom-table index
public static const uint BUCKET_CUTOUT = 1u;
public static const uint BUCKET_TRANSLUCENT = 2u;
public static const uint BUCKET_WATER = 3u;

// packHalf2x16 equivalent (Slang core module has no GLSL pack builtin; f32tof16 returns the half
// bit-pattern in the low 16 bits).
public uint packHalf2(float2 v) {
    return (f32tof16(v.y) << 16) | (f32tof16(v.x) & 0xFFFFu);
}

public float2 unpackHalf2(uint p) {
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}

// ---- Payload.surface0..2 accessors. Two views of the same twelve bytes; see the struct for why.
//
// The hit view straddles surface1 (albedo.b in its low half, normal.x in its high half), so the two
// setters are read-modify-write on that word. That is a couple of ALU ops on a value already in
// registers, and it buys a payload that did not have to grow. Each setter touches only its own halves,
// so writing one before the other is initialised is safe — nothing here does arithmetic on the half it
// is preserving.
public float3 unpackAlbedo(Payload p) {
    return float3(unpackHalf2(p.surface0), unpackHalf2(p.surface1).x);
}

public void packAlbedo(inout Payload p, float3 albedo) {
    p.surface0 = packHalf2(albedo.xy);
    p.surface1 = (p.surface1 & 0xFFFF0000u) | (f32tof16(albedo.z) & 0xFFFFu);
}

public float3 unpackNormal(Payload p) {
    return float3(unpackHalf2(p.surface1).y, unpackHalf2(p.surface2));
}

public void packNormal(inout Payload p, float3 normal) {
    p.surface1 = (p.surface1 & 0x0000FFFFu) | (f32tof16(normal.x) << 16);
    p.surface2 = packHalf2(normal.yz);
}

// Miss view: full fp32, so absolute photometric sky needs no pre-exposure round trip through the payload
// and no clamp to keep it finite. The fp16 ceiling still applies at the outImage store in world.rgen,
// where the finite-half clamp guards every radiance source.
public float3 unpackSky(Payload p) {
    return float3(asfloat(p.surface0), asfloat(p.surface1), asfloat(p.surface2));
}

public void packSky(inout Payload p, float3 sky) {
    p.surface0 = asuint(sky.x);
    p.surface1 = asuint(sky.y);
    p.surface2 = asuint(sky.z);
}

// The ray tracer's transport contract is scene-linear ACEScg (AP1/D60). Minecraft assets and
// captured vertex colours are authored as sRGB/BT.709 (D65), so colour inputs cross this OCIO-derived
// chromatic-adaptation + primary-conversion seam exactly once before they enter any BRDF, throughput,
// medium, light, guide, or radiance calculation.
public static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);
public static const float3 BT709_TO_ACESCG_R = float3(0.61309743, 0.33952314, 0.04737945);
public static const float3 BT709_TO_ACESCG_G = float3(0.07019372, 0.91635388, 0.01345240);
public static const float3 BT709_TO_ACESCG_B = float3(0.02061559, 0.10956977, 0.86981463);

public float3 bt709ToAcesCg(float3 color) {
    return float3(
            dot(color, BT709_TO_ACESCG_R),
            dot(color, BT709_TO_ACESCG_G),
            dot(color, BT709_TO_ACESCG_B));
}

public float3 srgbToLinear(float3 color) {
    float3 lo = color / 12.92;
    float3 hi = pow((color + 0.055) / 1.055, float3(2.4));
    return lerp(lo, hi, step(float3(0.04045), color));
}

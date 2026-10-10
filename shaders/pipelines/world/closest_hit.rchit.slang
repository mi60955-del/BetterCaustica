// Closest-hit shader. Geometry is per-section: InstanceID() indexes a section table (BDA array reached
// from the push constant) holding this section's {prim, uv} buffer addresses. Per-primitive
// {normal, tint} plus barycentric atlas UVs produce the payload guides; lighting is deferred to raygen.
//
// tableAddr/entityTableAddr are read from the inline push-constant lanes (pc), not through the
// WorldPush BDA struct: they're read on every single hit (the hottest chain in the frame), and going
// through worldPushAddr first would cost an extra global-memory load just to find the address of the
// load we actually want. The cold WorldPush struct is dereferenced only for the breaking overlay.
import world_common;
import math;
import bindings;

void payloadSetPacked(inout Payload payload, uint material, float roughness, float metalness,
        float emission, float sss, float ior, float transmission, uint emissionSource) {
    payload.flags = material | (emissionSource << PAYLOAD_EMISSION_SOURCE_SHIFT);
    payload.roughMetal = packHalf2(float2(roughness, metalness));
    payload.emissionSss = packHalf2(float2(emission, sss));
    payload.iorTransmission = packHalf2(float2(ior, transmission));
}

// Which side of a dielectric face this hit is on, shared by every hit path. Comes from the face
// orientation, so it is re-derived at each crossing instead of toggled (see PAYLOAD_DIELECTRIC_ENTERING).
void payloadSetDielectric(inout Payload payload, uint material, bool entering) {
    if (material != MATERIAL_WATER && material != MATERIAL_DIELECTRIC) return;
    if (entering) payload.flags |= PAYLOAD_DIELECTRIC_ENTERING;
}

uint materialEmissionSource(MaterialHeader header, float emission) {
    if (emission <= 0.0) return EMISSION_SOURCE_NONE;
    if ((header.features & MATERIAL_FEATURE_SPEC) != 0u) return EMISSION_SOURCE_LAB_PBR;
    if ((header.features & MATERIAL_FEATURE_HEURISTIC_EMISSION) != 0u) return EMISSION_SOURCE_HEURISTIC;
    return EMISSION_SOURCE_UNIFORM;
}

// Strength of the LabPBR _n ambient-occlusion (blue) applied to albedo. Kept mild because the path
// tracer already computes its own AO from sky-visibility rays (avoid double-darkening).
static const float NORMAL_AO_STRENGTH = 0.5;

// The two ends of the wetness response. A film turns any surface toward the water's own roughness
// regardless of what it covers; the albedo limit is what a fully soaked, fully porous texel reads once
// its pores fill — roughly what dry sand or soil does once wet. A drier world or a less porous surface
// lands proportionally short of both.
static const float WET_ROUGHNESS = 0.06;
static const float WET_ALBEDO_SCALE = 0.55;
// A vertical face still catches rain even though it pools nothing; only the up-facing end of the range
// is about standing water. Without this floor the response would switch off at exactly the faces that
// rain hits most.
static const float WET_GRAZING_FLOOR = 0.45;

static const float RAY_CONE_MIN_WIDTH = 1.0e-5;
static const float RAY_CONE_MIN_TEXEL_FOOTPRINT = 1.0e-4;
static const float RAY_CONE_MAX_LOD = 12.0;

// LabPBR stores height in the _n alpha as a depth: 255 is flush with the flat surface and 0 is 25% of
// a block deep. Steep parallax marches the view ray through that field and bisects the crossing, which
// is what recesses a face without any displacement geometry. How much of that 25% the march is allowed
// to reach is pc.parallaxDepth rather than a constant: it is the one knob that makes this whole path
// visible or invisible, and the only way to tell its artifacts from everything else on a surface.
//
// A hit shader has no dFdx/ddy, so the parallax scale cannot come from screen-space derivatives the way
// a rasterized POM would. It comes from the height field's own units instead: a sprite spans exactly one
// block face, so the spec's 25% depth is 0.25 of the [0,1] sprite UV the march already works in.
// Layer count trades banding against fetch count. The bisection removes most of the banding the layer
// count alone would leave, so a moderate count still resolves the crossing.
// The walk's layer budget is derived per hit rather than fixed (see parallaxLocalUv); these bound it.
// The floor keeps a shallow field from resolving its crossing too coarsely. The ceiling is only the
// compile-time bound the loop needs — the budget a frame actually spends is the Materials page's parallax
// quality, clamped into this range — so it is set past that control's own maximum.
static const int PARALLAX_MIN_LAYERS = 8;
static const int PARALLAX_MAX_LAYERS = 64;
static const int PARALLAX_REFINE_STEPS = 4;
// Tangent-space Z floor. Dividing the view's XY by Z projects it onto the surface, and a grazing view
// would stretch that to infinity without a floor; the offset stays inside the spec's 25% either way.
static const float PARALLAX_MIN_VZ = 0.2;
// Past this material LOD the displaced sample is smaller than the texel footprint it came from, so the
// march would spend its fetches to move the result by less than the error the mip already carries.
static const float PARALLAX_MAX_LOD = 2.0;
// The height drop across one texel, measured along the view's lateral direction, at which a displaced
// band counts as a full vertical wall. A binary height field (255 flush next to 0) reaches this; the
// shallower fields most packs author land well under it and keep most of their normal map, which is
// what keeps a smooth material from turning into flat vertical stripes where it happens to displace.
static const float PARALLAX_WALL_SLOPE = 0.35;
// The outer band of a sprite over which the reach is ramped to zero. A displaced sample must not be
// pulled across the sprite's border, where the page holds its neighbour's replicated gutter.
static const float PARALLAX_EDGE_FADE_MIN = 0.015;
static const float PARALLAX_EDGE_FADE_MAX = 0.09;
// Crossing depth, as a fraction of the material's full height range, at which the displacement reaches
// full strength. Below it the offset is ramped in, which is what keeps a slightly recessed texel from
// jumping to a full-texel step next to a flush one.
static const float PARALLAX_DEPTH_SOFTNESS = 0.30;

float2 texSizePx(Sampler2D t) {
    uint w, h;
    t.GetDimensions(w, h);
    return float2(w, h);
}

float rayConeHitWidth(uint rayCone) {
    float2 cone = unpackHalf2(rayCone);
    return max(cone.x + cone.y * max(RayTCurrent(), 0.0), RAY_CONE_MIN_WIDTH);
}

float edgeTexelsPerWorld(float3 p0, float3 p1, float2 uv0, float2 uv1, float2 textureSizePx) {
    float worldLen = length(p1 - p0);
    if (worldLen <= 1.0e-7) {
        return 0.0;
    }
    return length((uv1 - uv0) * textureSizePx) / worldLen;
}

float rayConeTextureLod(uint rayCone, float2 textureSizePx, float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2) {
    float texelsPerWorld = max(edgeTexelsPerWorld(p0, p1, uv0, uv1, textureSizePx),
            max(edgeTexelsPerWorld(p1, p2, uv1, uv2, textureSizePx),
                edgeTexelsPerWorld(p2, p0, uv2, uv0, textureSizePx)));
    float footprint = max(rayConeHitWidth(rayCone) * texelsPerWorld, RAY_CONE_MIN_TEXEL_FOOTPRINT);
    return clamp(log2(footprint), 0.0, RAY_CONE_MAX_LOD);
}

float rayConeUnitUvLod(uint rayCone, float2 textureSizePx) {
    float footprint = max(rayConeHitWidth(rayCone) * max(textureSizePx.x, textureSizePx.y), RAY_CONE_MIN_TEXEL_FOOTPRINT);
    return clamp(log2(footprint), 0.0, RAY_CONE_MAX_LOD);
}

// Build the hit triangle's tangent frame from its vertex positions (VK_KHR_ray_tracing_position_fetch)
// and UVs. `n` must already face the viewer, since the basis is built in that hemisphere. Returns false
// on a degenerate UV triangle, where the frame is undefined and every consumer falls back to the
// geometric normal and the flat UV. Instance transforms are translation-only, so object edges equal
// world edges and the frame needs no transform of its own.
bool tangentFrame(float3 n, float3 p0, float3 p1, float3 p2, float2 t0, float2 t1, float2 t2,
        out float3 T, out float3 B) {
    T = float3(0.0, 0.0, 0.0);
    B = float3(0.0, 0.0, 0.0);
    float2 g1 = t1 - t0;
    float2 g2 = t2 - t0;
    float det = g1.x * g2.y - g1.y * g2.x;
    if (abs(det) <= 1.0e-12) {
        return false;
    }
    float r = 1.0 / det;
    float3 traw = ((p1 - p0) * g2.y - (p2 - p0) * g1.y) * r;
    T = normalize(traw - n * dot(n, traw));       // Gram-Schmidt against the viewer-oriented normal
    B = cross(n, T) * (det < 0.0 ? -1.0 : 1.0);   // handedness from the UV winding
    return true;
}

// Rotate a LabPBR _n tangent-space normal into world space over a frame from tangentFrame. Clamps the
// result above the horizon so grazing perturbations don't invert it through the surface (the black-spot
// fix). Reports the blue channel's AO through `ao`.
float3 perturbNormal(float3 n, float3 T, float3 B, float3 vdir, float4 ntex, out float ao) {
    float2 nxy = ntex.xy * 2.0 - 1.0;                    // R,G = tangent-space X,Y
    float nz = sqrt(max(0.0, 1.0 - dot(nxy, nxy)));      // reconstruct Z
    float3 nm = normalize(nxy.x * T + nxy.y * B + nz * n);
    float NoV = dot(nm, vdir);
    if (NoV < 0.02) {
        nm = normalize(nm + vdir * (0.02 - NoV));        // keep above the horizon (no flip)
    }
    ao = lerp(1.0, ntex.b, NORMAL_AO_STRENGTH);
    return nm;
}

// Keep a sprite-local UV inside [0,1). The march only ever moves a couple of texels on the material
// maps this ships against, so this bound is rarely reached — but it must be a clamp and not a wrap:
// an offset that steps past the edge by a fraction of a texel would otherwise fold to the opposite
// side of the sprite, turning a sub-texel shift into a jump across the whole face. Clamping just holds
// the sample on the border texel, which is what the page's replicated gutter would have returned
// anyway.
float2 clampSpriteUv(float2 uv) {
    return clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));
}

// How many texels this sprite spans at a given LOD — i.e. the grid a displaced sample has to be
// quantised to. Derived from the page's dimensions and the sprite's share of it, halved per LOD, so
// the grid follows the resolution the height field is actually being read at rather than the mip-0
// one. Snapping to mip 0 at distance would quantise finer than the samples themselves and pop.
float2 materialSpriteGrid(uint page, float2 pageUvScale, float lod) {
    float2 pageTexels = texSizePx(materialNormalAoTex[NonUniformResourceIndex(page)]);
    return max(floor(pageUvScale * pageTexels / exp2(floor(lod)) + 0.5), float2(1.0, 1.0));
}

// Steep parallax plus a bisection over the LabPBR height field: walk the view ray through the height
// volume a layer at a time until the marched depth passes the stored surface, then bisect that last
// interval so the answer lands between layers instead of snapping to one. Returns the sprite-local UV
// to sample at, or the input unchanged when the ray never crosses — which is what a flush (height 255)
// surface should read as.
//
// `stepNormal` reports the band the displacement exposes, in the tangent space of the hit triangle:
// x,y is the unit wall direction and z is how much of a wall that band is (0 = none, so a caller that
// gets the flat mapping back has nothing to apply).
//
// Four things keep a displaced surface reading as geometry instead of as a smear. The last two are
// ports of the shipped 0.3.0 build's parallax, which solved the same problem with a texel-walking DDA
// behind an explicit envelope; the first two are that envelope.
//
//   * The reach is ramped to zero over the sprite's outermost texels. Without it a sample pulled
//     sideways off the sprite reads its neighbour's replicated gutter, so every tile boundary grows a
//     hard seam.
//   * The reach is ramped out as the material LOD approaches PARALLAX_MAX_LOD too, replacing the cliff
//     that used to switch the whole march off there. Past that LOD a displaced sample is no longer
//     resolved, so leaving the last stretch of distance on a ramp keeps it from popping.
//   * The march is budgeted against its own reach. A reach spanning more texels than the walk has
//     layers lands its crossing between texels and the band it uncovers smears. The budget is derived
//     per hit — one layer per texel of travel — and beyond its cap the sampling LOD is raised, which
//     halves the sprite's texel count, so an extreme reach is answered with a coarser reading of the
//     same height field instead of a smeared one. This is what makes the shipped build hold up at
//     grazing angles.
//   * The displacement is quantised to the shading grid — the sprite's texel grid at the LOD the
//     samples are read at — but quantised as an *offset*, and only once it spans a whole texel, so
//     that the step stays crisp without either dragging the flat mapping onto the lattice or cutting
//     the fades above into hard contours. See the block comment at the quantisation itself.
//
// And the band is told it is a wall: the march moves the sample away from the viewer, so the band it
// uncovers is the near wall of the step and that wall faces back along the viewer's lateral direction.
// How much of a wall it really is comes from the height field's own gradient across one texel along
// that direction, so a binary field reads full strength while a gently sloped one reads near zero and
// keeps its normal map — a hard switch would flatten every displaced band on a smooth material into
// a stripe.
float2 parallaxLocalUv(MaterialHeader header, uint page, float2 localUv, float lod, float maxLod,
        float3 n, float3 T, float3 B, float3 vdir, float parallaxDepth, out float3 stepNormal) {
    stepNormal = float3(0.0, 0.0, 0.0);
    float3 vTS = float3(dot(vdir, T), dot(vdir, B), dot(vdir, n));
    float vz = max(abs(vTS.z), PARALLAX_MIN_VZ);
    float2 slope = vTS.xy / vz;
    float2 pageUvMin = header.materialUv.xy;
    float2 pageUvScale = header.materialUv.zw;

    float2 edge = min(localUv, float2(1.0, 1.0) - localUv);
    float edgeFade = smoothstep(PARALLAX_EDGE_FADE_MIN, PARALLAX_EDGE_FADE_MAX, min(edge.x, edge.y));
    float lodFade = 1.0 - smoothstep(PARALLAX_MAX_LOD - 1.0, PARALLAX_MAX_LOD, lod);
    float reach = parallaxDepth * edgeFade * lodFade;
    if (reach <= 1.0e-5) {
        return localUv;
    }

    // Budget the walk against the reach: one layer per texel of lateral travel, so a shallow or head-on
    // surface pays for very few height fetches while a grazing one buys the resolution it needs to land
    // its crossing on the right texel. The Materials page's parallax quality caps that derived count;
    // deriving the count outright is what keeps the coarsening below from firing in ordinary play.
    float2 grid = materialSpriteGrid(page, pageUvScale, lod);
    float travelTexels = abs(slope.x) * reach * grid.x + abs(slope.y) * reach * grid.y;
    int budget = clamp(pc.parallaxLayers, PARALLAX_MIN_LAYERS, PARALLAX_MAX_LAYERS);
    int layers = clamp(int(ceil(travelTexels)) + 1, PARALLAX_MIN_LAYERS, budget);

    // Should the reach still want more texels than the cap allows, coarsen the *march's own* reading
    // of the height field — which halves the sprite's texel count — until a layer covers at most one
    // texel. A coarser reading of the same height field, rather than a smeared one across a finer one.
    //
    // This deliberately does not touch `grid`. That grid quantises the displacement below, and the
    // displaced sample is a *shading* sample, so its quantum has to be the resolution the shading is
    // actually read at. Quantising it on a coarser grid instead moves the sample several texels at a
    // time and makes the texture repeat at that coarser resolution.
    float coarsen = max(ceil(log2(max(travelTexels, 1.0) / float(layers))), 0.0);
    float parallaxLod = min(lod + coarsen, maxLod);

    float2 depthStep = slope * (reach / float(layers));
    float layerDepth = 1.0 / float(layers);
    // Depth is measured inward from the flat surface, so a stored height of 1 (flush) is depth 0.
    float2 uv = localUv;
    float depth = 0.0;
    float surface = 1.0
            - samplePageNormalAo(page, pageUvMin + clampSpriteUv(uv) * pageUvScale, parallaxLod).a;
    float2 prevUv = localUv;
    float prevDepth = 0.0;
    [loop]
    for (int i = 0; i < PARALLAX_MAX_LAYERS; i++) {
        if (i >= layers || depth >= surface) break;
        prevUv = uv;
        prevDepth = depth;
        uv -= depthStep;
        depth += layerDepth;
        surface = 1.0
                - samplePageNormalAo(page, pageUvMin + clampSpriteUv(uv) * pageUvScale,
                        parallaxLod).a;
    }
    // A flush texel — the ray never descended — is the only case that stays untouched, and it is the
    // degenerate one rather than a threshold. The threshold this replaces was `depth <= 1/layers`, and
    // `layers` itself shrinks as the reach shrinks, so lowering POM Depth moved the cutoff and grew the
    // untouched cells instead of shrinking the relief: a texel just past the cutoff got the full
    // displacement while its neighbour got none, painting the height field's own lattice onto the
    // surface as hard cells. The ramp below replaces that with a continuous one.
    if (depth <= 1.0e-6) {
        return localUv;
    }
    // Bisect the final interval: the crossing is between prevUv (still in front of the surface) and uv
    // (already behind it). The bracketing pair stays in unwrapped space; only the samples clamp.
    float2 lo = prevUv;
    float2 hi = uv;
    float dLo = prevDepth;
    float dHi = depth;
    for (int j = 0; j < PARALLAX_REFINE_STEPS; j++) {
        float2 mid = (lo + hi) * 0.5;
        float dMid = (dLo + dHi) * 0.5;
        float h = 1.0
                - samplePageNormalAo(page, pageUvMin + clampSpriteUv(mid) * pageUvScale,
                        parallaxLod).a;
        if (h < dMid) {
            hi = mid;
            dHi = dMid;
        } else {
            lo = mid;
            dLo = dMid;
        }
    }
    // Quantise the displacement, not the resulting UV, and only once it is a whole texel.
    //
    // Snapping the UV itself drags an *undisplaced* sample onto the texel lattice, so the whole
    // surface renders as a grid of flat texels — worst exactly where the displacement is smallest,
    // because there a sub-texel offset rounds the sample onto the nearest centre instead of leaving it
    // alone. On screen that reads as the surface cut into cells that hide the relief inside them, and
    // shrinking the POM Depth makes the cells *larger* rather than smaller, which is the giveaway.
    //
    // Rounding the offset instead fixes that, but rounding unconditionally puts a cliff at half a
    // texel: the reach ramps in the edge fade and the distance fade both drive the offset through that
    // point, so each of them would switch the parallax off along a hard contour — a flat frame around
    // every sprite, and a ring at the distance where the fade crosses. Rounding only from one texel up
    // moves the switch to where `round` and the continuous value already agree, so there is nothing to
    // see at the transition, and a sub-texel offset stays continuous — where it is harmless, being
    // smaller than the texel it would have been snapped to.
    // Ramp the displacement in with how deep the crossing is, so a texel that is only slightly
    // recessed moves only slightly. That is what keeps the height field's flush/recessed boundary from
    // reading as a hard cell edge.
    float depthFade = smoothstep(0.0, PARALLAX_DEPTH_SOFTNESS, depth);
    float2 offset = ((lo + hi) * 0.5 - localUv) * depthFade;
    float2 offsetTexels = floor(offset * grid + float2(0.5, 0.5));
    bool stepped = all(abs(offsetTexels) >= float2(1.0, 1.0));
    if (stepped) {
        offset = offsetTexels / grid;
    }
    float2 result = clampSpriteUv(localUv + offset);

    // Only a real step exposes a wall. Below one texel there is no band to orient, and reporting one
    // anyway is what used to fan the shading out of the point where the view faces the surface normal:
    // there the lateral direction collapses and its angle is numerically arbitrary, so neighbouring
    // pixels picked different neighbours and the normal span a full turn.
    float lateralLen = length(vTS.xy);
    if (stepped && lateralLen > 1.0e-5) {
        float2 lateralDir = vTS.xy / lateralLen;
        float2 texelStep = lateralDir / grid;
        float here = 1.0 - samplePageNormalAo(page, pageUvMin + clampSpriteUv(result) * pageUvScale,
                parallaxLod).a;
        float ahead = 1.0 - samplePageNormalAo(page,
                pageUvMin + clampSpriteUv(result - texelStep) * pageUvScale, parallaxLod).a;
        stepNormal = float3(lateralDir, saturate(abs(ahead - here) / PARALLAX_WALL_SLOPE));
    }
    return clampSpriteUv(result);
}

struct Surface {
    float3 normal;
    float3 f0;
    // Wetness-driven albedo multiplier. The caller owns the albedo fetch (the atlas sample is taken
    // before this evaluation), so the material path hands back only the factor that the final
    // post-normal-map orientation and the frame's wetness imply.
    float3 albedoScale;
    // Sprite-local UV this surface was actually sampled at. Parallax moves it off the flat mapping, and
    // the caller's albedo fetch has to follow or the texture would stay pinned to the flat surface.
    float2 localUv;
    // True when parallax moved localUv, so the caller re-fetches albedo at the displaced UV.
    bool albedoDisplaced;
    float roughness;
    float metalness;
    float emission;
    float sss;
    float ao;
};

// Canonical material pages use a compact page index shared by their three dedicated arrays.
float4 samplePageSurface0(uint page, float2 uv, float lod) {
    return materialSurface0Tex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}
float4 samplePageSurface1(uint page, float2 uv, float lod) {
    return materialSurface1Tex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}
float4 samplePageNormalAo(uint page, float2 uv, float lod) {
    return materialNormalAoTex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}

// The sole terrain/entity material evaluation path. LabPBR categorical bytes have already been
// decoded on the CPU; every sample below is a continuously filterable physical channel with semantic mips.
// The channel switches (sss, porosity, the two emission sources) come from the push constant, while
// parallaxDepth is a parameter because the entity host may switch the march off for its own prims.
Surface evaluateMaterial(MaterialHeader header, float2 atlasUv, float lod, float3 albedo,
        float3 geometricNormal, float3 p0, float3 p1, float3 p2,
        float2 uv0, float2 uv1, float2 uv2, float3 vdir,
        float fallbackRoughness, float fallbackMetalness, float fallbackEmission, bool allowSss,
        float parallaxDepth) {
    Surface surface;
    surface.normal = geometricNormal;
    surface.roughness = fallbackRoughness;
    surface.metalness = fallbackMetalness;
    surface.f0 = lerp(float3(0.04, 0.04, 0.04), albedo, fallbackMetalness);
    surface.albedoScale = float3(1.0, 1.0, 1.0);
    // Block-state emission (a glowing block state, or the primitive's light level where a block model
    // carries the mask). The Materials page can switch it off, which is how a surface's authored glow is
    // told apart from everything else on it; light already published to the RIS light buffer stays until
    // the section is re-meshed.
    surface.emission = (pc.materialToggles & MATERIAL_TOGGLE_BLOCK_EMISSION) != 0u ? fallbackEmission : 0.0;
    surface.sss = 0.0;
    surface.ao = 1.0;
    // LabPBR porosity, decoded on the CPU from the _s blue byte. It stays 0 for every material that
    // did not author one, which is what keeps the wetness block at the end of this function free.
    float porosity = 0.0;

    float2 localUv = clamp((atlasUv - header.albedoUv.xy) * header.albedoUv.zw,
            float2(0.0, 0.0), float2(1.0, 1.0));
    float maxMaterialLod = float((header.features >> MATERIAL_MAX_LOD_SHIFT) & MATERIAL_MAX_LOD_MASK);
    float materialLod = min(lod, maxMaterialLod);
    uint page = header.texturePage;
    surface.localUv = localUv;
    surface.albedoDisplaced = false;

    // A horizontal water face takes its normal from the animated wave field, which raygen applies after
    // this, so the wave owns its orientation and the flat plane is the answer for both parallax and the
    // normal map.
    bool waveOwned = header.model == MATERIAL_WATER && abs(geometricNormal.y) > 0.5;
    // One frame serves both the parallax march and the normal-map rotation.
    float3 T, B;
    bool hasTbn = tangentFrame(geometricNormal, p0, p1, p2, uv0, uv1, uv2, T, B);
    // Tangent-space wall reported by the march, applied after the normal map so it only overrides that
    // map where a step was actually crossed and only by how much of a wall the height gradient says.
    float3 stepNormal = float3(0.0, 0.0, 0.0);
    bool hasStepNormal = false;
    if (hasTbn && !waveOwned && (header.features & MATERIAL_FEATURE_HEIGHT) != 0u
            && materialLod < PARALLAX_MAX_LOD && parallaxDepth > 0.0) {
        // Parallax comes first: it moves where the surface actually is, so every channel below — and
        // the caller's albedo — has to be read from the moved UV.
        float2 displaced = parallaxLocalUv(header, page, localUv, materialLod, maxMaterialLod,
                geometricNormal, T, B, vdir, parallaxDepth, stepNormal);
        // A flush texel inside an otherwise displaced material marches nowhere, so it keeps the flat
        // mapping and skips the caller's extra albedo fetch.
        if (any(abs(displaced - localUv) > 1.0e-5)) {
            localUv = displaced;
            surface.localUv = displaced;
            surface.albedoDisplaced = true;
            hasStepNormal = stepNormal.z > 0.0;
        }
    }
    float2 pageUv = header.materialUv.xy + localUv * header.materialUv.zw;
    if ((header.features & (MATERIAL_FEATURE_SPEC | MATERIAL_FEATURE_HEURISTIC_EMISSION)) != 0u) {
        float4 surface0 = samplePageSurface0(page, pageUv, materialLod);
        if ((header.features & MATERIAL_FEATURE_SPEC) != 0u) {
            float4 surface1 = samplePageSurface1(page, pageUv, materialLod); // LabPBR F0
            surface.roughness = surface0.r;
            surface.metalness = surface0.g;
            // The pack's own emissiveness (_s alpha), the one channel a player can want off while keeping
            // the rest of the material: it is the only source of light the asset itself authors.
            surface.emission = (pc.materialToggles & MATERIAL_TOGGLE_TEXTURE_EMISSION) != 0u
                    ? surface0.b : 0.0;
            surface.sss = allowSss && (pc.materialToggles & MATERIAL_TOGGLE_SUBSURFACE) != 0u
                    ? surface0.a : 0.0;
            // Canonical LabPBR F0 is authored in the asset's BT.709 RGB basis. Fresnel is evaluated in
            // the transport basis, so coloured metal/custom F0 crosses the same boundary as albedo.
            surface.f0 = clamp(bt709ToAcesCg(surface1.rgb), float3(0.0), float3(1.0));
            // Porosity off leaves the wetness response with its smoothing but no absorption, which is what
            // separates "the world is wet" from "this material drinks it".
            porosity = (pc.materialToggles & MATERIAL_TOGGLE_POROSITY) != 0u ? surface1.a : 0.0;
        } else {
            // Eligibility is compiled from block-state semantics; the primitive's exact light level
            // supplies strength, while the page supplies only the normalized spatial mask.
            surface.emission = (pc.materialToggles & MATERIAL_TOGGLE_BLOCK_EMISSION) != 0u
                    ? surface0.b * fallbackEmission : 0.0;
        }
    }
    // Final HDR emitting-surface luminance (look-package baseline or absolute JSON cd/m² override), baked in Java at
    // material-compile time and packed unconditionally — see MATERIAL_EMISSION_STRENGTH_SHIFT. 0 for
    // non-emissive materials, so this multiply is always safe (and a no-op) when nothing above ran.
    float strength = float((header.features >> MATERIAL_EMISSION_STRENGTH_SHIFT)
            & MATERIAL_EMISSION_STRENGTH_MASK)
            * (MATERIAL_MAX_EMISSION_STRENGTH / float(MATERIAL_EMISSION_STRENGTH_MASK));
    surface.emission *= strength;
    // A pack's water_still normal map is 16x16 per block face, so on a horizontal face it would rule a
    // hard-edged static lattice that repeats block for block; the wave owns that orientation instead.
    // Flowing side faces keep their map.
    if ((header.features & MATERIAL_FEATURE_NORMAL) != 0u && !waveOwned && hasTbn) {
        float4 normalAo = samplePageNormalAo(page, pageUv, materialLod);
        surface.normal = perturbNormal(geometricNormal, T, B, vdir, normalAo, surface.ao);
    }
    // A crossed step is a vertical face, so its normal is horizontal in the tangent frame. This applies
    // whether or not the material carries a _n: the step is geometry the height field produced, not a
    // detail the normal map authored, so a material with height but no normal map still gets its walls
    // oriented — which is what stops the displaced band from reading as a stretched patch of texture.
    if (hasStepNormal) {
        float3 wallDir = normalize(stepNormal.x * T + stepNormal.y * B);
        surface.normal = normalize(lerp(surface.normal, wallDir, saturate(stepNormal.z)));
    }
    // A water film does the same thing to every surface it covers — it fills the micro-surface and
    // turns it reflective — so the film term carries no porosity. Only the darkening is porosity's
    // business: a material that cannot absorb water keeps its colour under a film, while a porous one
    // soaks and darkens in proportion to what it can hold. Splitting the two is what lets one setting
    // both wet a stone wall (brighter, smoother) and darken a sand path.
    if (pc.wetness > 0.0) {
        float wet = pc.wetness * lerp(WET_GRAZING_FLOOR, 1.0, saturate(surface.normal.y));
        // A surface that emits its own light is the one thing a film cannot improve: it would only
        // roughen and cool what is already a light source.
        wet *= surface.emission > 0.0 ? 0.0 : 1.0;
        surface.roughness = lerp(surface.roughness, WET_ROUGHNESS, wet);
        // Metalness is excluded here because LabPBR reserves the porosity byte on metals, so a pack
        // that stores something else there must not darken a conductor's reflection. The sqrt tracks
        // the pore area the water can actually reach, not the volume sitting behind it.
        float absorption = saturate(wet * sqrt(porosity) * (1.0 - surface.metalness));
        surface.albedoScale = lerp(float3(1.0, 1.0, 1.0),
                float3(WET_ALBEDO_SCALE, WET_ALBEDO_SCALE, WET_ALBEDO_SCALE), absorption);
    }
    return surface;
}

// Block-breaking overlay: shared by opaque terrain AND stained-glass/ice. Reconstructs the hit position
// in the same rebased space the breaking list was pushed in, nudges it into the solid block along
// -normal to get the block's integer coordinate, then decal-projects the local [0,1] position onto
// whichever axis pair the normal is most aligned with (same idea as vanilla's
// SheetedDecalTextureGenerator) and multiply-blends the matching destroy-stage crack texture in,
// mirroring vanilla's crumbling blend (DST_COLOR*SRC_COLOR, doubled).
float3 applyBreaking(float3 albedo709, uint rayCone, float3 rayOrigin, float3 rayDir, float hitT, float3 n) {
    ConstPtr<WorldPush> pcRef = ConstPtr<WorldPush>(pc.worldPushAddr);
    uint breakCount = pcRef[0].breakCount;
    if (breakCount == 0u) {
        return albedo709;
    }
    float3 hitPos = rayOrigin + rayDir * hitT;
    int3 blockPos = int3(floor(hitPos - n * 0.01));
    for (uint bi = 0u; bi < breakCount; ++bi) {
        int4 ps = pcRef[0].breaking[bi].posSlot;
        if (all(ps.xyz == blockPos)) {
            float3 local = hitPos - float3(blockPos);
            float3 an = abs(n);
            float2 decalUv = (an.x >= an.y && an.x >= an.z) ? local.zy
                           : (an.y >= an.z) ? local.xz
                           : local.xy;
            float decalLod = rayConeUnitUvLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(ps.w)]));
            float3 crack709 = srgbToLinear(
                    entityAlbedoTex[NonUniformResourceIndex(ps.w)].SampleLevel(decalUv, decalLod).rgb);
            return clamp(crack709 * albedo709, 0.0, 1.0);
        }
    }
    return albedo709;
}

// Vanilla composites the end portal and the end gateway in rendertype_end_portal.fsh: a base image plus a
// stack of transformed samples of a SECOND image, each scaled by a fixed colour table, and no lighting term
// anywhere. Sampling only the base is what makes them read as a flat purple cube, so the traced colour is
// built the same way here.
//
// Vanilla maps the composite in clip space, which pins the pattern to the screen. A traced surface cannot
// do that — it would slide around when the portal appeared in a reflection — so the coordinates below come
// from the world hit position instead, projected along the dominant normal axis so every face of the cube
// gets a non-degenerate mapping.
static const float3 END_PORTAL_COLORS[16] = {
        float3(0.022087, 0.098399, 0.110818),
        float3(0.011892, 0.095924, 0.089485),
        float3(0.027636, 0.101689, 0.100326),
        float3(0.046564, 0.109883, 0.114838),
        float3(0.064901, 0.117696, 0.097189),
        float3(0.063761, 0.086895, 0.123646),
        float3(0.084817, 0.111994, 0.166380),
        float3(0.097489, 0.154120, 0.091064),
        float3(0.106152, 0.131144, 0.195191),
        float3(0.097721, 0.110188, 0.187229),
        float3(0.133516, 0.138278, 0.148582),
        float3(0.070006, 0.243332, 0.235792),
        float3(0.196766, 0.142899, 0.214696),
        float3(0.047281, 0.315338, 0.321970),
        float3(0.204675, 0.390010, 0.302066),
        float3(0.080955, 0.314821, 0.661491)
};

// Blocks per texture repeat. Vanilla shows one repeat across the whole screen, so a face only ever reveals
// a window of the pattern; the layers then multiply their coordinates by up to ~4, which is why a plain
// one-block-per-repeat mapping reads as fine static. Raise it to enlarge the features.
static const float END_PORTAL_BLOCKS_PER_REPEAT = 16.0;

// Only the star layers are kept. Vanilla's composite also adds the base image (Sampler0 = the End's sky)
// at COLORS[0]; that base is a smooth violet sky, so it is what gives vanilla's portal its uniform dim
// field — and a uniform field is exactly what a linear renderer with an auto-exposure amplifies into a
// blue wash, since vanilla meant it as a display value and here it is a light source. Dropping it leaves
// a black void carrying vanilla's star pattern, so the layer sum needs this gain to put a saturated star
// back where vanilla's 8-bit clip lands (measured: field 4-19 -> 2-5 cd/m², star peak 420 -> 2000).
static const float END_PORTAL_STAR_GAIN = 2.5;

// Those fifteen layers are the same starfield at fifteen scales, and vanilla could afford to just add
// them because its base field covered the gaps. Summed over black they do not leave gaps: every texel
// sits near the faint tail of some layer's neighbourhood, so the surface reads as a dense churning net
// rather than as stars. The contrast curve keeps the star cores (the peak stays at the unlit baseline)
// and pushes that tail to black — measured coverage above 5% of the peak goes 7.1% -> 0.9%, i.e. the
// star density the starfield texture itself carries.
static const float END_PORTAL_STAR_CONTRAST = 3.0;

float2 endPortalLayerUV(float2 uv, float layer, float time) {
    float theta = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
    float c = cos(theta);
    float s = sin(theta);
    float k = (4.5 - layer / 4.0) * 2.0;
    // The fsh applies (scale * rotate), then the per-layer scroll, then SCALE_TRANSLATE (a 0.5 scale with a
    // 0.25 bias) - all in GLSL's row-vector order, so each step is applied to the right of the last.
    float2 p = mul(uv, float2x2(k * c, -k * s, k * s, k * c))
            + float2(17.0 / layer, (2.0 + layer / 1.5) * (time * 1.5));
    return p * 0.5 + 0.25;
}

float3 endPortalColor(float3 hitPos, float3 n, uint packedLayers, float time) {
    uint layerSlot = packedLayers & 0xFFFFu;
    int layers = int(packedLayers >> 16);
    float3 a = abs(n);
    float2 uv = (a.y > a.x && a.y > a.z) ? hitPos.xz : ((a.x > a.z) ? hitPos.yz : hitPos.xy);
    uv = uv * (1.0 / END_PORTAL_BLOCKS_PER_REPEAT);
    // Only the star layers (Sampler1) are summed; the base image is deliberately left out, so the void
    // behind the stars is black rather than vanilla's violet field (see END_PORTAL_STAR_GAIN). The shared
    // entity sampler clamps, and the layer transforms push coordinates far outside the texture, so the
    // wrap the fsh gets for free from its own addressing has to be applied explicitly.
    float3 color = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < layers; i++) {
        float2 layerUV = endPortalLayerUV(uv, float(i + 1), time);
        color += entityAlbedoTex[NonUniformResourceIndex(layerSlot)]
                .SampleLevel(layerUV - floor(layerUV), 0.0).rgb * END_PORTAL_COLORS[i];
    }
    // Vanilla sums these lanes in a float and writes the result into an 8-bit UNORM target, so its
    // composite CLIPS at 1.0 — the gain below is applied before that clip, so a saturated star lands on
    // white the way vanilla's does instead of running past it.
    return srgbToLinear(pow(saturate(color * END_PORTAL_STAR_GAIN), END_PORTAL_STAR_CONTRAST));
}

[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    float2 attribs = attr.barycentrics;
    uint rayCone = payload.rayCone;

    // Particle billboard: same geom-table/UV/bindless-atlas path as entities. Albedo is atlas texel *
    // per-particle colour; raygen treats material 2 as a primary-visible diffuse receiver with no
    // emission or specular response. Reached only by the primary ray (instance mask 0x02). Cutout: rahit.
    if ((InstanceID() & PARTICLE_BIT) != 0u) {
        uint idx = InstanceID() & IDX_MASK;
        EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[idx];
        uint pid = PrimitiveIndex() + g.triBase[GeometryIndex()];
        Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
        ConstPtr<uint> ib = ConstPtr<uint>(g.idxAddr);
        ConstPtr<float2> uvb = ConstPtr<float2>(g.uvAddr);
        uint p0 = ib[3u * pid + 0u];
        uint p1 = ib[3u * pid + 1u];
        uint p2 = ib[3u * pid + 2u];
        float3 pbary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
        float2 puv = pbary.x * uvb[p0] + pbary.y * uvb[p1] + pbary.z * uvb[p2];
        int pslot = int(pr.tint.w + 0.5);
        float3x3 o2w = (float3x3)ObjectToWorld3x4();
        float3 pp0 = mul(o2w, HitTriangleVertexPosition(0));
        float3 pp1 = mul(o2w, HitTriangleVertexPosition(1));
        float3 pp2 = mul(o2w, HitTriangleVertexPosition(2));
        float particleLod = rayConeTextureLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(pslot)]),
                pp0, pp1, pp2, uvb[p0], uvb[p1], uvb[p2]);
        float3 pn = normalize(pr.normal.xyz);
        if (dot(pn, WorldRayDirection()) > 0.0) {
            pn = -pn;
        }
        float3 particleAlbedo709 = srgbToLinear(
                entityAlbedoTex[NonUniformResourceIndex(pslot)].SampleLevel(puv, particleLod).rgb)
                * srgbToLinear(pr.tint.rgb);
        packAlbedo(payload, bt709ToAcesCg(particleAlbedo709));
        packNormal(payload, pn);
        payload.hitT = RayTCurrent();
        // Per-particle motion vector: interpolate the captured per-vertex displacement (uniform across
        // the billboard's verts) with the same indices/barycentrics as the UV. dispAddr == 0 falls back
        // to rigidDisp, which particles write as zero.
        if (g.dispAddr != 0) {
            ConstPtr<float4> pd = ConstPtr<float4>(g.dispAddr);
            payload.motionPrev = half3(pbary.x * pd[p0].xyz + pbary.y * pd[p1].xyz + pbary.z * pd[p2].xyz);
        } else {
            payload.motionPrev = half3(g.rigidDisp.xyz);
        }
        payload.f0 = half3(0.0h, 0.0h, 0.0h);
        // A self-luminous particle carries the package's emitter baseline in the prim's emission lane
        // (RtEntities.SELF_LUMINOUS_PARTICLES), so it is emitted from its own colour rather than lit: a
        // billboard's normal points at the camera, which makes ndl ≈ 0 for the very light it sits in.
        float particleEmission = pr.normal.w;
        payloadSetPacked(payload, MATERIAL_PARTICLE, 1.0, 0.0, particleEmission, 0.0, 1.0, 0.0,
                particleEmission > 0.0 ? EMISSION_SOURCE_UNIFORM : EMISSION_SOURCE_NONE);
        return;
    }
    // Dynamic entities: instances with this custom-index flag bit carry real captured ModelPart
    // geometry; the low bits index the entity geometry table, not the section table.
    if ((InstanceID() & ENTITY_BIT) != 0u) {
        uint eidx = InstanceID() & ~ENTITY_BIT;
        EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[eidx];
        uint pid = PrimitiveIndex() + g.triBase[GeometryIndex()];
        Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
        // Rigid-reuse instances (still / spinning entities re-referencing a cached BLAS) carry a real
        // rotation in the instance transform; per-frame-built entities are identity. The cached prim
        // normal (and the TBN source positions below) are object-space — rotate them into world space.
        // Identity/translation instances make this a no-op.
        float3x3 entityO2w = (float3x3)ObjectToWorld3x4();
        // Orient the geometric normal toward the viewer first (so the _n TBN is built in the right
        // hemisphere — same fix as terrain).
        float3 vdir = -WorldRayDirection();
        float3 n = normalize(mul(entityO2w, pr.normal.xyz));
        bool entering = dot(WorldRayDirection(), n) < 0.0;
        if (dot(n, vdir) < 0.0) {
            n = -n;
        }
        // Interpolate the captured entity-texture UV (same scheme as terrain) and sample the bindless
        // per-type texture; tint.rgb is the model colour multiplier (white for most, tinted for sheep/…).
        ConstPtr<uint> eidxBuf = ConstPtr<uint>(g.idxAddr);
        ConstPtr<float2> euv = ConstPtr<float2>(g.uvAddr);
        uint e0 = eidxBuf[3u * pid + 0u];
        uint e1 = eidxBuf[3u * pid + 1u];
        uint e2 = eidxBuf[3u * pid + 2u];
        float3 ebary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
        float2 euvCoord = ebary.x * euv[e0] + ebary.y * euv[e1] + ebary.z * euv[e2];
        int texSlot = int(pr.tint.w + 0.5);
        float3 ep0 = mul(entityO2w, HitTriangleVertexPosition(0));
        float3 ep1 = mul(entityO2w, HitTriangleVertexPosition(1));
        float3 ep2 = mul(entityO2w, HitTriangleVertexPosition(2));
        float entityLod = rayConeTextureLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(texSlot)]),
                ep0, ep1, ep2, euv[e0], euv[e1], euv[e2]);
        float4 entityTexel = entityAlbedoTex[NonUniformResourceIndex(texSlot)].SampleLevel(euvCoord, entityLod);
        MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
        uint material = header.model;
        float3 baseAlbedo709;
        if ((pr.flags & ENTITY_PRIM_END_PORTAL) != 0u) {
            // The layer stack is the only reason this branch exists, so the world push (and its animation
            // clock) is loaded here rather than on the hot path shared by every entity surface.
            ConstPtr<WorldPush> push = ConstPtr<WorldPush>(pc.worldPushAddr);
            float3 portalPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
            baseAlbedo709 = endPortalColor(portalPos, n, pr.aux1, push[0].endPortalTime);
        } else {
            baseAlbedo709 = srgbToLinear(entityTexel.rgb) * srgbToLinear(pr.tint.rgb);
        }
        float3 opticalColor709 = material == MATERIAL_DIELECTRIC
                ? lerp(float3(1.0), baseAlbedo709, entityTexel.a) : baseAlbedo709;
        float3 albedo = bt709ToAcesCg(opticalColor709);
        Surface surface = evaluateMaterial(header, euvCoord, entityLod, albedo, n,
                ep0, ep1, ep2, euv[e0], euv[e1], euv[e2], vdir,
                header.params.x, header.params.y, pr.normal.w, material == MATERIAL_OPAQUE,
                (pc.materialToggles & MATERIAL_TOGGLE_ENTITY_PARALLAX) != 0u ? pc.parallaxDepth : 0.0);
        n = surface.normal;
        // Entity albedo UVs are the texture's own, so the parallax-displaced local UV is already the
        // coordinate to re-fetch at. The end portal is procedural — its colour comes from the layer
        // stack, not the texture — so it has no albedo to displace and is skipped.
        if (surface.albedoDisplaced && (pr.flags & ENTITY_PRIM_END_PORTAL) == 0u) {
            float4 displacedTexel = entityAlbedoTex[NonUniformResourceIndex(texSlot)]
                    .SampleLevel(surface.localUv, entityLod);
            float3 displaced709 = srgbToLinear(displacedTexel.rgb) * srgbToLinear(pr.tint.rgb);
            float3 displacedOptical = material == MATERIAL_DIELECTRIC
                    ? lerp(float3(1.0), displaced709, displacedTexel.a) : displaced709;
            albedo = bt709ToAcesCg(displacedOptical);
        }

        packAlbedo(payload, albedo * surface.ao * surface.albedoScale);
        packNormal(payload, n);
        payload.hitT = RayTCurrent();
        // Per-vertex motion vector: interpolate the captured per-vertex displacement with the same
        // indices/barycentrics used for the UV above. Rotation and skeletal/lid animation use a
        // per-vertex buffer; pure whole-object translation is packed into rigidDisp with no buffer.
        if (g.dispAddr != 0) {
            ConstPtr<float4> dd = ConstPtr<float4>(g.dispAddr);
            payload.motionPrev = half3(ebary.x * dd[e0].xyz + ebary.y * dd[e1].xyz + ebary.z * dd[e2].xyz);
        } else {
            payload.motionPrev = half3(g.rigidDisp.xyz);
        }
        payload.f0 = half3(surface.f0);
        // An unlit submission is drawn with no lighting term, so its prim carries the finished HDR
        // luminance in the emission lane (the collector stamps the package's emitter baseline there) and
        // the material's compiled strength — the same cd/m² quantity — must not multiply it again: doing
        // so squares the luminance (2000 × 2000) and overflows the half2 lane to +inf, which the dither in
        // any_hit then sprinkles across the surface as blown-white specks. The vertex alpha stays where it
        // already is — as the coverage any_hit gates the hit with, which is what keeps the sheet's
        // see-through composite (emission over whatever is behind it) instead of replacing it.
        bool unlitPrim = (pr.flags & ENTITY_PRIM_UNLIT) != 0u;
        float sss = material == MATERIAL_OPAQUE && !unlitPrim ? surface.sss : 0.0;
        float emission = material == MATERIAL_OPAQUE
                ? (unlitPrim ? pr.normal.w : surface.emission) : 0.0;
        payloadSetPacked(payload, material, surface.roughness, surface.metalness,
                emission, sss, header.params.z, header.params.w,
                materialEmissionSource(header, emission));
        payloadSetDielectric(payload, material, entering);
        if (unlitPrim) {
            payload.flags |= PAYLOAD_UNLIT;
        }
        return;
    }

    Section sec = ConstPtr<Section>(pc.tableAddr)[InstanceID()];
    // Any-hit/SBT opt: the section BLAS has one fixed geometry per material bucket (solid / cutout /
    // translucent / water). PrimitiveIndex() restarts at 0 per geometry, so re-add this geometry's
    // triangle base to land in the section's packed prim/uv arrays (concatenated in BUCKET order).
    // Water prims keep tint.w == 1, converted to the packed material enum below.
    uint pid = PrimitiveIndex() + sec.triBase[GeometryIndex()];
    TerrainPrim pr = ConstPtr<TerrainPrim>(sec.primAddr)[pid];
    MaterialHeader materialHeader = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
    float3 n = normalize(pr.normal.xyz);
    float3 tint = pr.tint.rgb;

    // Water prims are single-sided, wound outward from the fluid volume (RtFluidMesher), so the
    // UNFLIPPED geometric normal is a reliable "which side of the dielectric" signal: a ray travelling
    // against it is entering the volume, one travelling with it is exiting. Captured before the
    // toward-viewer flip below and carried in the payload so raygen derives the medium from hit
    // orientation instead of toggling parity per crossing, so one malformed face cannot desync the path.
    bool entering = dot(WorldRayDirection(), n) < 0.0;

    // Lever B: per-triangle corner UVs in primitive order. uvs[3*pid + k] is a contiguous,
    // directly-addressed load (no index buffer, no scattered vertex-UV gather), so it issues as soon as
    // pid is known and its latency overlaps the prim fetch instead of serialising behind an index load.
    // The index buffer remains BLAS-only; shading reads the expanded UV buffer directly.
    ConstPtr<float2> uvs = ConstPtr<float2>(sec.uvAddr);
    float2 uv0 = uvs[3u * pid + 0u];
    float2 uv1 = uvs[3u * pid + 1u];
    float2 uv2 = uvs[3u * pid + 2u];
    float3 bary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    float3 tp0 = HitTriangleVertexPosition(0);
    float3 tp1 = HitTriangleVertexPosition(1);
    float3 tp2 = HitTriangleVertexPosition(2);
    float blockLod = rayConeTextureLod(rayCone, texSizePx(blockAlbedoAtlas), tp0, tp1, tp2, uv0, uv1, uv2);

    // Orient the GEOMETRIC normal toward the viewer FIRST (double-sided geometry), so the normal-map TBN
    // is built in the correct hemisphere. Doing this before the map — rather than flipping the perturbed
    // normal afterward — is what avoids the grazing-angle black spots: a post-flip could invert a tipped
    // mapped normal through the surface and light the back face.
    float3 vdir = -WorldRayDirection(); // toward the viewer
    if (dot(n, vdir) < 0.0) {
        n = -n;
    }

    // Stained glass / ice: a thin colored filter resolved in raygen
    // (material 3). albedo carries the transmission tint = texel rgb mixed toward white by (1 − opacity),
    // so a more opaque texel tints transmitted light more strongly.
    if (materialHeader.model == MATERIAL_DIELECTRIC) {
        // A translucent block's own light level rides the same lane the opaque path reads: 0..1 plus a +2
        // flag for non-SOLID render layers. Almost every translucent block is 0, so this costs nothing on
        // glass and ice; the nether portal is level 11 and must not render dark.
        float ew = pr.normal.w;
        float stateEmission = ew >= 1.5 ? ew - 2.0 : ew;
        float4 gtex = blockAlbedoAtlas.SampleLevel(uv, blockLod);
        float3 gtex709 = srgbToLinear(gtex.rgb);
        float3 tint709 = srgbToLinear(tint);
        float3 opticalColor709 = lerp(float3(1.0), gtex709 * tint709, gtex.a);
        float3 glassAlbedo = bt709ToAcesCg(opticalColor709);
        Surface glassSurface = evaluateMaterial(materialHeader, uv, blockLod, glassAlbedo, n,
                tp0, tp1, tp2, uv0, uv1, uv2, vdir,
                materialHeader.params.x, materialHeader.params.y, stateEmission, false,
                pc.parallaxDepth);
        n = glassSurface.normal;
        // Follow the parallax-displaced UV with the albedo, or the pane's texture would stay pinned to
        // the flat plane while its shading moved. The alpha that decides the transmission filter comes
        // from the flat texel: it is a coverage value, and moving it would let parallax punch holes.
        if (glassSurface.albedoDisplaced) {
            float2 displacedUv = materialHeader.albedoUv.xy
                    + glassSurface.localUv / materialHeader.albedoUv.zw;
            float3 displaced709 = srgbToLinear(
                    blockAlbedoAtlas.SampleLevel(displacedUv, blockLod).rgb) * tint709;
            opticalColor709 = lerp(float3(1.0), displaced709, gtex.a);
        }
        // Translucent blocks (glass, ice, …) are breakable too — apply the same overlay here, reusing
        // gtex (already sampled above) rather than re-fetching blockAlbedoAtlas.
        packAlbedo(payload, bt709ToAcesCg(applyBreaking(
                opticalColor709 * glassSurface.ao * glassSurface.albedoScale, rayCone, WorldRayOrigin(),
                WorldRayDirection(), RayTCurrent(), n)));
        packNormal(payload, n);
        payload.hitT = RayTCurrent();
        payload.motionPrev = half3(0.0h, 0.0h, 0.0h);
        // A glowing pane glows with its TEXTURE's colour, not with its transmission filter. The albedo
        // lane above is deliberately mixed toward white so a clear texel absorbs nothing — correct for
        // refraction, wrong as an emission tint, and it would leave the surface a paler hue than the
        // light RtLightCollector casts from the same quad (that uses the raw premultiplied albedo). No
        // raygen path reads f0 on a dielectric hit, so an emitting one borrows the lane rather than
        // growing the payload; a non-emissive one keeps its authored F0.
        payload.f0 = glassSurface.emission > 0.0
                ? half3(bt709ToAcesCg(gtex709 * tint709)) : half3(glassSurface.f0);
        payloadSetPacked(payload, MATERIAL_DIELECTRIC, glassSurface.roughness,
                glassSurface.metalness, glassSurface.emission, 0.0, materialHeader.params.z,
                materialHeader.params.w,
                materialEmissionSource(materialHeader, glassSurface.emission));
        payloadSetDielectric(payload, MATERIAL_DIELECTRIC, entering);
        // RIS emitter-NEE membership: the raygen gates this pane's direct-hit emission term because RIS
        // already samples the same rectangle.
        if ((pr.flags & TERRAIN_PRIM_IN_LIGHT_BUFFER) != 0u) {
            payload.flags |= PAYLOAD_EMITTER_IN_LIST;
        }
        return;
    }

    // Water (tint.w == 1) carries the pure biome water tint (no grey water-texture multiply): raygen
    // shades the surface as a clear dielectric and only needs the tint to derive the per-channel
    // Beer–Lambert absorption. Opaque terrain uses textured albedo.
    float3 tint709 = srgbToLinear(tint);
    float3 albedo709 = srgbToLinear(blockAlbedoAtlas.SampleLevel(uv, blockLod).rgb) * tint709;
    packAlbedo(payload, bt709ToAcesCg(
            materialHeader.model == MATERIAL_WATER ? tint709 : albedo709));
    payload.hitT = RayTCurrent();
    payload.motionPrev = half3(0.0h, 0.0h, 0.0h); // static terrain: camera-only motion vector
    uint material = materialHeader.model == MATERIAL_WATER ? MATERIAL_WATER : MATERIAL_OPAQUE;

    // normal.w packs the 0..1 block-light level plus a +2 offset flag for non-SOLID (cutout /
    // translucent) render layers, set at extraction so the hit can opt SOLID terrain out of SSS below.
    float ew = pr.normal.w;
    bool nonSolid = ew >= 1.5;
    float emission = nonSolid ? ew - 2.0 : ew; // heuristic emission source (block light level)
    Surface surface = evaluateMaterial(materialHeader, uv, blockLod, unpackAlbedo(payload), n,
            tp0, tp1, tp2, uv0, uv1, uv2, vdir,
            materialHeader.params.x, materialHeader.params.y, emission, nonSolid,
            pc.parallaxDepth);
    n = surface.normal;
    // Parallax moved the sample point, so both the payload albedo and the breaking overlay's decal base
    // have to come from the displaced UV rather than the flat one sampled above. Water is excluded: it
    // carries the pure biome tint instead of a texture multiply, and its horizontal faces never
    // parallax anyway.
    if (surface.albedoDisplaced && material != MATERIAL_WATER) {
        float2 displacedUv = materialHeader.albedoUv.xy + surface.localUv / materialHeader.albedoUv.zw;
        albedo709 = srgbToLinear(blockAlbedoAtlas.SampleLevel(displacedUv, blockLod).rgb) * tint709;
    }
    packAlbedo(payload, unpackAlbedo(payload) * surface.ao * surface.albedoScale);
    // Block-breaking overlay: opaque/cutout terrain (water skips — fluids aren't breakable). Evaluate
    // it after normal mapping so decal projection uses the final shading orientation.
    if (material != MATERIAL_WATER) {
        packAlbedo(payload, bt709ToAcesCg(applyBreaking(
                albedo709 * surface.ao * surface.albedoScale, rayCone, WorldRayOrigin(),
                WorldRayDirection(), RayTCurrent(), n)));
    }
    packNormal(payload, n);
    payload.f0 = half3(surface.f0);
    payloadSetPacked(payload, material, surface.roughness, surface.metalness,
            surface.emission, surface.sss, materialHeader.params.z, materialHeader.params.w,
            materialEmissionSource(materialHeader, surface.emission));
    payloadSetDielectric(payload, material, entering);
    // RIS emitter-NEE membership: raygen gates this emitter's direct-hit emission term (RIS covers it).
    if ((pr.flags & TERRAIN_PRIM_IN_LIGHT_BUFFER) != 0u) {
        payload.flags |= PAYLOAD_EMITTER_IN_LIST;
    }
}

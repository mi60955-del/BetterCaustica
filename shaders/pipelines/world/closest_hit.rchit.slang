// Closest-hit shader. Geometry is per-section: InstanceID() indexes a section table (BDA array reached
// from the push constant) holding this section's {prim, uv} buffer addresses. Per-primitive
// {normal, tint} plus barycentric atlas UVs produce the payload guides; lighting is deferred to raygen.
//
// tableAddr/entityTableAddr are read from the inline push-constant lanes (pc), not through the
// WorldPush BDA struct: they're read on every single hit (the hottest chain in the frame), and going
// through worldPushAddr first would cost an extra global-memory load just to find the address of the
// load we actually want. The cold WorldPush struct is dereferenced only for the breaking overlay and for
// the sun direction a height-field self-shadow marches along.
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

// ---- Parallax relief -------------------------------------------------------------------------------
//
// LabPBR stores height in the _n alpha: 255 is flush with the face and 0 is as deep as the spec allows,
// a quarter of a block. No geometry moves for it. What moves is the sample — the shading UV is walked
// down the view ray until it meets the stored surface — so a groove shows the wall behind it and a face
// stops reading as wallpaper.
//
// Every length here is in the sprite's own [0,1] UV, which is exactly one block face, so the spec's
// quarter block is the number 0.25 and the user's depth control scales it from there. A hit shader has no
// screen-space derivatives to derive that scale from, which is the other reason it comes from the field's
// own units.
//
// The walk reads the field as one depth per texel — a staircase — and steps the view ray across texel
// boundaries (a DDA) until it arrives at the column it actually hits. The shaded sample lands on a
// texel centre, so relief reads as solid chunks with hard edges instead of a smeared gradient, and the
// answer is geometric: a pixel shows exactly the column the ray reaches, never a rounded neighbour.
//
// The style decides what a crossing turns into (see parallaxStyleOf). The two differ in two numbers:
//
//   Sharp  is for authored hard edges — plank grooves, brick bevels, cut stone. A crossed boundary turns
//          into a wall at full strength, from the height step across it. The relief reads as cut
//          geometry.
//   Smooth is for rounded or organic relief. The field is read a mip coarser, so the walk follows the
//          shape instead of the texels, and a wall claims only a third of the shading.
//
// Two bounds keep the walk bounded, neither optional:
//   * the reach fades out with distance, before a displaced sample is smaller than the texel it came
//     from;
//   * the column budget is the quality setting, clamped into the compile-time range — a view so grazing
//     that the ray outruns the budget lands on the last column it reached.
// Samples that leave the sprite wrap to the opposite edge instead of fading: a block texture repeats per
// face, so the wrapped texel is the continuation the neighbouring face shows. Keeping the reach at full
// strength up to the border is what makes the relief tile across blocks; a reach fade there compresses
// the border band instead — a deep field at a grazing view sweeps half the sprite's content across that
// band, and it reads as a flap curling off every block edge.

// Layer bounds. The floor keeps a shallow field from resolving its crossing too coarsely; the ceiling is
// only the compile-time bound the loop needs, since the budget a frame spends is the quality setting
// clamped into this range.
static const int PARALLAX_MIN_LAYERS = 8;
static const int PARALLAX_MAX_LAYERS = 64;
// Tangent-space view Z floor. Projecting the view onto the surface divides by it, and a grazing view
// would stretch that projection to infinity.
static const float PARALLAX_MIN_VIEW_Z = 0.2;
// Material LOD past which a displaced sample is smaller than the texel it came from, so the reach has
// faded to nothing.
static const float PARALLAX_MAX_LOD = 2.0;
// Height step across one texel boundary at which a crossed face counts as a full wall; the claim ramps
// in from a third of this. Sandstone strata (~0.14) and bark plateaus (~0.2) clear it, while stone's
// noise (≤0.04) and the flat interior of every plateau stay under the ramp-in and keep their normal
// map.
static const float PARALLAX_WALL_GRADIENT = 0.15;
// Extra mip levels the smooth style reads, and how much of a wall it still claims.
static const float PARALLAX_SMOOTH_MIP_BIAS = 0.75;
static const float PARALLAX_SMOOTH_WALL = 0.35;
// Self-shadow walk: steps from the shaded texel up to the surface plane, the sine of the light's
// elevation below which nothing is worth shadowing, the depth difference that saturates an occlusion, and
// the bias that keeps a texel from shading itself. Sixteen steps keep the stride under one texel across
// the walk's full span.
static const int PARALLAX_SHADOW_STEPS = 16;
static const float PARALLAX_SHADOW_MIN_ELEVATION = 0.10;
static const float PARALLAX_SHADOW_SOFTNESS = 0.15;
static const float PARALLAX_SHADOW_BIAS = 0.02;
// How far the walk lets one unit of stored depth reach laterally, in sprite widths, at a 45° light. The
// physical quarter-block conversion the march uses leaves the walk sub-texel on the fields packs
// author — sandstone's deepest texel sits at 17% of the range, a 0.7-texel walk at 45° that ends before
// it can cross onto a shallower column — so the contact shadow read nothing. One sprite width per depth
// unit is a 4x exaggeration at LabPBR's 16-px scale: one texel of depth casts one texel of shadow at a
// 45° light, and the ratio holds at other sprite sizes.
static const float PARALLAX_SHADOW_STRETCH = 1.0;

struct ParallaxSample {
    // Sprite-local UV to shade at.
    float2 uv;
    // Lateral direction of the band the displacement exposed, in the hit triangle's tangent space.
    float2 wallDir;
    // How much of a wall that band is; 0 when nothing was exposed.
    float wallAmount;
    // Fraction of the directional light that reaches the displaced texel through its own relief.
    float shadow;
};

ParallaxSample untouchedParallax(float2 uv) {
    ParallaxSample sample;
    sample.uv = uv;
    sample.wallDir = float2(0.0, 0.0);
    sample.wallAmount = 0.0;
    sample.shadow = 1.0;
    return sample;
}

// The style's two dials, resolved once per sample so the body below stays one algorithm rather than two.
// The offset itself is continuous in both styles: quantising it to the texel grid draws closed one-texel
// jump contours over any field whose crossing depth varies smoothly — on a steep multi-level pack
// texture those bands read as several copies of the texture stacked at whole-texel depths.
struct ParallaxStyle {
    float mipBias;
    float wallScale;
};

ParallaxStyle parallaxStyleOf(int style) {
    ParallaxStyle resolved;
    bool sharp = style == PARALLAX_STYLE_SHARP;
    resolved.mipBias = sharp ? 0.0 : PARALLAX_SMOOTH_MIP_BIAS;
    resolved.wallScale = sharp ? 1.0 : PARALLAX_SMOOTH_WALL;
    return resolved;
}

// Tile a sprite-local sample across its sprite. Block textures repeat per face, so a sample pulled past
// the border reads the opposite edge — the continuation the neighbouring face shows — rather than the
// page gutter's replicated border texel, which would smear every tile boundary. The page gutter holds the
// wrapped continuation too (see the material packer), so bilinear filtering keeps tiling across the seam.
float2 wrapSpriteUv(float2 uv) {
    return fract(uv);
}

// Texels this sprite spans at the LOD the shading is read at — the grid a displacement is quantised to.
// Following the LOD rather than mip 0 keeps the quantum at the resolution actually sampled; snapping to
// mip 0 at distance would quantise finer than the samples and pop.
float2 spriteTexelGrid(uint page, float2 pageUvScale, float lod) {
    float2 pageTexels = texSizePx(materialNormalAoTex[NonUniformResourceIndex(page)]);
    return max(floor(pageUvScale * pageTexels / exp2(floor(lod)) + 0.5), float2(1.0, 1.0));
}

// Directional self-shadow through the same height field. From the shaded texel the ray is walked back
// up toward the light; any stretch of field it passes over that sits above it blocks what reaches the
// texel. The walk stops at the surface plane, so only relief the face itself carries can contribute, and
// the effect reads as contact shading inside the relief rather than as a shadow thrown across the face.
//
// The lateral conversion is the walk's own, not the march's reach: stored depth is read at face value
// against the sprite (PARALLAX_SHADOW_STRETCH), because the physical quarter block puts the whole walk
// inside one texel for the fields packs actually author. The light is whichever body is up, sun first,
// both from the lanes the sky bakes from, so a shadow can never point somewhere the sky is not lit from.
// This is the one place a hit dereferences WorldPush; it sits behind the material switch and the
// height-field test, so a surface without relief never pays.
float parallaxShadow(uint page, float2 pageUvMin, float2 pageUvScale, float2 uv, float hitDepth, float lod,
        float3 n, float3 T, float3 B) {
    if ((pc.materialToggles & MATERIAL_TOGGLE_PARALLAX_SHADOW) == 0u || hitDepth <= 1.0e-4) {
        return 1.0;
    }
    ConstPtr<WorldPush> push = ConstPtr<WorldPush>(pc.worldPushAddr);
    float noonTilt = push[0].skyLook1.x;
    float3 sunDir = celestialDirection(push[0].celestial.x, noonTilt);
    float3 lightDir = sunDir.y > 0.0
            ? sunDir : celestialDirection(push[0].celestial.y, noonTilt);
    float3 lightTS = float3(dot(lightDir, T), dot(lightDir, B), dot(lightDir, n));
    // At or below this elevation the face is already at its terminator and the traced sun shadow owns that
    // transition; a second one drawn here would put a contour along every face that happens to sit there.
    if (lightTS.z <= PARALLAX_SHADOW_MIN_ELEVATION) {
        return 1.0;
    }
    float2 lateral = (lightTS.xy / lightTS.z) * PARALLAX_SHADOW_STRETCH;
    float stepDepth = hitDepth / float(PARALLAX_SHADOW_STEPS);
    float shadow = 1.0;
    [loop]
    for (int i = 1; i <= PARALLAX_SHADOW_STEPS; i++) {
        float climbed = stepDepth * float(i);
        float2 sampleUv = pageUvMin + wrapSpriteUv(uv + lateral * climbed) * pageUvScale;
        float field = 1.0 - samplePageNormalAo(page, sampleUv, lod).a;
        float blocked = (hitDepth - climbed - field - PARALLAX_SHADOW_BIAS) / PARALLAX_SHADOW_SOFTNESS;
        shadow = min(shadow, 1.0 - saturate(blocked));
    }
    return lerp(1.0, shadow,
            smoothstep(PARALLAX_SHADOW_MIN_ELEVATION, PARALLAX_SHADOW_MIN_ELEVATION * 2.0, lightTS.z));
}

// One column of the height field: the stored surface depth at a texel's centre, wrapped across the
// sprite. The walk treats the field as constant per column — a staircase — which is what makes its
// answer geometric: a pixel shades exactly the column the ray reaches.
float columnDepth(uint page, float2 pageUvMin, float2 pageUvScale, float lod, float2 grid, float2 texel) {
    float2 uv = (texel + 0.5) / grid;
    return 1.0 - samplePageNormalAo(page, pageUvMin + wrapSpriteUv(uv) * pageUvScale, lod).a;
}

// Where to shade a height-mapped face, and how much light its own relief lets through. A ray that
// never leaves its starting column keeps the flat mapping — no relief was exposed — but its column's
// own depth still self-shadows: a face-on view of a groove bottom is exactly where contact shading
// has to show.
ParallaxSample sampleParallax(MaterialHeader header, uint page, float2 localUv, float lod, float maxLod,
        float3 n, float3 T, float3 B, float3 vdir, float depthScale) {
    ParallaxSample result = untouchedParallax(localUv);

    float3 vTS = float3(dot(vdir, T), dot(vdir, B), dot(vdir, n));
    float2 slope = vTS.xy / max(abs(vTS.z), PARALLAX_MIN_VIEW_Z);
    float2 pageUvMin = header.materialUv.xy;
    float2 pageUvScale = header.materialUv.zw;

    float lodFade = 1.0 - smoothstep(PARALLAX_MAX_LOD - 1.0, PARALLAX_MAX_LOD, lod);
    float reach = depthScale * lodFade;
    if (reach <= 1.0e-5) {
        return result;
    }

    ParallaxStyle style = parallaxStyleOf(pc.parallaxStyle);
    float2 grid = spriteTexelGrid(page, pageUvScale, lod);
    float marchLod = min(lod + style.mipBias, maxLod);
    int layers = clamp(pc.parallaxLayers, PARALLAX_MIN_LAYERS, PARALLAX_MAX_LAYERS);

    // The DDA descends the ray through the staircase field one texel boundary at a time. Depth t is
    // measured inward from the flat surface as a fraction of the height range; per unit t the sample
    // moves -slope·reach in UV, which is -slope·reach·grid texels. `boundary` holds the t at which
    // each axis next crosses a texel boundary, so their minimum is the next step.
    float2 texelVel = -slope * reach * grid;
    float2 entryPos = localUv * grid;
    float2 texel = floor(entryPos);
    float field = columnDepth(page, pageUvMin, pageUvScale, marchLod, grid, texel);
    if (field <= 0.0) {
        return result;                         // flush entry: the ray starts on the surface
    }
    float2 sdir = sign(texelVel);
    float2 ardir = abs(1.0 / max(abs(texelVel), float2(1.0e-9, 1.0e-9)));
    float2 frac = entryPos - texel;
    float2 boundary = (sdir * (0.5 - frac) + 0.5) * ardir;

    float t = 0.0;                // descended depth at the last crossed boundary
    float prevField = field;
    float hitDepth = field;
    float2 move = float2(0.0, 0.0);
    bool wallHit = false;
    bool landed = false;
    [loop]
    for (int i = 0; i < layers; i++) {
        if (field <= t) {
            // The ray entered this column already below its ceiling: it struck the crossed face.
            // The shadow walk climbs from that ceiling, the top of the face — starting at the
            // boundary depth would shadow the wall by its own column.
            wallHit = true;
            hitDepth = field;
            landed = true;
            break;
        }
        float b = min(boundary.x, boundary.y);
        if (field <= b) {
            // The ceiling arrives while the ray is still over this column: a flat landing on it.
            hitDepth = field;
            landed = true;
            break;
        }
        // Step across the nearer boundary and advance that axis's next crossing.
        t = b;
        move = float2(0.0, 0.0);
        if (boundary.x <= boundary.y) {
            move.x = sdir.x;
            boundary.x += ardir.x;
        } else {
            move.y = sdir.y;
            boundary.y += ardir.y;
        }
        texel += move;
        prevField = field;
        field = columnDepth(page, pageUvMin, pageUvScale, marchLod, grid, texel);
    }

    if (!landed) {
        // The budget ran out with the ray still in air: the flat mapping is the answer, not a snapped
        // texel centre, and a sample that never met the field casts no self-shadow.
        return result;
    }

    // The shaded sample is the reached column's texel centre. A wall is the face the ray crossed to
    // enter it, facing back along that boundary's own axis; its strength is the height step across
    // the face, so sandstone strata (~0.14) and bark plateaus (~0.2) read as cut faces while stone's
    // noise (≤0.04) keeps its normal map.
    if (move.x == 0.0 && move.y == 0.0) {
        // The ray never left its starting column (a face-on view): the flat mapping stays — no relief
        // was exposed — but the column's own depth still casts a contact shadow toward a low light.
        result.shadow = parallaxShadow(page, pageUvMin, pageUvScale, localUv, hitDepth, marchLod,
                n, T, B);
        return result;
    }
    float2 displaced = wrapSpriteUv((texel + 0.5) / grid);
    result.uv = displaced;
    result.shadow = parallaxShadow(page, pageUvMin, pageUvScale, displaced, hitDepth, marchLod, n, T, B);
    if (wallHit) {
        result.wallDir = -normalize(move);
        result.wallAmount = smoothstep(PARALLAX_WALL_GRADIENT * 0.33, PARALLAX_WALL_GRADIENT,
                abs(field - prevField)) * style.wallScale;
    }
    return result;
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
    // Fraction of the directional light that reaches this texel through the surface's own height field.
    // Folded into albedo where the material's _n AO already goes: a path tracer has no per-light shader to
    // multiply, and this is the same kind of quantity — occlusion of the light by the surface itself.
    float shadow;
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
    surface.shadow = 1.0;
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
    // Relief comes first: it moves where the surface is shaded, so every channel below — and the caller's
    // albedo — has to be read from the moved UV.
    ParallaxSample relief = untouchedParallax(localUv);
    if (hasTbn && !waveOwned && (header.features & MATERIAL_FEATURE_HEIGHT) != 0u
            && materialLod < PARALLAX_MAX_LOD && parallaxDepth > 0.0) {
#ifdef CAUSTICA_LEAN_HIT
        // Diagnostic build only: the march compiled out of the hit shader, so the cost of carrying it can
        // be measured against a hit shader without it. `relief` already holds the untouched sample.
#else
        relief = sampleParallax(header, page, localUv, materialLod, maxMaterialLod, geometricNormal,
                T, B, vdir, parallaxDepth);
#endif
        surface.shadow = relief.shadow;
        // A flush texel inside an otherwise displaced material marches nowhere, so it keeps the flat
        // mapping and skips the caller's extra albedo fetch.
        if (any(abs(relief.uv - localUv) > 1.0e-5)) {
            localUv = relief.uv;
            surface.localUv = relief.uv;
            surface.albedoDisplaced = true;
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
    if (relief.wallAmount > 0.0) {
        float3 wallNormal = normalize(relief.wallDir.x * T + relief.wallDir.y * B);
        surface.normal = normalize(lerp(surface.normal, wallNormal, relief.wallAmount));
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

// Vanilla's enchantment overlay is an ordinary model submission plus a texture matrix
// (TextureTransform.setupGlintTexturing): the model UV is scaled by the transform's uniform scale,
// rotated by ten degrees and translated by that frame's scroll. The rotation is a fixed constant in
// vanilla and only keeps the pattern from reading as an axis-aligned scroll; the translation is the
// animation itself, so it comes from the push.
float2 glintPatternUV(float2 uv, float scale, float2 scroll) {
    const float glintCos = 0.98480775; // cos(10 degrees)
    const float glintSin = 0.17364818; // sin(10 degrees)
    float2 rotated = float2(uv.x * glintCos - uv.y * glintSin, uv.x * glintSin + uv.y * glintCos);
    return rotated * scale + scroll;
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
        // The capture attaches vanilla's enchantment overlay to the surface it was drawn over, so a flagged
        // prim's emission lane holds the overlay's UV scale rather than an emission strength.
        bool glintPrim = (pr.flags & ENTITY_PRIM_GLINT) != 0u;
        // payloadSetPacked below replaces the flag word, so the trace's own guide marker must be read here.
        bool guideTrace = (payload.flags & PAYLOAD_GUIDE_STABLE) != 0u;
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
        // Vanilla's hurt / spawn-flash overlay (see RtEntityOverlay). It is per submission, so a layer that
        // opted out of it — a cape, an eye layer, an item frame's contents — keeps its own colour while the
        // entity around it flashes.
        float4 entityOverlay = entityOverlayRgba(pr.flags);
        baseAlbedo709 = applyEntityOverlay(baseAlbedo709, entityOverlay);
        float3 opticalColor709 = material == MATERIAL_DIELECTRIC
                ? lerp(float3(1.0), baseAlbedo709, entityTexel.a) : baseAlbedo709;
        float3 albedo = bt709ToAcesCg(opticalColor709);
        Surface surface = evaluateMaterial(header, euvCoord, entityLod, albedo, n,
                ep0, ep1, ep2, euv[e0], euv[e1], euv[e2], vdir,
                header.params.x, header.params.y, glintPrim ? 0.0 : pr.normal.w, material == MATERIAL_OPAQUE,
                (pc.materialToggles & MATERIAL_TOGGLE_ENTITY_PARALLAX) != 0u ? pc.parallaxDepth : 0.0);
        n = surface.normal;
        // Entity albedo UVs are the texture's own, so the parallax-displaced local UV is already the
        // coordinate to re-fetch at. The end portal is procedural — its colour comes from the layer
        // stack, not the texture — so it has no albedo to displace and is skipped.
        if (surface.albedoDisplaced && (pr.flags & ENTITY_PRIM_END_PORTAL) == 0u) {
            float4 displacedTexel = entityAlbedoTex[NonUniformResourceIndex(texSlot)]
                    .SampleLevel(surface.localUv, entityLod);
            float3 displaced709 = srgbToLinear(displacedTexel.rgb) * srgbToLinear(pr.tint.rgb);
            displaced709 = applyEntityOverlay(displaced709, entityOverlay);
            float3 displacedOptical = material == MATERIAL_DIELECTRIC
                    ? lerp(float3(1.0), displaced709, displacedTexel.a) : displaced709;
            albedo = bt709ToAcesCg(displacedOptical);
        }

        packAlbedo(payload, albedo * surface.ao * surface.shadow * surface.albedoScale);
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
        // Vanilla draws the enchantment glint as an extra additive pass (BlendFunction.GLINT, src*src + dst)
        // over the very model face just shaded, which is why the capture attached it here instead of
        // capturing it as geometry of its own. Resolve its pattern against this face's own model UV, so it
        // lands exactly on the pixels this surface owns — vanilla's depth-EQUAL test against the face.
        if (glintPrim && !guideTrace) {
            ConstPtr<WorldPush> glintPush = ConstPtr<WorldPush>(pc.worldPushAddr);
            float4 glint = glintPush[0].glintParams;
            float2 patternUV = glintPatternUV(euvCoord, abs(pr.normal.w), glint.xy);
            // The shared entity sampler clamps; the pattern scrolls past the texture's edge every frame,
            // so the wrap the glint pipeline gets from its own addressing is applied here by hand.
            float4 glintTexel = entityAlbedoTex[NonUniformResourceIndex(int(pr.aux1))]
                    .SampleLevel(patternUV - floor(patternUV), 0.0);
            // glint.fsh multiplies by the submission's colour modulator, then by the option fade, and the
            // blender squares the result. The square is taken on the framebuffer's own encoding — the
            // capture keeps the texture bytes as they are, so this is the same quantity vanilla squares,
            // not the linearised texel. That term is an addition to the display value, and one unit of
            // display is 1/pre-exposure of radiance here, so the overlay contributes its squared pattern
            // divided by the frame's pre-exposure.
            float3 glintDisplay = glintTexel.rgb * pr.tint.rgb * glint.z;
            glintDisplay *= glintTexel.a < 0.1 ? 0.0 : 1.0; // vanilla discards on the pattern's own alpha
            payload.overlay = half3(bt709ToAcesCg(glintDisplay * glintDisplay)
                    * (glint.w / max(glintPush[0].preExposure, 1.0e-9)));
            payload.flags |= PAYLOAD_GLINT_OVERLAY;
        }
        // Model geometry is made of sheets: a translucent model face has the same medium on both sides, so
        // it cannot bound a volume. Marked here rather than per material because every model dielectric is
        // one — block items are drawn through this path as their flat layers, never as a closed solid.
        if (material == MATERIAL_DIELECTRIC) {
            payload.flags |= PAYLOAD_DIELECTRIC_THIN;
        }
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
                opticalColor709 * glassSurface.ao * glassSurface.shadow * glassSurface.albedoScale,
                rayCone, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), n)));
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
        // Glass, ice and stained glass are cell-filling volumes, so their faces do end a medium. A pane,
        // slab or stair of the same material does not, and the mesher tags those quads.
        if ((pr.flags & TERRAIN_PRIM_THIN_DIELECTRIC) != 0u) {
            payload.flags |= PAYLOAD_DIELECTRIC_THIN;
        }
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
    packAlbedo(payload, unpackAlbedo(payload) * surface.ao * surface.shadow * surface.albedoScale);
    // Block-breaking overlay: opaque/cutout terrain (water skips — fluids aren't breakable). Evaluate
    // it after normal mapping so decal projection uses the final shading orientation.
    if (material != MATERIAL_WATER) {
        packAlbedo(payload, bt709ToAcesCg(applyBreaking(
                albedo709 * surface.ao * surface.shadow * surface.albedoScale, rayCone, WorldRayOrigin(),
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

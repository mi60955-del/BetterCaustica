// Direct lighting: the emitter light grid, RIS reservoirs, and reservoir shading.
// INDIRECT PASS ONLY — the primary pass shades nothing. Depends on trace and math.

// contribution weight, M the effective sample count. wSum/phat are streaming scratch (not persisted).

import world_common;
import world_core;
import bindings;
import math;
import trace;

public struct Reservoir {
    public uint   lightIndex; // survivor's record in pc.lightBufAddr, so the sample can be re-derived
                              // against a later frame instead of shipping its geometry across frames
    public float2 suv;        // survivor's coordinates on the emitter rectangle, [-1,1]^2. Persisted
                              // alongside the index so a later frame re-derives the SAME point
    public float3 pos;  // light sample point (rebased world space)
    public float3 lnrm; // emitter outward normal
    public float3 le;   // emitter radiance
    public float area;  // emitter rectangle area (the area-light pdf factor)
    public float M;
    public float W;
    public float wSum;
    public float phat;
};

public Reservoir resEmpty() {
    Reservoir r;
    r.lightIndex = 0u;
    r.suv = float2(0.0, 0.0);
    r.pos = float3(0.0, 0.0, 0.0);
    r.lnrm = float3(0.0, 0.0, 0.0);
    r.le = float3(0.0, 0.0, 0.0);
    r.area = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.wSum = 0.0;
    r.phat = 0.0;
    return r;
}

// Unshadowed contribution of a fixed light sample point at surface (hitPos,n,v); out p-hat = its
// luminance (the resampling target). Evaluated from the UNBIASED hitPos — SURF_BIAS is applied only to
// the shadow-ray origin in shadeReservoir, on whichever side the survivor sample lands. Three receiver
// modes share one target function because they are mutually exclusive per candidate:
//   - front hemisphere (ndl>0): the same Lambert(+GGX) BRDF split the sun NEE uses;
//   - twoSided (particles): billboards have no back face, ndl=abs(ndl) folds both sides into the front;
//   - back hemisphere with sss>0 (LabPBR leaves/grass): HG transmission phase — cosT = dot(wi,rd) peaks
//     when the ray points from the light through the slab toward the camera (same as the sun SSS term).
public float3 evalSampleContrib(float3 sp, float3 lnrm, float3 le, float area, float3 hitPos, float3 n,
                         float3 v, float3 rd, float3 diffAlb, float3 F0, float rough,
                         bool twoSided, float sss, out float phat) {
    phat = 0.0;
    float3 toL = sp - hitPos;
    float dist2 = dot(toL, toL);
    if (dist2 < 1.0e-6) {
        return float3(0.0, 0.0, 0.0);
    }
    float dist = sqrt(dist2);
    float3 wi = toL / dist;
    float cosL = max(0.0, dot(lnrm, -wi));
    if (cosL <= 0.0) {
        return float3(0.0, 0.0, 0.0); // receiver behind the emitter face
    }
    float ndl = dot(n, wi);
    if (twoSided) {
        ndl = abs(ndl);
    }
    float3 contrib = float3(0.0, 0.0, 0.0);
    if (ndl > 0.0) {
        float G = ndl * cosL / dist2; // area-light geometry term (x area)
        float3 brdf = diffAlb * INV_PI;
        // twoSided callers are particle billboards, which pass rough=1/F0=0 and want diffuse-only
        // shading: fresnelSchlick still returns up to full white at grazing angles even with F0=0 (that
        // IS the Fresnel effect), so skipping the term here — not just zeroing F0 — is what keeps
        // billboards from picking up a rim highlight they were never meant to have.
        if (!twoSided) {
            float3 h = normalize(wi + v);
            float ndh = max(0.0, dot(n, h));
            float ndv = max(1.0e-4, dot(n, v));
            float vdh = max(0.0, dot(v, h));
            float D = ggxD(ndh, rough);
            float Gs = ggxG1(ndv, rough) * ggxG1(ndl, rough);
            float3 Fs = fresnelSchlick(vdh, F0);
            brdf += (D * Gs) * Fs / (4.0 * ndv * ndl);
        }
        contrib = brdf * le * (G * area);
    } else if (sss > 0.0) {
        float backNdl = -ndl;
        float G = backNdl * cosL / dist2;
        float cosT = dot(wi, rd);
        contrib = diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * le * (G * area);
    }
    phat = luminance(contrib);
    return contrib;
}

// Select one light from the emitted-power distribution in O(1).
public void selectGlobalLight(inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(worldPush.lightCount);
    uint column = min(uint(aliasSample), worldPush.lightCount - 1u);
    if (pc.lightAliasAddr != 0) {
        LightAlias a = ConstPtr<LightAlias>(pc.lightAliasAddr)[column];
        bool self = aliasSample - float(column) < a.accept;
        lightIndex = self ? column : a.aliasIndex;
    } else {
        lightIndex = column;
    }
}

public bool findLightGridCell(float3 p, out LightGridCell cell, out int3 cellCoord) {
    cell.spanOffset = 0u;
    cell.spanCount = 0u;
    cell.invWeightSum = 0.0;
    cellCoord = int3(0, 0, 0);
    if (pc.lightGridCellAddr == 0 || pc.lightGridSpanAddr == 0) return false;
    int3 coord = int3(floor((p - worldPush.lightGridOrigin.xyz) / worldPush.lightGridOrigin.w));
    if (coord.x < 0 || coord.y < 0 || coord.z < 0
            || coord.x >= worldPush.lightGridDims.x
            || coord.y >= worldPush.lightGridDims.y
            || coord.z >= worldPush.lightGridDims.z) return false;
    uint linear = (uint(coord.z) * uint(worldPush.lightGridDims.y) + uint(coord.y))
            * uint(worldPush.lightGridDims.x) + uint(coord.x);
    cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linear];
    cellCoord = coord;
    return cell.spanCount > 0u;
}

public void selectSectionLight(uint firstLight, uint lightCount, inout uint proposalSeed,
                        out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(lightCount);
    uint column = min(uint(aliasSample), lightCount - 1u);
    LightAlias alias = ConstPtr<LightAlias>(pc.lightLocalAliasAddr)[firstLight + column];
    bool self = aliasSample - float(column) < alias.accept;
    uint localIndex = self ? column : alias.aliasIndex;
    lightIndex = firstLight + localIndex;
}

public float unpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

public float3 lightRadiance(Light light) {
    uint packed = light.le;
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

public int3 lightSectionCoord(Light light) {
    uint packed = light.section;
    return int3(int(packed & 0x3ffu), int((packed >> 10u) & 0x3ffu),
            int((packed >> 20u) & 0x3ffu));
}

// The half axes share three lanes two-at-a-time; see the Light record in world_common.
public float3 lightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

public float3 lightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

// U x V drives both the emitter normal and the rectangle area, so it is computed once and shared.
public float3 lightCrossUV(Light light) {
    return cross(lightHalfU(light), lightHalfV(light));
}

// The rect spans 2U x 2V, so its area is 4|U x V|.
public float lightArea(Light light) {
    return 4.0 * length(lightCrossUV(light));
}

public float3 lightGeometricNormal(Light light) {
    float3 normal = normalize(lightCrossUV(light));
    return (light.section & 0x40000000u) != 0u ? -normal : normal;
}

public float proposalPdf(Light light, float3 le, LightGridCell cell, int3 cellCoord,
                  float localProbability) {
    float power = max(0.0, lightArea(light) * luminance(le));
    float globalPdf = pc.lightAliasAddr != 0
            ? power * worldPush.lightRebase.w : 1.0 / float(worldPush.lightCount);
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = lightSectionCoord(light) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return localProbability * localPdf + (1.0 - localProbability) * globalPdf;
}

public void selectLightGridSpanLight(LightGridCell cell, inout uint proposalSeed,
                              out uint lightIndex) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
    float aliasSample = rndf(proposalSeed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint lightCount = self ? (span.packedLightCounts & 0xffffu)
            : (span.packedLightCounts >> 16u);
    selectSectionLight(firstLight, lightCount, proposalSeed, lightIndex);
}

// Sample the exact mixture q = alpha*qCell + (1-alpha)*qGlobal. qCell first selects a nearby section
// by emitted power and section distance, then its section-local power alias. Every light in
// the neighborhood is represented without a fixed cap; qGlobal gives distant lights full support.
// Alias spans make weighted local selection O(1). The mixture PDF is reconstructed after the single
// Light48 load: section power cancels with the section-local light PDF.
public void selectLightGridLight(LightGridCell cell, bool useLocal, inout uint proposalSeed,
                          out uint lightIndex) {
    if (useLocal) {
        selectLightGridSpanLight(cell, proposalSeed, lightIndex);
    } else {
        selectGlobalLight(proposalSeed, lightIndex);
    }
}

// ---- RIS cost profile. Temporary probes pinned the candidate walk out of the shader to bound what
// memory-side RIS work could ever be worth. Against an
// 18.0ms frame at M=8, removing every dependent load (span -> alias -> record) and all lane divergence
// saved 5.9ms — a third of the frame. Split by vertex that was ~4.3ms at secondary vertices against
// ~1.1ms at the primary hit, and forcing coherent selection recovered at most 1.3ms of it. So the cost
// was chasing depth, not divergence, and it lived at secondary vertices. A presampled candidate pool
// would attack the same 5.9ms structurally, but see the note on
// proposalSeed in tracePath for why it should share the pool rather than the seed.

// Initial RIS over M power-weighted candidates -> one resampled sample, no shadow ray yet. The chosen
// sample's W = wSum / (M * p-hat); M counts every candidate, including zero-weight ones.
public Reservoir risInitial(float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb, float3 F0,
                     float rough, bool twoSided, float sss, inout uint seed, inout uint proposalSeed) {
    Reservoir r = resEmpty();
    // The light buffer is published asynchronously with the light hierarchy: on the frames before the
    // first light is published it is null, and dereferencing it would fault the device.
    if (worldPush.lightCount == 0u || pc.lightBufAddr == 0ull) {
        return r;
    }
    LightGridCell gridCell;
    int3 gridCellCoord;
    float3 gridLookup = hitPos;
    if (pc.lightGridCellAddr != 0) {
        // Stochastically blend across hard section boundaries. Every cell proposal retains full global
        // support, so conditioning on this independently jittered lookup remains unbiased.
        gridLookup += (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5)
                * worldPush.lightGridOrigin.w;
    }
    bool hasGridCell = findLightGridCell(gridLookup, gridCell, gridCellCoord);
    uint candidateCount = worldPush.risCandidates;
    r.M = float(candidateCount);
    // Deterministically stratify the proposal mixture. At the default M=8 this schedules exactly six
    // local and two global candidates, with every lane taking the same branch for a given candidate.
    // Counts that are not divisible by four use the nearest practical split with at least one global
    // candidate; M=1 is global-only so the estimator retains full support.
    uint globalCandidateCount = hasGridCell
            ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
    uint localCandidateCount = candidateCount - globalCandidateCount;
    float localProbability = float(localCandidateCount) / float(candidateCount);
    for (uint c = 0u; c < candidateCount; c++) {
        uint li;
        uint globalsBefore = (c * globalCandidateCount) / candidateCount;
        uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
        bool useLocal = hasGridCell && globalsAfter == globalsBefore;
        selectLightGridLight(gridCell, useLocal, proposalSeed, li);
        Light lg = ConstPtr<Light>(pc.lightBufAddr)[li];
        // Soft shadows: uniform point on the emitter rectangle ((s,t) in [-1,1]^2, pdf 1/area).
        float s = rndf(seed) * 2.0 - 1.0;
        float t = rndf(seed) * 2.0 - 1.0;
        float3 sp = lg.pos + worldPush.lightRebase.xyz
                + s * lightHalfU(lg) + t * lightHalfV(lg);
        float3 le = lightRadiance(lg);
        float3 lightNormal = lightGeometricNormal(lg);
        float area = lightArea(lg);
        float phat;
        evalSampleContrib(sp, lightNormal, le, area, hitPos, n, v, rd,
                diffAlb, F0, rough, twoSided, sss, phat);
        if (phat <= 0.0) {
            continue;
        }
        float sourcePdf = proposalPdf(lg, le, gridCell, gridCellCoord, localProbability);
        float w = phat / max(sourcePdf, 1.0e-20);
        r.wSum += w;
        if (rndf(seed) * r.wSum < w) { // weighted reservoir update
            r.lightIndex = li;
            r.suv = float2(s, t);
            r.pos = sp;
            r.lnrm = lightNormal;
            r.le = le;
            r.area = area;
            r.phat = phat;
        }
    }
    r.W = r.phat > 0.0 ? (r.wSum / (r.M * r.phat)) : 0.0;
    return r;
}

// Reproject a current-frame pixel into last frame's reservoir image, in texels. False when it leaves the
// image, which is also the case on the frame after a resize if the motion buffer has not been refilled.
public bool prevTexelOf(uint2 pix, inout float2 prevTexel) {
    uint2 size;
    gMotion.GetDimensions(size.x, size.y);
    // gMotion holds (previous - current) in render-resolution pixels, so the history texel lies ahead of
    // this one by exactly that offset.
    prevTexel = float2(pix) + gMotion[pix].xy + 0.5;
    return prevTexel.x >= 0.0 && prevTexel.y >= 0.0
            && prevTexel.x < float(size.x) && prevTexel.y < float(size.y);
}

// One texel of last frame's reservoir, re-expanded and re-evaluated for the current shading point, or an
// empty reservoir when it cannot be used. The image lanes are (light index as float, W, M, linear depth of
// the point that was shaded) and the companion pair holds the survivor's (s,t) on the emitter rectangle.
//
// The stored index is remapped through pc.lightTranslationAddr before it is dereferenced, because a global
// light index carries no meaning across a republish. The sample point is then re-derived from this frame's
// emitter record rather than re-drawn: re-drawing it would leave W and p-hat describing different samples.
//
// Depth rejects a surface that has slid in front of or behind this one: absolute at close range, relative
// far away, since a fixed epsilon rejects every distant surface and a purely relative one accepts a wall
// moving through the frame. The normal catches what depth cannot — a reprojection that stayed on one plane
// but crossed a feature edge, where the history carries a different lobe and reusing it reads as grain
// rather than as blur.
public bool storedReservoirAt(uint2 texel, float depth, float3 hitPos, float3 n, float3 v, float3 rd,
                             float3 diffAlb, float3 F0, float rough, float sss, inout Reservoir stream) {
    stream = resEmpty();
    float4 stored = reservoirPrev[texel];
    if (stored.z <= 0.0 || stored.y <= 0.0) {
        return false;
    }
    if (abs(stored.w - depth) > max(0.25, 0.02 * depth) || dot(gNormal[texel].xyz, n) <= 0.8) {
        return false;
    }
    uint historyIndex = uint(stored.x);
    int translated = -1;
    if (worldPush.lightTranslationValid != 0u) {
        if (historyIndex < worldPush.lightTranslationCount) {
            translated = ConstPtr<int>(pc.lightTranslationAddr)[historyIndex];
        }
    } else if (historyIndex < worldPush.lightCount) {
        // No republish since the frame that wrote this history: its index is already current.
        translated = int(historyIndex);
    }
    if (translated < 0 || uint(translated) >= worldPush.lightCount) {
        return false;
    }
    Light lg = ConstPtr<Light>(pc.lightBufAddr)[uint(translated)];
    float2 histSuv = reservoirSamplePrev[texel].xy;
    stream.lightIndex = uint(translated);
    stream.suv = histSuv;
    stream.pos = lg.pos + worldPush.lightRebase.xyz
            + histSuv.x * lightHalfU(lg) + histSuv.y * lightHalfV(lg);
    stream.lnrm = lightGeometricNormal(lg);
    stream.le = lightRadiance(lg);
    stream.area = lightArea(lg);
    stream.M = stored.z;
    stream.W = stored.y;
    evalSampleContrib(stream.pos, stream.lnrm, stream.le, stream.area, hitPos, n, v, rd,
            diffAlb, F0, rough, false, sss, stream.phat);
    return stream.phat > 0.0;
}

// Admit one reused stream into a running reservoir.
//
// The stream enters as p-hat_now(y) * W_stream * admitted rather than as a stored weight sum. W_stream is
// the normalisation the stream itself was selected under, so this divides that p-hat out and this frame's
// in, which is what accounts for a texel that over- or under-sampled the survivor relative to the current
// shading point. Where the two p-hats agree — a still camera over an unchanged emitter record — the product
// is exactly that stream's weight sum.
//
// `admitted` is how many of the stream's own candidates this pixel trusts. Admitting an unbounded history
// while the caller's ceiling stops M growing makes W = wSum / (M * p-hat) climb without bound the longer a
// pixel stands still, which reads as fireflies that worsen over time.
public void reuseInto(inout Reservoir best, Reservoir stream, float admitted, inout uint seed) {
    if (admitted <= 0.0) {
        return;
    }
    float w = stream.W * stream.phat * admitted;
    if (w <= 0.0) {
        return;
    }
    float wSum = best.wSum + w;
    float m = best.M + admitted;
    if (rndf(seed) * wSum < w) {
        stream.M = m;
        stream.wSum = wSum;
        stream.W = wSum / (m * stream.phat);
        best = stream;
    } else {
        // best.phat is positive whenever best.wSum is: a candidate with no contribution is skipped before
        // it can enter the weight sum, and only a contributing candidate becomes the survivor.
        best.M = m;
        best.wSum = wSum;
        best.W = wSum / (m * best.phat);
    }
}

// Screen-space temporal reuse of the bounce-0 diffuse reservoir, and the write-back that feeds next frame.
// The ceiling never drops below this frame's own candidate count: at that point the history contributes
// zero and the merge degenerates exactly to single-frame shading.
public Reservoir risTemporal(Reservoir cur, uint2 pix, float depth, float3 hitPos, float3 n, float3 v,
                            float3 rd, float3 diffAlb, float3 F0, float rough, float sss,
                            inout uint seed) {
    Reservoir best = cur;
    if (worldPush.risTemporalReuse != 0u) {
        float2 prevTexel;
        Reservoir hist;
        if (prevTexelOf(pix, prevTexel)
                && storedReservoirAt(int2(prevTexel), depth, hitPos, n, v, rd, diffAlb, F0, rough, sss,
                        hist)) {
            reuseInto(best, hist, min(hist.M, max(float(worldPush.risTemporalHistory), cur.M) - cur.M),
                    seed);
        }
    }
    reservoirCur[pix] = float4(float(best.lightIndex), best.W, best.M, depth);
    reservoirSampleCur[pix] = best.suv;
    return best;
}

// How many of last frame's neighbouring reservoirs one pixel borrows from, and the radius of that
// neighbourhood in render-resolution texels.
public static const uint RIS_SPATIAL_NEIGHBOURS = 3u;
public static const float RIS_SPATIAL_RADIUS_PX = 20.0;

// Spatial reuse: merge last frame's reservoirs from RIS_SPATIAL_NEIGHBOURS random texels inside a disc of
// RIS_SPATIAL_RADIUS_PX around this pixel's reprojected texel, so a pixel that drew a poor candidate this
// frame can still end up on the emitter its surroundings agreed on.
//
// Neighbours come from the PREVIOUS frame's image, not this one's: a raygen dispatch has no global barrier,
// so this frame's neighbours may not have written their reservoir yet. Every borrowed sample therefore
// crosses the same generation boundary the temporal merge does, and goes through the translation table for
// the same reason.
//
// Each neighbour is trusted with at most one frame's worth of its own candidates, so no single neighbour
// can dominate the survivor and the accumulated W cannot run away the way an uncapped history's does.
//
// This is the biased form of spatial reuse, matching ReSTIR's biased mode: a neighbour's proposal pdf
// covers its own cell of the light grid rather than this pixel's, and no visibility ray is traced for a
// borrowed sample, so the survivor can be one the neighbour could see and this point cannot. The visible
// cost of that trade is darkening at occlusion boundaries, which is what the depth and normal gates in
// storedReservoirAt limit.
//
// worldPush.risTemporalReuse gates this too, not just risTemporal: every borrowed sample comes from a
// previous frame, so the switch means "reuse last frame's reservoirs at all" and turning it off has to
// leave a pixel with nothing but this frame's own candidates.
public Reservoir risSpatial(Reservoir cur, uint2 pix, float depth, float3 hitPos, float3 n, float3 v,
                           float3 rd, float3 diffAlb, float3 F0, float rough, float sss,
                           inout uint seed) {
    Reservoir best = cur;
    if (worldPush.risTemporalReuse == 0u) {
        return best;
    }
    float2 centre;
    uint2 size;
    gMotion.GetDimensions(size.x, size.y);
    if (!prevTexelOf(pix, centre)) {
        return best;
    }
    for (uint j = 0u; j < RIS_SPATIAL_NEIGHBOURS; j++) {
        // A random point in the disc rather than a fixed box around the pixel: deterministic neighbour
        // patterns read as tile-shaped lighting noise.
        float angle = rndf(seed) * 6.28318530718;
        float radius = RIS_SPATIAL_RADIUS_PX * sqrt(rndf(seed));
        int2 texel = int2(centre + float2(cos(angle), sin(angle)) * radius);
        if (texel.x < 0 || texel.y < 0 || texel.x >= int(size.x) || texel.y >= int(size.y)) {
            continue;
        }
        Reservoir stream;
        if (storedReservoirAt(texel, depth, hitPos, n, v, rd, diffAlb, F0, rough, sss, stream)) {
            reuseInto(best, stream, min(stream.M, cur.M), seed);
        }
    }
    return best;
}

// Shade a finalized reservoir: recompute the survivor's contribution at (hitPos,n,v), cast ONE shadow
// ray, and return throughput-free radiance contrib*vis*W. The one ray serves whichever term fired for
// the survivor (front BRDF, twoSided billboard, or SSS backscatter) — the origin is biased toward the
// sample's side of the surface.
public float3 shadeReservoir(Reservoir s, float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb,
                      float3 F0, float rough, bool twoSided, float sss, int hitDepth) {
    if (s.W <= 0.0 || s.phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float phat;
    float3 contrib = evalSampleContrib(s.pos, s.lnrm, s.le, s.area, hitPos, n, v, rd,
            diffAlb, F0, rough, twoSided, sss, phat);
    if (phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    // Pick the shadow-ray origin on the sample's side of the surface FIRST, then measure the ray from
    // that origin — measuring dist from hitPos while tracing from the biased origin shifts the ray tip
    // up to SURF_BIAS*ndl toward the emitter plane, beating the 0.001*dist stop-short margin for lights
    // closer than ~SURF_BIAS/0.001 blocks -> shadow ray hits the emitter's own face -> per-texel black.
    float side = dot(n, s.pos - hitPos) >= 0.0 ? 1.0 : -1.0;
    float3 origin = hitPos + side * n * SURF_BIAS;
    float3 toL = s.pos - origin;
    float dist = length(toL);
    // Deeper vertices may skip the ray: their emitter light is indirect and dim, so losing its
    // occlusion reads as slightly leaked far-bounce glow, while each traced ray stalls the whole warp.
    float3 vis = float3(1.0);
    if (worldPush.shadowSkipDepth == 0u || uint(max(hitDepth, 0)) < worldPush.shadowSkipDepth) {
        // Stop just short of the sample point so the ray doesn't self-occlude on the emitter's own face.
        VisibilityResult shadow = visibility(origin, toL / dist, dist * 0.999);
        vis = shadow.transmittance;
    }
    return contrib * vis * s.W;
}

// TLAS instance-mask bits (set on the Java side, ANDed against these per-ray cull masks):
//   bit 0 (0x01) — secondary rays: shadows, GI bounces, reflections.
//   bit 1 (0x02) — the primary camera ray.
// Terrain / entities / block entities keep 0xFF (both bits). Particles use 0x02: primary-visible

#ifndef CAUSTICA_NO_PROBE_PATH
// --- Indirect radiance probes -------------------------------------------------------------
// Camera-centred world-space L1 SH grid. A probe stores the radiance leaving the surface it sits on,
// refreshed every frame from that surface's own screen pixel; the indirect walk samples the grid at its
// first indirect vertex, so the second and later bounces of a camera path cost one trilinear gather
// instead of another traced continuation. The grid keeps two copies: the frame writes the copy opposite
// worldPush.probeParity and every query reads the copy probeParity names, so a path never sees a probe
// that this frame has only half refreshed.

public static const uint PROBE_DIM_X = 48u;
public static const uint PROBE_DIM_Y = 16u;
public static const uint PROBE_DIM_Z = 48u;
public static const uint PROBE_COUNT = PROBE_DIM_X * PROBE_DIM_Y * PROBE_DIM_Z;
public static const uint PROBE_BYTES = 64u;
// Radiance is stored in the pre-exposure units outImage holds, so a query can add it to the walk as it is.
public static const float PROBE_SH_C0 = 0.282095;
public static const float PROBE_SH_C1 = 0.488603;

public struct ProbeSH {
    public float4 shR;
    public float4 shG;
    public float4 shB;
    // x confidence 0..1, y unused, z the frame of the last accepted sample, w unused.
    public float4 meta;
};

float probeShChannel(float4 sh, float3 n) {
    return sh.x * PROBE_SH_C0 + PROBE_SH_C1 * dot(sh.yzw, float3(n.y, n.z, n.x));
}

float3 probeShRadiance(ProbeSH probe, float3 n) {
    // A single-direction EMA leaves the L1 lobe free to go negative on the side the sample did not see;
    // the gather below clamps instead of subtracting light.
    return max(float3(probeShChannel(probe.shR, n),
                      probeShChannel(probe.shG, n),
                      probeShChannel(probe.shB, n)), float3(0.0));
}

// A probe slot is addressed by world cell, not by window position: the same cell always lands in the
// same slot, so moving the camera does not reassign contents already stored. Slots wrap modulo the grid
// dims, and meta.y carries the cell tag so a slot reused by a cell one period away is rejected instead
// of answering with another place's radiance.
uint probeSlotOf(int3 cell) {
    int3 dims = int3(int(PROBE_DIM_X), int(PROBE_DIM_Y), int(PROBE_DIM_Z));
    int3 m = cell % dims;
    m = m + int3(m.x < 0 ? dims.x : 0, m.y < 0 ? dims.y : 0, m.z < 0 ? dims.z : 0);
    return uint(m.z) * PROBE_DIM_X * PROBE_DIM_Y + uint(m.y) * PROBE_DIM_X + uint(m.x);
}

uint probeTagOf(int3 cell) {
    return (uint(cell.x + 512) & 1023u) | ((uint(cell.y + 512) & 1023u) << 10)
            | ((uint(cell.z + 512) & 1023u) << 20);
}

// First cell of the camera window, in the same rebased space the probes and paths use.
int3 probeWindowBase() {
    float cellSize = max(worldPush.probeOrigin.w, 1.0e-3);
    return int3(int(floor(worldPush.probeOrigin.x / cellSize + 0.5)),
                int(floor(worldPush.probeOrigin.y / cellSize + 0.5)),
                int(floor(worldPush.probeOrigin.z / cellSize + 0.5)));
}

// Refresh the probe this pixel owns. Only the first PROBE_COUNT pixels of the dispatch take part; the
// rest return immediately. The probe is valid when its centre projects inside the frame and lands on the
// surface the depth guide reports there, which rejects probes inside geometry or outside the view.
public void updateProbeCache(uint2 pix, uint2 dimensions) {
    if (worldPush.probeEnabled == 0u) {
        return;
    }
    uint probeIndex = uint(pix.y) * dimensions.x + uint(pix.x);
    if (probeIndex >= PROBE_COUNT) {
        return;
    }
    uint3 offset = uint3(probeIndex % PROBE_DIM_X,
                         (probeIndex / PROBE_DIM_X) % PROBE_DIM_Y,
                         probeIndex / (PROBE_DIM_X * PROBE_DIM_Y));
    int3 cell = probeWindowBase() + int3(offset);
    float cellSize = max(worldPush.probeOrigin.w, 1.0e-3);
    float3 probePos = worldPush.probeOrigin.xyz + (float3(offset) + 0.5) * cellSize;

    // Ping-pong: the write half is the one no query is reading this frame, and the previous contents it
    // blends with are the same half's own last write.
    uint writeHalf = worldPush.probeParity != 0u ? 0u : PROBE_COUNT;
    uint readHalf = worldPush.probeParity != 0u ? PROBE_COUNT : 0u;
    uint slot = probeSlotOf(cell);
    uint64_t writeAddr = worldPush.probeGridAddr
            + uint64_t((writeHalf + slot) * PROBE_BYTES);
    uint64_t readAddr = worldPush.probeGridAddr
            + uint64_t((readHalf + slot) * PROBE_BYTES);
    ProbeSH previous = ConstPtr<ProbeSH>(readAddr)[0];
    // A wrapped slot may hold a cell one period away; its radiance must not seed this cell's average.
    if (uint(previous.meta.y) != probeTagOf(cell)) {
        previous = ProbeSH(float4(0.0), float4(0.0), float4(0.0), float4(0.0, 0.0, 0.0, 0.0));
    }

    bool valid = false;
    float3 radiance = float3(0.0);
    float3 towardEye = float3(0.0, 1.0, 0.0);
    // curViewProj is camera-relative, so the probe is projected from its offset to the eye.
    float4 clip = mul(worldPush.curViewProj, float4(probePos - worldPush.camOffset, 1.0));
    if (clip.w > 1.0e-4) {
        float2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (all(uv >= 0.0) && all(uv < 1.0)) {
            uint2 screenPix = uint2(clamp(uv * float2(dimensions), float2(0.0),
                                          float2(dimensions) - 1.0));
            // The probe is valid when the surface the pixel shows sits within a cell and a half of it:
            // that is the neighbourhood whose radiance a probe at this corner can speak for. A raw depth
            // comparison would be meaningless here, because the probe sits up to a cell off the surface
            // and reversed-Z depth is not linear in distance.
            float2 ndc = (float2(screenPix) + 0.5) / float2(dimensions) * 2.0 - 1.0;
            float4 surfaceH = mul(worldPush.invViewProj, float4(ndc, gDepth[screenPix], 1.0));
            float3 surface = surfaceH.xyz / surfaceH.w;
            if (length(surface - (probePos - worldPush.camOffset)) <= cellSize * 3.0) {
                radiance = outImage[screenPix].rgb * (1.0 / max(worldPush.preExposure, 1.0e-6));
                towardEye = normalize(worldPush.camOffset - probePos);
                valid = true;
            }
        }
    }

    float confidence = previous.meta.x;
    if (valid) {
        // First sighting snaps harder so a probe lights up within a few frames; after that the refresh
        // is slow enough to average the pixel's own noise.
        float alpha = confidence < 0.05 ? 0.5 : 0.125;
        float4 basis = float4(PROBE_SH_C0, PROBE_SH_C1 * towardEye.yzx);
        ProbeSH updated;
        updated.shR = previous.shR * (1.0 - alpha) + radiance.r * basis * alpha;
        updated.shG = previous.shG * (1.0 - alpha) + radiance.g * basis * alpha;
        updated.shB = previous.shB * (1.0 - alpha) + radiance.b * basis * alpha;
        updated.meta = float4(min(1.0, confidence + 0.25), asfloat(probeTagOf(cell)),
                float(worldPush.frameIndex), 0.0);
        Ptr<ProbeSH>(writeAddr)[0] = updated;
    } else if (float(worldPush.frameIndex) - previous.meta.z > 120.0) {
        // Age-based expiry: a probe is only stale once nothing has refreshed it for two seconds.
        // Decaying on every missed screen projection would erase the ray fill's confidence, because
        // every probe has exactly one screen owner and unseen probes always miss that projection.
        ProbeSH decayed = previous;
        decayed.meta.x = confidence * 0.9;
        Ptr<ProbeSH>(writeAddr)[0] = decayed;
    }
}

// Trilinear gather of the eight probes around p, evaluated against n. The sum is confidence-weighted,
// so a texel whose probes are stale or absent pulls the result toward the live path instead of the
// cache.
public float3 sampleProbeRadiance(float3 p, float3 n, out float confidence) {
    confidence = 0.0;
    if (worldPush.probeEnabled == 0u) {
        return float3(0.0);
    }
    float cellSize = max(worldPush.probeOrigin.w, 1.0e-3);
    float3 grid = (p - worldPush.probeOrigin.xyz) / cellSize - 0.5;
    int3 base = int3(floor(grid));
    float3 frac = grid - float3(base);
    int3 windowBase = probeWindowBase();
    uint readBase = worldPush.probeParity != 0u ? PROBE_COUNT : 0u;

    float3 sum = float3(0.0);
    float weights = 0.0;
    for (int dz = 0; dz < 2; dz++) {
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int3 corner = base + int3(dx, dy, dz);
                if (any(corner < int3(0)) || corner.x >= int(PROBE_DIM_X)
                        || corner.y >= int(PROBE_DIM_Y) || corner.z >= int(PROBE_DIM_Z)) {
                    continue;
                }
                float3 weight = float3(dx != 0 ? frac.x : 1.0 - frac.x,
                                       dy != 0 ? frac.y : 1.0 - frac.y,
                                       dz != 0 ? frac.z : 1.0 - frac.z);
                float w = weight.x * weight.y * weight.z;
                if (w <= 0.0) {
                    continue;
                }
                int3 worldCell = windowBase + corner;
                uint index = probeSlotOf(worldCell);
                ProbeSH probe = ConstPtr<ProbeSH>(worldPush.probeGridAddr
                        + uint64_t((readBase + index) * PROBE_BYTES))[0];
                if (uint(probe.meta.y) != probeTagOf(worldCell)) {
                    continue;
                }
                float probeConfidence = clamp(probe.meta.x, 0.0, 1.0);
                sum += probeShRadiance(probe, n) * (w * probeConfidence);
                weights += w * probeConfidence;
            }
        }
    }
    confidence = weights;
    return weights > 1.0e-4 ? sum / weights : float3(0.0);
}

// True when p lies inside the probe grid. The cutoff uses this instead of a confidence test: a
// per-lane confidence threshold makes neighbouring lanes disagree, and a wave that keeps one
// tracing lane pays for the whole continuation anyway. Grid membership changes only across cells,
// so lanes in a warp usually agree and the break actually removes work.
public bool probeGridContains(float3 p) {
    float3 grid = (p - worldPush.probeOrigin.xyz) / max(worldPush.probeOrigin.w, 1.0e-3);
    return all(grid >= float3(0.0)) && all(grid < float3(float(PROBE_DIM_X), float(PROBE_DIM_Y), float(PROBE_DIM_Z)));
}

// Amortized probe-ray fill.
//
// The screen-space writeback can only refresh probes whose projected pixel is visible, and most
// bounce-1 vertices sit on surfaces the camera cannot see. A small, rotating subset of rays therefore
// samples the grid directly from the probe centres: each frame traces PROBE_RAY_COUNT rays spread over
// the grid, so every probe is revisited once per PROBE_COUNT / PROBE_RAY_COUNT frames. The samples are
// isotropic (a probe in air has no hemisphere), shaded with the same emitter NEE the path vertex uses,
// and folded into the same ping-pong half as the screen writeback.
public static const uint PROBE_RAY_COUNT = 2048u;
public static const float PROBE_RAY_GAIN = 4.0;

public void fillProbeRay(uint2 pix, uint2 dimensions) {
    if (worldPush.probeEnabled == 0u) {
        return;
    }
    uint pixelIndex = uint(pix.y) * dimensions.x + uint(pix.x);
    if (pixelIndex < PROBE_COUNT || pixelIndex >= PROBE_COUNT + PROBE_RAY_COUNT) {
        return;
    }
    uint raySlot = pixelIndex - PROBE_COUNT;
    uint probeIndex = (raySlot + uint(worldPush.frameIndex) * PROBE_RAY_COUNT) % PROBE_COUNT;
    uint3 offset = uint3(probeIndex % PROBE_DIM_X,
                         (probeIndex / PROBE_DIM_X) % PROBE_DIM_Y,
                         probeIndex / (PROBE_DIM_X * PROBE_DIM_Y));
    int3 cell = probeWindowBase() + int3(offset);
    float cellSize = max(worldPush.probeOrigin.w, 1.0e-3);
    float3 probePos = worldPush.probeOrigin.xyz + (float3(offset) + 0.5) * cellSize;

    uint seed = probeIndex * 9781u + uint(worldPush.frameIndex) * 6271u + 17u;
    seed = pcg(seed);
    float2 u = float2(rndf(seed), rndf(seed));
    float cosTheta = u.x * 2.0 - 1.0;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = u.y * 6.28318530718;
    float3 dir = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    traceRadiance(CULL_SECONDARY, probePos, 0.0, dir, 10000.0, true, 0.0, 0.0, false);
    float3 radiance;
    if (payload.hitT < 0.0) {
        radiance = payloadSky();
    } else {
        float3 hitPos = probePos + dir * payload.hitT;
        float3 albedo = payloadAlbedo();
        radiance = albedo * payloadEmission();
        if (payloadMaterial() == MATERIAL_OPAQUE) {
            float3 n = payloadNormal();
            float rough = payloadRoughness();
            float metal = payloadMetalness();
            float3 diffAlb = albedo * (1.0 - metal);
            float3 p = hitPos + n * SURF_BIAS;
            uint proposalSeed = seed ^ (probeIndex * 2654435761u);
            proposalSeed = pcg(proposalSeed);
            Reservoir reservoir = risInitial(p, n, -dir, dir, diffAlb, payload.f0, rough,
                    false, 0.0, seed, proposalSeed);
            radiance += shadeReservoir(reservoir, p, n, -dir, dir, diffAlb, payload.f0, rough,
                    false, 0.0, 0);
        }
    }

    uint writeHalf = worldPush.probeParity != 0u ? 0u : PROBE_COUNT;
    uint readHalf = worldPush.probeParity != 0u ? PROBE_COUNT : 0u;
    uint probeSlot = probeSlotOf(cell);
    uint64_t writeAddr = worldPush.probeGridAddr + uint64_t((writeHalf + probeSlot) * PROBE_BYTES);
    uint64_t readAddr = worldPush.probeGridAddr + uint64_t((readHalf + probeSlot) * PROBE_BYTES);
    ProbeSH previous = ConstPtr<ProbeSH>(readAddr)[0];
    if (uint(previous.meta.y) != probeTagOf(cell)) {
        previous = ProbeSH(float4(0.0), float4(0.0), float4(0.0), float4(0.0, 0.0, 0.0, 0.0));
    }
    ProbeSH updated;
    // The screen writeback stores one radiance direction per probe and the query weights it by about
    // 1/pi; averaging isotropic directions cancels that dipole term, so the ray samples carry the
    // reciprocal of the query kernel's self-response to keep both fills on one scale.
    float4 basis = float4(PROBE_SH_C0, PROBE_SH_C1 * dir.yzx);
    float alpha = previous.meta.x < 0.05 ? 0.5 : 0.2;
    updated.shR = lerp(previous.shR, radiance.r * basis * PROBE_RAY_GAIN, alpha);
    updated.shG = lerp(previous.shG, radiance.g * basis * PROBE_RAY_GAIN, alpha);
    updated.shB = lerp(previous.shB, radiance.b * basis * PROBE_RAY_GAIN, alpha);
    updated.meta = float4(min(1.0, previous.meta.x + 0.2), asfloat(probeTagOf(cell)),
            float(worldPush.frameIndex), 0.0);
    Ptr<ProbeSH>(writeAddr)[0] = updated;
}
#endif

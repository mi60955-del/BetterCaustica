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
                      float3 F0, float rough, bool twoSided, float sss) {
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
    // Stop just short of the sample point so the ray doesn't self-occlude on the emitter's own face.
    VisibilityResult shadow = visibility(origin, toL / dist, dist * 0.999);
    float3 vis = shadow.transmittance;
    return contrib * vis * s.W;
}

// TLAS instance-mask bits (set on the Java side, ANDed against these per-ray cull masks):
//   bit 0 (0x01) — secondary rays: shadows, GI bounces, reflections.
//   bit 1 (0x02) — the primary camera ray.
// Terrain / entities / block entities keep 0xFF (both bits). Particles use 0x02: primary-visible

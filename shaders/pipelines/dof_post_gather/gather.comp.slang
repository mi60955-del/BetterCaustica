// Post-process depth of field, gather pass: the near and far defocus fields of the frame the display
// mapper just wrote.
//
// The frame arrives pinhole-sharp at display resolution in display code values, and the traced lens is
// off -- this pass defocuses that image instead of the trace. Per pixel it asks what the lens would
// have done: a surface at axial range s spreads over aperture * |1/s - 1/f| of the frame's tangent
// span, the same thin-lens answer the traced camera walks rays with, read here from the depth the
// primary pass published for its own blur ceiling. That depth is why the defocus is worth doing this
// way at all: entities, particles, water faces and the recessed side of a carved block hold a depth no
// depth-buffer probe could name.
//
// The gather runs scatter-as-gather in the correct direction: a sample's light spreads over ITS OWN
// disc, so a tap's weight is how much of the source pixel's circle of confusion covers this pixel --
// not how much of this pixel's disc covers the source, which is the approximation that bleeds blurred
// background over the sharp thing standing in front of it. Near and far accumulate separately, each
// with a depth gate (a nearer sample cannot light the far field it occludes; a farther one cannot be
// covered by the near field it sits behind), and the recombine lays near over far over sharp. A
// foreground edge therefore carries its own bokeh over the background without a halo, and a background
// blur never eats the in-focus object before it.
//
// Runs at half the display resolution: a defocus field is band-limited by the discs it accumulates, so
// the recombine's bilinear fetch costs less than the second half of every kernel would. Both display
// channels are accumulated by this one dispatch -- the HDR work is gated by a push constant rather
// than by an unbound descriptor, so the descriptor set never changes shape with the setting.

import display_common;
import bindings;

[[vk::push_constant]] DofPostGatherPush pc;

static const float PI = 3.14159265;
// Ring-to-ring rotation of the kernel, so two rings' taps cannot line up into visible spokes.
static const float GOLDEN_ANGLE = 2.39996323;
// Kernel geometry, in half-res texels: about one ring per RING_STEP of radius, and about one tap per
// RING_STEP of circumference, so the tap count tracks the disc's area rather than its radius squared.
static const float RING_STEP = 2.0;
static const int MAX_RINGS = 8;
static const int MAX_TAPS = 24;

// One channel's accumulation: the far field as (blurred colour, how much of it to take) and the near
// field as (blurred colour, how much of the pixel it covers).
struct FieldResult {
    float3 farColor;
    float farMix;
    float3 nearColor;
    float nearAlpha;
};

// The axial distance, in blocks, the render-res texel under this half-res coordinate shows. Nearest
// lookup: r32f is not linearly filterable, and a half-texel of depth quantisation is invisible next to
// a blur disc measured in texels.
float axialBlocks(float2 halfCoord, float2 halfSize) {
    float2 renderCoord = (halfCoord * 2.0 + 0.5) * pc.renderTexels / (halfSize * 2.0);
    int2 texel = int2(clamp(renderCoord, float2(0.0), pc.renderTexels - 1.0));
    return focusDepth[texel];
}

// The circle of confusion's radius, in display pixels, for a surface at `blocks`. Zero for a texel no
// frame has written (whatever the allocator held is not a surface), and capped at the kernel's bound.
float cocRadius(float blocks) {
    if (blocks <= 0.0 || pc.tangentPerPixel <= 0.0) {
        return 0.0;
    }
    float blur = pc.aperture * abs(1.0 / blocks - 1.0 / pc.focus) / pc.tangentPerPixel;
    return min(pc.maxCoc, blur);
}

// How much of a unit disc the aperture's opening covers at the normalized offset `unit`: 1 inside,
// 0 outside, with a rim one source texel wide so bokeh edges do not alias. Blades < 3 is the circle.
// A blade polygon is renormalised to the circle's area, so changing the shape changes nothing about
// the gathered energy.
float apertureCoverage(float2 unit, float edge) {
    float r = length(unit);
    if (pc.blades < 3) {
        return saturate((1.0 - r) / edge + 0.5);
    }
    float sector = 2.0 * PI / float(pc.blades);
    float theta = atan2(unit.y, unit.x) - pc.bladeRotation;
    float local = theta - sector * floor(theta / sector);
    // The regular polygon's radius along this direction: the apothem over the cosine of the angle
    // from the sector's centre line, scaled so the polygon's area equals the circle's.
    float areaScale = sqrt(2.0 * PI / (float(pc.blades) * sin(2.0 * PI / float(pc.blades))));
    float rim = areaScale * cos(PI / float(pc.blades)) / cos(local - sector * 0.5);
    return saturate((rim - r) / edge + 0.5);
}

// The relative depth gate: two surfaces within this share of each other's range are one layer, so
// noise between neighbouring render texels cannot cut a sample off from the pixel it legitimately
// covers. Beyond it the sample is a different layer, and the occlusion rule decides.
float layerGate(float blocks) {
    return 0.05 * blocks + 0.25;
}

// One full gather over one display channel. The ring kernel is walked once for the far field (radius
// from this pixel's own far blur) and once for the near field (radius from the widest near disc the
// neighbourhood shows, because a foreground disc larger than this pixel's own blur is exactly the one
// that lands on it).
FieldResult gatherField(Sampler2D source, int2 pix, float2 halfSize) {
    float2 centre = float2(pix);
    float2 centreUv = (centre + 0.25) / halfSize;
    float blocks = axialBlocks(centre, halfSize);
    float coc = cocRadius(blocks);
    float3 centreColor = source.SampleLevel(centreUv, 0.0).rgb;

    // ---- far field: the blur of everything at or beyond this pixel's layer ----
    // The kernel radius is this pixel's own disc: a farther layer's wider disc is truncated at depth
    // discontinuities, which is the accepted cost of running without a tile pass to widen it. The mix
    // ramp keeps a pixel inside a texel of the focal plane exactly sharp.
    FieldResult result;
    result.farMix = saturate((coc - 0.75) / 1.5);
    result.farColor = centreColor;
    float radiusHalf = coc * 0.5;
    if (result.farMix > 0.0 && radiusHalf > 0.5) {
        float3 sum = centreColor;
        float weightSum = 1.0;
        int rings = clamp(int(ceil(radiusHalf / RING_STEP)), 1, MAX_RINGS);
        for (int k = 1; k <= rings; ++k) {
            float ringRadius = radiusHalf * float(k) / float(rings);
            int taps = clamp(int(ceil(2.0 * PI * ringRadius / RING_STEP)), 4, MAX_TAPS);
            float angle = float(k) * GOLDEN_ANGLE;
            for (int t = 0; t < taps; ++t) {
                float theta = angle + 2.0 * PI * float(t) / float(taps);
                float2 offset = ringRadius * float2(cos(theta), sin(theta));
                float2 uv = (centre + offset + 0.25) / halfSize;
                float sampleBlocks = axialBlocks(centre + offset, halfSize);
                // A nearer sample is in front of this pixel's layer: its light is the near field's
                // business, and bleeding it here would paint foreground over background.
                if (blocks - sampleBlocks > layerGate(blocks)) {
                    continue;
                }
                float sampleCoc = cocRadius(sampleBlocks) * 0.5;
                if (sampleCoc < 0.5) {
                    continue;
                }
                float coverage = apertureCoverage(offset / sampleCoc, 1.0 / sampleCoc);
                sum += coverage * source.SampleLevel(uv, 0.0).rgb;
                weightSum += coverage;
            }
        }
        result.farColor = sum / weightSum;
    }

    // ---- near field: the foreground discs that land on this pixel ----
    // The kernel radius needs the widest near disc in the neighbourhood, not this pixel's own: a sharp
    // background pixel under a blurred foreground edge gathers that edge's disc. Twelve depth probes
    // stand in for the tile pass a full pipeline would run; they bound the miss to one probe ring's
    // spacing instead of allowing an unbounded foreground to vanish.
    float nearCoc = (blocks > 0.0 && blocks < pc.focus) ? coc : 0.0;
    float probeMax = 0.0;
    for (int p = 0; p < 12; ++p) {
        float probeRadius = (p < 8 ? 0.35 : 0.75) * pc.maxCoc * 0.5;
        float probeAngle = float(p) * (2.0 * PI / 12.0) + 0.35;
        float2 offset = probeRadius * float2(cos(probeAngle), sin(probeAngle));
        float probeBlocks = axialBlocks(centre + offset, halfSize);
        if (probeBlocks > 0.0 && probeBlocks < pc.focus) {
            probeMax = max(probeMax, cocRadius(probeBlocks) * 0.5);
        }
    }
    // The kernel radius is the widest near disc the neighbourhood shows: this pixel's own, or the
    // widest one a probe found reaching toward it.
    float kernelHalf = max(nearCoc * 0.5, probeMax);
    result.nearAlpha = 0.0;
    result.nearColor = centreColor;
    if (kernelHalf > 0.5) {
        float3 sum = float3(0.0);
        float weightSum = 0.0;
        // What a fully covered kernel disc would accumulate: every tap tested against the kernel
        // radius itself. The centre tap is this pixel's own layer membership, counted separately
        // below, so the ratio reaches 1 exactly where the near layer covers the pixel outright.
        float fullSum = 0.0;
        int rings = clamp(int(ceil(kernelHalf / RING_STEP)), 1, MAX_RINGS);
        for (int k = 1; k <= rings; ++k) {
            float ringRadius = kernelHalf * float(k) / float(rings);
            int taps = clamp(int(ceil(2.0 * PI * ringRadius / RING_STEP)), 4, MAX_TAPS);
            float angle = float(k) * GOLDEN_ANGLE + 0.9;
            for (int t = 0; t < taps; ++t) {
                float theta = angle + 2.0 * PI * float(t) / float(taps);
                float2 offset = ringRadius * float2(cos(theta), sin(theta));
                fullSum += apertureCoverage(offset / kernelHalf, 1.0 / kernelHalf);
                float2 uv = (centre + offset + 0.25) / halfSize;
                float sampleBlocks = axialBlocks(centre + offset, halfSize);
                float sampleCoc = (sampleBlocks > 0.0 && sampleBlocks < pc.focus)
                        ? cocRadius(sampleBlocks) * 0.5 : 0.0;
                if (sampleCoc < 0.5) {
                    continue;
                }
                // A farther sample sits behind this pixel's layer: whatever covers it is its own
                // gather's business, and the near field may not borrow it.
                if (sampleBlocks - blocks > layerGate(sampleBlocks)) {
                    continue;
                }
                float coverage = apertureCoverage(offset / sampleCoc, 1.0 / sampleCoc);
                sum += coverage * source.SampleLevel(uv, 0.0).rgb;
                weightSum += coverage;
            }
        }
        if (nearCoc > 0.0) {
            // The pixel is itself foreground: its own sharp value is the centre of its layer.
            sum += centreColor;
            weightSum += 1.0;
        }
        if (weightSum > 0.0) {
            result.nearColor = sum / weightSum;
            result.nearAlpha = saturate(weightSum / max(fullSum, 1.0));
        }
    }
    return result;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    farImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    float2 halfSize = float2(w, h);
    FieldResult sdr = gatherField(sourceImage, pix, halfSize);
    farImage[pix] = float4(sdr.farColor, sdr.farMix);
    nearImage[pix] = float4(sdr.nearColor, sdr.nearAlpha);
    // Skipped while HDR output is off: the slots then hold the 1x1 placeholder, and writing through
    // them would store a garbage texel no reader is entitled to.
    if (pc.hdrEnabled != 0) {
        FieldResult hdr = gatherField(hdrSourceImage, pix, halfSize);
        hdrFarImage[pix] = float4(hdr.farColor, hdr.farMix);
        hdrNearImage[pix] = float4(hdr.nearColor, hdr.nearAlpha);
    }
}

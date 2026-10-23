// Adds back the fine detail DLSS-RR's upscale averages away, in display code values.
//
// Sharpening runs last: the display mapper writes its unsharpened SDR (sRGB-encoded rgba8) and PQ HDR
// images here, and this pass writes the finished pair the present paths consume. A high pass taken before
// the display transform would have to evaluate ACES/AgX on every tap, because the edge being restored only
// exists in the graded image; here the kernel costs nine loads.
//
// Per channel: out = saturate(centre + (centre - blur) * amount), blur a 3x3 binomial low pass. The high
// pass has zero DC gain, so flat areas and overall brightness are untouched and only the texel-to-neighbour
// difference grows. PQ, like sRGB, is encoded into [0,1], so one gain and one clamp serve both images.

import display_common;
import bindings;

[[vk::push_constant]] SharpenPush pc;

// 1 2 1 / 2 4 2 / 1 2 1, normalised by 16. Every weight is non-negative, so the blur cannot ring on its
// own: the overshoot at an edge comes only from the boosted difference, and saturate() bounds it.
float3 sharpenDetail(float3 centre, float3 crossSum, float3 cornerSum, float amount) {
    float3 blur = (centre * 4.0 + crossSum * 2.0 + cornerSum) * (1.0 / 16.0);
    return saturate(centre + (centre - blur) * amount);
}

// Load() ignores the sampler's address mode, so the border taps are clamped by hand. `last` is the final
// texel coordinate, which keeps the window inside the image at every pixel including the frame border,
// where the one-sided neighbourhood must not wrap to the opposite edge.
float3 sharpenChannel(Sampler2D source, int2 pix, int2 last, float amount) {
    float3 centre = source.Load(int3(pix, 0)).rgb;
    float3 crossSum = float3(0.0);
    crossSum += source.Load(int3(clamp(pix + int2(0, -1), int2(0), last), 0)).rgb;
    crossSum += source.Load(int3(clamp(pix + int2(0, 1), int2(0), last), 0)).rgb;
    crossSum += source.Load(int3(clamp(pix + int2(-1, 0), int2(0), last), 0)).rgb;
    crossSum += source.Load(int3(clamp(pix + int2(1, 0), int2(0), last), 0)).rgb;
    float3 cornerSum = float3(0.0);
    cornerSum += source.Load(int3(clamp(pix + int2(-1, -1), int2(0), last), 0)).rgb;
    cornerSum += source.Load(int3(clamp(pix + int2(1, -1), int2(0), last), 0)).rgb;
    cornerSum += source.Load(int3(clamp(pix + int2(-1, 1), int2(0), last), 0)).rgb;
    cornerSum += source.Load(int3(clamp(pix + int2(1, 1), int2(0), last), 0)).rgb;
    return sharpenDetail(centre, crossSum, cornerSum, amount);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    int2 last = int2(w, h) - 1;
    float amount = max(pc.amount, 0.0);
    outputImage[pix] = float4(sharpenChannel(sourceImage, pix, last, amount), 1.0);
    // Skipped entirely while HDR output is off: the dispatch still runs, but the slot is bound to the 1x1
    // placeholder then, so writing through it would store a garbage texel for a present path that is not
    // reading it.
    if (pc.hdrEnabled != 0) {
        hdrOutputImage[pix] = float4(sharpenChannel(hdrSourceImage, pix, last, amount), 1.0);
    }
}

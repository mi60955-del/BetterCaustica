// Post-process depth of field, recombine: scales the gather's half-res near and far fields back up
// over the compose pair, in place.
//
// Near layers over far layers over sharp is the order a real lens stacks them in: a foreground object
// blurs over whatever is behind it, the far field blurs behind both, and the focal plane keeps the
// image the pinhole traced. Each layer's weight arrived from the gather folded into its alpha, so this
// pass needs no lens state of its own -- only the two resolutions it walks between.
//
// The compose images are both read and written here: each thread loads exactly its own texel before it
// stores exactly its own texel, and no other thread touches that texel, so the pair is an in-place
// filter rather than a hazard. Writing in place is what lets the sharpen pass downstream keep reading
// the compose pair no matter which depth-of-field mode is set.
//
// In display code values, like every pass after the display mapper: the defocus mixes sRGB-encoded SDR
// and PQ-encoded HDR code values directly, where a scene-linear gather would have to evaluate the
// whole display transform per tap to reach the same place.

import display_common;
import bindings;

[[vk::push_constant]] DofPostRecombinePush pc;

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    // The gather's half-res fields, fetched at the bilinear sweet spot: a display-resolution uv lands
    // between half-res texels, which is the upscale itself.
    float2 uv = (float2(pix) + 0.5) / (pc.halfTexels * 2.0);
    float4 farField = farImage.SampleLevel(uv, 0.0);
    float4 nearField = nearImage.SampleLevel(uv, 0.0);
    float3 composed = outputImage[pix].rgb;
    composed = lerp(composed, farField.rgb, saturate(farField.a));
    composed = lerp(composed, nearField.rgb, saturate(nearField.a));
    outputImage[pix] = float4(composed, 1.0);
    // Skipped while HDR output is off: the slot then holds the 1x1 placeholder, and writing through it
    // would store a garbage texel no reader is entitled to.
    if (pc.hdrEnabled != 0) {
        float4 hdrFarField = hdrFarImage.SampleLevel(uv, 0.0);
        float4 hdrNearField = hdrNearImage.SampleLevel(uv, 0.0);
        float3 hdrComposed = hdrOutputImage[pix].rgb;
        hdrComposed = lerp(hdrComposed, hdrFarField.rgb, saturate(hdrFarField.a));
        hdrComposed = lerp(hdrComposed, hdrNearField.rgb, saturate(hdrNearField.a));
        hdrOutputImage[pix] = float4(hdrComposed, 1.0);
    }
}

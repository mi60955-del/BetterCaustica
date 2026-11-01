// One accumulated still image: every frame of a run is summed into the same buffer and the run's sample
// count divides it back out at the end.
//
// The pass sits at the finished-display-image seam, so what converges is the picture the screen is already
// showing — Ray Reconstruction, bloom, the tone curve and sharpening are all inside it — and the average
// removes only what varies from one frame to the next: the jitter's sub-pixel placement, and the noise the
// denoiser leaves behind. Summing frames rather than exponentially blending them is what makes the reset on
// a camera move exact and the result an unweighted mean.
//
// The accumulator is never cleared: mode 1 stores the first frame instead of adding it, which is the only
// thing that has to be different about the first sample of a run.

import bindings;
import display_common;

[[vk::push_constant]] StillPush pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    accumulatorImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    if (pc.mode == 2) {
        // The trip to 8 bits happens once, for the whole run, so every sample keeps its share of the
        // accumulator's precision up to this line. Alpha is a constant: the PNG this lands in has nothing
        // to composite against.
        //
        // The row flip is what makes the saved file the right way up. A readback hands over memory rows in
        // Vulkan image order, where row zero is the bottom of the picture, while a PNG's first row is its
        // top. Only the output image takes the flip: the accumulator stays in image order, so the adds that
        // read it texel-for-texel are unaffected.
        int2 store = int2(pix.x, int(h) - 1 - pix.y);
        outputImage[store] = float4(max(accumulatorImage[pix].rgb * pc.weight, float3(0.0)), 1.0);
        return;
    }
    float3 frame = saturate(sourceImage.Load(int3(pix, 0)).rgb);
    if (pc.mode == 1) {
        accumulatorImage[pix] = float4(frame, 1.0);
    } else {
        accumulatorImage[pix] = float4(accumulatorImage[pix].rgb + frame, 1.0);
    }
}

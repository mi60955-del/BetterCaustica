// Where the lens should focus, metered off the frame that was just traced.
//
// The input is the depth the primary pass publishes for its own blur ceiling: the axial distance each
// pixel's pinhole ray travelled, to the surface it actually drew. That is the number autofocus wants, and
// it is the number a CPU raycast through the block state cannot produce — a block-shape probe answers with
// one block's voxel face, so it walks straight through entities, particles, painting, water surfaces and
// the half of a block a fence or a stair actually occupies, and it reports the surface behind them as what
// the frame shows.
//
// That depth is also what a mirror shows past itself, where the reflector's own distance is not the
// picture: with reflection focus on, the primary pass publishes the reflected image's distance on those
// pixels instead, so a lake's reflection focuses at the mountain it is carrying rather than at the water
// surface it is reflected in. This pass cannot tell the two apart and does not need to — one distance per
// pixel is one answer per pixel.
//
// The statistic is the window's mean DIOPTRIC distance inverted, which is its harmonic mean: a lens bends
// light so that a surface at range s lands off the focal plane by |1/s - 1/f|, so 1/s is the quantity the
// focal plane is chosen against. Averaging ranges instead would average the wrong thing — one block at
// 0.5 and nine at 50 would answer 45 blocks, putting the plane in front of the near surface and behind
// nothing, while the dioptric mean answers 4.6 blocks, which is the near surface with the field pulling it
// out. Sky is the far stop, and it contributes almost nothing to that mean, which is what infinity is for.

import display_common;
import bindings;

[[vk::push_constant]] DofFocusPush pc;

// One 16x16 workgroup covers the whole window: each thread strides across it, so the window can be any
// size the host asks for without a second dispatch or a global accumulation buffer, and the reduction is
// one barrier.
groupshared float2 localSum[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    float diopters = 0.0;
    float taps = 0.0;
    int2 thread = int2(dispatchId.xy);
    int2 origin = int2(pc.originX, pc.originY);
    for (int y = thread.y; y < pc.extentY; y += 16) {
        for (int x = thread.x; x < pc.extentX; x += 16) {
            float range = focusDepth[origin + int2(x, y)];
            // A texel no trace has written yet holds whatever the allocator returned, and NaN or a
            // negative distance is not a surface. Rejecting it leaves the tap out of the mean rather than
            // letting one unwritten pixel decide where the frame focuses; a window with no valid tap
            // answers the zero below, which the host reads as "no measurement yet".
            if (range > 0.0 && range < 1.0e7) {
                diopters += 1.0 / clamp(range, pc.nearBlock, pc.farBlock);
                taps += 1.0;
            }
        }
    }

    localSum[groupIndex] = float2(diopters, taps);
    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0u) {
        float totalDiopters = 0.0;
        float totalTaps = 0.0;
        for (uint i = 0u; i < 256u; ++i) {
            totalDiopters += localSum[i].x;
            totalTaps += localSum[i].y;
        }
        focusMeter[0] = totalTaps > 0.0 && totalDiopters > 0.0
                ? float2(totalTaps / totalDiopters, totalTaps) : float2(0.0, 0.0);
    }
}

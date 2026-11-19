// Full-resolution reconstruction of a subsampled indirect trace.
//
// When the frame asks for subsampled indirect, the indirect raygen traces only some pixels and stores the
// pre-exposed radiance plus that pixel's linear depth. With factor 2 the traced pixels are those with
// x % 2 == 0 and y % 2 == 0; with factor 3 they are the two-pixel groups x % 3 < 2 and y % 3 < 2. The
// traced pixels are packed into a compact texture, so this pass maps every full-resolution pixel back into
// compact coordinates, bilinearly gathers the four surrounding texels, and weights each by how close its
// depth is to this pixel's depth guide — a background texel across a silhouette gets a near-zero weight
// instead of bleeding over the foreground. A pixel that was itself traced lands exactly on a compact texel
// and copies its own value.
//
// The output is the same rgba16f radiance the indirect raygen writes on full-resolution frames: the same
// pre-exposure, the same fp16 clamp, alpha 1. DLSS-RR sees no difference other than the resolution of the
// lighting it denoises.
import world_common;
import world_core;
import bindings;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint subsample = worldPush.indirectSubsample;
    if (subsample < 2u) {
        return;
    }
    uint2 pix = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint2 compactDims = ((dimensions + subsample - 1u) / subsample) * (subsample - 1u);
    uint2 group = pix / subsample;
    uint2 offset = pix % subsample;
    // Traced pixels occupy compact positions group * (s - 1) + offset. A dropped pixel on one axis sits
    // halfway between the last texel of its group and the first texel of the next, hence the -0.5.
    float2 compact = float2(group * (subsample - 1u)) + float2(offset);
    compact.x = (offset.x == subsample - 1u) ? compact.x - 0.5 : compact.x;
    compact.y = (offset.y == subsample - 1u) ? compact.y - 0.5 : compact.y;
    if (offset.x != subsample - 1u && offset.y != subsample - 1u) {
        float3 own = indirectHalf[uint2(compact)].rgb;
        outImage[pix] = float4(clamp(own, float3(0.0), float3(HALF_MAX)), 1.0);
        return;
    }
    float depth = gDepth[pix];
    int2 base = int2(floor(compact));
    float2 frac = compact - float2(base);
    float3 sum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [unroll]
    for (int j = 0; j < 2; ++j) {
        [unroll]
        for (int i = 0; i < 2; ++i) {
            int2 coord = clamp(base + int2(i, j), int2(0, 0), int2(compactDims) - 1);
            float spatial = (i == 0 ? 1.0 - frac.x : frac.x) * (j == 0 ? 1.0 - frac.y : frac.y);
            float halfDepth = indirectHalfDepth[coord];
            // Linear depth in blocks: a texel one block behind still contributes (it is the same surface at
            // a grazing angle); a texel metres away contributes almost nothing.
            float weight = spatial / (1.0e-3 + abs(halfDepth - depth)) + 1.0e-8;
            sum += indirectHalf[coord].rgb * weight;
            weightSum += weight;
        }
    }
    float3 radiance = sum / weightSum;
    outImage[pix] = float4(clamp(radiance, float3(0.0), float3(HALF_MAX)), 1.0);
}

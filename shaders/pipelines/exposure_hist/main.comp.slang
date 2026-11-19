// Strided log2-luminance histogram over the post-RR trace color. Metering after RR avoids the
// Jensen bias that Monte-Carlo noise introduces into the logarithmic average.

import display_common;
import bindings;

[[vk::push_constant]] ExposureHistPush pc;

static const float LOG_MIN = -12.0;
static const float LOG_MAX = 12.0;
static const float INV_LOG_RANGE = 1.0 / (LOG_MAX - LOG_MIN);
static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

// One 16x16 workgroup == 256 threads == one thread per luminance bin. Each thread initializes and
// flushes all three population bins. Per-pixel atomics stay in shared memory; only non-empty bins
// pay global atomics.
groupshared uint localBins[768];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    localBins[groupIndex] = 0u;
    localBins[256u + groupIndex] = 0u;
    localBins[512u + groupIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();

    int2 pix = int2(dispatchId.xy * pc.stride);
    uint w, h;
    colorImage.GetDimensions(w, h);
    if (pix.x < int(w) && pix.y < int(h)) {
        // The histogram bins log2 luminance, and log is concave: Monte-Carlo noise makes the average
        // log lower than the log of the average (Jensen), which biases the metered EV and with it the
        // exposure. DLSS-RR normally hides that by supplying a denoised image; with RR off this pass
        // sees the raw trace. Average a 4x4 block per sample first, which cuts the noise variance (and
        // the bias) by sixteen while keeping the sample grid and the population weighting unchanged.
        // Blocks overlap when stride < 4; that only smooths the metered distribution further.
        const int BOX = 4;
        int2 maxPix = int2(int(w), int(h)) - 1;
        float3 rgb = float3(0.0);
        uint valid = 0u;
        for (int dy = 0; dy < BOX; dy++) {
            for (int dx = 0; dx < BOX; dx++) {
                float3 tap = colorImage[min(pix + int2(dx, dy), maxPix)].rgb;
                if (all(isfinite(tap))) {
                    rgb += max(tap, float3(0.0));
                    valid++;
                }
            }
        }
        // Invalid radiance is not a meaningful metering sample. Skipping it is safe because resolve
        // derives its population from the bins instead of assuming image area.
        if (valid > 0u) {
            rgb /= float(valid);
            // The reconstructed scene image is linear ACEScg; meter it in the transport basis.
            float lum = dot(rgb, ACESCG_LUMA);
            float logLum = clamp(log2(max(lum, 1.0e-5)), LOG_MIN, LOG_MAX);
            uint bin = min(uint(floor((logLum - LOG_MIN) * INV_LOG_RANGE * 256.0)), 255u);

            float2 uv = (float2(pix) + 0.5) / float2(w, h);
            float2 centered = uv * 2.0 - 1.0;
            float sigma = max(pc.centerWeightSigma, 0.01);
            float gaussian = exp(-0.5 * dot(centered, centered) / (sigma * sigma));
            float weight = lerp(clamp(pc.centerWeightFloor, 0.0, 1.0), 1.0, gaussian);
            uint fixedWeight = uint(round(weight * 256.0));

            uint depthW, depthH;
            depthImage.GetDimensions(depthW, depthH);
            int2 guidePix = min(int2(uv * float2(depthW, depthH)), int2(depthW, depthH) - 1);
            bool sky = depthImage[guidePix] <= 1.0e-6;
            bool emissive = !sky && albedoImage[guidePix].a > 0.5;
            uint populationOffset = sky ? 256u : (emissive ? 512u : 0u);
            uint weightedBin = bin + populationOffset;
            if (fixedWeight > 0u) {
                uint unused;
                InterlockedAdd(localBins[weightedBin], fixedWeight, unused);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    uint surfaceCount = localBins[groupIndex];
    if (surfaceCount > 0u) {
        uint unusedSurface;
        InterlockedAdd(histBins[groupIndex], surfaceCount, unusedSurface);
    }
    uint skyCount = localBins[256u + groupIndex];
    if (skyCount > 0u) {
        uint unusedSky;
        InterlockedAdd(histBins[256u + groupIndex], skyCount, unusedSky);
    }
    uint emissiveCount = localBins[512u + groupIndex];
    if (emissiveCount > 0u) {
        uint unusedEmissive;
        InterlockedAdd(histBins[512u + groupIndex], emissiveCount, unusedEmissive);
    }
}

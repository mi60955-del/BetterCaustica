// The accumulated still render: every frame of a run is summed into the same scene-linear buffer, and the
// run's sample count divides that sum back out into the display-res mean the display chain renders.
//
// The pass sits before exposure, bloom and the tone curve, so what converges is the light the display
// chain is about to map, not the mapped picture. A tone curve is not linear, so averaging its output
// darkens and desaturates as the sample count grows; averaging radiance and mapping once is the long
// exposure a still render is meant to be.
//
// Two estimators share this pass, chosen once per run by `pc.legacy`:
//
//   The current one clamps nothing. A clamp that can only lower a value lowers the estimate, and it lowers
//   it most where the local noise is largest: it erodes the convex edges and corners that block geometry is
//   made of, and because the clamped set follows the current noise draw, the displayed mean flickers while
//   it converges. Outliers are handled where they are identifiable instead — the display filter compares a
//   tap against the pixel's own measured noise, which is a continuous function of the data and biases
//   nothing.
//
//   The previous one did clamp, against the pixel's own running mean. It is kept because a long run of the
//   earlier builds produced it and a picture taken then should stay reproducible: a sample is limited to
//   LEGACY_FIREFLY_FACTOR times max(running mean, LEGACY_FIREFLY_FLOOR) per channel. The cap is
//   self-referential — a sample above the mean raises the mean, which raises the cap for everything that
//   follows — so a spike that lands on the first frame is never pulled back, and the clamp trims far more
//   than outliers: on a dim frame much of the bright half sits above eight times its pixel's mean.
//
// The current estimator's second product is measured noise. The accumulator's alpha carries the running
// second moment of luminance (Welford's M2, in the same pre-exposed radiance units as the sum), which the
// resolve turns into the relative standard deviation of the mean. Keeping it in the alpha the sum does not
// use keeps the whole run in one image instead of two. The legacy path leaves the alpha at one, because
// nothing reads it there.
//
// The accumulator is never cleared: mode 1 stores the first frame instead of adding it, which is the only
// thing that has to be different about the first sample of a run. The resolve writes image order — no row
// flip — because the result is a texture the display chain reads, not a PNG; the final display frame is
// flipped when it is captured for the file.

import bindings;
import display_common;

[[vk::push_constant]] StillPush pc;

static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);
// Luminance a mean is floored at before its noise is expressed as a ratio: the quotient stops being
// meaningful as the mean approaches zero, and a pixel that dark has no signal left to protect.
static const float MEAN_FLOOR = 1.0e-4;
// Legacy cap: a sample may reach this multiple of its pixel's running mean. The floor is in pre-exposed
// radiance, so in a dark texel — whose mean sits below it — the cap is effectively a flat factor of it.
static const float LEGACY_FIREFLY_FACTOR = 8.0;
static const float LEGACY_FIREFLY_FLOOR = 0.01;

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
        float count = max(float(pc.count), 1.0);
        float3 mean = max(accumulatorImage[pix].rgb * pc.weight, float3(0.0));
        if (pc.legacy != 0) {
            // No measured spread in this mode: the filter's gate is a fixed multiple of 1/sqrt(N) instead.
            outputImage[pix] = float4(mean, 1.0);
            return;
        }
        // Standard error of the mean: the sample variance M2 / (n - 1) divided by n, expressed relative to
        // the mean because that is what the display filter's tap gate compares. A single sample has no
        // spread to report, and reporting zero is right: there is nothing yet to average it with.
        float m2 = max(accumulatorImage[pix].a, 0.0);
        float variance = count > 1.0 ? m2 / ((count - 1.0) * count) : 0.0;
        float noise = sqrt(max(variance, 0.0)) / max(dot(mean, ACESCG_LUMA), MEAN_FLOOR);
        outputImage[pix] = float4(mean, noise);
        return;
    }
    // Scene-linear radiance. No saturate: values above one are the highlights the tone curve exists to roll
    // off, and the ceiling only keeps a pathological sample from reaching infinity.
    float3 frame = min(max(sourceImage.Load(int3(pix, 0)).rgb, float3(0.0)), float3(60000.0));
    if (pc.mode == 1) {
        accumulatorImage[pix] = float4(frame, 1.0);
        return;
    }
    if (pc.legacy != 0) {
        // pc.count is the sample count before this frame joins the sum, so the running mean divides by it.
        float3 runningMean = accumulatorImage[pix].rgb / max(float(pc.count), 1.0);
        float3 cap = LEGACY_FIREFLY_FACTOR
                * max(runningMean, float3(LEGACY_FIREFLY_FLOOR, LEGACY_FIREFLY_FLOOR, LEGACY_FIREFLY_FLOOR));
        accumulatorImage[pix] = float4(accumulatorImage[pix].rgb + min(frame, cap), 1.0);
        return;
    }
    // Welford's update for the luminance second moment, with the running mean taken from the sum this pass
    // already stores. pc.count is the sample count before this frame joins it, so its divisor matches the
    // sum the running mean is derived from.
    float3 sum = accumulatorImage[pix].rgb;
    float count = max(float(pc.count), 1.0);
    float lumaBefore = dot(sum, ACESCG_LUMA) / count;
    float lumaSample = dot(frame, ACESCG_LUMA);
    float lumaAfter = (dot(sum, ACESCG_LUMA) + lumaSample) / (count + 1.0);
    float m2 = accumulatorImage[pix].a + (lumaSample - lumaBefore) * (lumaSample - lumaAfter);
    accumulatorImage[pix] = float4(sum + frame, max(m2, 0.0));
}

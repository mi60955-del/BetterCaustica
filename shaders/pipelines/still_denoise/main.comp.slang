// Spatial filtering over scene-linear radiance, shared by the still render's accumulated mean and the
// single traced frame DLSS-RR reconstructs. Both sit before exposure, bloom and the tone curve, so a
// filter width in radiance stays meaningful across the whole exposure range.
//
// A still run averages independent frames, so what is left after N samples is white noise at roughly
// 1/sqrt(N) of a single frame — the mean is unbiased and needs no temporal history, only a filter that
// averages texels which see the same light. The surfel guides the world pass already wrote for DLSS-RR
// carry exactly that: a unit world-space normal, the diffuse albedo, and the endpoint depth.
//
// Four modes share one pipeline:
//   mode 0 demodulates the mean by the surface's own albedo, leaving the irradiance it receives. Texture
//          and material edges then differ in the residual, not in the signal, so the a-trous kernel below
//          can walk across an albedo edge without blurring it.
//   mode 1 runs one a-trous step over that residual, throwing away taps across a silhouette (depth and
//          normal) and taps that land on unrelated light (luminance).
//   mode 2 remodulates the filtered residual, which is exact where the filter did not touch it.
//   mode 3 clamps isolated single-texel outliers in the one-sample frame handed to DLSS-RR. That frame has
//          no history to measure a spread from, so the 3x3 neighbourhood stands in for it. Only a texel
//          brighter than every neighbour is clamped: a real highlight covers several texels and has to
//          survive untouched.
//
// The a-trous gate measures each pixel against its own noise rather than against a constant: the still
// pass's resolve writes the relative standard deviation of the mean into the scene image's alpha, and a tap
// is weighted by how far its luminance departs from the center in units of that spread. A dark pixel whose
// mean is still 100% noise therefore admits the neighbours it must average with, while a converged bright
// one rejects everything but its own value — one formulation that stays correct across the exposure range
// instead of a relative constant that is either too tight in shadow or too loose in light. A run that
// accumulated with the earlier estimator has no spread to report, so it passes `measured` off and the gate
// falls back to that constant: sigma times the local level.
//
// The demodulation is scalar (the albedo's luminance), not per channel: a per-channel divide blows up
// wherever a channel's albedo is near zero — a saturated red or green material — and the round trip has to
// stay exact for the unmodified texels.

import bindings;
import display_common;

[[vk::push_constant]] StillDenoisePush pc;

static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

// Albedo the demodulation divides by at least: a black surface's residual is meaningless (its radiance is
// emission or specular, not diffuse), and letting the divisor reach zero turns it into inf.
static const float ALBEDO_FLOOR = 0.02;
// A depth difference this large, relative to the nearer of the two endpoints, separates surfaces. The guide
// depth is reversed-Z NDC, whose relative difference is the relative view distance (z_ndc ~ near / z), so
// one tolerance serves the whole depth range. It is loose enough that a grazing surface's own taps, whose
// distance changes fast across a wide a-trous stride, still pass, and far tighter than a silhouette.
static const float DEPTH_TOLERANCE = 0.05;
// Cosine power of the normal gate: a tap is rejected as the surfaces' normals diverge, which is what keeps
// a floor from bleeding onto the wall it meets.
static const float NORMAL_POWER = 16.0;
// Luminance a gate's comparison is floored at, in the same units as the signal.
static const float LUMA_EPSILON = 1.0e-6;
// Relative spread a texel is credited with when the scene image carries less: a mean of a single sample has
// no measured spread, and a well-lit surface converges to a few thousandths, which is far below the point
// where a difference between two texels is worth keeping. Flooring the spread is what lets the kernel
// smooth a converged highlight instead of freezing every texel at its own value; shadow texels report far
// more than this and are unaffected by it.
static const float NOISE_FLOOR = 0.03;

// Mode 3's cap: a multiple of the 3x3 neighbourhood mean, its radiance floor, and how much brighter than
// the brightest neighbour a texel has to be before it counts as isolated. Eight times the mean is what a
// spike has to clear to be worth touching: a 1 spp frame in a dim scene carries relative noise well over
// 100%, so a low multiple describes a large share of the bright half of the histogram and pulls it down to
// the local level instead of removing an outlier — measured on a live dim frame, a factor of two dropped the
// image's mean by 3.6x and its 90th percentile from 170 to 56. The spikes this exists for sit orders of
// magnitude above their neighbours, and the isolation test is what keeps clustered highlights, thin bright
// geometry and silhouette edges out of it.
static const float RR_FIREFLY_FACTOR = 8.0;
static const float RR_FIREFLY_FLOOR = 0.01;
static const float RR_FIREFLY_ISOLATION = 1.5;

// Separable B3 weights of the 5x5 a-trous lattice: (1, 4, 6, 4, 1) / 16.
float b3(int offset) {
    int distance = offset < 0 ? -offset : offset;
    return distance == 0 ? 0.375 : (distance == 1 ? 0.25 : 0.0625);
}

float luminanceOf(float3 color) {
    return dot(max(color, float3(0.0)), ACESCG_LUMA);
}

// The albedo the demodulation divides by: a scalar, floored so a dark material cannot blow the residual up.
float demodulationOf(float4 albedo) {
    return max(luminanceOf(albedo.rgb), ALBEDO_FLOOR);
}

// 1 for two endpoints on the same surface, 0 across a silhouette. Sky and any other texel the guide never
// wrote carry a depth of zero, so they all gate each other in and everything else out.
float depthGate(float centerDepth, float tapDepth) {
    float scale = max(max(centerDepth, tapDepth), LUMA_EPSILON);
    return exp(-abs(centerDepth - tapDepth) / (DEPTH_TOLERANCE * scale));
}

// 1 for a tap within the center's own measured noise, fading with the square of how far outside it the tap
// sits. `noise` is the relative standard deviation of the mean in the scene image's alpha, and `scale` is
// the dispatcher's dial on how many of those it takes to count as different light. Both endpoints are
// credited with the larger of the two spreads, because they are neighbouring texels of one surface and
// their estimates are drawn from the same distribution; comparing against each one's own spread instead
// would let a texel that happens to be noisy talk itself into keeping all of its noise.
//
// With `measured` off, the spread is the dispatcher's sigma times the local level instead: the earlier
// estimator's fixed 1/sqrt(N) tolerance, kept so its filter can still be reproduced.
float lumaGate(float centerLuminance, float tapLuminance, float noise, float scale, bool measured) {
    float level = max(max(centerLuminance, tapLuminance), LUMA_EPSILON);
    float spread = measured ? max(noise, NOISE_FLOOR) * level * scale : level * scale;
    float relative = (tapLuminance - centerLuminance) / max(spread, LUMA_EPSILON);
    return exp(-relative * relative);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pixel = int2(dispatchId.xy);
    uint width, height;
    outputImage.GetDimensions(width, height);
    if (pixel.x >= int(width) || pixel.y >= int(height)) {
        return;
    }
    int2 last = int2(int(width) - 1, int(height) - 1);
    int2 here = clamp(pixel, int2(0), last);

    float3 center = max(sourceImage[here].rgb, float3(0.0));
    // The run's own noise estimate, carried in the alpha the still pass wrote with the mean.
    float noise = max(sourceImage[here].a, 0.0);
    float demodulation = demodulationOf(guideAlbedo[here]);

    if (pc.mode == 0) {
        outputImage[pixel] = float4(center / demodulation, noise);
        return;
    }
    if (pc.mode == 2) {
        outputImage[pixel] = float4(center * demodulation, 1.0);
        return;
    }
    if (pc.mode == 3) {
        float3 neighbourSum = float3(0.0, 0.0, 0.0);
        float neighbourMax = 0.0;
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                if (x == 0 && y == 0) continue;
                int2 tapped = clamp(pixel + int2(x, y), int2(0), last);
                float3 tap = max(sourceImage[tapped].rgb, float3(0.0, 0.0, 0.0));
                neighbourSum += tap;
                neighbourMax = max(neighbourMax, max(tap.r, max(tap.g, tap.b)));
            }
        }
        float3 neighbourMean = neighbourSum * (1.0 / 8.0);
        float cap = RR_FIREFLY_FACTOR
                * max(max(neighbourMean.r, max(neighbourMean.g, neighbourMean.b)), RR_FIREFLY_FLOOR);
        float magnitude = max(center.r, max(center.g, center.b));
        float3 result = center;
        if (magnitude > neighbourMax * RR_FIREFLY_ISOLATION && magnitude > cap) {
            result = center * (cap / magnitude);
        }
        outputImage[pixel] = float4(result, 1.0);
        return;
    }

    float3 centerNormal = guideNormal[here].xyz;
    float centerDepth = guideDepth[here];
    float centerLuminance = luminanceOf(center);
    float3 filtered = float3(0.0);
    float weightSum = 0.0;
    // A texel the guide never wrote (sky, or anything the world pass did not shade) has a zero normal,
    // which would gate every tap including the center one out; such a texel keeps its own value.
    if (dot(centerNormal, centerNormal) > 0.5) {
        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int2 tapped = clamp(pixel + int2(x, y) * pc.step, int2(0), last);
                float3 tap = max(sourceImage[tapped].rgb, float3(0.0));
                float weight = b3(x) * b3(y);
                weight *= pow(saturate(dot(centerNormal, guideNormal[tapped].xyz)), NORMAL_POWER);
                weight *= depthGate(centerDepth, guideDepth[tapped]);
                weight *= lumaGate(centerLuminance, luminanceOf(tap),
                        max(noise, sourceImage[tapped].a), pc.sigma, pc.measured != 0);
                filtered += tap * weight;
                weightSum += weight;
            }
        }
    }
    if (weightSum <= LUMA_EPSILON) {
        outputImage[pixel] = float4(center, noise);
        return;
    }
    filtered /= weightSum;
    // `amount` is the strength dial: it blends the filtered residual back over the input, so zero is an
    // exact copy and one is the whole kernel. Applying it per step is harmless — three steps at half
    // strength still keep every tap's contribution below the input's own weight. The noise estimate rides
    // along untouched: it describes the run, not the pass.
    outputImage[pixel] = float4(lerp(center, filtered, saturate(pc.amount)), noise);
}

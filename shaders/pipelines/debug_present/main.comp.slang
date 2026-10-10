// Computes and presents debugView content as a final inspection pass after ordinary path-traced
// radiance, RR/fallback upscale, exposure metering, and the ACES display transform. Keeping debug
// coloring out of the occupancy-bound primary raygen avoids register pressure on the normal render path.
//
// Reads the guide buffers directly at their real render resolution and nearest-samples them into the
// display-resolution output. Debug mode deliberately leaves DLSS-RR and its jitter enabled, so this
// pass shows the actual guide inputs used by RR rather than changing the renderer being inspected.

import display_common;
import bindings;

[[vk::push_constant]] DebugPresentPush pc;

static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

// OCIO cg-config-v4.0.0 ACES 2.0: ACEScg/AP1/D60 to Linear Rec.709/D65. Debug guide colors cross
// back to the rgba8 target here; semantic debug colors (normal, depth, motion, weight) are already
// authored as literal sRGB code values and bypass this conversion.
static const float3 ACESCG_TO_BT709_R = float3( 1.70505095, -0.62179214, -0.08325887);
static const float3 ACESCG_TO_BT709_G = float3(-0.13025641,  1.14080477, -0.01054832);
static const float3 ACESCG_TO_BT709_B = float3(-0.02400336, -0.12896897,  1.15297234);

float linearToSrgbChannel(float value) {
    value = clamp(value, 0.0, 1.0);
    return value <= 0.0031308
            ? 12.92 * value
            : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float3 acesCgToSrgb(float3 color) {
    float3 linearBt709 = float3(
            dot(color, ACESCG_TO_BT709_R),
            dot(color, ACESCG_TO_BT709_G),
            dot(color, ACESCG_TO_BT709_B));
    return float3(
            linearToSrgbChannel(linearBt709.r),
            linearToSrgbChannel(linearBt709.g),
            linearToSrgbChannel(linearBt709.b));
}

// Display-referred, deliberately discrete one-stop bands. `ev == 0` means the value entering ACES
// is 18% grey; cool colors are below it, warm colors above it. These values are literal SDR/sRGB
// debug colors written after the display transform, not scene radiance.
float3 exposureFalseColor(float ev) {
    float band = floor(ev + 0.5);
    if (band <= -6.0) return float3(0.01, 0.00, 0.03);
    if (band == -5.0) return float3(0.12, 0.00, 0.25);
    if (band == -4.0) return float3(0.10, 0.15, 0.75);
    if (band == -3.0) return float3(0.00, 0.55, 1.00);
    if (band == -2.0) return float3(0.00, 0.75, 0.55);
    if (band == -1.0) return float3(0.10, 0.80, 0.15);
    if (band ==  0.0) return float3(0.46, 0.46, 0.46);
    if (band ==  1.0) return float3(0.78, 0.78, 0.05);
    if (band ==  2.0) return float3(1.00, 0.50, 0.00);
    if (band ==  3.0) return float3(1.00, 0.05, 0.02);
    if (band ==  4.0) return float3(1.00, 0.00, 0.70);
    if (band ==  5.0) return float3(1.00, 0.55, 0.80);
    return float3(1.00, 1.00, 1.00);
}

// Exactly the metering weight used by the histogram: local Gaussian centre weighting multiplied by
// the same frame-global sky/emissive scale computed by resolve from separate populations.
float meteringWeight(float2 uv, float depth, bool emissive) {
    float2 centered = uv * 2.0 - 1.0;
    float sigma = max(pc.centerWeightSigma, 0.01);
    float gaussian = exp(-0.5 * dot(centered, centered) / (sigma * sigma));
    float centerWeight = lerp(clamp(pc.centerWeightFloor, 0.0, 1.0), 1.0, gaussian);
    float populationWeight = depth <= 1.0e-6 ? exposureState[0].meteringSkyScale
            : (emissive ? exposureState[0].meteringEmissiveScale : 1.0);
    return centerWeight * populationWeight;
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

    uint guideW, guideH;
    gNormal.GetDimensions(guideW, guideH);
    float2 uv = (float2(pix) + 0.5) / float2(w, h);
    int2 guidePix = min(int2(uv * float2(guideW, guideH)), int2(guideW, guideH) - 1);
    float2 motionToDisplay = float2(w, h) / float2(guideW, guideH);

    float4 normalRough = gNormal[guidePix];
    float3 dbg;
    if (pc.debugView == 8u) {
        float luminance = dot(max(sceneImage[pix].rgb, float3(0.0)), ACESCG_LUMA);
        float exposedLuminance = luminance * max(exposureImage[int2(0, 0)], 0.0);
        float evFromMidGrey = log2(max(exposedLuminance, 1.0e-6) / 0.18);
        dbg = exposureFalseColor(evFromMidGrey);
    } else if (pc.debugView == 9u) {
        float weight = meteringWeight(uv, gDepth[guidePix], gAlbedo[guidePix].a > 0.5);
        dbg = float3(weight, weight, weight);
    } else if (pc.debugView == 10u) {
        // r lane: fog transmittance along the camera ray (1 = clear air, 0 = fully opaque fog).
        float t = volumeDebug[guidePix].r;
        dbg = float3(t, t, t);
    } else if (pc.debugView == 11u) {
        // g lane: the fog in-scatter's luminance, Reinhard-scaled so daylight values stay readable.
        float insc = max(volumeDebug[guidePix].g, 0.0);
        float shown = insc / (1.0 + insc);
        dbg = float3(shown, shown, shown);
    } else if (pc.debugView == 12u) {
        // b lane: cloud density along the camera ray (inert zero until the cloud layers exist).
        float density = max(volumeDebug[guidePix].b, 0.0);
        dbg = float3(density, density, density);
    } else if (pc.debugView == 13u) {
        // a lane: cloud-shadow transmittance at the first cloud crossing (1 = unshadowed).
        float shadow = saturate(volumeDebug[guidePix].a);
        dbg = float3(shadow, shadow, shadow);
    } else if (pc.debugView == 1u) {
        dbg = normalRough.xyz * 0.5 + 0.5;
    } else if (pc.debugView == 2u) {
        dbg = acesCgToSrgb(gAlbedo[guidePix].rgb);
    } else if (pc.debugView == 3u) {
        float depth = gDepth[guidePix];
        dbg = float3(depth, depth, depth);
    } else if (pc.debugView == 4u) {
        dbg = float3(normalRough.w, normalRough.w, normalRough.w);
    } else if (pc.debugView == 6u) {
        dbg = acesCgToSrgb(gSpecAlbedo[guidePix].rgb);
    } else if (pc.debugView == 7u) {
        dbg = float3(clamp(0.5 + gSpecMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5);
    } else {
        dbg = float3(clamp(0.5 + gMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5);
    }
    outputImage[pix] = float4(clamp(dbg, 0.0, 1.0), 1.0);
}

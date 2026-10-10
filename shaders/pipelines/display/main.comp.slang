// Maps the display-res scene-linear ACEScg RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR.
// Tonemap seam: the path tracer and DLSS-RR use scene-linear ACEScg, and exposure is applied from the
// compositor-owned 1x1 image right here. pc.advancedGrading then selects the display chain: by default
// the image goes through the look package's scene-referred ACES look (LMT) and the baked ACES 2.0
// output-transform LUTs (see tools/bake_display_lut.py); with advanced grading on it goes through AgX
// instead, which needs no look package. Before that selection, pc.purkinjeEnabled re-weights the
// exposed image by the CIE mesopic response, so the dark-adaptation shift reaches every display path
// rather than being duplicated per grading (see applyPurkinje).

import display_common;
import bindings;

[[vk::push_constant]] DisplayPush pc;

// MUST exactly match SHAPER_LO_STOPS/SHAPER_HI_STOPS in tools/bake_display_lut.py -- these are the two
// halves of one shaper function, one run at bake time (Python), one at sample time (here); if they
// drift the LUT fetch reads the wrong shelf of the LUT everywhere, silently.
static const float LUT_SHAPER_LO_STOPS = -12.0;
static const float LUT_SHAPER_HI_STOPS = 12.0;

float3 shaperEncode(float3 linearColor) {
    float3 stops = log2(max(linearColor, float3(1.0e-6)));
    return clamp((stops - LUT_SHAPER_LO_STOPS) / (LUT_SHAPER_HI_STOPS - LUT_SHAPER_LO_STOPS), 0.0, 1.0);
}

// Edge-aligned LUT: texel 0's centre is uvw=0.0 and texel (lutSize-1)'s centre is uvw=1.0, but a
// sampler's normalized [0,1] domain spans texel *centres* at 0.5/N .. 1-0.5/N by default. Remap into
// that convention; RtToneLut's CLAMP_TO_EDGE then holds the boundary texel for any exposed value
// outside the shaper's +/-12 EV range instead of wrapping. toneLut/hdrToneLut share pc.lutSize;
// the independently sized lookLut passes pc.lookLutSize.
float3 lutTexCoord(float3 uvw, float size) {
    return uvw * ((size - 1.0) / size) + float3(0.5 / size);
}

float3 shaperDecode(float3 shaped) {
    return exp2(lerp(
            float3(LUT_SHAPER_LO_STOPS),
            float3(LUT_SHAPER_HI_STOPS),
            clamp(shaped, 0.0, 1.0)));
}

float3 applyLook(float3 exposedAcesCg) {
    if (pc.lookEnabled == 0) {
        return exposedAcesCg;
    }
    float3 uvw = shaperEncode(exposedAcesCg);
    float3 shaped = lookLut.SampleLevel(lutTexCoord(uvw, pc.lookLutSize), 0.0).rgb;
    return shaperDecode(shaped);
}

float3 sampleBloom(int2 outputPixel, uint outputWidth, uint outputHeight) {
    float2 uv = (float2(outputPixel) + 0.5) / float2(outputWidth, outputHeight);
    return bloomImage.SampleLevel(uv, 0.0).rgb;
}

// Artistic gamma after the view/display transform. Apply the curve to display luminance and scale
// RGB uniformly instead of exponentiating each channel: this lifts shadows/midtones without pulling
// chromaticities toward white. The uniform scale is limited at the display boundary so saturated
// colors preserve their channel ratios instead of clipping individual channels.
static const float3 BT709_LUMA = float3(0.2126, 0.7152, 0.0722);
static const float3 BT2020_LUMA = float3(0.2627, 0.6780, 0.0593);

float3 luminanceGamma(float3 linearDisplayColor, float3 lumaCoefficients) {
    float3 color = clamp(linearDisplayColor, 0.0, 1.0);
    float luminance = dot(color, lumaCoefficients);
    if (luminance <= 0.0) {
        return color;
    }

    float adjustedLuminance = pow(luminance, clamp(pc.gamma, 0.1, 5.0));
    float scale = adjustedLuminance / luminance;
    float maxChannel = max(color.r, max(color.g, color.b));
    float gamutLimitedScale = 1.0 / max(maxChannel, 1.0e-8);
    return color * min(scale, gamutLimitedScale);
}

float srgbDecodeChannel(float code) {
    return code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4);
}

float srgbEncodeChannel(float linearValue) {
    return linearValue <= 0.0031308
            ? 12.92 * linearValue
            : 1.055 * pow(linearValue, 1.0 / 2.4) - 0.055;
}

float3 srgbDecode(float3 code) {
    return float3(srgbDecodeChannel(code.r), srgbDecodeChannel(code.g), srgbDecodeChannel(code.b));
}

float3 srgbEncode(float3 linearColor) {
    return float3(
            srgbEncodeChannel(linearColor.r),
            srgbEncodeChannel(linearColor.g),
            srgbEncodeChannel(linearColor.b));
}

float3 displayGammaSdr(float3 displayColor) {
    float3 code = clamp(displayColor, 0.0, 1.0);
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return code;
    }
    return clamp(srgbEncode(luminanceGamma(srgbDecode(code), BT709_LUMA)), 0.0, 1.0);
}

static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

float3 pqDecode(float3 code) {
    float3 p = pow(clamp(code, 0.0, 1.0), float3(1.0 / PQ_M2));
    float3 y = max((p - PQ_C1) / max(PQ_C2 - PQ_C3 * p, float3(1.0e-6)), float3(0.0));
    return pow(y, float3(1.0 / PQ_M1)) * 10000.0;
}

float3 pqEncode(float3 nits) {
    float3 y = pow(max(nits, float3(0.0)) / 10000.0, float3(PQ_M1));
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), float3(PQ_M2));
}

float3 displayGammaHdr(float3 pqColor) {
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return clamp(pqColor, 0.0, 1.0);
    }
    float peak = max(pc.hdrPeakNits, 1.0);
    float3 normalizedNits = clamp(pqDecode(pqColor) / peak, 0.0, 1.0);
    float3 adjustedNits = peak * luminanceGamma(normalizedNits, BT2020_LUMA);
    return clamp(pqEncode(adjustedNits), 0.0, 1.0);
}

// Exposure -> log2 shaper -> trilinear LUT fetch. The LUT already bakes in ACES 2.0's gamut mapping to
// BT.709 and the sRGB OETF, so its output goes straight to outputImage. Single mip (RtToneLut), so an
// explicit level-0 SampleLevel is exact, not an approximation.
float3 tonemap(float3 lookedAcesCg) {
    float3 uvw = shaperEncode(lookedAcesCg);
    return displayGammaSdr(toneLut.SampleLevel(lutTexCoord(uvw, pc.lutSize), 0.0).rgb);
}

// Same shaper as tonemap() but sampling the HDR LUT, whose range is already PQ-encoded BT.2020 -- no
// separate encode or UI-brightness mapping needed.
float3 tonemapHdr(float3 lookedAcesCg) {
    float3 uvw = shaperEncode(lookedAcesCg);
    return displayGammaHdr(hdrToneLut.SampleLevel(lutTexCoord(uvw, pc.lutSize), 0.0).rgb);
}

// AgX display transform: an analytic alternative to the baked ACES 2.0 output LUT, used for the advanced
// grading. Its signature is a slow highlight rolloff that keeps bright colors legible instead of clipping
// them, and it needs no look package to be authored.
//
// The matrices are pre-composed for this pipeline's ACEScg input: OCIO's ACES 2.0 ACEScg/AP1/D60 ->
// linear BT.709/D65 rows (shared with debug_present) followed by AgX's BT.709 -> APM input transform
// with the Abney correction folded in, and APM -> linear BT.709 on the way out. They are stored as rows
// and applied with dot() so the build's column-major matrix layout can't transpose them.
static const float AGX_MID_GREY = 0.18;
static const float AGX_EV_RANGE = 14.0;

static const float3 AGX_FROM_ACESCG_R = float3( 1.42335736, -0.44430356,  0.02108223);
static const float3 AGX_FROM_ACESCG_G = float3(-0.04514778,  0.96591471,  0.07919589);
static const float3 AGX_FROM_ACESCG_B = float3( 0.02990423, -0.04710990,  1.01715782);

static const float3 AGX_TO_BT709_R = float3( 1.19687901, -0.09802088, -0.09902974);
static const float3 AGX_TO_BT709_G = float3(-0.05289685,  1.15190313, -0.09896118);
static const float3 AGX_TO_BT709_B = float3(-0.05297164, -0.09804345,  1.15107367);

// APM has imaginary primaries, and the APM -> BT.709 output matrix can go negative the same way, so both
// ends need a fit: AgX encodes with log2, where a negative channel becomes NaN and renders as a black hole
// in exactly the wide-gamut foliage and sky colors this transform is here to preserve. Scaling chroma about
// the brightest channel keeps neutrals exact and is continuous at the switch-off point (lo == 0 leaves the
// color untouched), so no contour appears at the gamut boundary.
float3 fitToPositiveGamut(float3 color) {
    float hi = max(color.r, max(color.g, color.b));
    float lo = min(color.r, min(color.g, color.b));
    if (lo >= 0.0) {
        return color;
    }
    float scale = hi / max(hi - lo, 1.0e-7);
    return hi + scale * (color - hi);
}

// AgX's published default contrast curve, as the 6th-order polynomial fit used by the real-time versions.
float3 agxContrast(float3 x) {
    return (((((15.5 * x - 40.14) * x + 31.96) * x - 6.868) * x + 0.4298) * x + 0.1191) * x - 0.00232;
}

float3 agxDisplayLinear(float3 exposedAcesCg) {
    float3 scaled = max(exposedAcesCg, float3(0.0)) * max(pc.agxGain, 1.0e-3);
    float3 apm = fitToPositiveGamut(float3(
            dot(scaled, AGX_FROM_ACESCG_R),
            dot(scaled, AGX_FROM_ACESCG_G),
            dot(scaled, AGX_FROM_ACESCG_B)));

    float halfRange = AGX_EV_RANGE * 0.5;
    float3 stops = clamp(log2(max(apm, float3(1.0e-7)) / AGX_MID_GREY), -halfRange, halfRange);
    float3 bt709 = agxContrast((stops + halfRange) / AGX_EV_RANGE);

    return fitToPositiveGamut(float3(
            dot(bt709, AGX_TO_BT709_R),
            dot(bt709, AGX_TO_BT709_G),
            dot(bt709, AGX_TO_BT709_B)));
}

// AgX ends at linear BT.709, so the sRGB OETF is applied here before the display-gamma stage, which -- like
// tonemap() -- works on display code values. Without it the Gamma slider would exponentiate linear light.
//
// Saturation scales chroma about BT.709 luminance in linear light, before the OETF, so it preserves hue and
// the luminance the exposure was matched to. Below 1 it is a desaturate; above 1, negative channels are
// floored rather than allowed through, since log-domain negatives would encode as NaN.
float3 agxTonemap(float3 exposedAcesCg) {
    float3 linearBt709 = agxDisplayLinear(exposedAcesCg);
    float luminance = dot(max(linearBt709, float3(0.0)), BT709_LUMA);
    float3 saturated = max(lerp(float3(luminance), linearBt709, pc.agxSaturation), float3(0.0));
    return displayGammaSdr(srgbEncode(saturated));
}

// Purkinje shift: rods peak near 507 nm while cones peak at 555 nm, so as the eye adapts to dim
// light a colour's apparent brightness moves from its photopic toward its scotopic luminance.
// Saturated red loses nearly everything while blue-green holds, which is why red flowers and
// torchlight go pale and drab at night while water and sky stay luminous.
//
// Applied to the exposed scene-linear triple as a UNIFORM scale, so the channel ratios survive: a
// poppy becomes a dark poppy rather than a cyan one. Both display paths read the result -- AgX and
// the look-plus-ACES-LUT chain on the SDR side, the PQ LUT on the HDR side -- and it runs before the
// tone curve, so a boosted blue enters the shoulder exactly as a brighter pixel of that colour
// would instead of clipping past it.
//
// The three colourimetric constants are baked by tools/bake_scotopic_basis.py from CIE 018:2019
// Tables 2 and 6 and MUST match its output: a drifted coefficient re-tints the night silently
// rather than failing. V'(lambda) is not in the span of the CIE 1931 colour-matching functions, so
// the scotopic row is a colourimetric approximation that cannot know the spectrum behind an RGB
// triple; the metamerism limits and the measured error are in that script's header.
static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);
static const float3 ACESCG_SCOTOPIC_LUMA = float3(-0.18596501, 0.60931744, 0.38383450);
static const float PURKINJE_NEUTRAL_RATIO = 0.80718694;

// CIE 191:2010's photopic weight of the mesopic blend, m = 0.7670 + 0.3334 * log10(L) (BIPM
// 'Principles Governing Photometry' 2nd ed., Metrologia 56 G1-2019, eq 11), restated on the EV100
// scale that ExposureState.evScene is already reported on (RtSceneUnits: EV100 = log2(8 L), with a
// scene value of 1.0 being 1 cd/m2). m reaches 1 at the standard's own 5 cd/m2 photopic limit and 0
// at its 0.005 cd/m2 scotopic limit, so a sunlit scene is left exactly as it was and only dim ones
// shift.
static const float MESOPIC_EV_INTERCEPT = 0.46590980;
static const float MESOPIC_EV_SLOPE = 0.10036340;

float3 applyPurkinje(float3 exposedAcesCg) {
    float photopicWeight = clamp(MESOPIC_EV_INTERCEPT
            + MESOPIC_EV_SLOPE * exposureState[0].evScene, 0.0, 1.0);
    float photopic = dot(exposedAcesCg, ACESCG_LUMA);
    // The fit extrapolates saturated red to a NEGATIVE rod response. Rods do not see deep red at
    // all, so zero is the honest value there, and falling to it is most of what the effect looks
    // like.
    float scotopic = max(dot(exposedAcesCg, ACESCG_SCOTOPIC_LUMA), 0.0);
    // Scale-free by construction: both rows are linear in the triple, so the auto-exposure
    // multiplier the image already carries cancels and this ratio needs no absolute units.
    float ratio = scotopic / max(photopic, 1.0e-6);
    float scotopicWeight = 1.0 - photopicWeight;
    // Divided by the same expression at the working-space neutral, which therefore keeps its
    // brightness exactly: the shift redistributes brightness between colours without moving the
    // overall level that auto-exposure was matched to.
    float gain = (photopicWeight + scotopicWeight * ratio)
            / (photopicWeight + scotopicWeight * PURKINJE_NEUTRAL_RATIO);
    // 1 is the CIE response; the control scales the deviation from the neutral, and a pixel cannot
    // go below black however far it is pushed.
    return exposedAcesCg * max(1.0 + (gain - 1.0) * max(pc.purkinjeStrength, 0.0), 0.0);
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

    float4 rt = rtImage[pix];
    float exposure = max(exposureImage[int2(0, 0)], 0.0);
    float3 exposedAcesCg = max(rt.rgb * exposure, float3(0.0));
    exposedAcesCg += sampleBloom(pix, w, h) * max(pc.bloomStrength, 0.0);
    // Before either display transform, so all three grading paths and the PQ image shift together.
    if (pc.purkinjeEnabled != 0) {
        exposedAcesCg = applyPurkinje(exposedAcesCg);
    }

    if (pc.advancedGrading != 0) {
        outputImage[pix] = float4(agxTonemap(exposedAcesCg), 1.0);
        // The PQ HDR image has no AgX path; unlooked ACES 2.0 is the closest equivalent there.
        if (pc.hdrEnabled != 0) {
            hdrImage[pix] = float4(tonemapHdr(exposedAcesCg), 1.0);
        }
    } else {
        float3 lookedAcesCg = applyLook(exposedAcesCg);
        outputImage[pix] = float4(tonemap(lookedAcesCg), 1.0);
        if (pc.hdrEnabled != 0) {
            hdrImage[pix] = float4(tonemapHdr(lookedAcesCg), 1.0);
        }
    }
}

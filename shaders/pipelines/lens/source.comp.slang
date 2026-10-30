// The lens flare's light source: how much light overflows the lens at each point of the frame.
//
// A flare is not made of the frame's own colours but of the excess light the lens fails to focus, which is
// why only sources far above white bloom into stars and rings while a lit wall does not. This pass reduces
// the exposed image to that excess once, at an eighth of the display size, so the flare gather can read it
// hundreds of times per texel without touching the full-resolution image.
//
// That eighth is fixed rather than following the flare image's resolution setting: a ray is as wide as the
// source texels that launch it, so a finer source would fan every star out into a wedge.
//
// Runs before the flare gather, which reads the texture this writes. The pair costs two small dispatches
// because the gather needs the *neighbourhood* of each texel: folding the box average into the gather would
// multiply its cost by the box size.

import display_common;
import source_bindings;

[[vk::push_constant]] LensSourcePush pc;

static const float3 ACESCG_LUMA = float3(0.2722287168, 0.6740817658, 0.0536895174);

// The box each output texel covers: 4x4 bilinear taps one display pixel apart, a 4-pixel footprint. The taps
// are dense enough that a sun smaller than an output texel still contributes its whole energy, which keeps the
// star from flickering as the camera moves, and tight enough that the source stays a compact point: a spike is
// as wide as the source that launches it, so a wide box would turn every star into a fan of fat wedges.
static const int BOX_TAPS = 4;
static const float BOX_STEP = 1.0;

// Exposed ACEScg luminance a surface must exceed before any of its light overflows the lens. Mid grey sits
// near 0.2 on this scale and exposed white near 1, so 6 is light the scene only produces in emitters, sun and
// specular: a lit wall lands at exactly 0 and cannot flare.
static const float HOT_THRESHOLD = 2.5;

// Soft ceiling on what a single source may put into the flare. Strictly increasing, so a brighter source still
// flares brighter, but a sun four orders of magnitude over white cannot outshine a lamp by that same factor --
// without it every flare of an exposed sun is an opaque white star.
static const float HOT_CEILING = 50.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint width, height;
    hotImage.GetDimensions(width, height);
    int2 pixel = int2(dispatchId.xy);
    if (uint(pixel.x) >= width || uint(pixel.y) >= height) {
        return;
    }

    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float2 step = pc.displayTexel * BOX_STEP;
    float3 sum = float3(0.0);
    for (int y = 0; y < BOX_TAPS; y++) {
        for (int x = 0; x < BOX_TAPS; x++) {
            float2 offset = (float2(x, y) - float(BOX_TAPS - 1) * 0.5) * step;
            sum += rtImage.SampleLevel(uv + offset, 0.0).rgb;
        }
    }

    float3 average = sum * (1.0 / float(BOX_TAPS * BOX_TAPS));
    float exposure = max(exposureImage[int2(0, 0)], 0.0);
    float3 exposed = max(average * exposure, float3(0.0));
    // Only the part of the radiance above the threshold flares, scaled onto the original colour so the
    // source keeps its hue instead of turning into a white mask.
    float luma = dot(exposed, ACESCG_LUMA);
    float above = max(luma - HOT_THRESHOLD, 0.0);
    float3 value = exposed * (above / max(luma, 1.0e-6));
    value = value / (1.0 + value / HOT_CEILING);
    hotImage[pixel] = float4(value, 1.0);
}

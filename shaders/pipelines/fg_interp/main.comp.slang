// Builds the frame halfway between the previous and current displayed frames by shifting each pixel back
// along its ray-traced motion vector and blending the two exposures.
//
// The guides are rendered at trace resolution while this pass runs at swapchain resolution, so a guide
// lookup goes uv -> guide texel, and the motion vector is expressed in guide pixels: dividing it by the
// guide extent turns it into uv units, which keeps the offset correct at any magnification.
import bindings;

struct FgInterpPush {
    int debugView;
};
[[vk::push_constant]] FgInterpPush pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    outputImage.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) return;

    int2 pix = int2(tid.xy);
    uint guideWidth, guideHeight;
    motionImage.GetDimensions(guideWidth, guideHeight);
    int2 guideSize = int2(guideWidth, guideHeight);

    float2 uv = (float2(pix) + 0.5) / float2(width, height);
    float3 current = currentImage.Load(int3(pix, 0)).rgb;

    int2 guidePix = clamp(int2(uv * float2(guideSize)), int2(0, 0), guideSize - 1);
    float2 motion = motionImage.Load(guidePix);

    // The guide pass stores (previous position - current position) in guide pixels, so the surface that
    // lands on this pixel now sat half that vector further along when the previous frame was captured.
    float2 previousUv = uv + 0.5 * motion / float2(guideSize);
    bool inRange = previousUv.x >= 0.0 && previousUv.y >= 0.0 && previousUv.x <= 1.0 && previousUv.y <= 1.0;
    float3 previous = inRange ? previousImage.SampleLevel(previousUv, 0.0).rgb : current;

    if (pc.debugView == 1) {
        // Motion magnitude normalised by the guide width, so a static scene is black and one guide pixel
        // of shift is a fixed grey. Direction is not shown here; use view 3 to judge alignment.
        float magnitude = length(motion) / float(max(guideSize.x, 1));
        outputImage[pix] = float4(float3(magnitude), 1.0);
    } else if (pc.debugView == 2) {
        outputImage[pix] = inRange ? float4(1.0, 1.0, 1.0, 1.0) : float4(1.0, 0.0, 0.0, 1.0);
    } else if (pc.debugView == 3) {
        outputImage[pix] = float4(previous, 1.0);
    } else {
        outputImage[pix] = float4(inRange ? 0.5 * (current + previous) : current, 1.0);
    }
}

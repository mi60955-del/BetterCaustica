// Rain streak, fragment stage: the streak's soft shape, the depth test, and a premultiplied write into
// the reconstructed scene image.
//
// There is no depth attachment on this pass — the streaks are drawn over the already-resolved scene and
// the only depth that exists is the guide buffer the trace wrote. That image holds the hardware depth a
// rasterizer would have stored (NDC z/w of the reversed-Z projection, near = 1, far = 0), so the test is
// the same comparison the hardware would do, just read from a texture: the streak is occluded whenever
// the scene's own depth is the larger of the two.
import rain;
import bindings;

[[vk::push_constant]] RainPush pc;

struct FragmentInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
    [[vk::location(1)]] float opacity;
    [[vk::location(2)]] float3 radiance;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
    // The streak's shape is resolved before either discard below: the across-axis footprint is a
    // derivative, and a derivative taken after a conditional discard is undefined for the fragments of a
    // 2x2 quad whose neighbours did not survive the test.
    //
    // Across the streak: a triangular profile, widened by the pixel's own footprint so a quad narrower
    // than a texel still covers a fraction of it instead of aliasing into a dashed line. That fraction
    // is what makes a distant streak fade out as its coverage shrinks rather than flicker between frames.
    float across = input.uv.x * 2.0 - 1.0;
    float footprint = max(fwidth(input.uv.x) * 2.0, 1.0e-5);
    float coverage = saturate((1.0 - abs(across)) / footprint);

    // Along the streak: both ends soften, because the quad's ends are an artefact of the quad and a hard
    // edge there reads as a line of dashes hanging in the air.
    float taper = saturate(input.uv.y * 8.0) * saturate((1.0 - input.uv.y) * 5.0);
    float alpha = saturate(coverage * taper * input.opacity * pc.intensity * pc.brightness);
    if (alpha <= 0.0) {
        discard;
    }

    // The guide is render-res while this pass is display-res, so the nearest sampler picks the texel
    // whose surface this pixel sits on. Sampling is skipped for the fragments already faded out, which is
    // most of them: the quads are two triangles each and the streaks occupy a small part of that area.
    float2 screenUv = input.position.xy / max(pc.displaySize, float2(1.0, 1.0));
    float sceneDepth = depthImage.SampleLevel(screenUv, 0.0).r;
    if (input.position.z <= sceneDepth) {
        discard;
    }

    // Premultiplied, matching the pass's blend state: the radiance is what the streak adds and alpha is
    // how much of what is behind it survives.
    return float4(input.radiance * alpha, alpha);
}

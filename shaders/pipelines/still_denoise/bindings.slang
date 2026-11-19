// The still render's spatial denoise: the accumulated mean, the surfel guides the world pass wrote for the
// frame that produced it, and the filtered result. Every slot is a storage image the pass loads unfiltered,
// so no sampler is needed and the pass has no resolution assumptions beyond the guides being one texel per
// mean texel.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> sourceImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> guideNormal;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> guideAlbedo;
[[vk::binding(3, 0)]] [format("r32f")] public RWTexture2D<float> guideDepth;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;

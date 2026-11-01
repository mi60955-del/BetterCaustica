// The recombine's inputs and outputs. The compose pair sits on both sides: sampled nowhere and stored
// as the pass's own in-place write target (each thread loads and stores exactly its own texel). The
// gather's half-res fields arrive through the linear sampler the host binds in these combined slots.
[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> hdrOutputImage;
[[vk::binding(2, 0)]] public Sampler2D farImage;
[[vk::binding(3, 0)]] public Sampler2D nearImage;
[[vk::binding(4, 0)]] public Sampler2D hdrFarImage;
[[vk::binding(5, 0)]] public Sampler2D hdrNearImage;

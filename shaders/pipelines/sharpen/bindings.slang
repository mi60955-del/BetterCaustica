// The display mapper's unsharpened output (compose images) on the sampled side, and the finished images the
// present paths consume on the storage side. Both channels are rgba8/rgba16f exactly as the display mapper
// wrote them, so this pass never converts: it only adds a high-pass term in the same code-value space.
[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> hdrOutputImage;
[[vk::binding(2, 0)]] public Sampler2D sourceImage;
[[vk::binding(3, 0)]] public Sampler2D hdrSourceImage;

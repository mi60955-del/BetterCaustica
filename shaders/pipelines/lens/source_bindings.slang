// Bindings of the lens chain's source pass (source.comp.slang). The exposed image is read through a sampler
// because the box average needs bilinear taps between texels; exposure stays a storage-image read so it
// needs no filtering.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> hotImage;
[[vk::binding(1, 0)]] public Sampler2D rtImage;
[[vk::binding(2, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;

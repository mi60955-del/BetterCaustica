// Bindings of the flare gather (main.comp.slang). The hot field and the bloom glow are sampled, not
// stored: both are read at many offsets per texel and need bilinear filtering.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> flareImage;
[[vk::binding(1, 0)]] public Sampler2D hotImage;
[[vk::binding(2, 0)]] public Sampler2D glowImage;

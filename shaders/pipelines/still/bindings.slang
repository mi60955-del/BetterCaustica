// The accumulated still render: a sum of the scene-linear image the display chain consumes (rrOutput),
// and the display-res mean the chain is then run on. The accumulator is float because it holds a sum of
// radiance, not a sum of already tone-mapped pixels: averaging the finished display image is biased by
// the tone curve, which is what made a long exposure darken and desaturate as samples grew.
[[vk::binding(0, 0)]] [format("rgba32f")] public RWTexture2D<float4> accumulatorImage;
[[vk::binding(1, 0)]] public Sampler2D sourceImage;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
